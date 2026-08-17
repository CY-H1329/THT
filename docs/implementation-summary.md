# Autonomous movement — what was implemented

Summary of the work done on the Memory-FPS agent. Scope: **movement only** —
the agent explores and maps the world on its own from pixels. `Agent.answer()`
is still a stub; the memory/QA layer is the next piece of work.

Design deep-dive: [navigation.md](navigation.md).

---

## 1. Starting point

The repo contained the environment (`memory_fps_env/`) and one agent module,
`agent/ocr.py` (HUD text reader). There was no navigation, no map, no agent
entry point.

Three properties of the environment were established by reading the world
generator and re-measuring from pixels — everything else is built on them:

| Property | Why it matters |
|---|---|
| The floor is the only achromatic surface (b/w checkerboard); walls and props are coloured | "First saturated pixel scanning up a column" = the nearest obstacle's contact point |
| Camera is fixed (`domain_rand=False`): height 1.5 m, fov_y 60°, pitch 0; step 0.15 m; turn 15° | Pixel row converts to metres exactly; odometry is near-free |
| Rooms are identical squares on a grid; doorways are always 2 m at the **midpoint** of a shared wall; room names are unique and shown on the HUD | Knowing one number (cell size) makes every door position in the world predictable |

---

## 2. Modules added

| File | Lines | Role |
|---|---:|---|
| `agent/__init__.py` | 22 | `Agent` entry point the evaluator imports |
| `agent/geometry.py` | ~200 | Camera model, IPM depth sensor, collision detection, colour helpers |
| `agent/localize.py` | ~150 | 360° scan ↔ room-rectangle matching; door/obstacle extraction |
| `agent/mapper.py` | ~290 | Room graph on the cell grid, routing, map-predicted range |
| `agent/pose.py` | ~75 | Dead-reckoned pose |
| `agent/perception.py` | ~110 | Per-step frame → `Percept`, with shared/cached computation |
| `agent/enemies.py` | ~190 | Mob detector (bearing, distance, confidence) |
| `agent/explorer.py` | ~870 | Exploration state machine, driving, combat policy |
| `agent/ocr.py` | ~300 | (existing) HUD reader — optimised, see below |

Plus tooling outside the submission package:

- `tmp/watch_agent.py` — live viewer with speed control, pause/step, and full
  run recording.
- `tmp/run_explore.py` — scores the agent's map against env ground truth.
- `docs/navigation.md` — design notes.

---

## 3. Sensing

**Depth by inverse perspective mapping** (`geometry.py`). For a floor contact
point at row `y`, column `x`:

```
dz   = cam_height · f / (y − horizon_row)     f = (H/2)/tan(fov_y/2) = 207.85 px
dist = dz · sqrt(1 + u²)                      u = (x − cx)/f
```

Pitch is 0, so `horizon_row` is exactly the image centre. Verified against a
ground-truth ray cast: **median error 0.1 m, max 0.5 m** over 2-15 m, across
all 12 wall colours.

Hard limit: the bottom image row maps to **2.6 m**. Anything closer occludes
the floor completely, leaving no contact row to measure. The camera cannot
pitch down, so this is resolved by two other channels — the map's predicted
range, and collision feedback.

**Collision feedback.** Miniworld movement is all-or-nothing. Comparing the
lower half of consecutive frames separates moved from blocked with a huge
margin — mean absolute difference ≥ 31 versus ≤ 0.03.

**HUD OCR speedup.** The existing reader cost 255 ms/frame. Grouping glyphs by
pixel width and comparing a whole group in one broadcast, then locking the
vertical offset after the first character, brought it to **21 ms** with
byte-identical output on an 85-frame regression set. It is now called on a
schedule and skipped when the HUD bar is unchanged; heading is integrated from
actions instead of read.

---

## 4. Localization and mapping

Position comes from **scan matching**: for a candidate position in a
`cell × cell` square, each ray's expected range is closed-form, so the agent
scores a 0.25 m grid of candidates against ~320 rays and takes the most
inliers. Median error **0.15 m**, ~12 ms per match.

This beats averaging wall distances because standing in front of a doorway
sends *every* ray toward that wall through the opening — the average is not
noisy but wrong. Matching discards those as outliers; they become the door
detector instead.

Cell size is recovered the same way (candidates 6-10 m, scores accumulated
over the first three rooms) and was correct on every seed tested.

The map is a dict of grid cell → room with wall states
(`unknown / wall / door / locked`), wall colour and static obstacle points.

---

## 5. Exploration

```
SCAN (24 × 15°) → update map → nearest unexplored door → DOORCHECK → GOTO
      → ENTER → SCAN … ↓ nothing left: BFS to nearest frontier … ↓ none: sentry
```

- **SCAN** completes when all 24 heading buckets are seen, not after 24 turns,
  so a combat interruption cannot corrupt it.
- **DOORCHECK** faces a door from 2.3 m before committing — this exists solely
  because of the locked door.
- **ENTER** takes the new room's grid cell from *which door was used*, never
  from the drifting position estimate.
- Failed traversals are blacklisted, so a mis-detected door cannot trap it.
- **Obstacle avoidance** is a committed sidestep: turn 60° toward the freer
  side and drive ~2.4 m before re-aiming. Committing to translation matters —
  a rotate-only version deadlocks, because within 2.6 m every heading reports
  "near".

**The locked door** (one per episode) is identified by, in order of
reliability: the hint banner (it fires only at that door), being blocked at a
doorway midpoint, and the lock panel's colour as a supporting cue.

---

## 6. Enemies

`agent/enemies.py` recognises the mob itself: primary-palette saturation,
**contact with the floor** (the cue that separates it from wall photographs,
which are often *more* saturated but hang at 0.75-2.25 m), height consistent
with 1.5 m / 2.0 m, and 2-3 stacked colour bands.

Measured over 216 frames on 7 seeds against ground truth: **bearing error
0.6° mean / 1.5° max**, distance error 0.2 m, 73% recall per frame, 0.11 false
positives per frame in attack range, 1.5 ms per call.

Combat policy: attack what is identified, never blind-spin, sweep *while
attacking* when hit from outside the field of view, budget stationary combat
(70 steps) and then disengage — the agent moves ~8× faster than an enemy, so
standing still is the losing move.

---

## 7. Bugs found and fixed

Each was found by instrumenting against ground truth, not by inspection.

| Bug | Evidence | Fix |
|---|---|---|
| Pose snapped 1.8 m past the door on entry | Belief vs truth diverged 1.75 m at one step | Removed the snap; re-anchor by scan matching |
| Rotate-only avoidance deadlocked | 4800 steps at one spot, heading oscillating ±15° | Committed sidestep that translates |
| Depth sensor read walls as obstacles | Dodged every wall it approached | Compare against map-predicted range |
| Retried the locked door forever | ~800 steps per attempt, repeatedly | Banner + blocked-at-midpoint ⇒ mark locked |
| Scan marked unseen walls as solid | Coverage collapsed to 3-5 rooms | Decide only on positive evidence; probe the rest |
| Attacked props while enemies killed it | 345 of 358 attacks whiffed | Width filter, wall-colour rejection, static memory |
| Spun without ever attacking | User-observed death; three separate code paths | Sweep attacks while turning; relax filters when hit |
| **Zero valid hits** | 413 attacks, all outside the ±20° cone, 0 kills | Enemies attack from outside the FOV — sweep to find them |
| Mob in the cone ignored | Recorded frame: mob at 1.8 m scored 0.80, threshold 0.95 | Score by cues *available* — unscanned room no longer caps it |
| Unbounded sweeping | HP 13 → 1 while stationary 130 steps in a doorway | Stationary-combat budget, then disengage |

---

## 8. How to run

```bash
# watch it play (recording is on by default)
.venv/bin/python tmp/watch_agent.py 42 --fps 8
#   SPACE pause · n single-step · [ ] speed · q quit

# score the map against ground truth
.venv/bin/python tmp/run_explore.py --limit 120 0 1 7 42 99

# OCR regression (expects 85/85)
.venv/bin/python tmp/test_ocr.py
```

Each run writes `tmp/runs/<timestamp>_seed<N>/` containing `frames/` (every
10th step plus every event, tagged in the filename: `hpdrop`, `bump`, `room`,
`state`, `final`), `log.jsonl` (per-step state) and `summary.json` (final map,
stats, hints, locked door).

---

## 9. Current results

| Metric | Result |
|---|---|
| Cell size / room placement / door edges | correct; **0 wrong edges on any seed** |
| Depth / localization / enemy bearing | 0.1 m / 0.15 m / 0.6° |
| `agent.act` | 7.5 ms mean, ~67 steps/s (budget is 5 s) |
| Rooms mapped | 3-8 of 9-12 |
| Survival | improved substantially, still not reliable |

---

## 10. Limitations and next steps

Honest state: **mapping is reliable; survival is not**, and coverage tracks
survival almost exactly.

1. **Cut standing time** — scan while approaching the room centre, and shorten
   re-scans of known rooms to a half sweep. Highest-leverage survival fix with
   no coverage cost.
2. **Diagnose seed 99** — still dies at ~660 steps with 500 attacks all
   off-cone and 0 kills. Different failure from the one fixed today.
3. **Capture per-room observations for QA** during the scan that already
   happens: wall photographs per wall index, prop and enemy appearance. The
   geometry needed to attribute an image to a wall is already computed.
4. **Key retrieval** — parse the recorded hint against the map to pick the
   target room, fetch the key, return to the locked door. The hint text and
   the door's location are already recorded.
5. **Then the memory/QA layer** — `answer()` is a stub; what the explorer
   records today (room names, grid layout, wall colours, door graph, hints,
   HP events) is scaffolding for it, not a memory.
