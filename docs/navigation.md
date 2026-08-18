# Autonomous movement: design notes

How the agent walks around on its own, using nothing but the `(240, 320, 3)`
RGB frame. Written as a working note; the numbers come from the dev harness
(`tmp/run_explore.py`) and the sensor tests described below.

## What the environment hands us for free

Three properties of `memory_fps_env` do most of the heavy lifting. They were
confirmed by reading the world generator and re-measuring from pixels:

1. **The floor is the only achromatic surface.** Floors are a black/white
   checkerboard (`floor_tiles_bw`); walls come from a 12-colour palette and
   3D props are coloured meshes. So "first saturated pixel scanning up a
   column" is the contact point between the floor and the nearest obstacle.
2. **The camera is fixed.** `domain_rand=False`, so `cam_height = 1.5 m`,
   `fov_y = 60°`, `pitch = 0`, and one forward step is exactly `0.15 m`,
   one turn exactly `15°`.
3. **Rooms are a uniform grid.** Every room in an episode is the same square
   cell, aligned to a grid (`origin = (gx·cell, gz·cell)`), neighbours share a
   flush wall, and every doorway is a 2 m opening at the **midpoint** of the
   shared wall. There are no corridor rooms. Room names are unique per
   episode and printed on the HUD.

(3) is the reason this agent is a metric planner rather than a reactive
wanderer: once `cell` is known, every door location in the world is
predictable, and exploration reduces to a graph search over grid cells.

## Sensing

### Depth by inverse perspective mapping — `agent/geometry.py`

For a floor contact point at pixel row `y`, column `x`:

```
dz   = cam_height · f / (y − horizon_row)      f = (H/2)/tan(fov_y/2) = 207.85 px
dist = dz · sqrt(1 + u²)                       u = (x − cx)/f
```

`horizon_row` is exactly the image centre because pitch is 0. The scan is
restricted to rows below the horizon, which makes it impossible to run up
into the (also achromatic) concrete ceiling.

*Measured*: against a ground-truth ray cast through `wall_segs`, the median
error is 0.1 m and the max 0.5 m over the 2-15 m band, across all 12 wall
colours and 7 seeds.

*Known limit*: the bottom image row maps to **2.6 m**
(`FLOOR_MIN_RANGE = h·f/119.5`). Anything closer occludes the floor
completely, so there is no contact row left to measure and the sensor
saturates to "something is near" with no distance. Pitch is fixed, so no
camera-side fix exists. Everything closer than 2.6 m is therefore resolved
by two other channels: the map's predicted range, and collision feedback.

### Collision feedback

Miniworld movement is all-or-nothing: a blocked `MOVE_FORWARD` moves the
agent zero distance. Comparing the lower half of consecutive frames
separates the two cases with an enormous margin — mean absolute difference
≥ 31 when the agent moved, ≤ 0.03 when it was blocked (the checkerboard is a
strong texture, and one step shifts it a lot). This is the ground truth for
odometry and for discovering obstacles the depth sensor cannot resolve.

### HUD text — `agent/ocr.py`

Glyph-template matching against Pillow's bundled font, which the env uses to
draw the HUD. Gives room name, remaining seconds, HP, heading and the
`[KEY]` flag. It was 255 ms/frame; grouping glyphs by pixel width and
comparing a whole width-group in one broadcast, then locking the vertical
offset after the first character, brought it to **21 ms** with byte-identical
output on an 85-frame regression set.

21 ms is still the most expensive thing in the loop, so `agent/perception.py`
reads it on a schedule (every 12 steps normally, every 4 while crossing a
doorway) and skips the call entirely when the HUD bar pixels are unchanged.
Heading does not need OCR at all — turns never fail, so it is integrated from
the actions and merely re-synced when the HUD is read.

## Localization — `agent/localize.py`

Position comes from matching a 360° scan against the room rectangle. For a
candidate position `(x, z)` inside a `cell × cell` square, each ray's expected
range is a closed form, so scoring a candidate is one vectorised comparison.
The agent evaluates a 0.25 m grid of candidates against ~320 rays and takes
the position with the most inliers (|measured − expected| ≤ 0.4 m).

Why not simply average wall distances: when the agent stands in front of a
doorway, *every* ray toward that wall passes through the opening, so the
"distance to that wall" statistic is not merely noisy, it is wrong. Scan
matching drops those rays as outliers and fits on the other three walls.
Rays that overshoot a wall are, in fact, the door detector; rays that fall
short are props or enemies.

*Measured*: median position error **0.15 m**, mean 0.40 m, p90 1.05 m over 28
room/seed combinations; ~12 ms per match.

Cell size is recovered the same way — the match is run for each candidate
size 6..10 and scores are accumulated across the first three rooms, which
fixes the ~11% of single-room estimates that picked a neighbouring size.

Between scans the pose is dead-reckoned: heading from the HUD, `0.15 m` per
successful forward step. Every room entry re-anchors it with a fresh match,
so drift never accumulates across rooms.

## Mapping and exploration — `agent/mapper.py`, `agent/explorer.py`

The map is a dict of grid cell → room (name, wall states, wall colour, static
obstacle points). Wall states are `unknown / wall / door / locked`. Because
doors are always at wall midpoints, a door's world position, its approach
point and its exit point are all derived from the cell index.

The loop is the DFS in the task diagram:

```
SCAN (24 × 15°) → update map → nearest unexplored door → DOORCHECK → GOTO → ENTER → SCAN …
                                     ↓ none left
                            route to nearest frontier room (BFS over known doors)
                                     ↓ none left
                                  sentry posture
```

- **SCAN** finishes when all 24 heading buckets have been observed, not after
  24 turns, so an interruption (combat) does not corrupt it.
- **DOORCHECK** faces the door from 2.3 m before committing. This exists
  because of the locked door.
- **ENTER** waits for the HUD room name to change; the new room's grid cell is
  taken from the traversal (which door we walked through), never from the
  drifting position estimate.
- A traversal that fails is blacklisted for 4000 steps and, after three
  failures, the wall is recorded as solid — so a mis-detected door cannot
  trap the agent in a retry loop.

### Obstacle avoidance

Steering is a bug-style local planner. Aim at the waypoint, walk; when
blocked — by a bump, or by the depth sensor reporting something near where
the map predicts open space — commit to a **sidestep**: turn 60° toward
whichever side has more clearance and drive ~2.4 m before re-aiming.

The commitment matters. An earlier version only rotated when it saw an
obstacle, which deadlocks: within 2.6 m the sensor cannot resolve distance,
so every heading reports "near" and the agent oscillates ±15° forever
without translating. One seed burned 4800 steps that way.

### The locked door

Every episode has exactly one locked doorway, which cannot be passed without
the key. It is recognised by three signals, in order of reliability:

1. **The hint banner.** It appears only when touching the locked door, so if
   it fires while the agent is near a door, that door is locked.
2. **Being blocked at a doorway midpoint** — an open doorway never blocks.
3. The yellow lock panel's colour, as a supporting cue. Lighting scales the
   texture unpredictably (measured (153,140,78) to (196,180,100) for a
   source colour of (235,215,120)), and the `olive` wall colour is nearly
   the same hue, so this is matched by channel ratio plus the dark lock
   glyph inside the panel — and it is never the sole evidence.

### Enemies — `agent/enemies.py`

Enemies are recognised directly rather than inferred as "something unexpected
is near". A mob is six MiniWorld primary-palette boxes plus a face quad, and
three cues identify it:

- **Saturation.** Mob boxes render at 100-255; the room wall palette never
  exceeds ~75 under the same lighting.
- **Floor contact** — the decisive cue. Wall photographs are real images and
  can be *more* saturated than a mob, but they hang at 0.75-2.25 m; a mob's
  colour runs down to the floor. The detector samples the band just above the
  floor-contact row from the same IPM pass.
- **Height and layering.** The silhouette top row plus the IPM distance gives
  a height, which rejects barrels (1.0 m), cones (0.6 m) and duckies (0.4 m);
  mobs are 1.5 m or 2.0 m. Closer than 2.6 m the floor contact is off-screen,
  so the relation is inverted — known mob height plus head-top row gives the
  distance. Shirt/pants/skin also stack into 2-3 colour bands, which single-
  colour props do not.

*Measured* over 216 frames on 7 seeds against ground-truth enemy positions:
bearing error **0.6° mean, 1.5° max**, distance error 0.2 m, 73% recall per
frame, 0.11 false positives per frame within attack range, 1.5 ms per call.

Confidence is scored as "of the cues that could be checked, how many passed",
so an unscanned room (wall colour unknown) does not cap the score — that bug
cost an episode 12 of 13 HP (see the case study below).

### Combat policy

Fighting exists to protect observation time. The agent moves ~8× faster than
an enemy in simulated terms, so standing still is what kills it:

- Attacks have no cooldown and enemies have 1-4 HP, so an identified enemy
  dies in a handful of steps.
- On taking damage it does not blind-spin: it engages a confident detection,
  and otherwise sweeps *while attacking* so a rotation always does damage.
- Stationary combat is budgeted (70 steps per encounter). Past that it
  disengages for 200 steps and returns to the route, still firing at anything
  already inside the ±20° cone but refusing to stop and aim.
- With the map complete it parks in a room corner and turns slowly, which
  limits the arc it must watch.

## Cost

`agent.act` averages **5.6 ms** (p50 3.8 ms, p95 24 ms on the frames where
OCR runs), against ~6.8 ms for `env.step` — about 81 steps/s, i.e. ~48,000
steps in a 600 s episode. The per-call budget is 5 s, so there is a large
margin for a heavier memory/QA stage later.

## Measured behaviour (public seeds, dev harness)

`tmp/run_explore.py` runs the agent against the real env and compares its map
to `env._graph` ground truth.

| metric | result |
|---|---|
| cell size recovered | correct on every seed |
| room→grid-cell placement | correct on every mapped room |
| door edges | **0 wrong edges on any seed tested** |
| locked door | identified and avoided |
| depth sensor | median error 0.1 m, max 0.5 m (2-15 m) |
| localization | median 0.15 m, p90 1.05 m |
| enemy bearing | 0.6° mean, 1.5° max |
| `agent.act` cost | 7.5 ms mean → ~67 steps/s |
| rooms mapped | 3-8 of 9-12, varies with survival |
| survival | improved but not reliable (seed 99 still dies early) |

Coverage tracks survival closely: runs that avoid an early enemy encounter map
most of the world; runs that meet one near spawn die with 3-4 rooms.

## Case study: one recorded episode

`tmp/runs/20260818_004854_seed42` is worth reading as a worked example of the
tooling. The agent mapped four rooms cleanly, then froze at exactly
`(30.00, 4.95)` — a doorway plane — for 130 steps, taking a hit every 24 steps
(the enemy cooldown) until HP fell 13 → 1.

The event-tagged frame `f_000336_hpdrop.png` showed a mob filling the centre of
the view at ~1 m, inside the attack cone. Replaying the detector on that exact
frame returned bearing +11.4°, distance 1.80 m, **score 0.80** — while the
under-attack path demanded 0.95. The room had not been scanned yet, so the
wall-colour cue was unavailable and 0.95 was unreachable. Two fixes followed:
proportional scoring, and the stationary-combat budget.

## Limitations

- **Survival is the binding constraint, and it is not solved.** Enemies chase
  at 1.5 m/s while the agent covers ~12 m/s of simulated distance, so
  disengaging is cheap — but the agent still stands still to scan (24 turns
  per room) and to aim, and that is where the damage lands. Deaths cluster in
  the first minute when an enemy spawns near the route.
- Position is only re-anchored on a scan; a long traversal with several
  sidesteps can drift ~1 m before arrival. It self-corrects on the next
  scan, but a door approach begun from a drifted pose can miss the opening.
- Scan-based door classification is 58/64 correct in isolation (0 missed
  doors, 1 false door, 5 undecided) but degrades when props block the view of
  a wall's midpoint. Undecided walls are physically probed, which costs
  ~150-300 steps each.
- Room identity relies on HUD names being unique, with prefix matching for
  names the HUD truncates. Two rooms sharing a long prefix would be
  conflated; the grid cell is a cross-check but the name wins.
- The key is never retrieved, so the room behind the locked door is never
  seen. The hint text and the door's location are recorded for that step.
- `Agent.answer()` is still a stub — this work covers movement only. What the
  explorer records today (room names, grid layout, wall colours, door graph,
  hint text, HP events) is a scaffold for the memory layer, not a memory
  layer.

## The premature-parking bug

A watched run of seed 7 ended with the agent spinning in place in the third
room while the episode clock ran. It had decided it was finished. Dumping its
map at that moment:

```
Sable Vestibule (0,0)  E=door  S=wall  W=door   N=door
Vermilion Galle (1,0)  E=door  S=wall  W=door   N=door
Russet Wing     (2,0)  E=locked S=wall W=door   N=wall
```

Three of those doors — Sable W, Sable N, Vermilion N — lead to cells that had
never been visited. Yet `frontier_rooms()` returned `[]`, so `_choose_goal`
fell through to `_park_pick()` and switched to sentry mode with two thirds of
the world unseen.

The cause was that "has work left" was defined as *unscanned, or has a wall of
unknown state*. A room that is fully scanned with all four walls classified
failed both tests — even when one of its doors opened onto a cell that was
never entered. Those cells are not in `self.rooms`, so iterating the map could
never reach them either. The one place that did check for a door to an
unvisited neighbour was step 1 of `_choose_goal`, which only looks at the
**current** room; once the agent walked away, that door became invisible to
the planner forever.

`frontier_rooms()` now also counts a room with an open door onto an unvisited
or unscanned cell, and takes the explorer's blacklist filter so a door that
repeatedly failed to traverse cannot keep re-attracting the planner.

Effect over ten public seeds (40 s each): rooms visited **50 → 66**, enemies
killed 29 → 35. Deaths rose 2 → 5, which is the direct consequence of no
longer standing in a safe corner: the agent now spends the second half of the
episode moving through contested rooms. Two of the five deaths (seeds 1 and
42) happened after covering 7/9 and 8/9 rooms and cost almost nothing; two
(seeds 0 and 99) were early and did cost coverage. Since the README grades
what you can answer rather than survival, more total observation is the right
trade — but the early deaths are the strongest argument for the flee-at-low-HP
step below.

## Next steps, in the order I would do them

1. **Cut standing time.** Scan while approaching the room centre rather than
   from a standstill, and shorten re-scans of already-known rooms to a half
   sweep. This is the highest-leverage survival fix and costs no coverage.
2. **Flee explicitly at low HP** toward an already-cleared room instead of
   continuing the route through contested space.
3. **Capture per-room observations for QA** during the scan that already
   happens: wall photographs per wall index, prop appearance, enemy
   appearance. The scan geometry needed to attribute an image to a wall is
   already computed.
4. **Key retrieval**: parse the recorded hint against the map (wall colour /
   image count / prop) to pick the target room, fetch the key, return to the
   locked door.
