# Enemy detection and killing: design notes

How the agent finds, fights, and counts enemies using nothing but the
`(240, 320, 3)` RGB frame — and why the method survives the held-out
evaluation assets.

## The generalization problem

The README is explicit that evaluation swaps the asset pools:

> Evaluation uses different, held-out names, images, objects, and enemy
> faces. Categories in the held-out pool are disjoint from the public pool.

So the first question is *what actually changes*. Reading the world
generator settles it:

| Thing | Held out? | Where it is defined |
|---|---|---|
| Enemy body colours | **No** — fixed 6-colour list | `world/enemies.py` `_SHIRT/_PANTS/_SKIN_COLORS` |
| Enemy body geometry (6 `Box`, scale 0.75/1.0) | **No** — hardcoded | `world/layout.py` |
| Enemy face texture | Yes | `world/faces.py` |
| 3D objects (`MeshEnt` `.obj`) | **Yes** | `world/objects.py` |
| Wall images, room names | Yes | `world/images.py`, `world/names.py` |

The enemy *body* is not an asset at all — it is constructed in code. The
real exposure runs the other way: **held-out 3D meshes creating false
positives**. The previous detector (`agent/enemies.py`) gated on saturation,
silhouette height in `[1.15, 2.45]` m, and "≥2 vertical colour layers" — all
of which an unknown 1.8 m multi-coloured mesh passes trivially. The agent
would then stand still hitting a prop while a real enemy killed it.

## The observation the design rests on

**The renderer is bit-exact deterministic.** `domain_rand=False`, no
animation, no noise. With every enemy dead, advancing the sim clock 3.6 s
and differencing the frames gives:

```
changed_px = 0     max_delta = 0
```

Not "low noise" — identical bytes. And the only entities that move in this
world are enemies; walls, wall photos, 3D props, the key, and doors are all
static.

Therefore: **while the agent is stationary, any non-zero pixel delta below
the HUD is an enemy.** This uses no colour, no texture, no shape, and no
face, so no held-out asset can affect it. That is `agent/motion.py`.

### What counts as "stationary"

- `NO_OP`.
- `ATTACK` — it only raycasts. Measured over 50 consecutive attacks:
  `agent_moved = 0.000 m`, `dir_changed = 0.000 rad`, `changed_px = 0`.
  So detection runs **for free while attacking**.
- A *blocked* `MOVE_FORWARD/BACK` — Miniworld has no partial movement.

Turns never collide, so they can never form a stationary pair.

### Sensitivity

Measured with a 4-step hold (0.05 simulated seconds):

| Enemy distance | 4 m | 6 m | 8 m | 12 m | 16 m |
|---|---|---|---|---|---|
| changed px | 279 | 87 | 182 | 1205 | 0 (behind a wall) |

One step (0.012 sim s) already gives 76 px at 5 m. A "look" is **two steps**,
not a few seconds. Detection range is also far longer than the appearance
detector's 2.4–3 m — an enemy crossing a room is seen from the doorway.

### The one blind spot

Inside 1.5 m an enemy enters `attack` state and stops completely — measured
movement over 2.4 sim s is `0.000 m`. Motion detection is blind exactly when
it is most dangerous. Three things cover that window:

1. The track is *sticky*: the enemy was seen approaching from 6–12 m, and
   since it is now frozen its last known position stays correct.
2. The HP-drop signal (`_track_hp`) proves an enemy is within 1.5 m.
3. The blind sweep below needs no detection at all.

## Why stopping is nearly free

Simulated time advances by the **real wall-clock elapsed between `step()`
calls** (`env.py`), not per step. At the measured ~12 ms per step:

- a 2-step look costs 0.024 sim s,
- a full turn-attack-attack revolution (72 steps) costs 0.86 sim s,
- enemy attack cooldown is 0.8–2.0 s.

So a whole revolution of blind attacking costs at most one incoming hit.
Meanwhile the agent's own attack has **no cooldown**, 3 m range and a ±20°
cone, and enemy HP caps at 4. This is why killing is decoupled from
recognition: you do not need to identify an enemy to kill it, you only need
to know one is near.

## How the pieces fit

```
frame ─┬─► perception.Percept ─► explorer (nav FSM)
       │
       └─► motion.MotionDetector ─► mapper.EnemyTrack ─► explorer._threat
                    │                                          │
                    └─► kill events ─► memory["enemies"]        └─► _combat
```

**`agent/motion.py`** holds the previous frame plus the pose it was taken at,
and diffs only when the pose is unchanged. Changed pixels are binned into
columns, grouped into blobs (1-bin gaps bridged), and each blob's lowest row
is back-projected to a distance through the existing IPM.

**`mapper.EnemyTrack`** accumulates blobs into per-enemy tracks in world
coordinates. Association radius grows with the age of the track, because an
enemy walks 1.5 m/s while unobserved.

**`explorer._threat`** now consults tracks *first* and the appearance
detector second. A track needs no one-step confirmation delay — a static
scene differences to 0 px, so a one-frame false positive does not exist.

**Cadence.** Driving never produces stationary pairs, so looks are inserted
deliberately:

- `_scan` runs `TURN → ATTACK → ATTACK` per 15° bucket. The two attacks form
  a stationary pair (free detection) *and* damage anything in range.
- `_goto` holds `LOOK_HOLD` steps every `LOOK_EVERY` steps.
- `_sweep_action` (used when the agent is hit but has no target) is
  `ATTACK, ATTACK, TURN`, so the blind sweep also sees.

## Counting kills

Counting *dead tracks* failed badly. Enemies move and back-projected
positions carry error, so one enemy fragments into several tracks, and every
fragment earned its own death verdict: **19 claimed kills against 3 real**
on seed 0.

Two fixes, in order of importance:

1. **Dead tracks still absorb new motion.** Motion where something was
   thought dead means the verdict was wrong, so the track is revived instead
   of a new one being created. Without this, "verdict → re-detect → new
   track → verdict" ran as a treadmill.
2. **Count events, not tracks.** When an enemy dies its mesh is removed
   immediately (`env._resolve_agent_attack`), so on that frame the diff
   covers the whole silhouette and **the new frame has no palette-coloured
   pixels there**. Walking away looks different: the mob's pixels are still
   palette-coloured at the new position. One `ATTACK` can kill at most one
   enemy (the raycast takes the nearest in-cone target), so the counter is
   capped at one kill per step.

The palette test used for "is the silhouette still there" is itself
held-out-safe: mob parts are Miniworld `Box` entities coloured from
`miniworld.entity.COLORS`, so their RGB *direction* (lighting only scales
magnitude) lands within 6° of one of six fixed vectors. Measured fraction of
chromatic pixels matching a palette direction:

| | enemies | public 3D objects |
|---|---|---|
| palette fraction | 0.81 – 0.94 | 0.00 – 0.03 |

The one exception, the rubber duckie at 0.79, is a single pure hue and is
excluded by the height gate. Held-out meshes are textured `.obj` files, which
do not systematically land on those directions — but this is a *prior*, not a
proof, which is why targeting rests on motion and the palette test is used
only for appearance read-out and the vanish check.

**Phantom tracks.** If a track absorbs `TRACK_GIVEUP` (24) aimed attacks
inside range and cone without dying or moving, its position must be wrong —
24 hits would kill any enemy three times over. Such a track is suspended from
targeting for `PHANTOM_STEPS`. The suspension is deliberately *timed*, not
permanent: an earlier permanent version blinded the agent once every track in
a room was flagged, and it was beaten to death in 793 steps.

## Phantom lock-on, and why it mattered

A watched run (seed 7) showed **41% of all actions were `ATTACK`**, with
bursts of exactly `TRACK_GIVEUP` length. Instrumenting the same seed against
ground truth explained it: of 320 aimed attacks, **312 went at tracks 3.5–11.8 m
from any real enemy**. One track with a *single* sighting absorbed 168.

Four causes, all now fixed:

1. **Aiming at stale tracks.** `TRACK_FRESH` (400 steps ≈ 4.8 sim s) was used
   for targeting as well as memory. An enemy walks ~7 m in that time, so the
   stored position was meaningless; one track was still being aimed at 458
   steps after its last sighting. Targeting now uses `TRACK_AIM_FRESH`.
2. **Single-sighting tracks were targetable.** Near the horizon one pixel row
   is over a metre, so a distant enemy projects to a plausible-looking but
   wrong nearby position. Aiming now needs `TRACK_MIN_SIGHTINGS`; a genuinely
   approaching enemy accumulates sightings quickly (12 in the measured case).
3. **The give-up rule reset itself.** `note_motion` cleared both
   `phantom_until` and `attacked` on any matched sighting, so the suspension
   re-armed endlessly into repeating 24-attack cycles. `attacked_total` is now
   never reset and drives permanent retirement at `TRACK_RETIRE`.
4. **Merged blobs got a nearest-fragment distance.** `rows.max()` over a
   `GAP_BINS`-bridged column range took the bottom row from the *nearest*
   fragment, so a far enemy inherited a near one's depth. The bottom row is
   now the median of per-column bottoms.

**The counter-intuitive part.** The first attempt at this fix made things
*worse* — deaths went 1 → 4 over ten seeds. The reason is that the env's
raycast hits whatever is actually inside the 3 m / ±20° cone, regardless of
what the agent believes it is aiming at, so "phantom" swings were still
landing on real enemies. The waste was never the swinging — it was *standing
still* to aim at nothing. The fix therefore keeps the anti-phantom gates but
restores defensive volume in a track-independent way: the periodic look during
`_goto` is now an `ATTACK` rather than a `NO_OP` (identical stationarity, free
damage).

## Measured results

All ten public seeds, 40 s each, every episode in its own process, run
sequentially (simulated time is driven by wall-clock, so parallel runs would
distort it).

| seed | rooms | HP left | outcome | real kills | claimed |
|---|---|---|---|---|---|
| 0 | 5/9 | 2/10 | ok | 2/5 | 0 |
| 1 | 5/9 | 7/8 | ok | 2/6 | 2 |
| 7 | 5/9 | 8/10 | ok | 2/6 | 1 |
| 42 | 5/9 | 5/13 | ok | 5/10 | 3 |
| 99 | 3/12 | 8/10 | ok | 2/8 | 2 |
| 137 | 5/12 | 8/14 | ok | 2/5 | 1 |
| 256 | 5/12 | 0/12 | died | 3/7 | 2 |
| 512 | 3/12 | 0/9 | died | 2/12 | 1 |
| 1024 | 6/10 | 5/8 | ok | 2/6 | 1 |
| 1337 | 8/11 | 8/13 | ok | 7/11 | 2 |

| Total over 10 seeds | baseline | motion work | **final** |
|---|---:|---:|---:|
| deaths | 5 | 1 | **2** |
| rooms visited | 49 | 52 | **50** |
| real enemies killed | 21 | 30 | **29** |
| aimed attacks, seed 7 | — | 320 | **56** |

Run-to-run variance at n=10 is substantial (episodes are wall-clock bounded),
so the death and kill totals should be read as "unchanged within noise"
between the last two columns. The attack-spam reduction is the robust result —
it was measured directly against ground truth on a fixed seed.

Kill *counting* got slightly worse in the process (15 claimed against 29 real,
versus 19 against 30 before). It still never over-counts on any seed, but it
under-reports more, because more kills now land during blind sweeps at the
edge of the cone where the kill-event gates reject them. This is an open item.

## Limitations

- **Overlapping enemies undercount.** The vanish test works on a column
  range; if a second enemy stands in the same columns, the palette pixels do
  not drop and the kill is missed. Seed 42 (10 enemies) undercounts for this
  reason.
- **Kill counting is biased low by design.** Every gate errs toward missing a
  kill rather than inventing one, because a wrong number is worse for QA than
  a conservative one.
- **Position error grows with distance.** Near the horizon one pixel row is
  ~1 m, so motion beyond `MOTION_TRACK_MAX` (7 m) is not turned into a track
  at all. It is still visible as motion, just not localizable.
- **Occlusion is not modelled.** An enemy that steps behind a prop while we
  attack can, in principle, read as a kill. The cone and range gates make
  this narrow but not impossible.
- **The 1.5 m freeze** remains the structural blind spot; it is covered by
  stickiness and the HP-drop signal rather than solved.
