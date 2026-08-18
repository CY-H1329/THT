# Locked door, hint, and key retrieval

How the agent opens the one locked door per episode, and why every step of it
survives the held-out asset pools.

## The mechanic

From the README and `env.py`:

1. Come within `door_touch_radius` (1.5 m) of the locked door. On the **first**
   touch a key spawns in some other room, and a hint banner appears for ~3
   simulated seconds. Re-touching re-shows the banner.
2. Walk within `key_pickup_radius` (0.8 m) of the key. The HUD gains `[KEY]`.
3. Walk into the door while holding the key. It unlocks and the key is consumed.

Four question types depend on this: what the hint said, which room held the
key, whether the door was unlocked, and what was in the room behind it.

## What was wrong before

The agent never triggered any of it — deliberately. [explorer.py](../agent/explorer.py):

```python
DOOR_STANDOFF = 2.3        # 문 앞 확인 지점 (배너 유발 반경 1.5m 밖)
```

"The door-inspection point, **outside** the 1.5 m banner-trigger radius." The
agent recognised the yellow lock panel at 2.3–2.4 m, marked the wall `LOCKED`,
and turned away. Since the key spawn is gated on first touch, **the key never
existed**. A watched run of seed 7 recorded `locked_door` but `hints: []`.

And `read_hint_text` was broken independently: it returned `''` on **all ten
public seeds**. Each text line was cropped to its tight ink band (19–21 px)
before matching, but the reference glyphs span the whole alphabet's
ascender-to-descender extent (27 px). There were never enough rows to compare,
so every glyph scored below threshold. Passing the full inner array with
`y_off = line_start` fixes it.

## Why the hint is solvable under held-out assets

The README says the hint vocabulary is held out. One exception matters, and
`world/walls.py` states it outright:

> The palette is intentionally shared across public and heldout sessions:
> forcing a heldout palette would only test color-name matching, not vision.

So the twelve wall-colour names **and** their RGBs are stable. And the env
prefers templates in a fixed order, picking the first that identifies exactly
one room — `color` is rank 1. Empirically **all ten public seeds used rank 1**.

| rank | template | resolvable? |
|---|---|---|
| 1 | `the room with sage walls` | yes — colour name → RGB → match room's scanned wall colour |
| 2 | `two images on its east wall` | yes — count + direction, no vocabulary at all |
| 3 | `two tiger images` | count only; category is held out |
| 4 | `the room with a barrel` | no — object name is held out |
| 5 | compound (colour + object) | colour half resolvable |

`agent/hints.py` carries a copy of the palette (the agent may not import
`memory_fps_env.world.*`) and parses the sentence into a `Target`.

**OCR noise is handled by fuzzy matching, not by fixing the matcher.** At font
size 28 the glyph matcher makes systematic substitutions — measured: `s→a`,
`t→I`, `i→I`, `d→r`, `k→I:`. Real reads look like `the room with muaIard
walla`. Since there are only twelve candidate colours and they are well
separated, nearest-string matching recovers the right one. On the ten real OCR
outputs it scores **10/10**.

Two traps found while building it, both fixed:

- Folding characters aggressively helps colours but wrecks number words —
  `"the"` matched `"three"` at exactly 0.75, so `"the room with a barrel"`
  parsed as count = 3. Counts and wall names now use a narrower fold plus a
  stopword list.
- The `image_count_wall` branch must run **before** the colour branch. When OCR
  mangles `"two"` into `"lwo"` the count branch fails silently and the colour
  branch then mistakes `"Images"` for a colour name.

## Finding the key

The key is `Key(color="yellow")` — fixed in env code, so "the key is yellow" is
not a held-out fact. `agent/keys.py` detects it with four constraints, because
any one alone fails:

- **Pure yellow.** Miniworld's yellow is `(1,1,0)`, so blue collapses. Wall
  palette yellows (`sand`, `mustard`) keep blue alive. Seed 7's key room has
  sand walls, and a brightness-only filter classified the entire wall as key.
- **Below the horizon.** The key is 0.35 m tall, so it always projects below
  the centre row.
- **Small.** Its pixel height is bounded by `FOCAL·0.35/d`. On seed 0 a tree's
  yellow foliage occupied 2699 px and swallowed the key entirely.
- **Level footing.** Per-column lowest-yellow rows are consistent for something
  resting on the floor and ragged for foliage.

Detection is an **accelerator, not a requirement** — measured 26/40 across
seeds and viewpoints, with accurate ranges (±0.4 m) but some false positives.
The fallback is a 3×3 waypoint sweep of the candidate room; the 0.8 m pickup
radius plus the sweep finds the key without any detection at all.

Note a blind zone: below ~2 m the key projects entirely under the frame, so it
cannot be seen while being approached. This is harmless because pickup happens
by proximity, not by sight.

## Candidate rooms, not one room

Colour matching returns a **ranked list**. `terracotta`, `brick` and `rust` are
nearly collinear in RGB (cosine 0.999), so lighting-invariant hue matching
cannot separate them — measured 29/40 rooms named correctly, with essentially
all the errors inside that trio. Rather than guess, the agent visits candidates
in score order. Each visit is a room traversal it would often make anyway.

## Ordering: explore first, then fetch

The quest is deliberately **not** top priority. Running it first cost more than
it gained — at a 40 s budget, rooms visited fell 66 → 40 and enemies killed
35 → 22, trading roughly six rooms for one. Room coverage feeds most question
types; the door feeds a few.

So the split is:

- **Touching the door runs immediately.** The agent is already standing next to
  it, the detour is ~10 steps, and it buys the hint text plus the key spawn.
- **Fetching the key waits** until exploration reports no frontier left — the
  slot that previously went to standing in a corner on sentry duty.

This means short test episodes never reach the fetch phase, which is exactly
what the measurements below show.

## Two control-hijack bugs found while integrating

Both were the same shape as the enemy phantom lock-on: an unverified detection
taking over navigation with no budget.

1. **The key chase bypassed the stuck guard.** `_key_in_view` was checked at the
   top of `_goto`, before `_drive`. Whenever the quest sat in SEEK, any yellow
   false positive — tree foliage, a `sand` or `mustard` wall — steered the agent,
   and because `_drive` never ran, `_wp_steps` never incremented and
   `STUCK_LIMIT` never fired. One room transition consumed **5429 steps**;
   seed 7 spent 6054 of 6633 steps in GOTO and scanned 3 rooms. Chasing is now
   restricted to the candidate-room sweep and capped by `KEY_CHASE_BUDGET`.
   Rooms scanned went 3 → 8 on that seed.
2. **The return-to-door path retried forever.** If the door approach point was
   unreachable, `_drive` returned "giveup", which re-entered `_choose_goal`,
   which re-issued the same goal — with a full 72-step rescan each cycle. The
   agent went back to the door **27 times** and scanned **50 times**, burning
   5301 steps while holding the key. Now capped by `QUEST_DOOR_TRIES`, and the
   quest no longer triggers a rescan when it exhausts a goal. That fell to 1
   door approach and 5 scans.

## Measured

Ten public seeds. Note the harness budget is far shorter than a real 600 s
episode, so these understate the fetch phase.

| stage | result |
|---|---|
| door touched | 9/10 |
| hint banner read | 9/10 (was **0/10** — OCR returned empty) |
| hint parsed to the correct colour | 9/9 of those read |
| key retrieved + door unlocked | 2/10 at a 110 s budget, quest-first ordering |

The hint pipeline is the solid part and it is what most of the QA value sits
in: the hint text and the key room are now recorded in `memory["key_hint"]`
whether or not the door is ever opened.

Cost to the rest of the agent, ten seeds at a 40 s budget:

| | before locked-door work | quest-first | deferred + loops fixed |
|---|---:|---:|---:|
| rooms visited | 66 | 40 | **55** |
| real kills | 35 | 22 | **27** |
| deaths | 5 | 2 | **4** |

So the feature still costs roughly ten rooms of coverage at this budget. Some
of that is real (the agent spends time on the door) and some is run-to-run
noise — identical configurations have varied between 50 and 66 rooms across
runs. It has not been measured at the real 600 s episode length, where the
fixed cost of the quest would be a much smaller fraction.

## Limitations

- **Unlock rate is low under a short budget**, and the deferred ordering means
  it is zero unless exploration completes. On seeds where exploration does not
  finish, the door stays shut by design.
- Object-template hints (rank 4) cannot be resolved by name at all; the agent
  falls back to visiting rooms and looking for the key.
- Image-count hints (rank 2) are parsed correctly but not yet *used* — per-wall
  image counts are not recorded during the scan, so those hints degrade to an
  unranked room list. That is the obvious next piece of work.
- The terracotta/brick/rust ambiguity costs extra room visits when the answer
  is one of those three.
