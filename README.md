# Memory-FPS Environment (THT)

A first-person 3D environment for evaluating exploration, survival, and
long-term memory in a vision-based agent.

## Task

Implement an `Agent` that plays the game, then answers questions about
what it observed. **You will not be told the questions in advance.**

## Install

```
pip install -e .[dev]
```

CPU-only, ≥4 GB RAM. Tested on Python 3.10+. Runs on Windows, macOS, and
Linux.

The 3D rendering (Miniworld via pyglet) needs an active local graphics
session — it works on a normal desktop login on any of the three OSes,
but not over a plain SSH connection with no display. On Linux, run
headless servers under `xvfb-run`; macOS and Windows have no headless
equivalent, so use a desktop session there. The HUD font is Pillow's
bundled font, so the HUD renders identically on every platform.

> **Windows note:** On Windows, constructing more than one `MemoryFPSEnv`
> instance per Python process can crash the pyglet backend. If you need
> to develop across multiple seeds, run each in a fresh subprocess.
> macOS and Linux do not have this limitation, but running each seed in
> its own process is still the recommended pattern for clean isolation.

## Agent contract

```python
class Agent:
    def act(self, obs: np.ndarray) -> int:
        """Return an Action ∈ {0..6} given the current RGB observation."""

    def answer(self, question: str) -> str:
        """Return a free-text answer to a question about the episode."""
```

The evaluator imports your `Agent` class, instantiates it, calls `act`
each step during the play phase, and calls `answer` for each question
during the QA phase.

## Observation

A `(240, 320, 3)` uint8 RGB frame. A black HUD bar at the top displays:

- The **room name** (or `"— corridor —"` when between rooms).
- The **time remaining in simulated seconds**, e.g. `T-587s`.
- The **current/max HP**, e.g. `HP 8/12`.
- The **facing heading** in integer degrees, e.g. `dir 135°` — the
  agent's absolute orientation. Use it to tell which wall of a room
  you are looking at (see the table below).
- A `[KEY]` tag — appended to the HUD line only while you are
  carrying the locked-door key (see "Locked door + key + hint" below).

These values are part of the visual frame only. They are not provided
via `info` or any structured channel. Your agent must extract them from
pixels.

The heading changes by 15° per turn action (matching the action space)
and maps to room walls as follows; intermediate values fall between two
walls:

| `dir` | Wall you are facing |
|------:|---------------------|
| 0°    | east wall  |
| 90°   | south wall |
| 180°  | west wall  |
| 270°  | north wall |

## Action space

| ID | Action |
|----|--------|
| 0 | turn left (15°) |
| 1 | turn right (15°) |
| 2 | move forward |
| 3 | move back |
| 4 | attack (forward raycast, ~3 m range) |
| 5 | no-op |
| 6 | end episode (voluntarily start QA) |

## Episode termination

The episode ends if any of the following happens:

1. You call `end_episode` (action 6).
2. The wall-clock time limit expires (default 600 simulated seconds).
   The env tracks the real wall-clock between your `step()` calls and
   advances enemy AI by that elapsed time. If you call `step()` slowly
   (e.g. a VLM agent that takes 1 s per decision), enemies will move
   ≈1.5 m between calls. If you call `step()` rapidly, each call
   advances the world less. Either way, the simulated clock keeps
   counting toward the 600-second cap.
3. Your HP reaches zero.

After termination, `info["phase"]` flips from `"play"` to `"qa"` and
the final frame shows a "QA PHASE" banner. The evaluator then asks
questions via `agent.answer(question)`.

## What questions might ask

Questions are authored by the evaluator. They will draw on what the
env recorded — room names, wall images, 3D objects in rooms, enemies
you encountered or killed, where things happened, how the game ended.
Examples of the kinds of things you might be asked (illustrative, not
exhaustive):

- What wall image was in some specific room.
- What 3D objects were in some specific room.
- What color a particular object or enemy was.
- How many enemies you killed.
- Whether you visited a particular room.
- What the key hint said, or which room contained the key.
- Whether you found the key and unlocked the door.
- What was in the room behind the locked door (if you reached it).

Design your agent's memory to capture observations that *might* matter,
not just what you guessed the questions would be.

## Public assets

- `memory_fps_env/assets/names_public.txt` — public name pool.
- `memory_fps_env/assets/images/public/` — 40 categories × 10 real
  photos (Pixabay, licensed for commercial use). Open the directory to
  see the categories shipped with your distribution.
- `memory_fps_env/assets/enemy_faces/public/` — public enemy face textures.

**Evaluation uses different, held-out names, images, objects, and
enemy faces. Categories in the held-out pool are disjoint from the
public pool, so you cannot memorize specific labels during
development. You receive only a final score.**

## Visual style

- **Wall images** are real photos. Each wall holds 0, 1, or 2 images
  drawn from a weighted distribution (`images_per_wall_weights` in
  `difficulty.yaml`; defaults give a mean of 1 image per wall).
- **Enemies** render as Minecraft-style 6-cube humanoid mobs with a
  face texture quad. The mob walks toward you when it sees you, hugs
  walls to find doorways, and wanders in a random direction when you
  leave its line of sight. The visible mesh disappears when you kill
  the enemy.
- **Rooms** each get a solid wall color sampled from a shared 12-color
  palette (see `memory_fps_env/world/walls.py::WALL_PALETTE`). Colors are
  drawn independently per room, so two rooms in the same episode may
  share a color — wall color alone is not a unique room identifier.

## Locked door + key + hint

One doorway per episode is **locked** — a yellow panel with a lock
icon across it that you cannot pass through. When you first come
close to the door, three things happen:

1. A **key** spawns in some other room.
2. A **natural-language hint** describing that room appears as a
   centered overlay on top of your view for ~3 simulated seconds
   (e.g., `"the room with sage walls"` or `"the room with two cat
   images"`). The 3D view and the HUD stay visible around the
   overlay. The overlay has a thick yellow border so it's easy to
   detect that one is currently being shown. Read the hint before it
   auto-dismisses; you can return to the door to re-trigger it.
3. The HUD bar gains a `[KEY]` indicator once you pick up the key.

To unlock the door, walk into it while holding the key. Both the key
pickup and the door unlock happen automatically — no new actions to
press. The key is consumed by the unlock.

Hints describe the key's room using one of these templates (one per
seed):

- **Wall color** — `"the room with sage walls"`
- **Image count and wall** — `"the room with two images on its
  east wall"`
- **Image category and count** — `"the room with two tiger images"`
- **3D object** — `"the room with a barrel"`

The category names above (`sage`, `tiger`, `barrel`, ...) come from
the public dev pool you see during training. At evaluation the
category vocabulary is held-out (different names), but the templates
themselves are unchanged — so your agent needs to map each template
shape to a verifiable visual feature, not memorize specific category
labels.

## Examples

The `examples/` directory ships three files. We suggest starting with
manual play to get a feel for the env, then fork the skeleton to build
your agent.

- **`examples/manual_play.py`** — a WASD-controlled OpenCV window
  that lets you walk through a fresh episode yourself. Use this
  first: it's the fastest way to see what the HUD looks like, how
  the held-out wall images and 3D objects render at agent
  eye-height, how enemies move and attack, and what the locked-door
  / key / hint overlay looks like in motion. Install OpenCV first
  (gated under the `[play]` extra so it's not needed for headless
  agent runs):

  ```
  pip install -e .[play]
  python examples/manual_play.py
  ```

  Controls are printed when the window launches: `w/s` to move
  forward/back, `a/d` to turn, `SPACE` to attack, `n` for no-op,
  `q` to end the episode (start QA), `ESC` to abort the window.

- **`examples/agent_skeleton.py`** — the template `Agent` class with
  the exact contract the evaluator expects. `act(obs)` and
  `answer(question)` are stubbed; copy this file into your own
  `agent/` package (see "Submission" below) and fill in the bodies.

- **`examples/random_agent.py`** — a minimal baseline that picks
  random actions (skewed toward `MOVE_FORWARD`) and always replies
  `"I did not pay attention to that."` Useful as a smoke test that
  your env install works end-to-end, and as a lower-bound reference
  point on QA accuracy — any meaningful agent should outscore it.

## Public training seeds

```
0, 1, 7, 42, 99, 137, 256, 512, 1024, 1337
```

Run your agent against these to develop. The eval harness uses a
disjoint set of seeds.

## Rules

- Your `Agent` class may not import from `memory_fps_env.world.*`.
  Memory must be built from `act`'s `obs` argument alone.
- Do not write to disk during `act` or `answer` except to your own
  agent's working files inside a `tmp/` directory you control.
- Wall-clock time per `act` call should be reasonable (≤ ~5 s for a
  VLM-backed agent).
- **Memory is frozen at the start of the QA phase.** The evaluator
  stops calling `act` once the episode terminates, so no new
  observations arrive. `answer(question)` should *query* a memory
  you finalized during play — it should not accumulate new
  inference state across questions.
- **Each `answer(question)` call has a 10-second wall-clock
  budget.** Answers that exceed it are scored zero, regardless of
  correctness. Design your memory for cheap query at QA time —
  heavy per-question inference is not viable.

## Compute and external resources

There are no restrictions on what your agent uses — local models,
classical CV, hosted APIs, anything you find useful is fair game. The
evaluated run has network access.

To make hosted models practical, we provide an **OpenRouter API key
with US $200 of credit** for the duration of the task; use it for both
development and the evaluated run. The key is delivered to you
separately as a `key.env` file (not bundled in this package) — load the
key from that file rather than hard-coding it.

Keep the `act` time budget in mind: one network round-trip fits
comfortably within the limit, but calling a heavy model on every frame
is slow and can exhaust the credit before the run finishes.

## Submission

Submit a single ZIP file containing **both** the code and a short
report:

```
your_submission.zip
├── agent/
│   ├── __init__.py      # exports `Agent`
│   └── ...              # your modules
├── report.md            # or report.pdf / report.txt
└── requirements.txt     # any deps beyond the env's own (optional)
```

- The evaluator runs `from agent import Agent` and uses the class for
  the play phase (`act`) and the QA phase (`answer`). Keep imports
  light — large weight loads or network calls at import time eat into
  your per-`act` time budget.
- Anything outside `agent/` that your code reads at runtime should
  also be in the ZIP (local model weights, prompt templates, etc.).

## Evaluation

Two graded axes:

| Axis | Weight | What's measured |
|---|---:|---|
| **QA accuracy** | 60% | Fraction of evaluator-authored questions your `Agent.answer(question)` answers correctly on the heldout seed set. |
| **Report** | 40% | Clarity of your approach, soundness of design choices, honest treatment of limitations. Not graded by length or polish — graded by whether a reasonable engineer can read it and understand what you did and why. |

Final score = `0.6 × QA + 0.4 × Report`, averaged across seeds.
Episode survival and room-exploration coverage are not directly
graded; they only matter to the extent that dying early or missing
rooms makes you unable to answer questions about them.

### Suggested report structure

≤ 4 pages, in any of `.md` / `.pdf` / `.txt`. The structure below is a
guide, not a requirement:

- **Approach** — how the agent decides to act, how it builds memory
  from pixels, how it answers questions.
- **Architecture** — brief module map and data flow.
- **What worked / what didn't** — tried-but-discarded ideas count.
- **Limitations + next steps** — be honest about edge cases.
