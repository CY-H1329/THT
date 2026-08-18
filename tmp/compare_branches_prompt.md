# Prompt — head-to-head evaluation of `DB_code` vs `DB_code-2`

Paste everything below the line into Claude Code, running in `/home/tototime/Desktop/THT`
on a desktop session (the env needs `DISPLAY`; there is no Xvfb on this machine).

---

You are evaluating two branches of this repo — `DB_code` and `DB_code-2` — to decide which
one better satisfies `README.md` (the task spec). Work autonomously end to end and produce a
written verdict. Do not modify either branch's `agent/` code; you are measuring them, not
fixing them.

## 0. Ground rules

- Read `README.md` first and treat it as the grading rubric. The things that decide the
  verdict are, in order: (1) the evaluator runs `from agent import Agent` and calls
  `act(obs)` / `answer(question)` — a branch that cannot be imported and driven this way
  scores zero regardless of internal quality; (2) QA accuracy = 60% of the grade, so what
  matters is what a *frozen* memory can answer, not what the explorer walked past;
  (3) the hard budgets: ≤ ~5 s per `act`, ≤ 10 s per `answer`, memory frozen at the start of
  QA, no imports from `memory_fps_env.world.*` inside `agent/`, writes only under `tmp/`.
- Never edit files in a branch to make it work better. If a branch needs glue to be runnable
  at all, put that glue in the *harness*, keep it minimal, and record in the report that the
  glue was necessary — the need for it is itself a finding.
- Both branches ship the whole `memory_fps_env` package and it is byte-identical between
  them, so the environment is a controlled variable. Confirm this with
  `git diff --stat DB_code DB_code-2 -- memory_fps_env/` (expect empty output).

## 1. Setup

1. The working tree is dirty (`agent/enemies.py`, `agent/explorer.py`, `agent/ocr.py`,
   `tmp/ocr_readings.json`). Commit it to a scratch branch `DB_code-2-wip` *or* stash it —
   ask me which if it is not obvious — and confirm `git status` is clean before continuing.
   Losing that work is not acceptable.
2. Create isolated worktrees so the two branches can be run without switching:
   `git worktree add /tmp/cmp/db1 DB_code` and `git worktree add /tmp/cmp/db2 DB_code-2`.
   (If I committed the WIP, add `/tmp/cmp/db3 DB_code-2-wip` as a third arm.)
3. Use the existing interpreter `/home/tototime/Desktop/THT/.venv/bin/python`. Put the
   worktree root first on `PYTHONPATH` so `agent` and `memory_fps_env` resolve from the
   branch under test, and verify per arm that
   `python -c "import agent, memory_fps_env; print(agent.__file__)"` prints a path inside
   that worktree. A run that silently imports the wrong branch invalidates everything.
4. Cost guard: captions go to OpenRouter (`agent/config.py`: `VLM_MODEL`,
   `VLM_MAX_CALLS_PER_EPISODE = 60`). Print per-run call count and USD, keep a running
   total, and stop and report if the total passes $5.

## 2. Establish contract compliance first (cheap, decides a lot)

For each arm, before any episode: `python -c "from agent import Agent; a = Agent(); print(a.act, a.answer)"`.
Record pass/fail and the exact traceback on failure. Then check the rules mechanically:
`grep -rn "memory_fps_env.world" agent/` and any writes outside `tmp/`.

If an arm has no `Agent` class in `agent/`, you must decide how it can be measured at all.
The honest options, in order of preference:
  a. If the branch ships a play-time ingest path *inside* `agent/`, write a thin harness-side
     shim that wires its own modules together — importing nothing from the other branch.
  b. If its QA pipeline lives in `tmp/` experiment scripts, you may run those with `tmp/` on
     `sys.path`, but label the results clearly as "reconstructed offline pipeline, not a
     submittable agent".
  c. If neither works, report exactly what is missing and fall back to Tier B only (below).
Never port the other branch's code in to fill the gap — that would erase the very difference
you are measuring.

## 3. Runs

Seeds: `0, 1, 7, 42, 99, 137, 256, 512, 1024, 1337` (README's public set). Start with
`0, 1, 7, 42, 99`; extend to all ten only if the first five leave the verdict genuinely
ambiguous. **One fresh subprocess per (arm, seed)** — README requires it and it isolates
crashes. Run sequentially; do not run more than two episodes at once against the single
display. Give each run a wall-clock timeout (~15 min) and treat a timeout as a data point,
not a bug to debug.

Write ONE harness script, shared by every arm, at `tmp/cmp/run_seed.py`, taking
`--worktree`, `--seed`, `--out`. It must:

- Build `MemoryFPSEnv(seed=...)`, `reset()`, loop `agent.act(obs)` → `env.step(action)` until
  termination, capping steps (1500) as a safety net.
- Time **every** `act` call and every `answer` call; record max and p95.
- On termination, read `info["phase"]`, and ask a fixed question battery covering every
  category README names — per visited room: wall image, 3D objects, wall color, enemies
  present; plus: how many enemies killed, what the key hint said, which room had the key,
  was the door unlocked, what was behind it, how many rooms visited, how the episode ended,
  and one control question about a room that does not exist (to test whether the agent
  hallucinates instead of declining). Use the *same* battery for both arms.
- Dump one JSON per run with: steps, simulated seconds survived, died vs. timeout vs.
  voluntary end, rooms discovered (names, in order), the full frozen memory (per-room wall
  color, images with descriptions, objects, enemies), key-hint text, key room, door-unlocked
  flag, kill count, every question with its answer and latency, VLM calls and cost, and any
  exception raised.

## 4. What to compare

Score the arms on these axes and show the evidence, not just conclusions:

1. **Submittable as-is** — does `from agent import Agent` work and drive a whole episode?
   (Binary; a failure here dominates everything else.)
2. **Information actually captured** — for each arm and seed, how many rooms have a wall
   color / ≥1 image description / objects / enemies recorded. Coverage of *content*, not
   just rooms walked through. Note empty or obviously-wrong fields.
3. **Answer quality** — go question by question on 2–3 seeds and mark each answer
   correct / wrong / declined, checking against the run's own recorded memory and, where you
   need ground truth, against the env's internal state read *from the harness only* (never
   from inside `agent/`). Report the hallucination rate on the nonexistent-room control.
4. **Budgets** — worst `act` time vs. 5 s, worst `answer` time vs. 10 s, whether any answer
   would score zero on time alone.
5. **Robustness** — crashes, exceptions swallowed, deaths, timeouts, runs that explored zero
   rooms.
6. **Exploration** — rooms/episode, survival time, hint captured, door unlocked, kills.
   Secondary: README says exploration is graded only through its effect on QA.

## 5. Deliverable

Write `tmp/cmp/COMPARISON.md`:

- A per-seed table (rows = seed, columns = arm × the metrics above), plus per-arm means.
- A short section per axis stating which arm wins and by how much, with the numbers inline.
- **Verdict**: which branch best realises the README instructions, stated plainly, with the
  one or two facts that decide it. If one arm is not submittable, say so first and do not
  bury it under exploration statistics.
- **Caveats**: any glue you had to write, seeds that failed to run, cost or time limits that
  cut the sample short, and anything the comparison does *not* establish.
- If the losing arm still contains something the winner lacks, list those items concretely —
  they are porting candidates, and I want them named.

Report back with the verdict and the table; do not just tell me the file was written.
