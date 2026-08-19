# Report — Memory-FPS Agent

(작성 중 — 초안 노트. 최종 제출 전 정리 예정.)

## 2. Architecture

The evaluator imports `Agent` from `agent/__init__.py`. Play and QA are
the same object, two methods, a hard cut between them. `act(obs)` returns
an action and may enqueue work; it never waits on a room-caption VLM call,
because the README's ≤ ~5 s per-`act` budget cannot absorb
`vlm.update_room_db` over several frames. `answer(question)` is a lookup.
The freeze is the first `answer` call: `Agent` sets `_finalized`, runs
`ContentIngest.finalize` (flush the last room, `wait` the worker up to
`_FINALIZE_TIMEOUT_S` = 6.0 s, then `ContentDB.freeze`), stores
`(content, scene)` in `_frozen`, and every later question only reads that
pair. The evaluator has already stopped calling `act`, so no new frames
arrive.

**Perception (pixels → readings).** `ocr.py` template-matches HUD glyphs
(`read_hud`: room, `T-`, HP, `dir`, `[KEY]`) and the yellow-border hint
banner (`hint_banner_active` / `read_hint_text`). `vision.py` samples
wall colour and tests `is_blocked` from frame motion (`detect_motion` is
defined and unused). `geometry.py` turns the floor checkerboard into a
depth profile / `cone_clearance` / `lock_visible`. `enemies.py` is the
combat detector (`detect`, `close_range_bearing`); ~1–2 ms, no VLM.

**Policy (readings → action).** `explorer.py` is the FSM.
`ExplorerPolicy._arbitrate` ranks FLEE (HP drop) > one-tick proactive
ATTACK > `GOTO_HINT` > DFS explore (`INIT` → `RECENTER` → `SURVEY` →
`SEEK` / `HINT_CAPTURE` → `RETURN`) > `_default_wander` (`DONE`).
`vlm.classify_heading` runs synchronously in `SURVEY` (one ~3 s call per
cardinal, fits the act budget); `vlm.locate_door` is the 4th SEEK give-up
check; `vlm.locate_enemy` is only a post-hit sweep recheck, not the
aimer — CV aiming was kept because the VLM's left/center/right bins
oscillated inside the ±20° attack cone.

**Memory (readings → facts).** `memory.py` `SceneGraph` is the live
topology (rooms, cardinal exits, `exit_leads_to`, DFS stack, hint
location). `RoomNode.images` / `objects` / `enemies_seen` are unused
stubs; content lives in `content_db.py`. `ContentDB` is the QA catalog:
per-room colour / images / objects / enemies gated at
`content_db.CONFIDENCE_THRESHOLD` (0.8), plus HUD-derived HP, `[KEY]`
edges, and kill estimates. `content_ingest.py` is the only writer of
that catalog during play.

**QA (question → string).** `qa.py` `answer_question` is regex routing
over the frozen pair. `hint_resolve.py` slot-fills the four README hint
templates against `ContentDB` during play (`GOTO_HINT`) and once more in
`finalize`; `qa.py` then reads `key_hint_room`, it does not re-resolve.
If no rule matches, `_llm_fallback` calls `vlm.qa_fallback` once
(DeepSeek, text-only, 7.5 s hard deadline).

**Support.** `vlm.py` is the OpenRouter client. Vision calls
(`update_room_db`, `classify_heading`, `locate_door`, `locate_enemy`)
use `VLM_MODEL` = `google/gemini-3.1-flash-lite`. The QA fallback uses
`QA_FALLBACK_MODEL` = `deepseek/deepseek-v4-flash`. urllib only; no
import-time network. `config.py` loads `OPENROUTER_API_KEY` or a
`key*.env` glob (never a hardcoded key). `palette.py` copies the public
12-colour `WALL_PALETTE` so `nearest_color_name` does not import
`memory_fps_env.world.*`.

```
Agent.act(obs)                              # agent/__init__.py
├─ ExplorerPolicy.step
│  ├─ _read_hud_cached → ocr.read_hud
│  ├─ _note_hud_transitions                 # HP↓ → FLEE; [KEY] edge
│  └─ _arbitrate
│     ├─ FLEE → enemies.detect / close_range_bearing
│     │         [sweep recheck only] vlm.locate_enemy
│     ├─ proactive ATTACK ← enemies.detect
│     ├─ GOTO_HINT → SceneGraph.shortest_path + _navigate_step
│     └─ explore: SURVEY (sample_wall_color, classify_heading)
│                 SEEK (depth_profile, is_blocked)
│                   lock_visible → HINT_CAPTURE → read_hint_text
│                   4th fail → locate_door
└─ ContentIngest.on_frame                   # never waits on VLM
   ├─ reuse HUD cache; record HP / [KEY] / damage
   └─ room exit → ThreadPoolExecutor(1) → vlm.update_room_db
                  → ContentDB.apply_update  (CONFIDENCE_THRESHOLD)

Agent.answer(q)   [first call only]
├─ ContentIngest.finalize  wait ≤ 6 s → ContentDB.freeze
└─ qa.answer_question(q, frozen ContentDB, SceneGraph)
```

The ingest split is the load-bearing decision. `on_frame` keeps SURVEY
frames (15° buckets, cap 6) plus a couple of combat frames, and
`_schedule_flush` fires on room *exit* onto a single worker so
`act` returns immediately. `finalize` spends up to 6 s of the 10 s
first-`answer` budget draining that worker, then `freeze` drops
lock-wall “paintings”, adjacent-room image/enemy leaks, and estimates
kills. Remaining `answer` calls are milliseconds; exceeding 10 s scores
zero regardless of correctness, so there is no per-question model.

Locked door spans every layer: SEEK `geometry.lock_visible` diverts to
`HINT_CAPTURE`, which walks into `door_touch_radius`, reads the banner
with `ocr`, and marks `exits[heading] = "locked"` only if the banner
actually appeared (colour lock alone is not trusted).
`hint_resolve.resolve_hint_room` matches the four fixed templates
(wall colour / count+wall / category+count / 3D object) to rooms
already in `ContentDB` — held-out labels, never a memorized vocabulary;
ties and scores below `_MIN_SCORE` return `None`. `GOTO_HINT` then BFS
via `SceneGraph.shortest_path` (only walked edges) to the key room
(`leg="key"`), sweeps until `[KEY]`, routes back (`leg="door"`), and
walks into the panel. Tick caps abort quietly to DFS; the path is
fragile when the key room was never surveyed, which is why ingest
retries `GOTO_HINT` every tick in `DONE` as background VLM fills catch
up.

Constraints in code, not policy comments: no `memory_fps_env.world.*`
imports (`Action` comes from `memory_fps_env.env` inside
`ExplorerPolicy.__init__`). Disk writes are only the optional
`tmp/qa_out/db.json` dump in `finalize`. Heavy modules load in
`Agent.__init__` / first `answer`, not at `from agent import Agent`.
`act`/`answer` swallow exceptions and still return a legal action or
the baseline “I did not pay attention to that.”

**Data flow.** Play: `obs` → HUD OCR names the room → cheap CV / FSM
decides the action; SURVEY frames are buffered; on room *exit* a
background worker groups them. Freeze (first `answer`): drain that
worker → `vlm.update_room_db` (rules + current DB JSON) →
`ContentDB.freeze`. QA: question → room/keyword regex → exact lookup;
unmatched intent → one text-only LLM call over the frozen JSON.

**Model selection (time / cost / accuracy).** The eval box is CPU-only
(≥4 GB RAM) and the brief forbids heavy loads at import, so a local
open-weight VLM (Qwen-VL, LLaVA) cannot fit a forward pass inside
~5 s/`act`. OpenRouter is the envelope that works; the choice is *which*
hosted model, and *when* to call it.

Vision (ingest + `classify_heading` / `locate_door` / `locate_enemy`)
uses **`google/gemini-3.1-flash-lite`**. It is multimodal, supports the
structured JSON schema `update_room_db` needs, and in a live OpenRouter
bench on our door-check prompt it averaged 1.52 s vs 1.71 s for
2.5-flash-lite, 3.59 s for 3.7-flash, and 8.04 s for `gpt-5-nano`.
Accuracy on that small door/wall set was tied; Gemini stayed because we
already knew its schema traps (integer `enum`+`null` breaks
`wall_dir`) and had a working prompt.

The default QA path is **no model** — regex lookup on frozen facts,
milliseconds, zero credit. The last-resort fallback is text-only
(question + JSON dump, no images), so a VLM is wasted spend. It uses
**`deepseek/deepseek-v4-flash`**: cheap, non-reasoning, instruction-
following. We do not use DeepSeek-R1 or `gpt-5-nano` here — both spend
the 10 s `answer` budget on hidden reasoning tokens (a 200-token cap
on nano returned empty `content` before the JSON). `qa.py` still wraps
the call in a 7.5 s hard deadline.

## Limitations (draft notes)

- **Doorway bleed-through when VLM-captioning wall content.** When the agent
  stands close to and faces straight through a doorway, the adjacent room's
  walls/images/objects can appear just as large and sharp in the frame as
  content actually in the current room, so a naive "near/large = current
  room, far/small = other room" heuristic fails in that specific case
  (confirmed on real data: `seed0_step00216.png`, HUD says "Ebon Annex" but
  the frame is dominated by a different room's blue walls, two wall images,
  and a floor object, seen straight through the doorway). Our own pixel-based
  wall-color check (`agent/vision.py::sample_wall_color`) was not fooled
  (still measured the current room's true color by raw pixel count), but
  that doesn't stop the VLM from describing the clearly-visible content
  behind the doorway as if it belonged to the current room.
  Mitigations applied: (1) the VLM prompt explicitly instructs it to treat
  anything framed inside a rectangular doorway/portal opening as belonging
  to the other room, regardless of how near/sharp it looks; (2) the live
  agent only triggers a room's VLM capture after moving a bit away from the
  doorway it entered through, not right at the threshold. Neither is a full
  fix — a few frames captured while turning toward a doorway can still leak
  a neighboring room's content into the current room's record.
