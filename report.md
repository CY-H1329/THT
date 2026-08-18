# Report — Memory-FPS Agent

## 1. Approach

The agent is **rule-based perception + a small state machine + one cheap VLM call
per room**. Nothing is learned; everything is derived from the pixels of `act`'s
`obs`, and the only network use is a background call that captions the rooms we
have already left.

Three ideas shape the design:

1. **Read the HUD exactly, not approximately.** Room name, HP, heading, remaining
   time and the `[KEY]` tag are the only *ground truth* the game gives us, and
   they are rendered with Pillow's bundled font — the same glyphs on every
   platform. So instead of a general OCR engine (Tesseract was tried and dropped:
   heavy dependency, worse on 16-px HUD glyphs), `agent/ocr.py` rasterises that
   font once and does template matching per glyph. It is exact, dependency-free,
   and reproducible. HUD pixels rarely change between steps, so the result is
   cached on a hash of the HUD bar — OCR was the single biggest per-frame cost
   (~140 ms vs ~1.5 ms for everything else), and caching removes it from most ticks.

2. **Geometry from the floor, not from a depth model.** The camera is fixed
   (`domain_rand=False`), so the row at which the checkerboard floor ends in a
   column maps analytically to a distance in metres (`agent/geometry.py`). That
   gives a per-column depth profile at ~1.4 ms/frame, which drives obstacle
   avoidance, "how far can I walk in this direction", and enemy distance/bearing
   in degrees (`agent/enemies.py`) — accurate enough to aim the 40°-wide attack
   cone without any model call.

3. **Two memories with different update rates.** `agent/memory.py` (`SceneGraph`)
   is the *exploration* state — rooms as nodes, doors as directed edges — updated
   every step from the HUD. `agent/content_db.py` (`ContentDB`) is the *content*
   memory — wall colours, wall images, 3D objects, enemies, key/hint/door facts —
   updated asynchronously by the VLM and by CV, and frozen once at the start of
   QA. They are joined only by the room-name string, so an unstable exploration
   run still leaves a usable content DB, which is what the 60 % QA score depends on.

**Acting.** `agent/explorer.py` runs a DFS over rooms: enter → walk to roughly the
room centre (`RECENTER`) → face each of the four cardinal walls and sample them
(`SURVEY`) → try an unexplored direction (`SEEK`) → recurse, backtracking through
the DFS stack when a room is exhausted. Doors only exist at 0/90/180/270°, so no
arbitrary angle is ever probed. Priorities are made explicit in `_arbitrate()`:

```
flee/counter-attack  >  opportunistic attack  >  go to key/locked door  >  explore  >  wander
```

**Building memory.** `agent/content_ingest.py` buffers a few frames per room —
one per 15° heading bucket, taken during `SURVEY` (i.e. standing inside the room,
facing a wall head-on), plus up to two frames in which an enemy was actually
detected. When the agent leaves the room, those frames plus *the room's current
DB entry* are handed to a single background worker thread, which asks
`google/gemini-3.1-flash-lite` for a refined JSON record (`agent/vlm.py`). `act`
never waits for a reply. Everything that does not need a model — wall colour by
pixel vote, which room the HP dropped in, where the `[KEY]` tag turned on and off,
which door had a lock panel, the hint text — is written directly from pixels, so
the memory is still useful if the network fails entirely.

**Answering.** `agent/qa.py` is pure Python over the frozen structures: it
resolves a room name out of the question by token overlap (tolerating the HUD's
truncated `"Vermilion Libra…"` names), classifies the question into one of the
README's categories with regexes, and reads the answer out of `ContentDB`. There
is **no model call in `answer()`** — every answer returns in single-digit
milliseconds, far inside the 10 s budget, and repeated questions always get the
same answer. When nothing matches, the agent says `"I did not pay attention to
that."` rather than guessing.

## 2. Architecture

```
act(obs)
  ├── ocr.read_hud            room name / HP / heading / time / [KEY]   (cached)
  ├── explorer.ExplorerPolicy._arbitrate
  │     ├── FLEE          enemies.close_range_bearing → aim/attack, else 24×15° sweep
  │     ├── attack        enemies.detect  (cone ±20°, range 3 m)
  │     ├── GOTO_HINT     memory.SceneGraph.shortest_path + _navigate_step
  │     ├── explore       INIT→RECENTER→SURVEY→SEEK→(HINT_CAPTURE)→RETURN
  │     └── wander        _navigate_step toward the current heading
  │           ├── geometry.depth_profile / cone_clearance   (look before stepping)
  │           └── vision.is_blocked                          (frame diff: did I move?)
  └── content_ingest.ContentIngest.on_frame
        ├── direct facts → content_db (damage per room, key pickup/consume, wall-colour votes)
        └── on room exit → ThreadPoolExecutor(1) → vlm.update_room_db → content_db.apply_update

answer(q)
  ├── (first call only) ContentIngest.finalize → drain worker → ContentDB.freeze(scene)
  └── qa.answer_question   regex/keyword routing over the frozen DB
```

Supporting modules: `agent/palette.py` (the 12 wall colours), `agent/hint_resolve.py`
(hint template → room), `agent/config.py` (API key discovery, budgets).

**Hint handling.** When a blocked direction shows the lock panel's warm-gold
colour, the agent switches to `HINT_CAPTURE`: it walks into the door's 1.5 m touch
radius until the yellow-bordered banner actually appears — the banner is a
100 %-reliable confirmation, whereas the colour heuristic alone is not — reads the
text, then retreats the same number of steps. `hint_resolve.py` then slot-fills the
four README templates (wall colour / count + wall / category + count / 3D object)
against `ContentDB`, so no category vocabulary is memorised: "the room with two
X images" is answered by counting images whose description contains X, whatever X
turns out to be in the held-out pool. If it resolves to a room we have seen, the
`GOTO_HINT` state routes there via BFS over the room graph, sweeps the room to
step on the key, then routes back to the locked door and walks into it — which is
the only way to answer "what was behind the locked door".

**Confidence discipline.** Every VLM-reported item carries a confidence; anything
below 0.8 is discarded rather than stored (`content_db.CONFIDENCE_THRESHOLD`).
Counts that the game's own generator cannot produce (>2 objects or >8 images in a
room) evict the lowest-confidence entry instead of accumulating.

## 3. What worked / what didn't

**Worked**

- *Exact HUD OCR + caching.* Removed the dominant per-frame cost and gave the
  whole system a trustworthy anchor (room identity) that never needs a model.
- *Floor-geometry depth.* Cheap, deterministic, and precise enough to both steer
  and aim.
- *Deterministic combat sweep.* The attack cone is ±20° and one turn is 15°, so
  "attack twice, turn 15°, repeat 24 times" is *geometrically guaranteed* to put
  any surviving enemy inside the cone at some point. This is the safety net under
  the CV aimer and it is why combat does not depend on the model being right.
- *Giving the VLM the current DB and asking it to refine.* Independently
  captioning frames and merging text afterwards failed badly: the same picture
  came back as "chimp"/"ape"/"monkey" and de-duplication never fired. Showing the
  model what is already recorded made it distinguish "new" from "already known".
- *Deriving facts from the HUD instead of from the model where possible.* Key
  room, damage location, and wall colour are model-independent and survive an
  outage or a bad caption.

**Didn't work (and was removed)**

- *Retreating from enemies.* The intuitive "back off, then counter-attack" loses:
  measured during development, the mob's chase speed matches our backward speed,
  so the gap stays at 1.3–1.5 m — inside its attack range — and the agent simply
  absorbs more hits. The agent now fights in place, which is sound here precisely
  because the enemy is *already* closer than our own 3 m attack range.
- *Waiting for out-of-range enemies to approach.* On seed 7 this cost the whole
  episode: the agent froze in an enemy-dense room and died at ~640 steps, versus
  surviving the full ~4000 steps with the behaviour disabled (direct A/B).
- *Using the VLM to aim.* Its left/center/right verdict uses a much narrower
  notion of "center" than the real 40° cone, so the agent oscillated between two
  headings for 76+ ticks while being hit, never firing. The VLM is now used in
  combat only for a yes/no "is it still there", where that ambiguity does not exist.
- *Trusting the depth profile as a veto on moving.* A doorway narrower than the
  15° safety cone reads as blocked, so a confirmed door was abandoned without a
  single step being tried. The rule is now: depth *suggests*, the frame-diff
  `is_blocked` check *decides* — and a direction is always physically probed once.
- *Colour-difference enemy detection ("anything not wall-coloured").* Lighting
  darkens the same wall by up to ×0.65, so plain walls were repeatedly flagged as
  enemies. Replaced by the shape/height/floor-contact test in `agent/enemies.py`.

## 4. Limitations and next steps

- **Doorway bleed-through when captioning wall content.** Standing near a doorway
  and looking straight through it makes the *next* room's walls, pictures and
  objects appear as large and sharp as the current room's, so the naive
  "near/large ⇒ current room" heuristic fails (confirmed on a captured frame:
  `seed0_step00216.png`, HUD says "Ebon Annex" while the frame is dominated by
  another room's blue walls, two of its images and one of its floor objects).
  Our pixel wall-colour vote was not fooled, but the VLM describes what it sees.
  Three mitigations are in place: the prompt explicitly assigns anything framed
  inside a portal-shaped opening to the other room; only frames taken inside the
  room during `SURVEY` are sent; and at freeze time, an image recorded on the very
  wall that leads to an adjacent room *and* also recorded in that neighbour is
  dropped as a leak. It is a reduction, not a fix — frames captured while turning
  toward a doorway can still leak.
- **Kill count is an estimate, not a count.** The game never tells us an enemy
  died. We count "rooms where HP actually dropped **and** we fought a bout in
  which we swung", capped at the two-enemies-per-room structural maximum. A bout
  that ended because the sweep ran out (enemy alive) is indistinguishable from a
  kill, so the number can over-count; conversely enemies killed opportunistically
  without ever hitting us are not counted at all. A better signal would be the
  enemy mesh disappearing between two consecutive frames at a known bearing.
- **Enemy appearance is thinly sampled.** Enemy colours only enter the DB if a
  combat frame happens to reach the VLM. Rooms where we were attacked from behind
  usually record damage but no description; the agent then says so honestly
  instead of inventing one.
- **Room identity is a string.** Two rooms whose HUD names both truncate to the
  same prefix would merge. The canonicaliser tolerates ellipsis truncation but
  cannot detect a genuine collision.
- **Locked-door reasoning is single-shot.** We assume exactly one locked door per
  episode (as documented) and use the room where the banner fired as its location.
  If the banner is missed, the lock is only known from the colour heuristic.
- **Exploration can end early.** Death ends everything, and the QA-relevant cost
  of dying is the rooms never seen. The agent survives noticeably better since
  combat stopped retreating, but a run that meets two enemies in one room can
  still end at ~350 steps.

**Next steps, in order of expected QA gain:** (1) a second VLM pass at freeze time
for rooms whose record is still empty, using the frames we already buffered — the
budget (`VLM_MAX_CALLS_PER_EPISODE`) is barely used today; (2) per-wall image
counting from geometry rather than from the VLM's `wall_dir`, which is the weakest
field in the schema and the one the "two images on its east wall" hint template
depends on; (3) frame-differencing at the moment of an attack to confirm kills.
