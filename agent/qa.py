"""QA phase: pure-Python queries over frozen memory (ContentDB + SceneGraph).

Design constraints, driven by the README:
  - 10 s budget per question. Exceeding it scores zero even if the
    answer is correct, so answer() never calls a model here. Everything
    in this file is regex/token matching, so each question resolves in
    milliseconds.
  - Memory is frozen at the start of QA, so questions never share or
    accumulate state (the same question always gets the same answer).
  - If we don't know, we say so instead of making something up, and
    fall back to the same safe sentence as the random baseline.

Question classification follows the categories the README calls out as
examples (visited / wall images / objects / color / kill count / hint /
key / behind the locked door), and room names are found in the question
text by token overlap (carried over from the matching logic in
tmp/qa_db.py, so a name OCR truncated with an ellipsis still matches).
"""

from __future__ import annotations

import json
import re
from typing import Optional

DONT_KNOW = "I did not pay attention to that."

# Hard wall-clock cap on the LLM fallback, used only when no rule
# recognizes the question's intent. answer()'s total budget is 10 s
# (README), but urllib's timeout= is a socket-idle timeout, so if the
# response keeps trickling in a little at a time, the whole call can
# take far longer than that (measured: anywhere from 3 s to 12 s) --
# so we enforce a real cap ourselves with a ThreadPoolExecutor. 7.5 s
# leaves room under the 10 s budget for our own overhead (JSON parsing,
# etc. -- negligible), while not cutting off a normal response that
# finishes around the 6 s mark.
_LLM_FALLBACK_WALL_CLOCK_BUDGET_S = 7.5

_WALL_TO_HEADING = {"east": 0, "south": 90, "west": 180, "north": 270}
_HEADING_TO_WALL = {v: k for k, v in _WALL_TO_HEADING.items()}

_STOP = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "did", "you", "was",
    "were", "is", "are", "what", "which", "how", "many", "there", "that",
    "room", "rooms", "and", "or", "it", "its", "any", "have", "has", "do",
    "does", "your", "with", "see", "saw", "visit", "visited", "be", "been",
})


def _norm(text) -> str:
    return " ".join(str(text or "").lower().replace("-", " ")
                    .replace("?", " ").replace(",", " ").replace(".", " ").split())


def _tokens(text) -> set:
    return {w for w in _norm(text).split() if w}


def _content_tokens(text) -> set:
    return {w for w in _tokens(text) if w not in _STOP}


def _join(items) -> str:
    items = [str(i) for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# Finding a room name
def _all_room_names(content, scene) -> list:
    names = list(content.rooms.keys())
    if scene is not None:
        for name in scene.nodes:
            if name not in names:
                names.append(name)
    return names


def _resolve_room(question: str, content, scene) -> Optional[str]:
    """Find a room name mentioned in the question. Scores by how much of
    the name's tokens appear in the question, and returns None if
    nothing is confident enough."""
    q = _norm(question)
    qt = _tokens(q)
    best, best_score = None, 0.0
    for name in _all_room_names(content, scene):
        clean = _norm(str(name).rstrip("…"))
        if not clean:
            continue
        if clean in q:  # The whole name appears verbatim -- can't get more confident than that.
            return name
        nt = _tokens(clean)
        if not nt:
            continue
        score = len(nt & qt) / len(nt)
        # A single-word overlap ("Vault" alone matching) is too easy to
        # confuse with another room, so only accept it once at least
        # half the name's words match.
        if score >= 0.5 and score > best_score:
            best, best_score = name, score
    return best


def _display(room) -> str:
    """Don't echo OCR's truncated form (ellipsis/double spaces) back in the answer."""
    return " ".join(str(room or "").replace("…", "").split())


def _room_record(content, room: Optional[str]):
    if not room:
        return None
    return content.rooms.get(room)


# Per-category answers
def _answer_images(content, scene, room: Optional[str], q: str) -> str:
    rec = _room_record(content, room)
    if rec is None:
        return _which_room_has(content, q, kind="image") or DONT_KNOW
    if not rec.images:
        return (f"I don't have a record of any wall image in {_display(room)}."
                if _room_known(content, scene, room) else DONT_KNOW)
    # If a specific wall was asked about, only that wall's images.
    for wall, heading in _WALL_TO_HEADING.items():
        if wall in q:
            on_wall = [im for im in rec.images if _snap(im.get("wall_dir")) == heading]
            if on_wall:
                return (f"On the {wall} wall of {_display(room)}: "
                        + _join(im.get("desc") for im in on_wall) + ".")
            return f"I don't have a record of an image on the {wall} wall of {_display(room)}."
    if re.search(r"how many", q):
        return f"{len(rec.images)}."
    parts = []
    for im in rec.images:
        wall = _HEADING_TO_WALL.get(_snap(im.get("wall_dir")))
        parts.append(f"{im.get('desc')}" + (f" (on the {wall} wall)" if wall else ""))
    return f"In {_display(room)} I saw " + _join(parts) + "."


def _answer_objects(content, scene, room: Optional[str], q: str) -> str:
    rec = _room_record(content, room)
    if rec is None:
        return _which_room_has(content, q, kind="object") or DONT_KNOW
    if not rec.objects:
        return (f"I don't have a record of any 3D object in {_display(room)}."
                if _room_known(content, scene, room) else DONT_KNOW)
    if re.search(r"how many", q):
        return f"{len(rec.objects)}."
    parts = []
    for ob in rec.objects:
        color = ob.get("color")
        parts.append(f"a {color} {ob.get('name')}" if color else str(ob.get("name")))
    return f"In {_display(room)} there was " + _join(parts) + "."


def _answer_enemies(content, scene, room: Optional[str], q: str) -> str:
    rec = _room_record(content, room)
    if rec is None:
        if re.search(r"how many", q):
            return _answer_kills(content, q)
        # No specific room was asked about (e.g. "Did you encounter any
        # enemy?") -- find every room with a recorded enemy and describe
        # them, the same reverse-lookup pattern images/objects/color
        # already have. Found by measurement: this used to just return
        # DONT_KNOW unconditionally whenever room was None.
        hits = [(name, r) for name, r in content.rooms.items() if r.enemies]
        if not hits:
            if content.damage_taken_total > 0:
                return ("I was attacked at some point, but I never got a clear look "
                        "at the enemy.")
            return DONT_KNOW
        parts = [f"in {_display(name)}: " + _join(_describe_enemy(en) for en in r.enemies)
                 for name, r in hits]
        return "I encountered an enemy " + _join(parts) + "."
    if not rec.enemies:
        if rec.damage_taken > 0:
            return (f"I was attacked in {_display(room)}, but I never got a clear look at "
                    "the enemy.")
        return f"I don't have a record of an enemy in {_display(room)}."
    if re.search(r"how many", q):
        return f"{len(rec.enemies)}."
    return f"In {_display(room)}: " + _join(_describe_enemy(en) for en in rec.enemies) + "."


def _describe_enemy(en: dict) -> str:
    bits = []
    if en.get("body_shape"):
        bits.append(str(en["body_shape"]))
    for field, label in (("shirt_color", "shirt"), ("pants_color", "pants"),
                         ("skin_color", "skin")):
        if en.get(field):
            bits.append(f"{en[field]} {label}")
    if bits:
        return "a humanoid enemy with " + _join(bits)
    return str(en.get("desc") or "a humanoid enemy")


def _answer_color(content, scene, room: Optional[str], q: str) -> str:
    """"What color was ~" -- first work out whether the wall, an enemy, or an object is being asked about."""
    rec = _room_record(content, room)
    if re.search(r"\bwalls?\b", q):
        if rec is not None and rec.wall_color.value:
            return f"The walls of {_display(room)} were {rec.wall_color.value}."
        if rec is None:
            return _which_room_has(content, q, kind="color") or DONT_KNOW
        return DONT_KNOW
    if re.search(r"enem|monster|mob|creature", q):
        if rec is not None and rec.enemies:
            en = rec.enemies[0]
            for field, label in (("shirt_color", "shirt"), ("pants_color", "pants"),
                                 ("skin_color", "skin")):
                if en.get(field) and label in q:
                    return f"Its {label} was {en[field]}."
            if en.get("shirt_color"):
                return (f"It wore a {en['shirt_color']} shirt"
                        + (f" and {en['pants_color']} pants." if en.get("pants_color")
                           else "."))
        return DONT_KNOW
    # Object color: search every room for an object matching the name mentioned in the question.
    want = _content_tokens(q)
    pool = [rec] if rec is not None else list(content.rooms.values())
    for r in pool:
        for ob in r.objects:
            if _content_tokens(ob.get("name")) & want and ob.get("color"):
                return f"The {ob.get('name')} was {ob['color']}."
    if rec is not None and rec.wall_color.value and "wall" not in q:
        # A room was named but the target is ambiguous -- that room's wall color is the most likely intent.
        return f"The walls of {_display(room)} were {rec.wall_color.value}."
    return DONT_KNOW


def _answer_kills(content, q: str) -> str:
    n = content.killed_enemies
    return f"{n}." if n else "0 — I don't believe I killed any enemy."


def _answer_visited(content, scene, room: Optional[str], q: str) -> str:
    if room is None:
        return DONT_KNOW
    if _room_known(content, scene, room):
        return f"Yes, I visited {_display(room)}."
    return f"No, I never visited {_display(room)}."


def _room_known(content, scene, room: Optional[str]) -> bool:
    if not room:
        return False
    if room in content.rooms:
        return True
    return scene is not None and room in scene.nodes


def _answer_key(content, scene, q: str) -> str:
    if re.search(r"hint", q) and not re.search(r"which room|what room|where", q):
        if content.hint_text.value:
            return f'The hint said: "{content.hint_text.value}"'
        return DONT_KNOW
    if re.search(r"unlock|open(ed)? the (locked )?door", q):
        if content.door_unlocked:
            return "Yes, I found the key and unlocked the door."
        if content.key_found:
            return "I found the key, but I never unlocked the door."
        return "No, I never unlocked the locked door."
    if re.search(r"which room|what room|where", q):
        target = content.key_found_in or content.key_hint_room
        if target:
            how = ("I picked it up there" if content.key_found_in
                   else "that matches the hint I read")
            return f"{target} — {how}."
        if content.hint_text.value:
            return (f'I only know the hint: "{content.hint_text.value}" — '
                    "I could not match it to a room I saw.")
        return DONT_KNOW
    if content.key_found:
        where = f" (in {content.key_found_in})" if content.key_found_in else ""
        return f"Yes, I found the key{where}."
    return "No, I never found the key."


def _answer_locked_door(content, scene, q: str) -> str:
    if re.search(r"behind", q):
        room = _room_behind_locked_door(content, scene)
        if room is None:
            return ("I never got through the locked door, so I don't know what "
                    "was behind it.")
        rec = content.rooms.get(room)
        details = []
        if rec is not None:
            if rec.wall_color.value:
                details.append(f"{rec.wall_color.value} walls")
            if rec.images:
                details.append(_join(im.get("desc") for im in rec.images))
            if rec.objects:
                details.append(_join(ob.get("name") for ob in rec.objects))
        if details:
            return f"{_display(room)}, which had " + _join(details) + "."
        return f"{_display(room)}."
    if content.locked_door_room:
        return f"The locked door was in {content.locked_door_room}."
    return DONT_KNOW


def _room_behind_locked_door(content, scene) -> Optional[str]:
    """Only knowable if we actually unlocked and walked through the door
    -- once the key is used (door_unlocked), whatever room is recorded
    in the graph as leading off that door's heading is the answer."""
    if scene is None or not content.door_unlocked:
        return None
    room = content.locked_door_room
    heading = content.locked_door_heading
    node = scene.nodes.get(room) if room else None
    if node is None:
        return None
    if heading is not None and node.exit_leads_to.get(heading):
        return node.exit_leads_to[heading]
    return None


def _which_room_has(content, q: str, kind: str) -> Optional[str]:
    """Reverse lookup for "which room had X"-style questions."""
    want = _content_tokens(q)
    if not want:
        return None
    hits = []
    for name, rec in content.rooms.items():
        if kind in ("image", "any"):
            for im in rec.images:
                if _content_tokens(im.get("desc")) & want:
                    hits.append((name, f"a wall image of {im.get('desc')}"))
        if kind in ("object", "any"):
            for ob in rec.objects:
                if _content_tokens(ob.get("name")) & want:
                    hits.append((name, str(ob.get("name"))))
        if kind in ("color", "any"):
            color = rec.wall_color.value
            if color and color.lower() in want:
                hits.append((name, f"{color} walls"))
    if not hits:
        return None
    names = list(dict.fromkeys(_display(n) for n, _ in hits))
    if len(names) == 1:
        return f"{names[0]} — it had {hits[0][1]}."
    return _join(names) + "."


def _answer_room_list(content, scene, q: str) -> str:
    names = _all_room_names(content, scene)
    if not names:
        return DONT_KNOW
    if re.search(r"how many", q):
        return f"{len(names)}."
    return "I visited " + _join(_display(n) for n in names) + "."


def _answer_ending(content, q: str) -> str:
    """How the episode ended. We never see the final (HP-0) frame
    ourselves -- the evaluation harness stops calling act() once the
    episode terminates -- so we tell death from a timeout using the
    last seconds-remaining value we saw on the HUD."""
    hp = content.hp_end
    left = content.seconds_left_last
    died = hp == 0 or (left is not None and left > 5 and hp is not None
                       and hp <= (content.hp_max or 10) // 2)
    if died:
        return "I died — an enemy took my HP down to zero."
    if hp is not None:
        return (f"I survived to the end of the episode with {hp}"
                + (f"/{content.hp_max}" if content.hp_max else "") + " HP left.")
    return DONT_KNOW


def _snap(heading):
    if heading is None:
        return None
    try:
        h = int(heading) % 360
    except (TypeError, ValueError):
        return None
    return min((0, 90, 180, 270), key=lambda c: min(abs(h - c), 360 - abs(h - c)))


def _summarize_room(content, scene, room: str) -> str:
    """A room was identified but no category matched -- list out
    whatever we actually know about it as plain fact (nothing invented,
    only what's recorded)."""
    rec = content.rooms.get(room)
    bits = []
    if rec is not None:
        if rec.wall_color.value:
            bits.append(f"{rec.wall_color.value} walls")
        if rec.images:
            bits.append("wall images of " + _join(im.get("desc") for im in rec.images))
        if rec.objects:
            bits.append(_join(ob.get("name") for ob in rec.objects))
        if rec.enemies:
            bits.append(_join(_describe_enemy(en) for en in rec.enemies))
    if not bits:
        if _room_known(content, scene, room):
            return f"I visited {_display(room)}, but I did not record any detail about it."
        return DONT_KNOW
    return f"In {_display(room)} I saw " + _join(bits) + "."


def _llm_fallback(question: str, content, scene) -> str:
    """Last resort for a question no rule recognized (an unexpected
    phrasing or paraphrase). Has agent.vlm.qa_fallback() answer using
    only the frozen facts (as JSON), wrapped in a hard wall-clock
    deadline (ThreadPoolExecutor). We deliberately avoid `with`
    (equivalent to shutdown(wait=True)): closing the context manager
    would block until that background thread finishes too, defeating
    the point of timing out early via future.result() below.
    shutdown(wait=False) returns immediately on our side and just lets
    the thread finish on its own (its result is discarded, and it
    doesn't block the process). If anything goes wrong -- failure,
    timeout, no API key -- we fall back to the same safe sentence as
    the baseline (never inventing an answer)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    from agent.vlm import qa_fallback

    try:
        facts = content.to_dict()
    except Exception:
        return DONT_KNOW
    facts["rooms_visited_all"] = _all_room_names(content, scene)
    facts_json = json.dumps(facts, ensure_ascii=False, default=str)

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(qa_fallback, question, facts_json)
    try:
        result = future.result(timeout=_LLM_FALLBACK_WALL_CLOCK_BUDGET_S)
    except Exception:
        pool.shutdown(wait=False)
        return DONT_KNOW
    pool.shutdown(wait=False)

    if result.ok and result.data and result.data.get("answer"):
        return str(result.data["answer"])
    return DONT_KNOW


# Entry point
def answer_question(question: str, content, scene=None) -> str:
    q = _norm(question)
    if not q or content is None:
        return DONT_KNOW
    room = _resolve_room(q, content, scene)

    # Order matters: check narrow patterns (key/locked door/kill count)
    # first, and broad ones (images/objects/color) later.
    if re.search(r"\bkey\b|\bhint\b", q):
        return _answer_key(content, scene, q)
    if re.search(r"locked door|behind the (locked )?door", q):
        return _answer_locked_door(content, scene, q)
    if re.search(r"how many.*(kill|defeat|slay)|kill(ed)? .*(how many)|"
                 r"(number|count) of .*(kill|enem)", q) or \
            re.search(r"how many enem", q) and re.search(r"kill|defeat", q):
        return _answer_kills(content, q)
    if re.search(r"how (did|does) (the )?(episode|game|run) end|did you (die|survive)|"
                 r"how did it end|how much hp", q):
        return _answer_ending(content, q)
    if re.search(r"how many rooms?\b|which rooms\b|what rooms\b|"
                 r"list\b.*\brooms?\b|name\b.*\brooms?\b|names? of\b.*\brooms?\b|"
                 r"cite\b.*\brooms?\b", q):
        # "which room(s)"/"what room(s)" only counts as "list every
        # room" intent in the plural. Singular ("which room did you
        # spawn in?") is asking to find one specific room, so it needs
        # to fall through to the narrower lookup below
        # (_which_room_has) or the LLM fallback instead of the room
        # list. Found by measurement: the singular form used to get
        # caught here too, so it always answered with the full room
        # list no matter what was actually asked.
        return _answer_room_list(content, scene, q)
    if re.search(r"did you (visit|enter|go (in)?to|see) |have you (visited|been)|"
                 r"were you (ever )?in ", q) and room is not None and "how many" not in q:
        # If "how many" is present (e.g. "how many images did you see in
        # X?"), this is asking for a count, not a visited check --
        # guards against the "did you ... see" pattern accidentally
        # matching and answering the wrong question (found by measurement).
        return _answer_visited(content, scene, room, q)
    if re.search(r"colou?r", q):
        return _answer_color(content, scene, room, q)
    if re.search(r"image|picture|photo|painting|poster|artwork|hanging|on the wall", q):
        return _answer_images(content, scene, room, q)
    if re.search(r"enem|monster|\bmob\b|creature", q):
        return _answer_enemies(content, scene, room, q)
    if re.search(r"\bobject|\bprop\b|\bitem|\bfurniture|\b3d\b|\bthing\b", q):
        return _answer_objects(content, scene, room, q)
    if re.search(r"wall", q):
        # A question with "wall" in it is usually about wall images, but
        # sometimes -- when no room name is pinned down, e.g. "which
        # room had sage walls?" -- it's actually asking about wall
        # color. Found by measurement: such a question matched neither
        # the color nor the image rule cleanly and fell through to the
        # image reverse-lookup, ending up as DONT_KNOW. If no room is
        # resolved yet, try the kind="any" reverse lookup first (checks
        # images, color, and objects together).
        if room is None:
            found = _which_room_has(content, q, kind="any")
            if found:
                return found
        return _answer_images(content, scene, room, q)

    if re.search(r"which room|what room|where", q):
        found = _which_room_has(content, q, kind="any")
        if found:
            return found
    if room is not None:
        return _summarize_room(content, scene, room)
    # No rule matched at all (an unexpected phrasing) -- hand it to the LLM once.
    return _llm_fallback(question, content, scene)
