"""Locked-door hint text -> the name of "the room with the key."

The README pins down exactly 4 templates (one per seed), with only the
category vocabulary held out. So instead of memorizing vocabulary, we
break the template's "shape" apart into slots and match it against
room features we've already observed (ContentDB):

  1) wall color         -- "the room with sage walls"
  2) count + wall        -- "the room with two images on its east wall"
  3) category + count    -- "the room with two tiger images"
  4) 3D object           -- "the room with a barrel"

If the template can't be parsed (the phrasing doesn't match what we
expect), it falls back to the original scoring approach from the
tmp/qa_db.py prototype (score up if a color/object/image token shows up
in the text). Either way this never crashes -- None just means "don't
know."

The same function is used both during play (GOTO_HINT: navigate to the
key's room) and during QA (a question asking which room had the key).
"""

from __future__ import annotations

import re
from typing import Optional

# The README's heading <-> wall table.
_WALL_TO_HEADING = {"east": 0, "south": 90, "west": 180, "north": 270}

_NUMBER_WORDS = {
    "no": 0, "zero": 0, "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
}

_STOPWORDS = frozenset({
    "the", "room", "with", "on", "its", "it", "a", "an", "of", "and",
    "image", "images", "picture", "pictures", "photo", "photos",
    "wall", "walls", "that", "has", "have", "in",
})

_MIN_SCORE = 2  # below this, treat it as "no match" and return None (matches tmp/qa_db.py)


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().replace("-", " ").split())


def _tokens(text: str) -> set:
    return {w for w in _norm(text).split() if w and w not in _STOPWORDS}


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _count_images_on_wall(room, heading: int) -> int:
    total = 0
    for im in room.images:
        wd = im.get("wall_dir")
        if wd is None:
            continue
        try:
            wd = int(wd) % 360
        except (TypeError, ValueError):
            continue
        if wd == heading:
            total += 1
    return total


def _image_texts(room) -> list:
    return [str(im.get("desc") or "") for im in room.images]


def _object_texts(room) -> list:
    return [str(ob.get("name") or "") for ob in room.objects]


# Per-template scoring
def _score_wall_color(room, color: str) -> int:
    value = _norm(room.wall_color.value or "")
    if not value:
        return 0
    return 4 if value == color or color in value or value in color else 0


def _score_count_on_wall(room, count: int, heading: int) -> int:
    seen = _count_images_on_wall(room, heading)
    if seen == count:
        return 4
    # If wall direction was never filled in for any image (VLM didn't
    # give a wall_dir for this room), fall back to just the total count.
    if all(im.get("wall_dir") is None for im in room.images) and len(room.images) >= count:
        return 1
    return 0


def _score_category_count(room, category: str, count: Optional[int]) -> int:
    cat = _singular(category)
    matches = sum(1 for text in _image_texts(room)
                  if cat in {_singular(w) for w in _tokens(text)})
    if matches == 0:
        return 0
    if count is None or matches == count:
        return 4
    return 2  # right category, wrong count -- the VLM may have missed one image.


def _score_object(room, phrase: str) -> int:
    want = {_singular(w) for w in _tokens(phrase)}
    if not want:
        return 0
    for text in _object_texts(room):
        have = {_singular(w) for w in _tokens(text)}
        if want & have:
            return 4
    return 0


def _score_generic(room, text: str) -> int:
    """Fallback for when template parsing fails -- score up for any matching feature mentioned in the text."""
    score = 0
    color = _norm(room.wall_color.value or "")
    if color and color in text:
        score += 3
    for name in _object_texts(room):
        n = _norm(name)
        if n and n in text:
            score += 3
    for desc in _image_texts(room):
        for word in _tokens(desc):
            if len(word) > 3 and word in text:
                score += 2
                break
    for wall, heading in _WALL_TO_HEADING.items():
        if wall in text and _count_images_on_wall(room, heading) >= 2:
            score += 2
    return score


def _parse(text: str):
    """Hint text -> (scoring function, description). None if it isn't recognized."""
    body = text
    m = re.search(r"room (?:with|that has|containing) (.+)$", text)
    if m:
        body = m.group(1).strip(" .\"'")

    # 1) count + wall: "two images on its east wall"
    m = re.match(r"^(\w+)\s+(?:images?|pictures?|photos?)\s+on\s+(?:its|the)\s+"
                 r"(east|south|west|north)\s+wall$", body)
    if m:
        count = _NUMBER_WORDS.get(m.group(1))
        heading = _WALL_TO_HEADING[m.group(2)]
        if count is not None:
            return ((lambda room: _score_count_on_wall(room, count, heading)),
                    f"{count} images on heading {heading}")

    # 2) wall color: "sage walls" -- color names are one or two words, so we
    #    keep this narrow (otherwise "two images on its east wall" would
    #    also match this pattern).
    m = re.match(r"^((?:\w+\s+){0,1}\w+)\s+walls?$", body)
    if m:
        color = m.group(1).strip()
        color = re.sub(r"^(a|an|the)\s+", "", color)
        return (lambda room: _score_wall_color(room, color)), f"wall color {color!r}"

    # 3) category + count: "two tiger images"
    m = re.match(r"^(\w+)\s+(.+?)\s+(?:images?|pictures?|photos?)$", body)
    if m:
        count = _NUMBER_WORDS.get(m.group(1))
        category = m.group(2).strip()
        if count is not None:
            return ((lambda room: _score_category_count(room, category, count)),
                    f"{count} {category!r} images")
    m = re.match(r"^(.+?)\s+(?:images?|pictures?|photos?)$", body)
    if m:
        category = m.group(1).strip()
        return ((lambda room: _score_category_count(room, category, None)),
                f"{category!r} images")

    # 4) 3D object: "a barrel"
    m = re.match(r"^(?:a|an|the)\s+(.+)$", body)
    if m:
        phrase = m.group(1).strip()
        return (lambda room: _score_object(room, phrase)), f"object {phrase!r}"

    return None


def resolve_hint_room(hint_text: Optional[str], content_db) -> Optional[str]:
    """Match the hint text against the rooms observed so far and guess
    which one has the key. Returns None ("don't know") if there's no
    confident candidate."""
    text = _norm(hint_text)
    if not text or content_db is None or not content_db.rooms:
        return None

    parsed = _parse(text)
    scorer = parsed[0] if parsed else (lambda room: _score_generic(room, text))

    best_name, best_score, ties = None, 0, 0
    for name, room in content_db.rooms.items():
        try:
            score = scorer(room)
        except Exception:
            score = 0
        if parsed is not None:
            # Even when a template was recognized, add a small amount of
            # the fallback score to break ties (e.g. a color-template
            # hint where the room name itself also appears in the text).
            score += min(_score_generic(room, text), 2)
        if score > best_score:
            best_name, best_score, ties = name, score, 1
        elif score == best_score and score > 0:
            ties += 1

    if best_name is None or best_score < _MIN_SCORE:
        return None
    if ties > 1:
        # Multiple rooms tied for the top score -- don't guess. "I don't know" beats a wrong answer.
        return None
    return best_name
