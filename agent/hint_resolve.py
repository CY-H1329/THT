"""잠긴 문 힌트 문구 -> "열쇠가 있는 방" 이름 추정.

README가 명시한 템플릿은 4개뿐이고(seed마다 하나), 카테고리 어휘만
held-out으로 바뀐다. 그래서 어휘를 외우는 게 아니라 템플릿 "모양"을
슬롯으로 뜯어서, 우리가 이미 관측한 방 특징(ContentDB)과 대조한다:

  1) 벽 색      — "the room with sage walls"
  2) 개수+벽    — "the room with two images on its east wall"
  3) 카테고리+개수 — "the room with two tiger images"
  4) 3D 오브젝트 — "the room with a barrel"

템플릿 파싱에 실패하면(문구가 예상과 다르면) tmp/qa_db.py의 원래
점수식(색/오브젝트/이미지 토큰이 문구에 등장하면 가점)으로 폴백한다 —
어떤 경우에도 크래시하지 않고 None이면 "모른다"로 처리된다.

Play 중(GOTO_HINT: 열쇠 방으로 이동)과 QA 중(열쇠 방을 묻는 질문)
양쪽에서 같은 함수를 쓴다.
"""

from __future__ import annotations

import re
from typing import Optional

# README의 heading <-> 벽 대응표.
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

_MIN_SCORE = 2  # 이 미만이면 "매칭 실패"로 보고 None (tmp/qa_db.py와 동일)


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


# --- 템플릿별 점수 -----------------------------------------------------
def _score_wall_color(room, color: str) -> int:
    value = _norm(room.wall_color.value or "")
    if not value:
        return 0
    return 4 if value == color or color in value or value in color else 0


def _score_count_on_wall(room, count: int, heading: int) -> int:
    seen = _count_images_on_wall(room, heading)
    if seen == count:
        return 4
    # 방향 정보가 하나도 없으면(VLM이 wall_dir을 못 채운 방) 총 개수라도 본다.
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
    return 2  # 카테고리는 맞는데 개수가 다름 — VLM이 한 장 놓쳤을 수 있다.


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
    """템플릿 파싱 실패 시 폴백 — 문구에 등장하는 특징이 있으면 가점."""
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
    """힌트 문구 -> (스코어 함수, 설명). 못 알아보면 None."""
    body = text
    m = re.search(r"room (?:with|that has|containing) (.+)$", text)
    if m:
        body = m.group(1).strip(" .\"'")

    # 1) 개수 + 벽: "two images on its east wall"
    m = re.match(r"^(\w+)\s+(?:images?|pictures?|photos?)\s+on\s+(?:its|the)\s+"
                 r"(east|south|west|north)\s+wall$", body)
    if m:
        count = _NUMBER_WORDS.get(m.group(1))
        heading = _WALL_TO_HEADING[m.group(2)]
        if count is not None:
            return ((lambda room: _score_count_on_wall(room, count, heading)),
                    f"{count} images on heading {heading}")

    # 2) 벽 색: "sage walls" — 색 이름은 한두 단어짜리라 그렇게 좁힌다
    #    (안 좁히면 "two images on its east wall"까지 이 패턴에 걸린다).
    m = re.match(r"^((?:\w+\s+){0,1}\w+)\s+walls?$", body)
    if m:
        color = m.group(1).strip()
        color = re.sub(r"^(a|an|the)\s+", "", color)
        return (lambda room: _score_wall_color(room, color)), f"wall color {color!r}"

    # 3) 카테고리 + 개수: "two tiger images"
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

    # 4) 3D 오브젝트: "a barrel"
    m = re.match(r"^(?:a|an|the)\s+(.+)$", body)
    if m:
        phrase = m.group(1).strip()
        return (lambda room: _score_object(room, phrase)), f"object {phrase!r}"

    return None


def resolve_hint_room(hint_text: Optional[str], content_db) -> Optional[str]:
    """힌트 문구와 지금까지 관측한 방들을 대조해 열쇠가 있는 방 이름을
    추정한다. 확신할 만한 후보가 없으면 None(=모른다)."""
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
            # 템플릿을 알아본 경우에도 폴백 점수를 약하게 더해 동점을 깬다
            # (예: 색 힌트인데 방 이름 자체가 문구에 들어있는 경우).
            score += min(_score_generic(room, text), 2)
        if score > best_score:
            best_name, best_score, ties = name, score, 1
        elif score == best_score and score > 0:
            ties += 1

    if best_name is None or best_score < _MIN_SCORE:
        return None
    if ties > 1:
        # 같은 점수의 방이 여러 개면 찍지 않는다 — 틀린 답보다 "모른다"가 낫다.
        return None
    return best_name
