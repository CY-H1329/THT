"""QA 단계: 동결된 기억(ContentDB + SceneGraph)에 대한 순수 파이썬 질의.

설계 제약(README):
  - 질문당 10초 예산. 초과하면 정답이어도 0점 -> answer() 안에서는 어떤
    모델 호출도 하지 않는다. 여기 있는 건 전부 정규식/토큰 매칭이라
    한 질문당 밀리초 단위다.
  - 기억은 QA 시작 시점에 동결된다 -> 질문끼리 상태를 공유/누적하지
    않는다(같은 질문엔 항상 같은 답).
  - 모르면 지어내지 않고 baseline과 같은 안전 문장을 돌려준다.

질문 분류는 README가 예시로 든 카테고리(방문 여부/벽 그림/오브젝트/색/
처치 수/힌트/열쇠/잠긴 문 뒤)를 그대로 따르고, 방 이름은 질문 문장에서
토큰 겹침으로 찾아낸다(tmp/qa_db.py의 매칭을 옮겨온 것 — OCR이 말줄임표로
자른 이름도 맞물리도록).
"""

from __future__ import annotations

import re
from typing import Optional

DONT_KNOW = "I did not pay attention to that."

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


# --- 방 이름 찾기 ------------------------------------------------------
def _all_room_names(content, scene) -> list:
    names = list(content.rooms.keys())
    if scene is not None:
        for name in scene.nodes:
            if name not in names:
                names.append(name)
    return names


def _resolve_room(question: str, content, scene) -> Optional[str]:
    """질문 문장에서 방 이름을 찾아낸다. 이름 토큰이 질문에 얼마나
    들어있는지로 점수를 매기고, 확실하지 않으면 None."""
    q = _norm(question)
    qt = _tokens(q)
    best, best_score = None, 0.0
    for name in _all_room_names(content, scene):
        clean = _norm(str(name).rstrip("…"))
        if not clean:
            continue
        if clean in q:  # 이름이 통째로 들어있으면 그 이상 확실할 수 없다.
            return name
        nt = _tokens(clean)
        if not nt:
            continue
        score = len(nt & qt) / len(nt)
        # 한 단어짜리 겹침("Vault" 하나만 맞음)은 다른 방과 헷갈리기 쉬워서
        # 이름의 절반 이상이 맞을 때만 인정한다.
        if score >= 0.5 and score > best_score:
            best, best_score = name, score
    return best


def _display(room) -> str:
    """OCR이 잘라 표시한 이름(말줄임표/이중 공백)을 답변에 그대로 쓰지 않게."""
    return " ".join(str(room or "").replace("…", "").split())


def _room_record(content, room: Optional[str]):
    if not room:
        return None
    return content.rooms.get(room)


# --- 카테고리별 답 -----------------------------------------------------
def _answer_images(content, scene, room: Optional[str], q: str) -> str:
    rec = _room_record(content, room)
    if rec is None:
        return _which_room_has(content, q, kind="image") or DONT_KNOW
    if not rec.images:
        return (f"I don't have a record of any wall image in {_display(room)}."
                if _room_known(content, scene, room) else DONT_KNOW)
    # 특정 벽을 물었으면 그 벽 것만.
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
        return DONT_KNOW
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
    """"~는 무슨 색이었나" — 벽/적/오브젝트 중 무엇을 묻는지 먼저 가른다."""
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
    # 오브젝트 색: 질문에 나온 물건 이름과 맞는 것을 모든 방에서 찾는다.
    want = _content_tokens(q)
    pool = [rec] if rec is not None else list(content.rooms.values())
    for r in pool:
        for ob in r.objects:
            if _content_tokens(ob.get("name")) & want and ob.get("color"):
                return f"The {ob.get('name')} was {ob['color']}."
    if rec is not None and rec.wall_color.value and "wall" not in q:
        # 방만 지정하고 대상이 모호하면(그 방의) 벽 색이 가장 흔한 의도다.
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
    """잠긴 문을 실제로 열고 지나갔을 때만 알 수 있다 — 열쇠를 쓴 뒤
    (door_unlocked) 그 문 방향으로 이어진 방이 그래프에 기록돼 있으면 그 방."""
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
    """"어느 방에 X가 있었나" 류의 역방향 조회."""
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
    """에피소드가 어떻게 끝났는지. 마지막(HP 0인) 프레임은 우리에게 오지
    않으므로 — 평가 하네스는 종료 후 act()를 더 부르지 않는다 — 마지막으로
    본 HUD의 남은 시간으로 사망/시간초과를 가른다."""
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
    """카테고리를 못 가렸지만 방은 특정된 경우 — 그 방에 대해 아는 것을
    사실 그대로 나열한다(지어내지 않고, 아는 것만)."""
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


# --- 진입점 ------------------------------------------------------------
def answer_question(question: str, content, scene=None) -> str:
    q = _norm(question)
    if not q or content is None:
        return DONT_KNOW
    room = _resolve_room(q, content, scene)

    # 순서가 중요하다: 좁은 패턴(열쇠/잠긴 문/처치 수)부터 보고, 넓은
    # 패턴(그림/오브젝트/색)은 나중에 본다.
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
    if re.search(r"how many rooms|which rooms|what rooms|list .*rooms", q):
        return _answer_room_list(content, scene, q)
    if re.search(r"did you (visit|enter|go (in)?to|see) |have you (visited|been)|"
                 r"were you (ever )?in ", q) and room is not None:
        return _answer_visited(content, scene, room, q)
    if re.search(r"colou?r", q):
        return _answer_color(content, scene, room, q)
    if re.search(r"image|picture|photo|painting|poster|artwork|hanging|on the wall", q):
        return _answer_images(content, scene, room, q)
    if re.search(r"enem|monster|mob|creature", q):
        return _answer_enemies(content, scene, room, q)
    if re.search(r"object|prop|item|furniture|3d|thing", q):
        return _answer_objects(content, scene, room, q)
    if re.search(r"wall", q):
        return _answer_images(content, scene, room, q)

    if re.search(r"which room|what room|where", q):
        found = _which_room_has(content, q, kind="any")
        if found:
            return found
    if room is not None:
        return _summarize_room(content, scene, room)
    return DONT_KNOW
