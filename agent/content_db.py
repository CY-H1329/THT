"""confidence 기반 콘텐츠 DB — VLM이 "본 것"을 방 단위로 누적 저장.

agent/memory.py(탐험용 SceneGraph)와 의도적으로 분리돼 있다:
  - SceneGraph는 Play 중 실시간으로(매 스텝) 갱신되는 "탐험 상태".
  - 이 ContentDB는 Play 종료 후(또는 answer() 첫 호출 시) VLM이 한 번에
    채우는 "관찰 콘텐츠"로, 갱신 시점과 성격이 완전히 다르다.
  - 방 이름(문자열)을 공통 key로 삼아 둘을 연결한다. 탐험이 불안정해도
    이 DB(QA 60%의 핵심)는 독립적으로 견고하게 유지된다.

핵심 규칙: confidence < CONFIDENCE_THRESHOLD인 관측은 절대 커밋하지 않는다
(불확실한 인식으로 DB를 오염시키지 않기 위함).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

CONFIDENCE_THRESHOLD = 0.8

# world/layout.py에서 확인한 실제 게임 구조 하한/상한(모든 seed 공통):
#   - 오브젝트: 방 하나당 정확히 1~2개 (`objects_rng.randint(1, 2)`).
#   - 이미지: 벽당 0~2개 × 4벽 = 최대 8개.
# 이 상한을 넘는 관측은 오염(문틈으로 다른 방 것이 섞여 들어옴)일 가능성이
# 커서, 넘으면 confidence가 제일 낮은 기존 항목을 새 항목으로 교체한다
# (그냥 버리면 초반의 잘못된 관측이 영구히 고정될 수 있어서, 교체가 더 안전).
MAX_OBJECTS_PER_ROOM = 2
MAX_IMAGES_PER_ROOM = 8


@dataclass
class ConfidentValue:
    """단일 값(예: 방 벽 색) + 그 값의 신뢰도."""
    value: Optional[str] = None
    confidence: float = 0.0

    def maybe_update(self, new_value: str, new_confidence: float) -> bool:
        """new_confidence가 임계값 미만이면 무시. 임계값 이상이면, 기존보다
        confidence가 높을 때만(또는 아직 값이 없을 때) 갱신. 갱신했으면 True."""
        if new_confidence < CONFIDENCE_THRESHOLD:
            return False
        if self.value is None or new_confidence >= self.confidence:
            self.value = new_value
            self.confidence = new_confidence
            return True
        return False


@dataclass
class RoomContent:
    name: str
    wall_color: ConfidentValue = field(default_factory=ConfidentValue)
    has_locked_door: ConfidentValue = field(default_factory=ConfidentValue)
    images: list = field(default_factory=list)   # [{desc, wall_dir, confidence}]
    objects: list = field(default_factory=list)  # [{name, color, confidence}]
    enemies: list = field(default_factory=list)  # [{desc, color, confidence}]
    frame_count: int = 0
    # 이 방에 있는 동안 실제로 깎인 HP 총량(agent/content_ingest.py가 매
    # 프레임 HUD HP 하락에서 채움). "적을 몇 마리 죽였나" 추정의 핵심
    # 근거다 — 맞았다는 건 그 방에 적이 실제로 있었다는 뜻이라, VLM이
    # 문틈으로 옆방 적을 잘못 본 경우와 구분된다(tmp/qa_db.py에서 이미
    # 검증된 휴리스틱).
    damage_taken: int = 0
    # 이 방에서 실제로 ATTACK을 낸 교전(FLEE strike) 횟수 —
    # 처치 수 추정의 상한 근거(방당 적은 최대 2마리).
    combat_bouts: int = 0

    def image_count_on_wall(self, wall_dir) -> int:
        return sum(1 for im in self.images if im.get("wall_dir") == wall_dir)


class ContentDB:
    def __init__(self) -> None:
        self.rooms: dict[str, RoomContent] = {}
        self.hint_text = ConfidentValue()
        self.locked_door_room: Optional[str] = None
        self.locked_door_heading: Optional[int] = None
        self.key_found: bool = False
        self.key_found_in: Optional[str] = None   # [KEY] 태그가 처음 켜졌을 때 있던 방
        self.door_unlocked: bool = False          # [KEY]가 다시 꺼짐 = 열쇠 소모 = 문 열림
        self.key_hint_room: Optional[str] = None
        self.killed_enemies: int = 0
        self.damage_taken_total: int = 0
        self.hp_start: Optional[int] = None
        self.hp_end: Optional[int] = None
        self.hp_max: Optional[int] = None
        # 마지막으로 읽은 HUD의 남은 시간(초). QA 시작 시점에 시간이 아직
        # 넉넉히 남아 있었다면 "시간 초과"가 아니라 "사망"으로 끝난 것이다.
        self.seconds_left_last: Optional[int] = None
        self.visited_order: list = []
        self.frozen: bool = False

    def get_or_create(self, room_name: str) -> RoomContent:
        if room_name not in self.rooms:
            self.rooms[room_name] = RoomContent(name=room_name)
            self.visited_order.append(room_name)
        return self.rooms[room_name]

    # --- Play 중 CV/HUD로 직접 채우는 사실들(VLM 없이도 항상 신뢰 가능) ---
    def record_damage(self, room_name: Optional[str], amount: int) -> None:
        if self.frozen or amount <= 0:
            return
        self.damage_taken_total += amount
        if room_name:
            self.get_or_create(room_name).damage_taken += amount

    def note_hp(self, hp: Optional[int], hp_max: Optional[int]) -> None:
        if self.frozen:
            return
        if hp_max is not None:
            self.hp_max = hp_max
        if hp is None:
            return
        if self.hp_start is None:
            self.hp_start = hp
        self.hp_end = hp

    def note_key_found(self, room_name: Optional[str]) -> None:
        """HUD의 [KEY] 태그가 꺼짐->켜짐. 열쇠를 집은 그 순간 있던 방이
        곧 열쇠가 있던 방이다(픽셀만으로 100% 확실한 사실)."""
        if self.frozen:
            return
        self.key_found = True
        if room_name and self.key_found_in is None:
            self.key_found_in = room_name

    def note_key_consumed(self) -> None:
        """[KEY] 태그가 켜짐->꺼짐. README: 열쇠는 문을 열 때만 소모된다."""
        if self.frozen:
            return
        self.door_unlocked = True

    def apply_update(self, room_name: str, update: dict) -> None:
        """vlm.update_room_db()가 반환한 JSON 하나를 이 방에 병합.

        update 형식(agent/vlm.py의 스키마와 일치):
          {
            "wall_color": {"value": str, "confidence": float} | None,
            "has_locked_door": bool | None (True/False 자체엔 confidence
                필드가 없을 수 있어, 있으면 이 값에도 0.8 게이트를 적용),
            "images": [{"desc": str, "wall_dir": int|None, "confidence": float}],
            "objects": [{"name": str, "confidence": float}],
            "enemies": [{"desc": str, "color": str, "confidence": float}],
            "hint_text": {"value": str, "confidence": float} | None,
          }
        """
        room = self.get_or_create(room_name)
        room.frame_count += 1

        wc = update.get("wall_color")
        if wc and wc.get("value"):
            room.wall_color.maybe_update(wc["value"], wc.get("confidence", 0.0))

        hld = update.get("has_locked_door")
        hld_conf = update.get("has_locked_door_confidence", 1.0 if hld is not None else 0.0)
        if hld is not None:
            room.has_locked_door.maybe_update(str(bool(hld)), hld_conf)
            if hld and hld_conf >= CONFIDENCE_THRESHOLD:
                self.locked_door_room = room_name

        for img in update.get("images", []) or []:
            if img.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
                continue
            if not _has_similar(room.images, img, text_key="desc"):
                _add_with_cap(room.images, img, MAX_IMAGES_PER_ROOM)

        for obj in update.get("objects", []) or []:
            if obj.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
                continue
            if not _has_similar(room.objects, obj, text_key="name"):
                _add_with_cap(room.objects, obj, MAX_OBJECTS_PER_ROOM)

        for en in update.get("enemies", []) or []:
            if en.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
                continue
            if not _has_similar(room.enemies, en, text_key="desc"):
                room.enemies.append(en)

        hint = update.get("hint_text")
        if hint and hint.get("value"):
            if self.hint_text.maybe_update(hint["value"], hint.get("confidence", 0.0)):
                self.key_hint_room = room_name


    # --- QA 시작 시점의 동결(freeze) ------------------------------------
    # README: "Memory is frozen at the start of the QA phase". 그래서 여기서
    # 딱 한 번, Play 중엔 알 수 없었던 "전체를 다 본 뒤에야 판단 가능한"
    # 정리를 한다. 전부 tmp/qa_db.py 프로토타입에서 검증된 규칙이다:
    #   1) 잠긴 문(자물쇠 판) 벽에 걸린 그림 기록 삭제 — 자물쇠 판 위엔
    #      아무것도 안 걸린다. VLM이 노란 판/자물쇠 아이콘을 "그림"으로
    #      본 경우를 잡는다.
    #   2) 인접한 두 방에 같은 피사체 그림이 있고, 그중 하나가 그 두 방을
    #      잇는 문이 있는 벽에 기록돼 있으면 문틈으로 새어 들어온 것으로
    #      보고 버린다(report.md에 기록된 doorway bleed-through의 직접
    #      대응).
    #   3) 같은 외형의 적이 인접한 두 방에 기록돼 있으면 실제로 맞은
    #      (damage_taken이 큰) 방만 남긴다 — 몹은 하나인데 문 너머로도
    #      보여서 두 방에 중복 기록되는 경우.
    def freeze(self, scene=None) -> None:
        if self.frozen:
            return
        adjacency = _adjacency_from_scene(scene)
        self._drop_lock_wall_images(scene)
        self._prune_adjacent_image_leaks(adjacency)
        self._prune_adjacent_enemy_leaks(adjacency)
        self._estimate_kills()
        self.frozen = True

    def _drop_lock_wall_images(self, scene) -> None:
        if scene is None:
            return
        for name, room in self.rooms.items():
            node = scene.nodes.get(name)
            if node is None:
                continue
            lock_dirs = {h for h, st in node.exits.items() if st == "locked"}
            if not lock_dirs:
                continue
            room.images = [im for im in room.images
                           if _snap_heading(im.get("wall_dir")) not in lock_dirs]

    def _prune_adjacent_image_leaks(self, adjacency: dict) -> None:
        for a, links in adjacency.items():
            room_a = self.rooms.get(a)
            if room_a is None:
                continue
            for dir_ab, b in links.items():
                room_b = self.rooms.get(b)
                if room_b is None:
                    continue
                # a의 "b로 통하는 벽"에 기록된 그림 중, b에도 같은 피사체가
                # 있는 것만 버린다(양쪽 다 진짜로 비슷한 그림을 걸고 있을
                # 수도 있으니, 문이 난 벽이라는 조건을 반드시 함께 본다).
                room_a.images = [
                    im for im in room_a.images
                    if not (_snap_heading(im.get("wall_dir")) == dir_ab
                            and _has_similar(room_b.images, im, text_key="desc"))
                ]

    def _prune_adjacent_enemy_leaks(self, adjacency: dict) -> None:
        for a, links in adjacency.items():
            room_a = self.rooms.get(a)
            if room_a is None:
                continue
            for b in links.values():
                room_b = self.rooms.get(b)
                if room_b is None or room_a is room_b:
                    continue
                # 실제로 맞은 쪽(damage_taken이 큰 쪽)이 진짜 그 적이 있던
                # 방이다. 동점이면 아무것도 안 지운다(근거 부족).
                if room_a.damage_taken >= room_b.damage_taken:
                    continue
                room_a.enemies = [
                    en for en in room_a.enemies
                    if not _same_enemy_in(room_b.enemies, en)
                ]

    def _estimate_kills(self) -> None:
        """처치 수 추정: "실제로 맞았고(=적이 확실히 그 방에 있었고), 그
        방에서 공격 교전을 벌였다"는 방만 센다. 방당 적은 최대 2마리라
        교전 횟수를 2로 자른다. 우리 정책은 도망치지 않고 그 자리에서
        끝까지 때리므로(explorer.py 상단 주석의 실측 근거), 교전 하나는
        대체로 처치 하나에 대응한다 — 완벽하진 않지만 QA에서 "0"이나
        "모름"보다 훨씬 낫다."""
        kills = 0
        for room in self.rooms.values():
            if room.damage_taken <= 0 or room.combat_bouts <= 0:
                continue
            kills += min(2, room.combat_bouts)
        self.killed_enemies = kills

    def to_dict(self) -> dict:
        return {
            "rooms": {
                name: {
                    "wall_color": r.wall_color.value,
                    "has_locked_door": r.has_locked_door.value,
                    "images": r.images,
                    "objects": r.objects,
                    "enemies": r.enemies,
                }
                for name, r in self.rooms.items()
            },
            "hint_text": self.hint_text.value,
            "locked_door_room": self.locked_door_room,
            "key_found": self.key_found,
            "key_found_in": self.key_found_in,
            "door_unlocked": self.door_unlocked,
            "key_hint_room": self.key_hint_room,
            "killed_enemies": self.killed_enemies,
            "damage_taken_total": self.damage_taken_total,
            "hp_start": self.hp_start,
            "hp_end": self.hp_end,
            "hp_max": self.hp_max,
            "seconds_left_last": self.seconds_left_last,
            "visited_order": self.visited_order,
            "frozen": self.frozen,
        }


def _add_with_cap(existing: list, new_item: dict, cap: int) -> None:
    """existing에 new_item을 추가하되, cap을 넘으면 confidence가 제일 낮은
    기존 항목을 새 항목으로 교체한다(둘 다 게임 구조상 정해진 상한을 넘는
    관측은 십중팔구 오염이므로, 무한정 쌓이게 두지 않는다)."""
    if len(existing) < cap:
        existing.append(new_item)
        return
    worst_idx = min(range(len(existing)), key=lambda i: existing[i].get("confidence", 0.0))
    if new_item.get("confidence", 0.0) > existing[worst_idx].get("confidence", 0.0):
        existing[worst_idx] = new_item
    # else: 새 항목이 기존 중 제일 낮은 것보다도 낮으면 그냥 버림(상한 유지).


def _has_similar(existing: list, new_item: dict, text_key: str, threshold: float = 0.35) -> bool:
    """new_item이 이미 있는 항목과 (같은 물건을 VLM이 다르게 표현한 것으로
    보일 만큼) 비슷하면 True. VLM이 현재 DB를 보고 스스로 중복을 피하는 게
    1차 방어선이고, 이건 그래도 새는 경우를 잡는 2차 안전망이다."""
    new_text = str(new_item.get(text_key, "")).lower()
    new_tokens = set(new_text.split())
    if not new_tokens:
        return False
    for item in existing:
        old_tokens = set(str(item.get(text_key, "")).lower().split())
        if not old_tokens:
            continue
        inter = len(new_tokens & old_tokens)
        union = len(new_tokens | old_tokens)
        if union and inter / union >= threshold:
            return True
    return False


CARDINAL_HEADINGS = (0, 90, 180, 270)


def _snap_heading(heading):
    """VLM이 준 wall_dir을 가장 가까운 기본 방위로 스냅. 문/벽은 오직
    0/90/180/270에만 있다(world/layout.py의 격자 구조)."""
    if heading is None:
        return None
    try:
        h = int(heading) % 360
    except (TypeError, ValueError):
        return None
    return min(CARDINAL_HEADINGS, key=lambda c: min(abs(h - c), 360 - abs(h - c)))


def _adjacency_from_scene(scene) -> dict:
    """SceneGraph -> {방: {나가는 헤딩: 이웃 방}}. 실제로 걸어서 통과한
    문만 들어있다(explorer가 exit_leads_to를 그때만 채움)."""
    if scene is None:
        return {}
    return {name: dict(node.exit_leads_to) for name, node in scene.nodes.items()}


def _same_enemy_in(existing: list, new_item: dict) -> bool:
    """외형(셔츠/바지/피부/체형)이 같으면 같은 몹으로 본다. 색이 하나도
    안 채워진 경우엔 설명 텍스트 유사도로 대신 판단한다."""
    key = tuple(new_item.get(k) for k in
                ("shirt_color", "pants_color", "skin_color", "body_shape"))
    if any(v for v in key):
        for old in existing:
            if tuple(old.get(k) for k in
                     ("shirt_color", "pants_color", "skin_color", "body_shape")) == key:
                return True
        return False
    return _has_similar(existing, new_item, text_key="desc")
