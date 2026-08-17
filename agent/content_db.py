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
    objects: list = field(default_factory=list)  # [{name, confidence}]
    enemies: list = field(default_factory=list)  # [{desc, color, confidence}]
    frame_count: int = 0


class ContentDB:
    def __init__(self) -> None:
        self.rooms: dict[str, RoomContent] = {}
        self.hint_text = ConfidentValue()
        self.locked_door_room: Optional[str] = None
        self.key_found: bool = False
        self.key_hint_room: Optional[str] = None
        self.killed_enemies: int = 0

    def get_or_create(self, room_name: str) -> RoomContent:
        if room_name not in self.rooms:
            self.rooms[room_name] = RoomContent(name=room_name)
        return self.rooms[room_name]

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
            "key_hint_room": self.key_hint_room,
            "killed_enemies": self.killed_enemies,
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
