"""content_db.py의 confidence 게이트 + 병합 로직을 가짜 데이터로 검증."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.content_db import ContentDB, CONFIDENCE_THRESHOLD

db = ContentDB()

# 1) confidence 미만은 커밋 안 됨.
db.apply_update("Ebon Annex", {
    "wall_color": {"value": "brick", "confidence": 0.5},  # 0.8 미만 -> 무시
    "images": [{"desc": "a picture of a horse", "wall_dir": 0, "confidence": 0.6}],  # 무시
})
room = db.rooms["Ebon Annex"]
assert room.wall_color.value is None, f"낮은 confidence인데 커밋됨: {room.wall_color}"
assert len(room.images) == 0, f"낮은 confidence인데 이미지 커밋됨: {room.images}"
print("[PASS] confidence 미만은 커밋 안 됨")

# 2) confidence 이상은 커밋됨.
db.apply_update("Ebon Annex", {
    "wall_color": {"value": "brick", "confidence": 0.9},
    "images": [{"desc": "a picture of a horse on a wall", "wall_dir": 0, "confidence": 0.85}],
})
assert room.wall_color.value == "brick"
assert len(room.images) == 1
print("[PASS] confidence 이상은 커밋됨")

# 3) 비슷한 이미지(캡션 겹침)는 중복 추가 안 됨.
db.apply_update("Ebon Annex", {
    "images": [{"desc": "a picture of a horse hanging on the wall", "wall_dir": 0, "confidence": 0.9}],
})
assert len(room.images) == 1, f"비슷한 이미지가 중복 추가됨: {room.images}"
print("[PASS] 비슷한 항목 중복 추가 안 됨")

# 4) 명확히 다른 이미지는 새로 추가됨.
db.apply_update("Ebon Annex", {
    "images": [{"desc": "a photo of a blue airplane in the sky", "wall_dir": 90, "confidence": 0.9}],
})
assert len(room.images) == 2, f"다른 이미지가 안 추가됨: {room.images}"
print("[PASS] 명확히 다른 항목은 추가됨")

# 5) 더 낮은 confidence로는 기존 값을 덮어쓰지 않음.
db.apply_update("Ebon Annex", {"wall_color": {"value": "rust", "confidence": 0.82}})
assert room.wall_color.value == "brick", f"낮은 confidence가 기존 값을 덮어씀: {room.wall_color}"
print("[PASS] 더 낮은 confidence는 기존 값 유지")

# 6) 힌트 텍스트 -> key_hint_room 자동 세팅.
db.apply_update("Pearl Vault", {"hint_text": {"value": "the room with sage walls", "confidence": 0.9}})
assert db.hint_text.value == "the room with sage walls"
assert db.key_hint_room == "Pearl Vault"
print("[PASS] 힌트 텍스트 -> key_hint_room 연결")

# 7) 잠긴 문 -> locked_door_room 자동 세팅.
db.apply_update("Ivory Gallery", {"has_locked_door": True, "has_locked_door_confidence": 0.95})
assert db.locked_door_room == "Ivory Gallery"
print("[PASS] 잠긴 문 감지 -> locked_door_room 연결")

print("\n=== 전체 통과 ===")
import json
print(json.dumps(db.to_dict(), indent=2, ensure_ascii=False))
