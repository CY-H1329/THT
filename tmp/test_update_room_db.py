"""vlm.update_room_db()를 실제 방 프레임으로 테스트 (v3).

v1 문제: 인덱스 균등 간격 -> 같은 장면 반복. v2에서 각도 버킷으로 해결.
v2 문제: 문 바로 앞 프레임이 섞여서 다른 방 내용(적/오브젝트/그림)이
잘못 들어감(실측 확인). v3에서는:
  1) 먼저 이 방의 진짜 벽색을 CV로 추정(다수결).
  2) 그 벽색이 화면의 50% 이상을 덮는 프레임만 후보로 남김(문 앞 프레임
     배제 — 문틈으로 다른 방이 크게 보이면 이 비율이 낮아짐).
  3) 그 후보들로만 각도 버킷 샘플링 + VLM 호출.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from agent.vlm import update_room_db
from agent.content_db import ContentDB
from agent.vision import sample_wall_color, wall_color_fraction

DATASET = ROOT / "tmp" / "vlm_dataset"
ROOM = sys.argv[1] if len(sys.argv) > 1 else "Ebon Annex"
BUCKET_DEG = 15
MAX_PER_BUCKET = 2
BATCH_SIZE = 8
COVERAGE_THRESHOLD = 0.95

scan = json.loads((ROOT / "tmp" / "dataset_scan.json").read_text(encoding="utf-8"))
room_records = [r for r in scan if r.get("room") == ROOM and r["ok"] and r["heading"] is not None]
print(f"{ROOM} 프레임 총 {len(room_records)}장")

# 1) 벽색 다수결로 확정.
color_votes: dict[str, int] = {}
for r in room_records:
    frame = np.array(Image.open(DATASET / r["file"]).convert("RGB"))
    c = sample_wall_color(frame)
    if c:
        color_votes[c] = color_votes.get(c, 0) + 1
room_color = max(color_votes, key=color_votes.get) if color_votes else None
print(f"이 방의 벽색(CV 다수결): {room_color} (투표 분포: {color_votes})")

# 2) 벽색 커버리지 50% 미만인 프레임(문 앞 등) 제외.
kept = []
excluded = 0
for r in room_records:
    frame = np.array(Image.open(DATASET / r["file"]).convert("RGB"))
    frac = wall_color_fraction(frame, room_color) if room_color else 1.0
    if frac >= COVERAGE_THRESHOLD:
        kept.append(r)
    else:
        excluded += 1
print(f"벽색 커버리지 >= {COVERAGE_THRESHOLD} 인 프레임: {len(kept)}장 (제외됨: {excluded}장)")

# 3) 남은 후보로 각도 버킷 샘플링.
by_bucket: dict[int, list[str]] = {}
for r in kept:
    b = (r["heading"] // BUCKET_DEG) * BUCKET_DEG
    by_bucket.setdefault(b, []).append(r["file"])

picked = []
for b in sorted(by_bucket):
    files_sorted = sorted(by_bucket[b])
    step = max(1, len(files_sorted) // MAX_PER_BUCKET)
    picked.extend(files_sorted[::step][:MAX_PER_BUCKET])

print(f"각도 버킷 {len(by_bucket)}개 중 총 {len(picked)}장 선택")
print(f"확보된 각도: {sorted(by_bucket.keys())}")

db = ContentDB()
total_cost = 0.0
for i in range(0, len(picked), BATCH_SIZE):
    batch_files = picked[i:i + BATCH_SIZE]
    frames = [np.array(Image.open(DATASET / f).convert("RGB")) for f in batch_files]
    current = db.to_dict()["rooms"].get(ROOM, {})
    r = update_room_db(frames, current, room_hint=ROOM)
    total_cost += r.cost_usd
    print(f"  배치 {i//BATCH_SIZE + 1} ({len(batch_files)}장): ok={r.ok} "
          f"cost={r.cost_usd:.5f} error={r.error}")
    if r.ok:
        db.apply_update(ROOM, r.data)

print(f"\n총 비용: ${total_cost:.4f}")
print(f"\n=== 최종 {ROOM} ===")
final = db.rooms[ROOM]
print("wall_color:", final.wall_color)
print("has_locked_door:", final.has_locked_door)
print(f"images ({len(final.images)}):")
for i in final.images:
    print(" -", i)
print(f"objects ({len(final.objects)}):")
for o in final.objects:
    print(" -", o)
print(f"enemies ({len(final.enemies)}):")
for e in final.enemies:
    print(" -", e)
