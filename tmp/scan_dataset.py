"""tmp/vlm_dataset/의 프레임들을 OCR(공짜)로 훑어서, VLM에 보낼 대표
프레임을 추린다.

선정 기준:
  1) 힌트 배너가 떠 있는 프레임 — 각도 상관없이 텍스트별로 전부 포함
     (놓치면 다시 못 구하는 귀한 정보라서 중복 제거 대상에서 제외).
  2) 방 이름 + 30도 헤딩 버킷 별 대표 프레임 1장씩(각도 커버리지용).
  3) 마지막에 "이 방에서 못 본 각도"가 있으면 따로 보고 — 놓친 방향이
     있으면 나중에 그 방향만 추가로 둘러보시면 됨.
내용(그림/오브젝트가 뭔지)은 전혀 안 본다 — 그건 다음 VLM 단계의 몫.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
from PIL import Image

from agent.ocr import read_hud, hint_banner_active, read_hint_text

DATASET = ROOT / "tmp" / "vlm_dataset"
OUT_JSON = ROOT / "tmp" / "dataset_scan.json"
SELECTED_JSON = ROOT / "tmp" / "dataset_selected.json"
HEADING_BUCKET_DEG = 30
ALL_BUCKETS = list(range(0, 360, HEADING_BUCKET_DEG))

files = sorted(DATASET.glob("*.png"))
print(f"총 {len(files)}장 스캔 중...")

records = []
for i, f in enumerate(files):
    frame = np.array(Image.open(f).convert("RGB"))
    hud = read_hud(frame)
    rec = {
        "file": f.name,
        "ok": hud.ok,
        "room": hud.room_name,
        "corridor": hud.is_corridor,
        "hp": hud.hp,
        "hp_max": hud.hp_max,
        "heading": hud.heading,
        "has_key": hud.has_key,
        "hint_active": hint_banner_active(frame),
    }
    if rec["hint_active"]:
        rec["hint_text"] = read_hint_text(frame)
    records.append(rec)
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(files)}")

OUT_JSON.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def canon(name, known):
    if name is None:
        return None
    if name in known:
        return name
    stripped = name.rstrip("…")
    for k in known:
        kk = k.rstrip("…")
        if stripped == kk or stripped.startswith(kk) or kk.startswith(stripped):
            return k
    return name


known_rooms = []
for r in records:
    if r["room"] and r["room"] not in known_rooms:
        c = canon(r["room"], known_rooms)
        if c not in known_rooms:
            known_rooms.append(c)

selected = []  # list of (file, reason)

# 1) 힌트 배너 프레임 — 텍스트별로 1장씩(같은 텍스트 반복은 중복이므로 제외).
seen_hint_texts = set()
for r in records:
    if r.get("hint_active") and r.get("hint_text"):
        if r["hint_text"] not in seen_hint_texts:
            seen_hint_texts.add(r["hint_text"])
            selected.append((r["file"], "hint"))

# 2) 방+각도 버킷 대표 프레임.
seen_buckets = {}
for r in records:
    if not r["ok"] or r["room"] is None or r["heading"] is None:
        continue
    room = canon(r["room"], known_rooms)
    bucket = (r["heading"] // HEADING_BUCKET_DEG) * HEADING_BUCKET_DEG
    key = (room, bucket)
    if key not in seen_buckets:
        seen_buckets[key] = r["file"]
        selected.append((r["file"], f"room_angle:{room}:{bucket}"))

SELECTED_JSON.write_text(
    json.dumps([f for f, _ in selected], ensure_ascii=False), encoding="utf-8"
)

print(f"\n정규화된 고유 방: {len(known_rooms)}개 -> {known_rooms}")

by_room = {}
for (room, bucket) in seen_buckets:
    by_room.setdefault(room, set()).add(bucket)

print("\n방별 각도 커버리지 (30도 단위):")
for room in known_rooms:
    covered = by_room.get(room, set())
    missing = [b for b in ALL_BUCKETS if b not in covered]
    print(f"  {room!r}: {len(covered)}/12 각도 확인함"
          + (f" | 못 본 방향: {missing}" if missing else " | 전 방향 확인 완료"))

print(f"\nVLM에 보낼 프레임: 총 {len(selected)}장 "
      f"(힌트 {len(seen_hint_texts)}장 + 방향 커버리지 {len(seen_buckets)}장), "
      f"원본 {len(files)}장에서 축소")
print(f"저장: {SELECTED_JSON}")

hints = [r for r in records if r.get("hint_active")]
print(f"\n힌트 배너 감지된 프레임: {len(hints)}장, 고유 텍스트 {len(seen_hint_texts)}개")
for t in seen_hint_texts:
    print(f"  - {t!r}")
