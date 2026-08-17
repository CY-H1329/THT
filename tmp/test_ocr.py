"""agent/ocr.py의 read_hud()를 debug_frames PNG 전체에 돌려서 결과를
JSON 하나로 저장하는 검증 스크립트. 시각화 이미지는 안 만들고, 구조화된
데이터(JSON)만 남긴다. env를 새로 실행하지 않고 이미 찍어둔 스크린샷만
읽는다 (움직임/행동 없음).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # memory_fps_env editable-install 우회 (dev_log.md 참고)

import numpy as np
from PIL import Image

from agent.ocr import read_hud

FRAMES_DIR = ROOT / "tmp" / "debug_frames"
OUT_JSON = ROOT / "tmp" / "ocr_readings.json"

results = []
for png in sorted(FRAMES_DIR.glob("*.png")):
    frame = np.array(Image.open(png).convert("RGB"))
    r = read_hud(frame)
    results.append({
        "file": png.name,
        "ok": r.ok,
        "room_name": r.room_name,
        "is_corridor": r.is_corridor,
        "seconds_remaining": r.seconds_remaining,
        "hp": r.hp,
        "hp_max": r.hp_max,
        "heading": r.heading,
        "has_key": r.has_key,
        "raw_text": r.raw_text,
    })

n_ok = sum(1 for r in results if r["ok"])
n_corridor = sum(1 for r in results if r["is_corridor"])
rooms = sorted({r["room_name"] for r in results if r["room_name"]})

OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{n_ok}/{len(results)} frames parsed ok, {n_corridor} corridor frames, "
      f"{len(rooms)} distinct room names seen: {rooms}")
print("failures:")
for r in results:
    if not r["ok"]:
        print(f"  {r['file']}: raw={r['raw_text']!r}")
print("saved ->", OUT_JSON)
