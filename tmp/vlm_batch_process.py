"""tmp/vlm_dataset/ 전체를 VLM으로 처리 + OCR로 방/방향 정보도 같이 붙여서
JSONL로 저장한다.

- 병렬 처리(스레드풀)로 시간 단축.
- 누적 비용이 COST_LIMIT_USD를 넘으면 새 작업 제출을 멈춘다(진행 중인
  건 마저 끝냄).
- 한 줄씩(JSONL) 즉시 저장 — 중간에 멈춰도 그때까지 결과는 안전함.
"""

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from agent.config import get_api_key
from agent.ocr import read_hud, hint_banner_active, read_hint_text
from agent.vlm import describe_view

DATASET = ROOT / "tmp" / "vlm_dataset"
OUT_JSONL = ROOT / "tmp" / "vlm_results.jsonl"
COST_LIMIT_USD = 5.0
N_WORKERS = 8

api_key = get_api_key()
if not api_key:
    print("API 키를 못 찾았습니다 (OPENROUTER_API_KEY 또는 key*.env 확인).")
    sys.exit(1)

SELECTED_JSON = ROOT / "tmp" / "dataset_selected.json"
if len(sys.argv) > 1 and sys.argv[1] == "--selected":
    names = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    files = [DATASET / n for n in names]
    print(f"방+각도 버킷 대표 프레임 {len(files)}장 사용 ({SELECTED_JSON.name})")
else:
    files = sorted(DATASET.glob("*.png"))
    if len(sys.argv) > 1:
        files = files[: int(sys.argv[1])]  # 테스트용: 앞에서부터 N장만
print(f"총 {len(files)}장, 워커 {N_WORKERS}개, 비용 상한 ${COST_LIMIT_USD}")

lock = threading.Lock()
total_cost = 0.0
stop_flag = False
done_count = 0
ok_count = 0
t0 = time.monotonic()


def process_one(path: Path):
    global total_cost, stop_flag, done_count, ok_count
    with lock:
        if stop_flag:
            return None

    frame = np.array(Image.open(path).convert("RGB"))
    hud = read_hud(frame)
    hint_active = hint_banner_active(frame)
    hint_text = read_hint_text(frame) if hint_active else None

    vlm = describe_view(frame, api_key=api_key)

    record = {
        "file": path.name,
        "room": hud.room_name,
        "corridor": hud.is_corridor,
        "hp": hud.hp,
        "hp_max": hud.hp_max,
        "heading": hud.heading,
        "has_key": hud.has_key,
        "hint_active": hint_active,
        "hint_text": hint_text,
        "vlm_ok": vlm.ok,
        "vlm_error": vlm.error,
        "vlm_cost": vlm.cost_usd,
        "vlm_data": vlm.data,
    }

    with lock:
        total_cost += vlm.cost_usd
        done_count += 1
        if vlm.ok:
            ok_count += 1
        if total_cost >= COST_LIMIT_USD:
            stop_flag = True
    return record


with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for f in files:
            with lock:
                if stop_flag:
                    print(f"\n비용 상한 도달, 남은 작업 제출 중단 (제출된 {len(futures)}개는 계속 진행)")
                    break
            futures[pool.submit(process_one, f)] = f

        for fut in as_completed(futures):
            rec = fut.result()
            if rec is None:
                continue
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if done_count % 100 == 0:
                elapsed = time.monotonic() - t0
                print(f"  {done_count}/{len(futures)} 완료, ok={ok_count}, "
                      f"누적비용=${total_cost:.3f}, 경과={elapsed:.0f}s")

elapsed = time.monotonic() - t0
print(f"\n=== 완료: {done_count}장 처리(ok={ok_count}), "
      f"총비용=${total_cost:.3f}, 소요={elapsed:.0f}s ===")
print(f"결과: {OUT_JSONL}")
