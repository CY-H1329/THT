"""한 방에 대해 1장씩 vs 사방 배치 vs 사방+extra 를 같은 프레임으로 비교.

사용:
  .venv/bin/python tmp/calibrate_qa.py              # 스핀이 가장 긴 방
  .venv/bin/python tmp/calibrate_qa.py "Room Name"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

import numpy as np
from PIL import Image

from agent.vision import sample_wall_color, wall_color_fraction
from qa_db import CARDINAL, EpisodeDB
from qa_vlm import survey_room, witness_enemies

DATASET = ROOT / "tmp" / "vlm_dataset" / "DB_maker"
SCAN = ROOT / "tmp" / "dbmaker_scan.json"
OUT = ROOT / "tmp" / "qa_calib.json"
COVERAGE_THRESHOLD = 0.50


def load_frame(name: str) -> np.ndarray:
    return np.array(Image.open(DATASET / name).convert("RGB"))


def pick_room(scan: dict, want: str | None) -> str:
    if want and want in scan["rooms"]:
        return want
    # 360 커버 + 방문 1회인 방이 캘리브레이션에 가장 깨끗함.
    for name, node in scan["rooms"].items():
        if len(node.get("visits") or []) == 1 and len(node.get("unique_headings") or []) >= 20:
            return name
    return next(iter(scan["rooms"]))


def files_for_room(scan: dict, room: str) -> list[str]:
    """scan은 파일 리스트를 안 남겼으므로 DB_maker를 다시 OCR하지 않고
    dataset_scan이 없으면 파일명 순서 + 이 스크립트가 직접 고른다.
    방 프레임은 run에서 heading과 함께 넘긴다 — 여기서는 glob+HUD."""
    return []


def _view_score(coverage: float) -> float:
    if coverage < COVERAGE_THRESHOLD:
        return -1.0
    return -abs(coverage - 0.68)


def select_views(records: list[dict]) -> tuple[list[dict], str | None]:
    votes: dict[str, int] = {}
    frames = []
    for r in records:
        fr = load_frame(r["file"])
        c = sample_wall_color(fr)
        if c:
            votes[c] = votes.get(c, 0) + 1
        frames.append((r, fr, c))
    majority = max(votes, key=votes.get) if votes else None
    best: dict[int, dict] = {}
    extras_by: dict[int, dict] = {}
    for r, fr, c in frames:
        h = r.get("heading")
        if h is None:
            continue
        cov = wall_color_fraction(fr, majority) if majority else 0.0
        if majority and cov < COVERAGE_THRESHOLD:
            continue
        bucket = min(CARDINAL, key=lambda d: min(abs(h - d), 360 - abs(h - d)))
        off = min(abs(h - bucket), 360 - abs(h - bucket))
        item = {"file": r["file"], "heading": h, "coverage": cov, "frame": fr, "offset": off}
        if off <= 8:
            old = best.get(bucket)
            if old is None or off < old["offset"] or (
                off == old["offset"] and _view_score(cov) > _view_score(old["coverage"])
            ):
                best[bucket] = item
        else:
            eb = (h // 15) * 15
            old = extras_by.get(eb)
            if old is None or _view_score(cov) > _view_score(old["coverage"]):
                extras_by[eb] = item
    extras = sorted(extras_by.values(), key=lambda x: -_view_score(x["coverage"]))
    cardinals = [best[d] for d in CARDINAL if d in best]
    return cardinals + extras[:2], majority


def summarize(db: EpisodeDB, room: str) -> dict:
    r = db.rooms.get(room)
    if not r:
        return {}
    return {
        "wall_color": r.wall_color.value,
        "images": r.images,
        "objects": r.objects,
        "doors": {str(d): {"status": s.status, "neighbor_color": s.neighbor_color}
                  for d, s in r.doors.items()},
        "enemies": r.enemies,
        "has_locked_door": r.has_locked_door,
        "image_count": len(r.images),
        "object_count": len(r.objects),
    }


def run_strategy(name: str, views: list[dict], majority: str | None, room: str) -> dict:
    from agent.ocr import read_hud
    cardinals = [v for v in views if min(abs(v["heading"] - d) for d in CARDINAL) <= 8 or
                 min((360 - abs(v["heading"] - d)) for d in CARDINAL) <= 8]
    # 위 필터가 느슨할 수 있어 버킷으로 다시
    by_b = {}
    extras = []
    for v in views:
        b = min(CARDINAL, key=lambda d: min(abs(v["heading"] - d), 360 - abs(v["heading"] - d)))
        off = min(abs(v["heading"] - b), 360 - abs(v["heading"] - b))
        if off <= 8:
            by_b[b] = v
        else:
            extras.append(v)
    cardinals = [by_b[d] for d in CARDINAL if d in by_b]

    db = EpisodeDB()
    if majority:
        db.apply_cv_wall_color(room, majority)
    cost = 0.0
    latency = 0.0
    calls = []
    ok = True

    def commit(result, label):
        nonlocal cost, latency, ok
        cost += result.cost_usd
        latency += result.latency_s
        calls.append({"label": label, "ok": result.ok, "error": result.error,
                      "cost": result.cost_usd, "latency": result.latency_s,
                      "n_frames": result.n_frames})
        if result.ok and result.data:
            db.apply_survey(room, result.data)
        else:
            ok = False

    current = db.to_dict()["rooms"].get(room, {})
    if name == "sequential_1":
        for v in cardinals:
            commit(survey_room([v["frame"]], [v["heading"]], room, current, majority),
                   f"one@{v['heading']}")
            current = db.to_dict()["rooms"].get(room, {})
    elif name == "batch_cardinal":
        commit(survey_room([v["frame"] for v in cardinals],
                           [v["heading"] for v in cardinals],
                           room, current, majority), "batch4")
    elif name == "batch_cardinal_plus_extra":
        pack = cardinals + extras[:2]
        commit(survey_room([v["frame"] for v in pack],
                           [v["heading"] for v in pack],
                           room, current, majority), "batch4+2")
    else:
        raise ValueError(name)

    return {
        "strategy": name,
        "ok": ok,
        "cost_usd": cost,
        "latency_s": latency,
        "n_calls": len(calls),
        "calls": calls,
        "n_input_frames": len(cardinals) + (2 if name.endswith("extra") else 0),
        "headings": [v["heading"] for v in (cardinals + extras[:2] if name.endswith("extra") else cardinals)],
        "db": summarize(db, room),
    }


def collect_room_records(scan: dict, room: str) -> list[dict]:
    """스캔에 저장된 visit 인덱스 구간만 OCR — 전체 5300장을 다시 안 읽음."""
    import re
    from agent.ocr import read_hud

    def step_of(p: Path) -> int:
        m = re.search(r"(\d+)\.png$", p.name)
        return int(m.group(1)) if m else 0

    files = sorted(DATASET.glob("*.png"), key=step_of)
    visits = scan["rooms"][room]["visits"]
    out = []
    for vis in visits[:1]:  # 첫 방문(360 한 바퀴)만 — 캘리브레이션용
        chunk = files[vis["start_i"]: vis["end_i"] + 1]
        print(f"  visit frames {vis['start_i']}..{vis['end_i']} ({len(chunk)} files)")
        for p in chunk:
            fr = np.array(Image.open(p).convert("RGB"))
            hud = read_hud(fr)
            if not hud.ok or hud.is_corridor or hud.heading is None:
                continue
            out.append({"file": p.name, "heading": hud.heading, "step": step_of(p)})
    print(f"  kept {len(out)} in-room frames")
    return out


def main():
    if not SCAN.exists():
        print(f"need {SCAN} — run tmp/scan_dbmaker.py first")
        sys.exit(1)
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    want = sys.argv[1] if len(sys.argv) > 1 else "Glacial Annex"
    room = pick_room(scan, want)
    print(f"calibrating on room {room!r}")
    print("scan summary:", {k: {"n": v["n_frames"], "color": v["wall_color"],
                                "spins": len(v.get("spins") or [])}
                            for k, v in scan["rooms"].items()})

    records = collect_room_records(scan, room)
    views, majority = select_views(records)
    print(f"CV wall color={majority} selected {len(views)} views")
    for v in views:
        print(f"  h={v['heading']:3d} cov={v['coverage']:.2f} {v['file']}")

    results = []
    for strat in ("sequential_1", "batch_cardinal", "batch_cardinal_plus_extra"):
        print(f"\n=== {strat} ===")
        r = run_strategy(strat, views, majority, room)
        print(f"  ok={r['ok']} cost=${r['cost_usd']:.4f} latency={r['latency_s']:.1f}s "
              f"images={r['db'].get('image_count')} objects={r['db'].get('object_count')}")
        print("  objects:", r["db"].get("objects"))
        print("  images:", [im.get("category") or im.get("desc") for im in r["db"].get("images") or []])
        results.append(r)

    payload = {
        "room": room,
        "cv_wall_color": majority,
        "n_room_frames": len(records),
        "results": results,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
