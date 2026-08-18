"""DB_maker 프레임을 OCR-only로 훑어 방 전환 / 360 스핀 / HP / 열쇠 / 힌트를 집계.

VLM 없음. 이후 캘리브레이션·파이프라인의 입력 인덱스.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from agent.ocr import hint_banner_active, read_hint_text, read_hud
from agent.vision import sample_wall_color

DATASET = ROOT / "tmp" / "vlm_dataset" / "DB_maker"
OUT = ROOT / "tmp" / "dbmaker_scan.json"


def _step(path: Path) -> int:
    m = re.search(r"(\d+)\.png$", path.name)
    return int(m.group(1)) if m else 0


def canon(name, known):
    if name is None:
        return None
    if name in known:
        return name
    stripped = name.rstrip("…").rstrip(".").rstrip()
    if len(stripped) < 6:
        return name
    for k in known:
        kk = k.rstrip("…").rstrip(".").rstrip()
        if stripped == kk or stripped.startswith(kk) or kk.startswith(stripped):
            return k
    return name


def detect_spins(headings: list[int], min_span: int = 18) -> list[dict]:
    """연속 heading이 ±15°로 도는 구간. min_span=18이면 약 270°+."""
    if len(headings) < min_span:
        return []
    spins = []
    i = 0
    n = len(headings)
    while i < n - 1:
        d = (headings[i + 1] - headings[i]) % 360
        if d == 15:
            sign = 1
        elif d == 345:
            sign = -1
        else:
            i += 1
            continue
        j = i
        while j + 1 < n:
            dd = (headings[j + 1] - headings[j]) % 360
            if sign == 1 and dd == 15:
                j += 1
            elif sign == -1 and dd == 345:
                j += 1
            else:
                break
        length = j - i + 1
        if length >= min_span:
            span = (length - 1) * 15
            spins.append({
                "start_idx": i,
                "end_idx": j,
                "length": length,
                "span_deg": span,
                "dir": "cw" if sign == 1 else "ccw",
                "start_heading": headings[i],
                "end_heading": headings[j],
            })
            i = j
        else:
            i += 1
    return spins


def main():
    files = sorted(DATASET.glob("*.png"), key=_step)
    print(f"scanning {len(files)} frames in {DATASET}")
    records = []
    known = []
    prev_hp = None
    prev_key = False
    prev_room = None
    events = []

    for i, path in enumerate(files):
        frame = np.array(Image.open(path).convert("RGB"))
        hud = read_hud(frame)
        room = None
        if hud.ok and hud.room_name and not hud.is_corridor:
            room = canon(hud.room_name, known)
            if room not in known:
                known.append(room)
        hint_on = hint_banner_active(frame)
        hint_text = read_hint_text(frame) if hint_on else None
        rec = {
            "i": i,
            "file": path.name,
            "step": _step(path),
            "ok": hud.ok,
            "room": room,
            "corridor": bool(hud.is_corridor),
            "heading": hud.heading,
            "hp": hud.hp,
            "hp_max": hud.hp_max,
            "has_key": hud.has_key,
            "hint": hint_on,
            "hint_text": hint_text,
            "wall_color": sample_wall_color(frame) if hud.ok and not hud.is_corridor else None,
        }
        records.append(rec)

        if hud.ok and hud.hp is not None and prev_hp is not None and hud.hp < prev_hp:
            events.append({
                "type": "damage",
                "step": rec["step"],
                "room": room or prev_room,
                "amount": prev_hp - hud.hp,
                "hp_after": hud.hp,
            })
        if hud.ok and hud.has_key and not prev_key:
            events.append({"type": "key_pickup", "step": rec["step"], "room": room})
        if hud.ok and prev_key and not hud.has_key:
            events.append({"type": "key_consumed", "step": rec["step"], "room": room})
        if hint_on and hint_text:
            events.append({"type": "hint", "step": rec["step"], "room": room, "text": hint_text})
        if prev_room and room and prev_room != room:
            events.append({
                "type": "transition",
                "step": rec["step"],
                "from": prev_room,
                "to": room,
                "heading": hud.heading,
            })
        if hud.ok:
            prev_hp = hud.hp
            prev_key = hud.has_key
            if room:
                prev_room = room
            elif hud.is_corridor:
                prev_room = None
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}")

    rooms = {}
    for rec in records:
        if not rec["room"]:
            continue
        node = rooms.setdefault(rec["room"], {
            "n_frames": 0,
            "headings": [],
            "files": [],
            "idxs": [],
            "color_votes": {},
            "hp_min": 99,
            "hp_max": 0,
        })
        node["n_frames"] += 1
        node["headings"].append(rec["heading"])
        node["files"].append(rec["file"])
        node["idxs"].append(rec["i"])
        if rec["wall_color"]:
            node["color_votes"][rec["wall_color"]] = node["color_votes"].get(rec["wall_color"], 0) + 1
        if rec["hp"] is not None:
            node["hp_min"] = min(node["hp_min"], rec["hp"])
            node["hp_max"] = max(node["hp_max"], rec["hp"])

    for name, node in rooms.items():
        votes = node["color_votes"]
        node["wall_color"] = max(votes, key=votes.get) if votes else None
        node["unique_headings"] = sorted(set(h for h in node["headings"] if h is not None))
        node["spins"] = detect_spins([h if h is not None else -1 for h in node["headings"]])
        # 방문 구간(연속 인덱스)
        visits = []
        start = node["idxs"][0]
        prev = node["idxs"][0]
        for idx in node["idxs"][1:]:
            if idx != prev + 1:
                visits.append({"start_i": start, "end_i": prev, "n": prev - start + 1})
                start = idx
            prev = idx
        visits.append({"start_i": start, "end_i": prev, "n": prev - start + 1})
        node["visits"] = visits
        # 출력에서 긴 리스트는 요약만
        node.pop("headings")
        node.pop("files")
        node.pop("idxs")

    # 힌트 텍스트 중복 제거
    hints = []
    seen_h = set()
    for e in events:
        if e["type"] == "hint" and e["text"] not in seen_h:
            seen_h.add(e["text"])
            hints.append(e)

    summary = {
        "n_frames": len(records),
        "n_rooms": len(rooms),
        "rooms": rooms,
        "hints": hints,
        "events_n": len(events),
        "damage_events": [e for e in events if e["type"] == "damage"],
        "transitions": [e for e in events if e["type"] == "transition"],
        "key_events": [e for e in events if e["type"] in ("key_pickup", "key_consumed")],
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nrooms={len(rooms)} frames={len(records)} -> {OUT}")
    for name, node in rooms.items():
        print(f"  {name!r:30s} n={node['n_frames']:4d} color={node['wall_color']} "
              f"headings={len(node['unique_headings']):3d} spins={len(node['spins'])} "
              f"visits={len(node['visits'])}")
    print("hints:", hints)
    print("key:", summary["key_events"])
    print("damage n:", len(summary["damage_events"]), "total",
          sum(e["amount"] for e in summary["damage_events"]))


if __name__ == "__main__":
    main()
