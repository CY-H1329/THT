"""image_count=0 인 방만 가장 긴 방문으로 재서베이. 통과한 문은 locked가 아님."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

import numpy as np
from PIL import Image

from agent.ocr import read_hud
from agent.vision import sample_wall_color, wall_color_fraction
from qa_db import CARDINAL, EpisodeDB
from qa_vlm import survey_room
from calibrate_qa import _view_score, COVERAGE_THRESHOLD

DATASET = ROOT / "tmp" / "vlm_dataset" / "DB_maker"
SCAN = ROOT / "tmp" / "dbmaker_scan.json"
DB_PATH = ROOT / "tmp" / "qa_out" / "db.json"


def step_of(p: Path) -> int:
    m = re.search(r"(\d+)\.png$", p.name)
    return int(m.group(1)) if m else 0


def main():
    dbj = json.loads(DB_PATH.read_text(encoding="utf-8"))
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    files = sorted(DATASET.glob("*.png"), key=step_of)
    extra_cost = 0.0

    empty = [n for n, r in dbj["rooms"].items() if r.get("image_count", 0) == 0]
    print("empty-image rooms:", empty)

    for room in empty:
        visits = scan["rooms"][room]["visits"]
        vis = max(visits, key=lambda v: v["n"])
        chunk = files[vis["start_i"]: vis["end_i"] + 1]
        recs = []
        votes = {}
        for p in chunk:
            fr = np.array(Image.open(p).convert("RGB"))
            hud = read_hud(fr)
            if not hud.ok or hud.heading is None:
                continue
            c = sample_wall_color(fr)
            if c:
                votes[c] = votes.get(c, 0) + 1
            recs.append((hud.heading, fr, c))
        majority = max(votes, key=votes.get) if votes else None
        best = {}
        for h, fr, c in recs:
            cov = wall_color_fraction(fr, majority) if majority else 0.0
            if majority and cov < COVERAGE_THRESHOLD:
                continue
            b = min(CARDINAL, key=lambda d: min(abs(h - d), 360 - abs(h - d)))
            off = min(abs(h - b), 360 - abs(h - b))
            if off > 8:
                continue
            item = {"heading": h, "frame": fr, "offset": off, "coverage": cov}
            old = best.get(b)
            if old is None or off < old["offset"] or (
                off == old["offset"] and _view_score(cov) > _view_score(old["coverage"])
            ):
                best[b] = item
        views = [best[d] for d in CARDINAL if d in best]
        print(f"  {room!r} longest visit n={vis['n']} cardinals={len(views)} color={majority}")
        tmp = EpisodeDB()
        if majority:
            tmp.apply_cv_wall_color(room, majority)
        current = tmp.to_dict()["rooms"].get(room, {})
        for v in views:
            r = survey_room([v["frame"]], [v["heading"]], room, current, majority)
            extra_cost += r.cost_usd
            print(f"    h={v['heading']} ok={r.ok} ${r.cost_usd:.4f} {r.error}")
            if r.ok and r.data:
                tmp.apply_survey(room, r.data)
                current = tmp.to_dict()["rooms"].get(room, {})
        rec = tmp.rooms[room]
        target = dbj["rooms"][room]
        target["images"] = rec.images
        target["image_count"] = len(rec.images)
        target["images_per_wall"] = {str(d): rec.image_count_on_wall(d) for d in CARDINAL}
        if rec.objects:
            target["objects"] = rec.objects
            target["object_count"] = len(rec.objects)
        if rec.enemies and not target.get("enemies"):
            target["enemies"] = rec.enemies
        print(f"    -> images={target['image_count']} objects={target['object_count']}")

    # 실제로 통과한 문은 locked가 될 수 없음
    for room in dbj["rooms"].values():
        for slot in room["doors"].values():
            if slot.get("leads_to"):
                slot["status"] = "open"
        room["door_count"] = sum(1 for s in room["doors"].values() if s["status"] in ("open", "locked"))
        room["has_locked_door"] = any(s["status"] == "locked" for s in room["doors"].values())

    locked = [n for n, r in dbj["rooms"].items() if r.get("has_locked_door")]
    dbj["locked_door_room"] = locked[0] if len(locked) == 1 else (locked[0] if locked else None)
    dbj["_meta"]["refine_cost_usd"] = extra_cost
    dbj["_meta"]["total_cost_usd"] = float(dbj["_meta"].get("total_cost_usd") or 0) + extra_cost

    DB_PATH.write_text(json.dumps(dbj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"refine cost ${extra_cost:.4f} -> {DB_PATH}")
    print("locked rooms:", locked)


if __name__ == "__main__":
    main()
