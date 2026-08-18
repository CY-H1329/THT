"""Replay DB_maker: OCR/frame, 2× GPT-4o parallèle / salle, images only.

  .venv/bin/python tmp/run_qa_db.py --live
  .venv/bin/python tmp/run_qa_db.py --live --limit 800
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

import numpy as np
from PIL import Image

from qa_ingest import QaIngest
from qa_vlm import SURVEY_MODEL

DATASET = ROOT / "tmp" / "vlm_dataset" / "DB_maker"
OUT_DIR = ROOT / "tmp" / "qa_out"


def _step(p: Path) -> int:
    m = re.search(r"(\d+)\.png$", p.name)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--live", action="store_true", help="1 dual-GPT/salle, on_frame non bloquant")
    ap.add_argument("--cost-limit", type=float, default=10.0)
    args = ap.parse_args()

    files = sorted(DATASET.glob("*.png"), key=_step)
    if args.limit:
        files = files[: args.limit]
    live = args.live
    print(f"{len(files)} frames live={live} model={SURVEY_MODEL} dry={args.dry_run}")

    events = []

    def on_event(kind, room, result, headings):
        rec = {
            "kind": kind,
            "room": room,
            "ok": result.ok,
            "error": result.error,
            "cost": result.cost_usd,
            "latency": result.latency_s,
            "n_frames": result.n_frames,
            "model": result.model,
            "headings": headings,
        }
        if result.data:
            rec["images"] = [
                x.get("category") or x.get("desc")
                for x in (result.data.get("images") or [])
            ]
            rec["objects"] = [x.get("name") for x in (result.data.get("objects") or [])]
            rec["enemies"] = result.data.get("enemies_visible") or result.data.get("enemies")
        events.append(rec)
        extra = ""
        if rec.get("images"):
            extra += f" img={rec['images']}"
        if rec.get("objects"):
            extra += f" obj={rec['objects']}"
        if rec.get("enemies"):
            extra += f" en={len(rec['enemies'])}"
        print(
            f"  [{kind}] {room!r} ok={result.ok} {result.model} "
            f"${result.cost_usd:.4f} {result.latency_s:.1f}s n={result.n_frames} err={result.error}{extra}"
        )

    ingest = QaIngest(
        dry_run=args.dry_run,
        cost_limit_usd=args.cost_limit,
        on_event=on_event,
        live_mode=live,
        survey_model=SURVEY_MODEL,
    )
    for i, path in enumerate(files):
        frame = np.array(Image.open(path).convert("RGB"))
        ingest.on_frame(frame, step=_step(path))
        if (i + 1) % 500 == 0:
            print(f"  streamed {i+1}/{len(files)} cost=${ingest.total_cost:.4f} surveys={ingest.n_surveys}")

    wait_s = 180.0 if live else 4.5
    db = ingest.finalize(wait_s=wait_s)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = db.to_dict()
    payload["_meta"] = {
        "n_frames": ingest.n_frames,
        "n_surveys": ingest.n_surveys,
        "n_verify": ingest.n_verify,
        "n_witness": ingest.n_witness,
        "total_cost_usd": ingest.total_cost,
        "live_mode": live,
        "model": SURVEY_MODEL,
        "image_only": True,
        "dual_gpt4o": True,
        "dry_run": args.dry_run,
        "events": events,
    }
    out = OUT_DIR / ("db_dry.json" if args.dry_run else "db.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n=== done cost=${ingest.total_cost:.4f} surveys={ingest.n_surveys} "
        f"verify={ingest.n_verify} frozen={db.frozen} -> {out}"
    )
    for name, room in payload["rooms"].items():
        print(
            f"  {name!r}: color={room['wall_color']} doors={room['door_count']} "
            f"img={room['image_count']} obj={room['object_count']} "
            f"en={len(room['enemies'])} dmg={room['damage_taken']}"
        )
    print(
        "hint:", payload.get("hint_text"), "key_found:", payload.get("key_found"),
        "key_room:", payload.get("key_found_in") or payload.get("key_hint_room"),
    )


if __name__ == "__main__":
    main()
