"""Compare GPT OpenRouter sur le même survey 6 frames (Vermilion, masqué).

  .venv/bin/python tmp/bench_gpt.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from exp_portal_mask import pick_frames, summarize
from qa_vlm import survey_room

OUT = ROOT / "tmp" / "qa_out" / "gpt_bench.json"

MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-5-nano",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.5",
]


def main():
    picks, color = pick_frames({0: 209, 90: 322, 180: 310, 270: 298}, n=6)
    frames = [p["frame"] for p in picks]
    headings = [p["heading"] for p in picks]
    print(f"color={color} headings={headings}")
    rows = []
    for model in MODELS:
        print(f"\n== {model} ==")
        r = survey_room(
            frames, headings, "Vermilion Salon", {}, color,
            model=model, mask_portals=True,
        )
        raw = summarize(r.data)
        rec = {
            "model": model,
            "ok": r.ok,
            "error": r.error,
            "latency_s": round(r.latency_s, 2),
            "cost_usd": round(r.cost_usd, 4),
            "images": raw.get("images"),
            "objects": raw.get("objects"),
            "enemies": raw.get("enemies"),
            "n_images": raw.get("n_images"),
            "n_objects": raw.get("n_objects"),
        }
        rows.append(rec)
        print(
            f"  ok={r.ok} {r.latency_s:.1f}s ${r.cost_usd:.4f} "
            f"img={raw.get('images')} obj={raw.get('objects')} en={raw.get('enemies')} "
            f"err={r.error}"
        )
    OUT.write_text(json.dumps({"color": color, "headings": headings, "runs": rows}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
