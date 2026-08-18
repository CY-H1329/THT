"""Portail leak + modèle rapide. 2 salles, 4 conditions.

  .venv/bin/python tmp/exp_portal_mask.py           # VLM
  .venv/bin/python tmp/exp_portal_mask.py --preview # masques seulement
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

import numpy as np
from PIL import Image

from agent.ocr import read_hud
from qa_db import EpisodeDB
from qa_portal import best_wall_color, mask_portals
from qa_vlm import survey_room

DATASET = ROOT / "tmp" / "vlm_dataset" / "DB_maker"
OUT = ROOT / "tmp" / "qa_out"
PREVIEW = OUT / "portal_preview"

ROOMS = {
    "Vermilion Salon": {
        "cardinals": {0: 209, 90: 322, 180: 310, 270: 298},
        "why": "4 portes — photos du voisin (Citrine/Glacial) visibles dans le portail",
    },
    "Citrine  Library": {
        "cardinals": {0: 1577, 90: 1537, 180: 1523, 270: 1434},
        "why": "a reçu flamingo+car de Vermilion dans db.json v2",
    },
}

CONDITIONS = [
    {"id": "4o_nomask", "model": "openai/gpt-4o", "mask": False},
    {"id": "4o_mask", "model": "openai/gpt-4o", "mask": True},
    {"id": "mini_mask", "model": "openai/gpt-4o-mini", "mask": True},
    {"id": "flash_mask", "model": "google/gemini-2.5-flash", "mask": True},
]


def _load(step: int) -> np.ndarray:
    return np.array(Image.open(DATASET / f"seed99_step{step:05d}.png").convert("RGB"))


def _side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    gap = np.full((a.shape[0], 8, 3), 40, dtype=np.uint8)
    return np.concatenate([a, gap, b], axis=1)


def pick_frames(cardinals: dict[int, int], n: int = 6) -> tuple[list[dict], str | None]:
    """4 cardinaux (là où sont les portes) + 2 extras adjacents."""
    items = []
    for h, step in sorted(cardinals.items()):
        fr = _load(step)
        items.append({"step": step, "heading": h, "frame": fr})
    extras = []
    for h, step in list(cardinals.items())[:2]:
        for d in (-1, 1):
            s = step + d
            path = DATASET / f"seed99_step{s:05d}.png"
            if path.exists():
                fr = _load(s)
                hud = read_hud(fr)
                extras.append({
                    "step": s,
                    "heading": hud.heading if hud.ok and hud.heading is not None else h,
                    "frame": fr,
                })
    picks = items + extras[: max(0, n - len(items))]
    color = best_wall_color([p["frame"] for p in picks])
    for p in picks:
        p["color"] = color
    return picks[:n], color


def summarize(data: dict | None) -> dict:
    if not data:
        return {"images": [], "objects": [], "enemies": 0, "doors": []}
    imgs = [
        f"{im.get('category') or im.get('desc')}@{im.get('wall_dir')}"
        for im in (data.get("images") or [])
        if float(im.get("confidence") or 0) >= 0.8
    ]
    objs = [
        o.get("name") for o in (data.get("objects") or [])
        if float(o.get("confidence") or 0) >= 0.8
    ]
    doors = data.get("doors") or {}
    open_d = [
        k.replace("dir_", "") for k, v in doors.items()
        if isinstance(v, dict) and v.get("status") in ("open", "locked")
    ]
    ens = data.get("enemies_visible") or []
    return {
        "images": imgs,
        "n_images": len(imgs),
        "objects": objs,
        "n_objects": len(objs),
        "enemies": len([e for e in ens if float(e.get("confidence") or 0) >= 0.8]),
        "doors": open_d,
    }


def preview():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    saved = []
    for room, meta in ROOMS.items():
        picks, color = pick_frames(meta["cardinals"], n=4)
        print(f"preview {room} color={color}")
        for p in picks:
            masked = mask_portals(p["frame"], color)
            combo = _side_by_side(p["frame"], masked)
            out = PREVIEW / f"{room.split()[0].lower()}_h{p['heading']}_s{p['step']}_{color}.png"
            Image.fromarray(combo).save(out)
            diff = float((p["frame"] != masked).any(-1).mean())
            print(f"  dir {p['heading']:>3} step {p['step']} mask_frac={diff:.3f} -> {out.name}")
            saved.append({
                "room": room, "step": p["step"], "heading": p["heading"],
                "color": color, "mask_frac": diff,
            })
    return saved


def run_vlm():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"rooms": {}, "conditions": CONDITIONS}
    for room, meta in ROOMS.items():
        print(f"\n=== pick {room} ===")
        picks, color = pick_frames(meta["cardinals"], n=6)
        print("  color", color, "frames:", [(p["step"], p["heading"]) for p in picks])
        frames = [p["frame"] for p in picks]
        headings = [p["heading"] for p in picks]
        if picks:
            combo = _side_by_side(frames[0], mask_portals(frames[0], color))
            Image.fromarray(combo).save(PREVIEW / f"batch_{room.split()[0].lower()}.png")
        room_out = {
            "why": meta["why"],
            "color": color,
            "picks": [{"step": p["step"], "heading": p["heading"]} for p in picks],
            "runs": {},
        }
        for cond in CONDITIONS:
            print(f"  -> {cond['id']} {cond['model']} mask={cond['mask']}")
            r = survey_room(
                frames, headings, room, {}, color,
                model=cond["model"], mask_portals=cond["mask"],
            )
            db = EpisodeDB()
            if r.ok and r.data:
                db.apply_survey(room, r.data)
            rec = db.to_dict()["rooms"].get(room, {})
            raw = summarize(r.data)
            room_out["runs"][cond["id"]] = {
                "ok": r.ok,
                "error": r.error,
                "latency_s": round(r.latency_s, 2),
                "cost_usd": round(r.cost_usd, 4),
                "model": r.model,
                "raw": raw,
                "db": {
                    "images": rec.get("images"),
                    "objects": rec.get("objects"),
                    "enemies": rec.get("enemies"),
                    "doors": rec.get("doors"),
                },
            }
            print(
                f"     lat={r.latency_s:.1f}s ${r.cost_usd:.4f} "
                f"imgs={raw['images']} objs={raw['objects']} ok={r.ok} {r.error or ''}"
            )
        results["rooms"][room] = room_out
    outp = OUT / "portal_exp.json"
    outp.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {outp}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    PREVIEW.mkdir(parents=True, exist_ok=True)
    preview()
    if not args.preview:
        run_vlm()


if __name__ == "__main__":
    main()
