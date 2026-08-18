"""QA 전용 VLM 전문가들. agent/vlm.py(전투 locate_enemy, Gemini flash-lite)와 분리.

역할:
  Surveyor (GPT-4o) — 각도별 프레임만 보고 그림/물건/문/적을 채움. DB 이전 기록 없음.
  Verifier (GPT-4o, parallèle) — autre jeu d'angles, même salle, images only.
  Witness  (GPT-4o) — 적 외형만 (combat frames).
"""

from __future__ import annotations

import base64
import io
import json
import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from agent.config import get_api_key
from qa_db import WALL_COLOR_NAMES
from qa_portal import prepare_survey_frames

_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Bench: gpt-4o ~2.7s / 4 photos justes; 5.4-mini même temps mais flower/taxi.
SURVEY_MODEL = "openai/gpt-4o"
LIVE_SURVEY_MODEL = "openai/gpt-4o"
VERIFY_MODEL = "openai/gpt-4o"
_PRICES = {
    "openai/gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "openai/gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "openai/gpt-5-nano": (0.05 / 1_000_000, 0.40 / 1_000_000),
    "openai/gpt-5.4-nano": (0.20 / 1_000_000, 1.25 / 1_000_000),
    "openai/gpt-5.4-mini": (0.75 / 1_000_000, 4.50 / 1_000_000),
    "openai/gpt-5.5": (5.00 / 1_000_000, 30.00 / 1_000_000),
    "google/gemini-2.5-flash": (0.30 / 1_000_000, 2.50 / 1_000_000),
    "anthropic/claude-sonnet-4.6": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "anthropic/claude-sonnet-4": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "anthropic/claude-3.7-sonnet": (3.00 / 1_000_000, 15.00 / 1_000_000),
}
QA_TIMEOUT_S = 60.0

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = None

_PALETTE = ", ".join(WALL_COLOR_NAMES)
_COLORS_SHIRT = ["red", "blue", "green", "yellow", "purple"]
_COLORS_PANTS = ["red", "blue", "green", "yellow", "purple", "grey"]
_COLORS_SKIN = ["yellow", "green", "purple", "grey"]


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@dataclass
class CallResult:
    ok: bool
    data: Optional[dict] = None
    cost_usd: float = 0.0
    error: Optional[str] = None
    latency_s: float = 0.0
    model: str = SURVEY_MODEL
    n_frames: int = 0


def _parse_content(raw) -> dict:
    if isinstance(raw, list):
        raw = "".join(
            (p.get("text") or "") if isinstance(p, dict) else str(p) for p in raw
        )
    if not isinstance(raw, str):
        raw = str(raw)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    return json.loads(raw)


def _call(messages: list, schema: dict, schema_name: str, *,
          model: str = SURVEY_MODEL,
          api_key: Optional[str] = None, timeout: float = QA_TIMEOUT_S,
          n_frames: int = 0) -> CallResult:
    import time
    key = api_key or get_api_key()
    if not key:
        return CallResult(ok=False, error="no_api_key", n_frames=n_frames, model=model)
    pin, pout = _PRICES.get(model, (2.50 / 1_000_000, 10.00 / 1_000_000))
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    }
    # gpt-5.x refuse souvent temperature=0
    if "gpt-5" not in model:
        payload["temperature"] = 0
    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        msg = body["choices"][0]["message"]
        data = _parse_content(msg.get("content"))
        usage = body.get("usage") or {}
        cost = (usage.get("prompt_tokens") or 0) * pin + (usage.get("completion_tokens") or 0) * pout
        return CallResult(ok=True, data=data, cost_usd=cost,
                          latency_s=time.monotonic() - t0, n_frames=n_frames, model=model)
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as e:
        return CallResult(ok=False, error=str(e), latency_s=time.monotonic() - t0,
                          n_frames=n_frames, model=model)


_DOOR = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["open", "locked", "wall", "unknown"]},
        "neighbor_color": {"type": ["string", "null"],
                           "enum": list(WALL_COLOR_NAMES) + [None]},
        "confidence": {"type": "number"},
    },
    "required": ["status", "neighbor_color", "confidence"],
    "additionalProperties": False,
}

SURVEY_SCHEMA = {
    "type": "object",
    "properties": {
        "wall_color": {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "null"], "enum": list(WALL_COLOR_NAMES) + [None]},
                "confidence": {"type": "number"},
            },
            "required": ["value", "confidence"],
            "additionalProperties": False,
        },
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "desc": {"type": "string"},
                    "category": {"type": "string"},
                    "wall_dir": {"type": ["integer", "null"]},
                    "confidence": {"type": "number"},
                },
                "required": ["desc", "category", "wall_dir", "confidence"],
                "additionalProperties": False,
            },
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "color": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                },
                "required": ["name", "color", "confidence"],
                "additionalProperties": False,
            },
        },
        "doors": {
            "type": "object",
            "properties": {
                "dir_0": _DOOR,
                "dir_90": _DOOR,
                "dir_180": _DOOR,
                "dir_270": _DOOR,
            },
            "required": ["dir_0", "dir_90", "dir_180", "dir_270"],
            "additionalProperties": False,
        },
        "has_locked_door": {"type": "boolean"},
        "has_locked_door_confidence": {"type": "number"},
        "locked_door_dir": {"type": ["integer", "null"]},
        "enemies_visible": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "shirt_color": {"type": ["string", "null"], "enum": _COLORS_SHIRT + [None]},
                    "pants_color": {"type": ["string", "null"], "enum": _COLORS_PANTS + [None]},
                    "skin_color": {"type": ["string", "null"], "enum": _COLORS_SKIN + [None]},
                    "body_shape": {"type": ["string", "null"], "enum": ["short", "tall", None]},
                    "desc": {"type": "string"},
                    "alive_looking": {"type": "boolean"},
                    "confidence": {"type": "number"},
                },
                "required": ["shirt_color", "pants_color", "skin_color", "body_shape",
                             "desc", "alive_looking", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "wall_color", "images", "objects", "doors",
        "has_locked_door", "has_locked_door_confidence", "locked_door_dir",
        "enemies_visible",
    ],
    "additionalProperties": False,
}

WITNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "enemies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "shirt_color": {"type": ["string", "null"], "enum": _COLORS_SHIRT + [None]},
                    "pants_color": {"type": ["string", "null"], "enum": _COLORS_PANTS + [None]},
                    "skin_color": {"type": ["string", "null"], "enum": _COLORS_SKIN + [None]},
                    "body_shape": {"type": ["string", "null"], "enum": ["short", "tall", None]},
                    "desc": {"type": "string"},
                    "alive_looking": {"type": "boolean"},
                    "confidence": {"type": "number"},
                },
                "required": ["shirt_color", "pants_color", "skin_color", "body_shape",
                             "desc", "alive_looking", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["enemies"],
    "additionalProperties": False,
}


_SURVEY_PROMPT = f"""You are the SURVEYOR specialist for a first-person 3D game memory DB.

You see several frames from ONE room, mostly a 360° turn. HUD at the top has room name, time, HP, heading (dir N°).

GAME FACTS (always true):
- Floor is two-tone checkerboard tiles. Ceiling is plain grey concrete. Do NOT list floor/ceiling as objects.
- Walls of THIS room are one solid color from: {_PALETTE}.
- Doors exist ONLY at headings 0° (east), 90° (south), 180° (west), 270° (north).
- Each of the 4 walls has 0, 1, or 2 real photographs hanging on it (never more than 2 per wall, never more than 8 in the room). A typical room has ABOUT 4 photos. Look at EVERY frame and list every distinct photo on THIS room's walls — do not stop after the first one.
- Each room has 1 or 2 freestanding 3D floor props (barrel, traffic cone, rubber duck, office chair, tree, or held-out equivalents — name what you actually see).
- Enemies are Minecraft-style 6-cube humanoids (head, torso, 2 arms, 2 legs) with a face on the head. Shirt/torso color ∈ {{red,blue,green,yellow,purple}}. Pants ∈ {{red,blue,green,yellow,purple,grey}}. Head/skin ∈ {{yellow,green,purple,grey}}. Body short or tall. There is NO health bar above them.
- Count an enemy ONLY if it stands on THIS room's checkerboard, in front of THIS room's walls — not inside a doorway gap, not on a neighbor's floor, not inside a black mask. A mob framed by a door jamb belongs to the NEXT room: omit it.
- Exactly one doorway in the episode is a LOCKED DOOR: a large yellow-gold rectangular PANEL filling the doorway, with a BLACK PADLOCK icon. It is a DOOR, never a wall photograph / painting / poster / plaque / object. Do NOT put it in images[] or objects[]. If that doorway is painted black here, still do not invent a photo for it.

CRITICAL — OTHER ROOMS THROUGH DOORWAYS:
A doorway looks like two large slabs of THIS room's wall color with a rectangular gap between them. Anything INSIDE that gap (different wall color, photos, props, enemies) belongs to a NEIGHBOR room. Do NOT add those to this room's images/objects/enemies/wall_color. You MAY record the neighbor wall color on that door slot (neighbor_color) and status=open.
A yellow-gold padlock panel filling a doorway is status=locked — never a photo, even if it looks like a picture of a lock.

BLACK RECTANGLES in some frames are CV-masked doorways: treat them as empty voids. Do NOT guess what was behind. Do NOT list photos/objects/enemies that appear only inside a doorway gap OR only in a blacked-out band.

Only report items you are ≥0.8 confident are IN THIS ROOM, painted on THIS room's walls or standing on THIS room's floor, not framed inside a portal.
If a photo is on a wall, set wall_dir to 0/90/180/270 using the HUD `dir` of the frame that faces that wall (or null if unsure).
category = a short STABLE noun (e.g. "tiger", "flamingo", "sailboat"). Same subject from two angles MUST reuse the same category — never list it twice.
Deduplicate: same photo/object seen from two angles = one entry. "feather" and "feathers" are the same.
Ignore the yellow hint banner if present (another specialist OCR's it).
Do not invent objects to fill the 1–2 quota — only what is clearly visible.
"""


_WITNESS_PROMPT = """You are the ENEMY WITNESS specialist. These frames were taken when the player took damage IN THIS ROOM.

Look only for Minecraft-style 6-cube humanoid mobs (head/torso/2 arms/2 legs, face texture) standing on THIS room's checkerboard.

Shirt/torso ∈ {red,blue,green,yellow,purple}. Pants ∈ {red,blue,green,yellow,purple,grey}. Head/skin ∈ {yellow,green,purple,grey}. short or tall.
Do NOT count a mob that is only visible inside a doorway gap or inside a black rectangle (that is the NEXT room).
Do NOT confuse wall photos, 3D props (barrel/cone/duck/chair/tree), or a yellow locked-door panel with an enemy.
If none are visible on THIS floor, return enemies=[].
alive_looking=true if the mesh is present. If you only see a corpse/empty floor where one was, omit it.
"""


def survey_room(
    frames: list[np.ndarray],
    headings: list[int],
    room_name: str,
    current_db: Optional[dict] = None,
    cv_wall_color: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = SURVEY_MODEL,
    mask_portals: bool = True,
) -> CallResult:
    if not frames:
        return CallResult(ok=False, error="no_frames")
    send = prepare_survey_frames(frames, cv_wall_color, mask=mask_portals)
    heading_note = ", ".join(f"frame{i+1}: dir {h}°" for i, h in enumerate(headings))
    cv_line = f"Cheap CV majority wall-color estimate for this room: {cv_wall_color}. Prefer this unless you are sure it is wrong.\n" if cv_wall_color else ""
    mask_line = (
        "Doorway interiors are painted black in these frames. Ignore those voids.\n"
        if mask_portals else ""
    )
    text = (
        _SURVEY_PROMPT
        + f"\nRoom name (HUD OCR): {room_name!r}\n"
        + cv_line
        + mask_line
        + f"Frame headings: {heading_note}\n"
        + "IMAGE-ONLY: ignore any prior memory. List only what you see in these frames.\n"
        + f"\n{len(send)} frame(s) follow, in time order. Multiple angles of the SAME room."
    )
    content = [{"type": "text", "text": text}]
    for fr in send:
        content.append({"type": "image_url", "image_url": {"url": encode_jpeg(fr)}})
    return _call(
        [{"role": "user", "content": content}],
        SURVEY_SCHEMA,
        "room_survey",
        model=model,
        api_key=api_key,
        n_frames=len(send),
    )


_VERIFY_PROMPT = (
    _SURVEY_PROMPT
    + "\nYou are a second independent SURVEYOR (verifier). You do NOT see another model's DB. "
      "Look only at the frames. List every photo/object/enemy that is IN THIS ROOM. "
      "DELETE from your answer anything that is only visible through a doorway or in a black mask.\n"
)


def verify_room(
    frames: list[np.ndarray],
    headings: list[int],
    room_name: str,
    current_db: Optional[dict] = None,
    cv_wall_color: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = VERIFY_MODEL,
    mask_portals: bool = True,
) -> CallResult:
    """Deuxième GPT-4o, images only — pas de draft DB."""
    if not frames:
        return CallResult(ok=False, error="no_frames")
    send = prepare_survey_frames(frames, cv_wall_color, mask=mask_portals)
    heading_note = ", ".join(f"frame{i+1}: dir {h}°" for i, h in enumerate(headings))
    cv_line = f"CV wall-color: {cv_wall_color}.\n" if cv_wall_color else ""
    text = (
        _VERIFY_PROMPT
        + f"\nRoom: {room_name!r}\n"
        + cv_line
        + "Doorway interiors are painted black. Ignore those voids.\n"
        + f"Headings: {heading_note}\n"
        + "IMAGE-ONLY. No previous DB.\n"
        + f"\n{len(send)} frames, diverse angles. Return the full JSON."
    )
    content = [{"type": "text", "text": text}]
    for fr in send:
        content.append({"type": "image_url", "image_url": {"url": encode_jpeg(fr)}})
    return _call(
        [{"role": "user", "content": content}],
        SURVEY_SCHEMA,
        "room_verify",
        model=model,
        api_key=api_key,
        n_frames=len(send),
    )


def survey_and_verify_parallel(
    frames_a: list[np.ndarray],
    headings_a: list[int],
    frames_b: list[np.ndarray],
    headings_b: list[int],
    room_name: str,
    cv_wall_color: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = SURVEY_MODEL,
    mask_portals: bool = True,
) -> tuple[CallResult, CallResult]:
    """Deux GPT-4o en parallèle sur deux jeux d'angles. Images only."""
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="qa-dual") as pool:
        fa = pool.submit(
            survey_room, frames_a, headings_a, room_name, None,
            cv_wall_color, api_key, model, mask_portals,
        )
        fb = pool.submit(
            verify_room, frames_b, headings_b, room_name, None,
            cv_wall_color, api_key, model, mask_portals,
        )
        return fa.result(), fb.result()


def witness_enemies(
    frames: list[np.ndarray],
    api_key: Optional[str] = None,
    model: str = SURVEY_MODEL,
    cv_wall_color: Optional[str] = None,
) -> CallResult:
    if not frames:
        return CallResult(ok=False, error="no_frames")
    send = prepare_survey_frames(frames, cv_wall_color, mask=True)
    text = (
        _WITNESS_PROMPT
        + "\nDoorway interiors (and locked-door panels) are painted black. Ignore those voids.\n"
        + f"{len(send)} frame(s) from THIS room during combat."
    )
    content = [{"type": "text", "text": text}]
    for fr in send:
        content.append({"type": "image_url", "image_url": {"url": encode_jpeg(fr)}})
    return _call(
        [{"role": "user", "content": content}],
        WITNESS_SCHEMA,
        "enemy_witness",
        model=model,
        api_key=api_key,
        n_frames=len(send),
    )
