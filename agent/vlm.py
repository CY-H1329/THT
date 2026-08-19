"""OpenRouter VLM client: sends several frames from one room, together
with the DB state built up for it so far, and has the VLM "incrementally
update" that room's content DB.

Why this shape: captioning frames independently and merging the text
afterward (the earlier approach) called the same picture "chimp" one
time, "ape" the next, "monkey" the time after that, and dedup failed
(confirmed by measurement). Instead, showing the VLM "here's what's
recorded so far" and asking it to "only add/refine what's newly
visible" lets it tell "already known" from "new" far more reliably.
Layered on top of that is the confidence gate (agent/content_db.py,
anything below 0.8 is dropped), which keeps shaky observations from
polluting the DB.

Never called synchronously inside act() during actual play (risks
blowing the 5 s budget) -- frames are buffered and handled all at once
when leaving a room (agent/content_ingest.py).

Uses only urllib (standard library) -- no extra dependency like
requests, and no network access or heavy loading at import time (only
touched when actually called).
"""

from __future__ import annotations

import base64
import io
import json
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from agent.config import (
    QA_FALLBACK_MODEL, QA_TIMEOUT_S, VLM_MODEL, VLM_TIMEOUT_S, get_api_key,
)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# The python.org distribution of Python on macOS doesn't wire up the
# system certificate chain by default (SSL: CERTIFICATE_VERIFY_FAILED,
# confirmed by hitting this directly), so we point it at certifi's
# certificate bundle explicitly. Handled defensively since the
# evaluation environment could hit the same issue.
try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = None

# gemini-2.5-flash-lite pricing (measured from OpenRouter's
# /api/v1/models as of 2026-08): $0.0000001/prompt token,
# $0.0000004/completion token.
_PRICE_PROMPT_PER_TOKEN = 0.0000001
_PRICE_COMPLETION_PER_TOKEN = 0.0000004

# Game-rules summary -- only facts confirmed from difficulty.yaml + the
# README that hold for every seed. Held-out categories (image subjects,
# 3D object types) are never included -- they're only ever mentioned as
# illustrative examples ("something in this vein"), never memorized labels.
RULES_TEXT = (
    "- Each episode has 8-12 rooms, each normally with 4 walls.\n"
    "- Each wall independently has 0-2 real photos hung on it (weighted "
    "toward 1), so a room has ~4 wall photos on average.\n"
    "- Each room has 1-2 freestanding 3D props resting on the floor "
    "(concrete real-world items, e.g. barrel/cone/tree/chair/duck-like "
    "things — the actual category varies per episode, don't assume these "
    "exact ones).\n"
    "- Each room (except the spawn room) has 0-2 enemies: a Minecraft-style "
    "6-box humanoid mob (head/torso/2 arms/2 legs) with a face texture. "
    "There is NO floating health bar above enemies — HP is only shown in "
    "the top HUD text, not as a 3D element.\n"
    "- Enemy shirt/pants/skin colors are drawn from a small fixed set: "
    "shirt in {red,blue,green,yellow,purple}, pants in {red,blue,green,"
    "yellow,purple,grey}, skin in {yellow,green,purple,grey}. Body shape "
    "is 'short' or 'tall'.\n"
    "- Room wall color is one solid color from a shared palette, sampled "
    "independently per room (two rooms CAN share the same color — color "
    "alone never identifies which room this is).\n"
    "- Exactly one door in the whole episode is locked: a yellow panel "
    "with a padlock icon blocking a doorway.\n"
    "- A hint banner (yellow-bordered box, centered) appears for a few "
    "seconds when the agent nears the locked door, describing which room "
    "holds the key.\n"
)

_TASK_INSTRUCTIONS = (
    "You are maintaining a memory database for a first-person 3D game, "
    "built from frames the agent saw while standing in ONE room (its name "
    "is given below). You will be called again later with more frames from "
    "other visits, each time given the database entry built so far — your "
    "job is to REFINE it, not restart it.\n\n"
    "CURRENT ROOM vs ADJACENT ROOMS: through a doorway you may see another "
    "room's walls in the background, with a DIFFERENT color. Only large, "
    "near walls that fill most of the frame belong to the CURRENT room. "
    "Small/distant patches seen through a doorway belong to a NEIGHBORING "
    "room — ignore them entirely for this room's wall_color/images/objects. "
    "Prefer whatever appears consistently large/near across the given "
    "frames.\n"
    "IMPORTANT CAVEAT: standing close to and facing straight through a "
    "doorway/portal opening can make the NEXT room's walls, pictures, and "
    "objects look just as large/near/sharp as if they were in the current "
    "room — 'near' alone is NOT reliable in that case. Whenever you see "
    "content framed inside a rectangular doorway/portal-shaped opening (as "
    "opposed to painted directly on a flat, unbroken wall of the current "
    "room), treat everything inside that opening as belonging to the OTHER "
    "room and exclude it entirely from this room's images/objects/"
    "wall_color.\n"
    "CONCRETE PATTERN TO WATCH FOR: a doorway typically looks like two "
    "large wall segments of the CURRENT room's color filling the left and "
    "right (or top/bottom) sides of the frame, with a narrower gap between "
    "them showing a different color/scene beyond. ANYTHING visible inside "
    "that narrow gap — a wall image, a 3D object, an enemy, a different "
    "wall color — belongs to the other room on the far side of the "
    "doorway. Ignore it completely, even if it looks sharp and clearly "
    "visible; do not add it to this room's images/objects/enemies.\n\n"
    "CONFIDENCE DISCIPLINE: every reported item needs a confidence in "
    "[0,1]. Only report something (or update an existing field) if you are "
    "at least 0.8 confident. If unsure, omit it or keep the existing value "
    "rather than guessing. If a frame just confirms something already in "
    "the current database, you don't need to repeat it unless you can "
    "improve its confidence or description.\n\n"
    "DEDUPE AGAINST CURRENT DB: before adding an image/object/enemy, check "
    "if it plausibly already exists in the current database (same subject, "
    "possibly described with different words or seen from a different "
    "angle). If so, don't add a duplicate entry.\n\n"
    "OBJECTS: only concrete, freestanding real-world floor props count "
    "(e.g. a barrel, chair, tree). The yellow door panel with a padlock "
    "icon is NOT an object — it's the locked door (report it via "
    "has_locked_door instead). If you can't concretely name what a shape "
    "is, leave it out rather than using a vague label. Also give each "
    "object's dominant color as a plain color word (e.g. 'red', 'brown'), "
    "or null if it is multi-colored or you cannot tell.\n\n"
    "If a yellow-bordered hint banner with text is visible in any frame, "
    "transcribe it verbatim into hint_text."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "wall_color": {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "null"]},
                "confidence": {"type": "number"},
            },
            "required": ["value", "confidence"],
            "additionalProperties": False,
        },
        "has_locked_door": {"type": "boolean"},
        "has_locked_door_confidence": {"type": "number"},
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "desc": {"type": "string"},
                    # Note: pairing an integer type with enum+null broke
                    # Gemini's structured output (confirmed directly --
                    # a "requires unspecified property" error). String
                    # enum+null works fine, so other fields are left as-is.
                    "wall_dir": {"type": ["integer", "null"]},
                    "confidence": {"type": "number"},
                },
                "required": ["desc", "wall_dir", "confidence"],
                "additionalProperties": False,
            },
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    # The README explicitly lists "what color a
                    # particular object was" as a QA category, so we
                    # take color as its own field (previously only
                    # enemies had one). It's a strict schema so this
                    # still has to be in `required`, but the prompt
                    # tells the model to use null when it isn't confident.
                    "color": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                },
                "required": ["name", "color", "confidence"],
                "additionalProperties": False,
            },
        },
        "enemies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "desc": {"type": "string"},
                    "shirt_color": {"type": ["string", "null"],
                                     "enum": ["red", "blue", "green", "yellow", "purple", None]},
                    "pants_color": {"type": ["string", "null"],
                                     "enum": ["red", "blue", "green", "yellow", "purple", "grey", None]},
                    "skin_color": {"type": ["string", "null"],
                                    "enum": ["yellow", "green", "purple", "grey", None]},
                    "body_shape": {"type": ["string", "null"], "enum": ["short", "tall", None]},
                    "confidence": {"type": "number"},
                },
                "required": ["desc", "shirt_color", "pants_color", "skin_color",
                             "body_shape", "confidence"],
                "additionalProperties": False,
            },
        },
        "hint_text": {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "null"]},
                "confidence": {"type": "number"},
            },
            "required": ["value", "confidence"],
            "additionalProperties": False,
        },
    },
    "required": ["wall_color", "has_locked_door", "has_locked_door_confidence",
                 "images", "objects", "enemies", "hint_text"],
    "additionalProperties": False,
}


@dataclass
class VlmResult:
    ok: bool
    data: Optional[dict] = None
    cost_usd: float = 0.0
    error: Optional[str] = None


def _encode_jpeg(frame: np.ndarray, quality: int = 80) -> str:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def update_room_db(
    frames: list[np.ndarray],
    current_db: dict,
    room_hint: Optional[str] = None,
    rules_text: str = RULES_TEXT,
    api_key: Optional[str] = None,
) -> VlmResult:
    """Sends frames (several taken in the same room) plus current_db
    (what's recorded for this room so far) and gets back the updated
    room-content JSON.

    Never raises on failure (no key / network / timeout / parse error)
    -- always returns VlmResult(ok=False) instead, so the caller's loop
    doesn't die. The returned VlmResult.data is in the exact shape
    agent.content_db.ContentDB.apply_update() expects.
    """
    key = api_key or get_api_key()
    if not key:
        return VlmResult(ok=False, error="no_api_key")
    if not frames:
        return VlmResult(ok=False, error="no_frames")

    room_line = f"Room name (from HUD OCR): {room_hint!r}\n\n" if room_hint else ""
    text = (
        "GAME RULES (always true, every episode):\n" + rules_text + "\n"
        + _TASK_INSTRUCTIONS + "\n\n"
        + room_line
        + "CURRENT DATABASE FOR THIS ROOM (refine this, don't restart it):\n"
        + json.dumps(current_db, ensure_ascii=False) + "\n\n"
        + f"Below are {len(frames)} frame(s) from this room, "
          "possibly different angles/distances of the same room."
    )

    content = [{"type": "text", "text": text}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": _encode_jpeg(frame)}})

    payload = {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "room_db_update",
                "strict": True,
                "schema": _SCHEMA,
            },
        },
        "temperature": 0,
    }
    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=VLM_TIMEOUT_S, context=_SSL_CONTEXT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data = json.loads(body["choices"][0]["message"]["content"])
        usage = body.get("usage", {})
        cost = (
            usage.get("prompt_tokens", 0) * _PRICE_PROMPT_PER_TOKEN
            + usage.get("completion_tokens", 0) * _PRICE_COMPLETION_PER_TOKEN
        )
        return VlmResult(ok=True, data=data, cost_usd=cost)
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError,
            json.JSONDecodeError) as e:
        return VlmResult(ok=False, error=str(e))


# Combat aiming (strike phase only)
# The earlier CV color rule (agent.vision.enemy_bearing, since removed)
# inferred the enemy from "pixels that differ from this room's stored
# wall color," but because lighting can render the same wall up to
# 0.65x darker, that measurably ended up mistaking a plain, empty wall
# for an enemy. So instead of "find what isn't the wall," we switched to
# actually looking at the image and identifying the enemy directly (the
# fixed color-palette 6-box humanoid, which is never held out). Measured:
# one image's VLM call averages ~2 s (fits inside act()'s 5 s budget). We
# call it every tick during the strike phase (up to
# _FLEE_STRIKE_MAX_TICKS ticks) and decide attack/turn-left/turn-right
# directly from that frame.
_LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "enemy_visible": {"type": "boolean"},
        "bearing": {"type": ["string", "null"], "enum": ["left", "center", "right", None]},
        "close_enough_to_attack": {"type": "boolean"},
    },
    "required": ["enemy_visible", "bearing", "close_enough_to_attack"],
    "additionalProperties": False,
}

_LOCATE_PROMPT = (
    "One frame from a first-person 3D game (HUD cropped out). The player "
    "is currently fighting a nearby blocky Minecraft-style humanoid "
    "monster (6 boxes: head/torso/2 arms/2 legs, a face texture, a "
    "colored shirt torso and colored pants legs), which may or may not be "
    "visible in this exact frame. Do not confuse a wall, door, or 3D prop "
    "(barrel/cone/duckie/chair/tree) for the monster. Is the monster "
    "visible? If so: is it on the left, center, or right side of the "
    "frame (center = roughly the middle third, good enough to attack "
    "straight ahead)? And does it look close enough to melee (large in "
    "frame, within a couple of meters) rather than far away across the "
    "room? Answer only the JSON fields, no extra text."
)


def locate_enemy(frame: np.ndarray, api_key: Optional[str] = None,
                  timeout: float = 3.0) -> VlmResult:
    """Ask once whether an enemy is visible in the current frame and
    roughly which side it's on. Used to set the initial bearing at the
    start of a fight (see the design note above for details). Never
    raises on failure -- returns VlmResult(ok=False), and the caller
    just falls straight back to CV-based aiming.
    """
    key = api_key or get_api_key()
    if not key:
        return VlmResult(ok=False, error="no_api_key")

    payload = {
        "model": VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _LOCATE_PROMPT},
                {"type": "image_url", "image_url": {"url": _encode_jpeg(frame)}},
            ],
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "locate_enemy", "strict": True, "schema": _LOCATE_SCHEMA},
        },
        "temperature": 0,
    }
    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data = json.loads(body["choices"][0]["message"]["content"])
        usage = body.get("usage", {})
        cost = (
            usage.get("prompt_tokens", 0) * _PRICE_PROMPT_PER_TOKEN
            + usage.get("completion_tokens", 0) * _PRICE_COMPLETION_PER_TOKEN
        )
        return VlmResult(ok=True, data=data, cost_usd=cost)
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError,
            json.JSONDecodeError) as e:
        return VlmResult(ok=False, error=str(e))


# Last-resort check on a blocked direction (navigation safety net)
# Right before agent.explorer gives up on a direction as "no door" based
# purely on physical collision checks (is_blocked), it asks the VLM once
# -- "is this really a wall, or a door?" (per explicit direction). At
# most one call per direction given up per room, so the round-trip
# latency (~2 s) never accumulates.
_DOOR_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "door_or_opening_visible": {"type": "boolean"},
        "bearing": {"type": ["string", "null"], "enum": ["left", "center", "right", None]},
    },
    "required": ["door_or_opening_visible", "bearing"],
    "additionalProperties": False,
}

_DOOR_CHECK_PROMPT = (
    "One frame from a first-person 3D game (HUD cropped out). The player "
    "has been trying to walk toward what might be a doorway/passage to "
    "another room, but kept failing to actually pass through. Look "
    "carefully: is there an actual door, open doorway, or passage to "
    "another room visible anywhere in this image? A flat picture/photo "
    "frame hanging on a wall, or a plain wall, does NOT count as a door "
    "or opening -- only an actual gap/passage/doorway leading further "
    "does. If a door/opening is visible, is it on the left, center, or "
    "right side of the frame? Answer only the JSON fields, no extra text."
)


def _call_vlm_json(frame: np.ndarray, prompt: str, schema: dict, schema_name: str,
                    api_key: Optional[str], timeout: float) -> VlmResult:
    """Shared request assembly + call used by locate_door/locate_enemy/classify_heading."""
    key = api_key or get_api_key()
    if not key:
        return VlmResult(ok=False, error="no_api_key")

    payload = {
        "model": VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _encode_jpeg(frame)}},
            ],
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
        "temperature": 0,
    }
    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data = json.loads(body["choices"][0]["message"]["content"])
        usage = body.get("usage", {})
        cost = (
            usage.get("prompt_tokens", 0) * _PRICE_PROMPT_PER_TOKEN
            + usage.get("completion_tokens", 0) * _PRICE_COMPLETION_PER_TOKEN
        )
        return VlmResult(ok=True, data=data, cost_usd=cost)
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError,
            json.JSONDecodeError) as e:
        return VlmResult(ok=False, error=str(e))


def locate_door(frame: np.ndarray, api_key: Optional[str] = None,
                 timeout: float = 3.0) -> VlmResult:
    """Last check right before giving up on this direction as "no door."

    A problem found by measurement: there's already a safety net that
    catches "mistook a wall-hung picture for a door and can't physically
    pass through," but it can't tell apart "there really is a door here,
    the passage is just narrow enough that we keep bumping into it" from
    "this is just a wall/picture and there's no door at all." Never
    raises on failure -- returns VlmResult(ok=False), and the caller
    just marks it a wall and moves to the next candidate.
    """
    return _call_vlm_json(frame, _DOOR_CHECK_PROMPT, _DOOR_CHECK_SCHEMA,
                           "locate_door", api_key, timeout)


# Asked during SURVEY (right after entering a new room, facing each of
# the 4 directions in turn), before actually walking that way. Unlike
# locate_door(), this isn't "already failed several times, check once
# as a last resort" -- it's "just turned to face this direction
# straight on, haven't taken a single step yet," so it gets its own
# prompt (checking with the VLM before every move, per explicit
# direction). Since we're already facing it head-on, bearing isn't needed.
_HEADING_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "is_door_or_opening": {"type": "boolean"},
    },
    "required": ["is_door_or_opening"],
    "additionalProperties": False,
}

_HEADING_CHECK_PROMPT = (
    "One frame from a first-person 3D game (HUD cropped out). The player "
    "is standing still, looking directly at one wall of the room, deciding "
    "whether it's worth walking that way. Is there an actual door, open "
    "doorway, or passage to another room straight ahead? A flat "
    "picture/photo frame hanging on the wall, a decorative object, or a "
    "plain solid wall does NOT count -- only a real gap/passage/doorway "
    "leading further does. Answer only the JSON field, no extra text."
)


def classify_heading(frame: np.ndarray, api_key: Optional[str] = None,
                      timeout: float = 3.0) -> VlmResult:
    """Judge door-vs-wall ahead of time from one face-on frame of a
    direction during SURVEY.

    The caller only treats ok and data["is_door_or_opening"] == True as
    a "likely a door" hint; if ok is False (network error / timeout /
    budget exhausted), it just falls back entirely to the existing
    physical-check (is_blocked-based SEEK) safety net -- this function
    never replaces SEEK, it only cuts down on wasted attempts walking
    into an obvious wall.
    """
    return _call_vlm_json(frame, _HEADING_CHECK_PROMPT, _HEADING_CHECK_SCHEMA,
                           "classify_heading", api_key, timeout)


# --- QA fallback (agent/qa.py): called only when rule-based routing
# can't classify the question's intent, exactly once, text-only (no image) ---
_QA_FALLBACK_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

# Questions qa.py's regex routing can't classify (an unexpected phrasing
# or paraphrase) get handed to this LLM instead. So the prompt only has
# to take care of two things: (1) give it enough game context to
# correctly interpret the question's intent, and (2) force it to answer
# using only the given JSON (the frozen memory), so it follows the same
# "don't invent an answer" philosophy as qa.py's other rules.
_QA_FALLBACK_INSTRUCTIONS = (
    "You are answering a question about ONE completed episode of a "
    "first-person exploration game, using ONLY the JSON memory given below "
    "-- this is the agent's complete, frozen record of everything it saw "
    "and did during play. You were not there yourself; the JSON is your "
    "only source of truth.\n\n"
    "HOW TO ANSWER:\n"
    "- Base every claim strictly on the JSON. Never invent a room name, "
    "color, image subject, object, enemy detail, or count that is not "
    "literally present in the data.\n"
    "- Counting questions (how many rooms/images/objects/enemies/kills, "
    "etc.): count the actual list length in the JSON -- don't estimate.\n"
    "- \"List / name / cite the rooms\" style questions: answer with the "
    "room names from visited_order (or the keys of \"rooms\"), joined "
    "naturally, e.g. 'I visited X, Y and Z.'\n"
    "- Questions may be phrased many different ways -- singular/plural, "
    "imperative, indirect, casual. Interpret the INTENT, not the exact "
    "wording: 'how many room did you visit', 'cite the name of the rooms "
    "did you visit', and 'which rooms did you explore' are all the same "
    "question.\n"
    "- If a specific room is named or implied in the question, match it to "
    "the closest room name in the JSON (OCR may have truncated some room "
    "names with '...').\n"
    "- If the JSON genuinely has no information to answer the question, "
    "say so plainly and briefly (e.g. 'I don't have a record of that') -- "
    "never guess or fabricate just to sound more complete. An honest "
    "'I don't know' is always better than a confident wrong answer.\n"
    "- Keep the answer short: one or two sentences of plain factual prose. "
    "No bullet points, no JSON, no meta-commentary about your reasoning "
    "process.\n"
    "- Answer in the voice of the player recalling the episode (\"I saw "
    "...\", \"I visited ...\", \"I killed ...\"), matching how the rest of "
    "this agent's answers are phrased.\n"
)


def qa_fallback(question: str, facts_json: str, api_key: Optional[str] = None,
                 timeout: float = QA_TIMEOUT_S) -> VlmResult:
    """Last resort used by agent/qa.py -- only called when regex routing
    can't classify the question's intent. Sends text only, no image, and
    the prompt forces the answer to be based solely on facts_json (the
    frozen ContentDB+SceneGraph). Assumes the caller (qa.py) wraps this
    in a hard wall-clock deadline, so timeout here is kept short --
    urllib's timeout= is a socket-idle timeout, so if the response keeps
    trickling in a little at a time, the whole call can run far longer
    than that (measured: one call took 9.46 s, most of the README's
    10 s per-answer budget) -- so qa.py adds a real hard cap of its own
    on top, via a ThreadPoolExecutor.
    """
    key = api_key or get_api_key()
    if not key:
        return VlmResult(ok=False, error="no_api_key")

    prompt = (
        "GAME RULES (always true, every episode):\n" + RULES_TEXT + "\n"
        + _QA_FALLBACK_INSTRUCTIONS + "\n"
        + f"MEMORY (JSON, the agent's complete frozen record):\n{facts_json}\n\n"
        + f"QUESTION: {question}"
    )
    payload = {
        "model": QA_FALLBACK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "qa_answer", "strict": True, "schema": _QA_FALLBACK_SCHEMA},
        },
        "temperature": 0,
    }
    # We tried a tight max_tokens cap to clip reasoning models
    # (gpt-5-nano) before they burned the whole budget on hidden
    # reasoning tokens without ever emitting the JSON answer.
    # QA_FALLBACK_MODEL is a non-reasoning chat model, so the cap is
    # left unset here, and latency is bounded by qa.py's wall-clock
    # deadline instead.
    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content_text = body["choices"][0]["message"]["content"]
        if not content_text:
            return VlmResult(ok=False, error="empty_content")
        data = json.loads(content_text)
        return VlmResult(ok=True, data=data)
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError,
            json.JSONDecodeError) as e:
        return VlmResult(ok=False, error=str(e))
