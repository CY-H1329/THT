"""OpenRouter VLM 클라이언트: 방 하나의 프레임 여러 장 + 현재까지 쌓인
DB 상태를 함께 보내서, VLM이 그 방의 콘텐츠 DB를 "누적 갱신"하게 한다.

왜 이 방식인가: 프레임을 하나씩 독립적으로 캡션하고 나중에 텍스트로
합치면(예전 방식), 같은 그림을 "chimp"/"ape"/"monkey"처럼 매번 다르게
불러서 중복 제거가 실패했다(실측 확인, dev_log.md). 대신 VLM에게
"지금까지 이렇게 기록돼 있다"를 보여주고 "새로 보이는 것만 추가/보정하라"고
시키면, VLM 스스로 "이미 아는 것"과 "새로운 것"을 훨씬 잘 구분한다.
거기에 confidence 게이트(agent/content_db.py, 0.8 미만은 폐기)까지 더해
불확실한 관측으로 DB가 오염되는 걸 막는다.

실제 플레이 중에는 act() 안에서 동기 호출하지 않는다(5초 예산 초과 위험) —
프레임을 버퍼링해뒀다가 answer() 첫 호출 시 한 번에 처리한다(agent.py).

urllib만 사용(표준 라이브러리) — requests 등 추가 의존성 없음, import
시점에는 네트워크/무거운 로딩 없음(호출 시점에만 접근).
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

from agent.config import VLM_MODEL, VLM_TIMEOUT_S, get_api_key

_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# macOS의 python.org 배포판 파이썬은 시스템 인증서 체인이 기본으로
# 연결 안 돼 있어서(SSL: CERTIFICATE_VERIFY_FAILED, 실측 확인됨),
# certifi의 인증서 번들을 명시적으로 지정한다. 평가 환경에서도 같은
# 문제가 있을 수 있어 견고하게 처리.
try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = None

# gemini-2.5-flash-lite 요금(2026-08 기준 OpenRouter /api/v1/models 실측):
# prompt $0.0000001/토큰, completion $0.0000004/토큰.
_PRICE_PROMPT_PER_TOKEN = 0.0000001
_PRICE_COMPLETION_PER_TOKEN = 0.0000004

# 게임 규칙 요약 — difficulty.yaml + README에서 확인한, 매 seed 공통인 사실만.
# held-out 카테고리(이미지 주제, 3D 오브젝트 종류)는 절대 안 넣는다 — 예시로만
# 언급해서 "이런 느낌의 것"이라는 감만 주고, 실제 라벨을 암기시키지 않는다.
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
    "is, leave it out rather than using a vague label.\n\n"
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
                    # 주의: 정수 타입에 enum+null을 같이 쓰면 Gemini 구조화 출력이
                    # 깨진다(실측 확인, "requires unspecified property" 오류).
                    # 문자열 enum+null은 정상 동작해서 다른 필드는 그대로 둠.
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
                    "confidence": {"type": "number"},
                },
                "required": ["name", "confidence"],
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
    """frames(같은 방에서 찍힌 여러 장) + current_db(이 방의 지금까지 기록)를
    보내서, 갱신된 방 콘텐츠 JSON을 받는다.

    실패(키 없음/네트워크/타임아웃/파싱 오류)해도 예외를 던지지 않고
    VlmResult(ok=False)를 반환한다 — 호출자 루프가 죽지 않도록.
    반환된 VlmResult.data는 agent.content_db.ContentDB.apply_update()에
    그대로 넘기면 되는 형식이다.
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
