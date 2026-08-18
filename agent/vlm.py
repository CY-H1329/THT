"""OpenRouter VLM 클라이언트: 방 하나의 프레임 여러 장 + 현재까지 쌓인
DB 상태를 함께 보내서, VLM이 그 방의 콘텐츠 DB를 "누적 갱신"하게 한다.

왜 이 방식인가: 프레임을 하나씩 독립적으로 캡션하고 나중에 텍스트로
합치면(예전 방식), 같은 그림을 "chimp"/"ape"/"monkey"처럼 매번 다르게
불러서 중복 제거가 실패했다(실측 확인, dev_log.md). 대신 VLM에게
"지금까지 이렇게 기록돼 있다"를 보여주고 "새로 보이는 것만 추가/보정하라"고
시키면, VLM 스스로 "이미 아는 것"과 "새로운 것"을 훨씬 잘 구분한다.
거기에 confidence 게이트(agent/content_db.py, 0.8 미만은 폐기)까지 더해
불확실한 관측으로 DB가 오염되는 걸 막는다.

answer()는 질문당 10초 예산이라 방이 여러 개면 거기서 한 번에 다 처리할
여유가 없다 — 대신 agent.py가 SURVEY 동안 프레임을 방별로 버퍼링해뒀다가,
그 방이 DFS상 "done"(4방향 다 확인 끝) 되는 순간 act() 안에서 그 방
1개분만 호출한다. 방 하나씩 분산되므로 개별 act() 호출은 여전히 5초
예산 안에 들고, QA 시작 시점엔 이미 다 채워져 있어 answer()는 조회만
한다.

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


# --- 전투 중 조준(strike 단계 전용) -----------------------------------------
# CV 색상 규칙(agent.vision.enemy_bearing)은 "저장된 방 벽색과 다른 픽셀"로
# 적을 추론하는데, 조명에 따라 같은 벽도 최대 0.65배까지 어둡게 렌더되는
# 문제 때문에(dev_log.md) 아무것도 없는 벽을 적으로 오판하는 게 실측으로
# 확인됐다 — 그래서 "벽이 아닌 것 찾기"가 아니라 실제로 이미지를 보고
# 적(정해진 색 팔레트의 6박스 humanoid, held-out 아님)을 직접 식별하는
# 방식으로 바꾼다. 실측: 이미지 1장 VLM 호출은 평균 ~2초(act() 5초 예산
# 안에 들어감). strike 단계(최대 _FLEE_STRIKE_MAX_TICKS틱) 동안 매 틱
# 호출해서 그 프레임 기준으로 공격/좌회전/우회전을 바로 결정한다.
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
    """지금 프레임에 적이 보이는지, 대략 어느 쪽인지 한 번만 물어본다.
    전투 시작 시 초기 방향을 잡는 용도(자세한 설계는 위 주석 참고).
    실패 시 예외 없이 VlmResult(ok=False) — 호출자는 CV 기반 조준으로
    바로 넘어가면 된다.
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


# --- 막힌 방향 최후 확인(길찾기 안전망) --------------------------------
# agent.explorer의 물리 충돌 확인(is_blocked)만으로 특정 방향을 "문 없음"
# 으로 포기하기 직전, 딱 한 번만 VLM에게 "이거 진짜 벽이야, 문이야?"를
# 물어본다(사용자 지시). 방마다 후보 방향 하나 포기할 때 최대 1회만
# 부르므로 왕복 지연(~2초)이 누적되지 않는다.
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
    """locate_door/locate_enemy/classify_heading이 공유하는 요청 조립+호출."""
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
    """이 방향을 "문 없음"으로 포기하기 직전 마지막으로 한 번 확인.

    실측으로 확인된 문제(dev_log.md): 벽에 걸린 그림을 문으로 착각하고
    물리적으로 못 지나가는 걸 감지하는 안전망은 있지만, 반대로 "저기 문이
    있는데 통로가 좁아서 계속 부딪히는 것"과 "그냥 벽/그림이라 문이 아예
    없는 것"을 구분은 못 한다. 실패 시 예외 없이 VlmResult(ok=False) —
    호출자는 그냥 벽으로 마킹하고 다음 후보로 넘어가면 된다.
    """
    return _call_vlm_json(frame, _DOOR_CHECK_PROMPT, _DOOR_CHECK_SCHEMA,
                           "locate_door", api_key, timeout)


# SURVEY(새 방 진입 직후 4방향을 하나씩 정면으로 바라보는 단계)에서, 그
# 방향으로 실제로 걸어가 보기 전에 먼저 물어본다. locate_door()와 달리
# "이미 여러 번 부딪혀서 실패한 뒤 마지막으로 확인"하는 맥락이 아니라
# "이제 막 이 방향을 정면으로 보고 있고 아직 한 걸음도 안 걸었다"는
# 맥락이라 프롬프트를 분리했다(사용자 지시: 매번 움직이기 전에 먼저
# VLM에게 문인지 확인). 이미 정면을 보고 있으므로 방향(bearing)은
# 필요 없다.
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
    """SURVEY 중 한 방향을 정면으로 본 프레임 한 장으로 문/벽을 미리 판단.

    호출자는 ok and data["is_door_or_opening"]가 True일 때만 "문일
    가능성 높음" 힌트로 쓰고, ok가 False(네트워크 오류/타임아웃/예산
    소진)면 기존 물리 확인(is_blocked 기반 SEEK) 안전망에 그대로
    맡긴다 — 이 함수는 SEEK를 대체하지 않고, 명백한 벽에 헛되이
    부딪혀보는 시도만 줄여준다.
    """
    return _call_vlm_json(frame, _HEADING_CHECK_PROMPT, _HEADING_CHECK_SCHEMA,
                           "classify_heading", api_key, timeout)


# VLM_RECOVER: 룰베이스 탐험(RECENTER/SURVEY/SEEK/RETURN)이 같은 행동을
# 반복하는데 화면도 안 바뀌면(=제자리에서 맴돔) 룰베이스를 잠깐 멈추고
# VLM에게 매 틱 이미지를 보여주며 직접 조종을 맡긴다(사용자 지시). 목표는
# "방의 트인 중앙 쪽으로 이동"으로 고정 — 방 중앙 자체는 도착 판정이
# 애매하므로, VLM 스스로 "막힘에서 벗어나 트인 곳에 도달했다"고 판단하면
# 그렇게 보고하게 하고, 그러면 룰베이스로 복귀한다.
_RECOVER_SCHEMA = {
    "type": "object",
    "properties": {
        "reached_open_area": {"type": "boolean"},
        "action": {"type": ["string", "null"],
                   "enum": ["turn_left", "turn_right", "move_forward", "move_back", None]},
        "reasoning": {"type": "string"},
    },
    "required": ["reached_open_area", "action", "reasoning"],
    "additionalProperties": False,
}

_RECOVER_PROMPT = (
    "One frame from a first-person 3D game. A rule-based navigation "
    "script has been repeating the same action without the view changing "
    "-- it's stuck (likely wedged against a wall, corner, or obstacle). "
    "You are taking over movement for a few steps. The goal is simply to "
    "get the agent moving again toward the open, walkable center of the "
    "room it's currently in (away from walls/corners/clutter), not to "
    "reach any door or exit. If the view already shows reasonably open "
    "space ahead with room to move (no wall or obstacle filling the "
    "frame up close), set reached_open_area=true and action=null -- "
    "control will be handed back to the rule-based script. Otherwise set "
    "reached_open_area=false and pick exactly one action (turn_left, "
    "turn_right, move_forward, move_back) that best helps it get "
    "unstuck. Answer only the JSON fields, no extra text."
)


def recover_action(frame: np.ndarray, api_key: Optional[str] = None,
                    timeout: float = 3.0) -> VlmResult:
    """제자리 맴돌기(정체) 감지 시 매 틱 호출 — 트인 곳으로 벗어날 때까지
    VLM이 직접 액션을 하나씩 골라준다. ok=False(네트워크 실패 등)면
    호출자가 즉시 룰베이스로 복귀하면 된다(이 기능 자체가 안전망이라,
    실패해도 원래 동작으로 돌아가면 그만 — 더 물러날 곳이 없음).
    """
    return _call_vlm_json(frame, _RECOVER_PROMPT, _RECOVER_SCHEMA,
                           "recover_action", api_key, timeout)


# --- QA 폴백(agent/qa.py): 키워드 규칙이 애매할 때 딱 1번, 이미지 없이
# 텍스트 전용 저가 모델에 "이 사실들만 근거로" 답하게 위임 ---------------
from agent.config import QA_FALLBACK_MODEL, QA_TIMEOUT_S  # noqa: E402

_QA_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def qa_fallback(question: str, facts_json: str, api_key: Optional[str] = None,
                 timeout: float = QA_TIMEOUT_S) -> VlmResult:
    """질문 의도를 키워드로 못 잡았을 때만 호출. facts_json 안의 사실만
    근거로 짧게 답하게 하고, 모르면 모른다고 말하게 지시한다(헛소리 방지).
    answer()의 10초 예산을 지키기 위해 timeout을 짧게 잡는다."""
    key = api_key or get_api_key()
    if not key:
        return VlmResult(ok=False, error="no_api_key")

    prompt = (
        "You are answering a question about a game episode a player just "
        "finished, using ONLY the JSON facts below (this is the player's "
        "complete memory of the episode -- nothing else is known). If the "
        "facts don't contain the answer, say so plainly instead of "
        "guessing. Answer in one short sentence.\n\n"
        f"FACTS:\n{facts_json}\n\nQUESTION: {question}"
    )
    payload = {
        "model": QA_FALLBACK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "qa_answer", "strict": True, "schema": _QA_SCHEMA},
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
        return VlmResult(ok=True, data=data)
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError,
            json.JSONDecodeError) as e:
        return VlmResult(ok=False, error=str(e))
