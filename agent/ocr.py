"""HUD 바 + 힌트 배너 텍스트를 픽셀에서 읽어내는 자체 OCR.

왜 tesseract 같은 범용 OCR을 안 쓰는가: env는 HUD/힌트 배너 텍스트를
Pillow 번들 기본 폰트(ImageFont.load_default)로, 안티에일리어싱이 거의
없는 단색 배경 위에 그린다(hud.py, env.py 참고). 우리는 정확히 같은
폰트/크기로 참조 글리프를 직접 렌더링할 수 있으므로, 왼쪽에서 오른쪽으로
한 글자씩 "다음에 올 후보 글자들 중 어느 게 지금 이 위치의 픽셀과 가장
비슷한가"를 비교하는 그리디 템플릿 매칭만으로 사실상 100% 정확하고
1ms 미만으로 끝나는 OCR이 된다. room 이름이나 힌트 문구의 단어는 개발/평가
풀이 달라(held-out) 사전에 알 수 없지만, 이 방식은 어떤 문자열이든
글자 단위로 읽어내므로 특정 단어를 몰라도 동작한다.

memory_fps_env.hud를 import하지 않는다 — "HUD도 픽셀에서 읽어야 한다"는
과제 취지에 맞춰, 폰트 로딩 로직도 독립적으로 재구현한다.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 폰트 -------------------------------------------------------------
# hud.py/env.py와 동일한 로딩 방식(Pillow >=10.1의 번들 스케일러블 기본
# 폰트). 크기만 맞으면 모든 OS에서 byte-identical하게 렌더된다고 env
# 코드 주석에 명시되어 있음 — 우리 쪽 참조 렌더링도 동일 보장을 받는다.
def _load_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


_HUD_FONT_SIZE = 16
_BANNER_FONT_SIZE = 28

# HUD/힌트 배너/QA 배너에 실제로 등장할 수 있는 문자 전체.
# (영문 대소문자 + 숫자 + 관련 기호. 여유 있게 몇 개 더 포함.)
_ALPHABET = (
    string.ascii_letters
    + string.digits
    + " .,:;'\"!?()-/°[]—·…"
)

_INK_THRESHOLD = 40  # 배경(검정/진남색) 대비 "글자 잉크"로 볼 밝기 임계값
_MATCH_MIN_SCORE = 0.80  # 이 이하 점수면 해당 위치는 매칭 실패로 간주


@dataclass
class _GlyphSet:
    font: ImageFont.FreeTypeFont
    chars: str
    masks: list  # list[np.ndarray[bool]], 문자별 잉크 마스크 (height, width)
    height: int
    # 폭이 같은 글리프끼리 묶어둔 것: width → (문자열, (n, h, w) bool 스택).
    # 한 후보 위치에서 80개 문자를 하나씩 비교하는 대신 폭 그룹별로
    # 한 번씩만 numpy 브로드캐스트 비교하면 되므로 ~30배 빠르다.
    by_width: dict = None


@lru_cache(maxsize=4)
def _build_glyph_set(size: int) -> _GlyphSet:
    """알파벳 전체를 한 번에 렌더링해서 문자별 참조 비트맵을 잘라낸다.

    한 번에 그려야 실제 HUD 렌더링과 동일한 advance-width 계산 방식을
    쓰게 되어, 개별 문자를 따로 그릴 때 생기는 베어링 오차가 없다.
    """
    font = _load_font(size)
    # 넉넉한 캔버스에 알파벳 전체를 한 줄로 렌더링.
    pad_top = size  # ascender/descender 여유
    canvas_h = size * 2
    canvas_w = int(font.getlength(_ALPHABET)) + 8
    img = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(img)
    draw.text((0, pad_top // 2), _ALPHABET, fill=255, font=font)
    arr = np.array(img)

    # 각 문자의 시작 x좌표는 그 앞까지 부분 문자열의 누적 advance width.
    xs = [0.0]
    for i in range(1, len(_ALPHABET) + 1):
        xs.append(font.getlength(_ALPHABET[:i]))

    # 세로 방향은 잉크가 실제로 존재하는 행 범위로 타이트하게 자른다.
    ink_rows = np.where((arr > _INK_THRESHOLD).any(axis=1))[0]
    y0, y1 = (int(ink_rows.min()), int(ink_rows.max()) + 1) if len(ink_rows) else (0, canvas_h)

    masks = []
    for i, ch in enumerate(_ALPHABET):
        x0, x1 = int(round(xs[i])), max(int(round(xs[i])) + 1, int(round(xs[i + 1])))
        glyph = arr[y0:y1, x0:x1]
        masks.append(glyph > _INK_THRESHOLD)

    by_width: dict = {}
    for ch, m in zip(_ALPHABET, masks):
        by_width.setdefault(m.shape[1], [[], []])
        by_width[m.shape[1]][0].append(ch)
        by_width[m.shape[1]][1].append(m)
    alpha_idx = {ch: i for i, ch in enumerate(_ALPHABET)}
    by_width = {
        w: (chars, np.stack(stack),
            np.array([alpha_idx[c] for c in chars], dtype=np.int32))
        for w, (chars, stack) in by_width.items()
    }
    return _GlyphSet(font=font, chars=_ALPHABET, masks=masks, height=y1 - y0,
                     by_width=by_width)


def _score(crop_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    """crop_mask와 ref_mask(둘 다 bool, 같은 shape로 이미 맞춰짐)의 유사도."""
    if crop_mask.size == 0:
        return 0.0
    return float((crop_mask == ref_mask).mean())


def _read_line(ink: np.ndarray, glyphs: _GlyphSet, x_start: int, x_end: int,
               y_off: int) -> str:
    """ink: (height, width) bool 배열(잉크=True). [x_start, x_end) 구간을
    왼쪽부터 그리디하게 한 글자씩 읽어 문자열로 복원한다.

    y_off: 참조 글리프(높이 glyphs.height, 알파벳 전체의 잉크 상단에 맞춰
    잘라둔 것)를 ink의 몇 번째 행부터 맞춰 비교할지. 호출자가 "이 ink에서
    실제 글자가 시작하는 맨 윗 행"을 넘겨줘야 한다(가운데 정렬 추정이
    아니라 실측값). 작은 정렬 오차를 흡수하기 위해, 매 스텝마다 커서
    위치를 -2..+2px 범위에서 같이 탐색해 가장 잘 맞는 (글자, 시작위치)
    조합을 고른다.
    """
    gh = glyphs.height
    out = []
    cursor = x_start
    stall_guard = 0
    dy_choices = (0, -1, 1)   # 첫 글자에서 정해지면 그 줄 내내 고정 (아래 참고)
    while cursor < x_end and stall_guard < 500:
        stall_guard += 1
        best = None  # (score, alphabet_index, char, width, actual_cursor, dy)
        for dx in (0, -1, 1, -2, 2):
            c0 = cursor + dx
            if c0 < 0:
                continue
            for dy in dy_choices:
                yy = y_off + dy
                if yy < 0 or yy + gh > ink.shape[0]:
                    continue
                # 폭 그룹 단위 브로드캐스트 비교. 그룹 내부는 알파벳 순서로
                # 쌓여 있으므로 argmax가 곧 "동점이면 앞 글자" 규칙이 된다.
                cand = None
                for w, (chars, stack, alpha) in glyphs.by_width.items():
                    if c0 + w > ink.shape[1]:
                        continue
                    crop = ink[yy:yy + gh, c0:c0 + w]
                    scores = (stack == crop).mean(axis=(1, 2))
                    k = int(scores.argmax())
                    key = (float(scores[k]), -int(alpha[k]))
                    if cand is None or key > cand[0]:
                        cand = (key, chars[k], w, int(alpha[k]))
                if cand is None:
                    continue
                (score, _), ch, w, aidx = cand
                if best is None or score > best[0]:
                    best = (score, aidx, ch, w, c0, dy)
        if best is None or best[0] < _MATCH_MIN_SCORE:
            break
        score, _aidx, ch, w, actual_cursor, dy_best = best
        # 세로 정렬은 줄 단위로 일정하다. 첫 글자에서 고른 dy를 고정하면
        # 이후 글자 비교량이 1/3로 줄고, 실측상 결과 문자열은 동일하다.
        dy_choices = (dy_best,)
        out.append(ch)
        cursor = actual_cursor + w
    return "".join(out).rstrip()


def _ink_mask(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.astype(np.int32).sum(axis=-1) / 3.0
    return gray > _INK_THRESHOLD


def _ink_x_bounds(ink: np.ndarray) -> Optional[tuple]:
    cols = np.where(ink.any(axis=0))[0]
    if len(cols) == 0:
        return None
    return int(cols.min()), int(cols.max()) + 1


# --- HUD 바 파싱 --------------------------------------------------------

_HUD_BAR_HEIGHT = 28  # hud.py::_BAR_HEIGHT와 동일 (실측 프레임으로 검증됨)

_HUD_RE = re.compile(
    r"^(?P<room>.*?)\s*[·.]\s*T-(?P<secs>\d+)s\s*[·.]\s*HP\s*(?P<hp>\d+)/(?P<hpmax>\d+)"
    r"\s*[·.]\s*dir\s*(?P<heading>\d+).?(?:\s*[·.]\s*\[KEY\])?$"
    # 마지막 '.?'는 도(°) 기호 자리 — 16px 크기에서 어퍼스트로피(')와
    # 시각적으로 거의 구분이 안 돼 OCR이 종종 다르게 읽는다. 어차피
    # 필요한 건 heading 숫자뿐이라 기호 자체는 검증하지 않는다.
)


@dataclass
class HudReading:
    ok: bool
    room_name: Optional[str] = None
    is_corridor: bool = False
    seconds_remaining: Optional[int] = None
    hp: Optional[int] = None
    hp_max: Optional[int] = None
    heading: Optional[int] = None
    has_key: bool = False
    raw_text: str = ""


def read_hud(frame: np.ndarray) -> HudReading:
    """frame: (240,320,3) uint8 전체 관찰 프레임. 상단 HUD 바만 읽는다."""
    bar = frame[:_HUD_BAR_HEIGHT, :, :]
    ink = _ink_mask(bar)
    bounds = _ink_x_bounds(ink)
    if bounds is None:
        return HudReading(ok=False)
    x0, x1 = bounds
    ink_rows = np.where(ink.any(axis=1))[0]
    y_off = int(ink_rows.min())
    glyphs = _build_glyph_set(_HUD_FONT_SIZE)
    text = _read_line(ink, glyphs, x0, x1, y_off)
    m = _HUD_RE.match(text)
    if not m:
        return HudReading(ok=False, raw_text=text)
    room = m.group("room").strip()
    is_corridor = room in ("— corridor —", "-- corridor --", "corridor")
    return HudReading(
        ok=True,
        room_name=None if is_corridor else room,
        is_corridor=is_corridor,
        seconds_remaining=int(m.group("secs")),
        hp=int(m.group("hp")),
        hp_max=int(m.group("hpmax")),
        heading=int(m.group("heading")),
        has_key="[KEY]" in text,
        raw_text=text,
    )


# --- 힌트 배너 파싱 -------------------------------------------------------
# env.py::_composite_hint_overlay와 동일한 결정론적 기하 계산.
_HINT_BORDER_PX = 8
_HINT_W_FRAC = 0.82
_HINT_H_FRAC = 0.55
_HINT_BORDER_RGB = (255, 220, 80)


def hint_banner_box(frame_shape) -> tuple:
    h, w = frame_shape[0], frame_shape[1]
    box_w = int(w * _HINT_W_FRAC)
    box_h = int(h * _HINT_H_FRAC)
    box_x = (w - box_w) // 2
    box_y = (h - box_h) // 2
    return box_x, box_y, box_w, box_h


def hint_banner_active(frame: np.ndarray) -> bool:
    """결정론적으로 계산되는 테두리 픽셀 좌표 몇 곳의 색만 확인 —
    CV/VLM 없이 100% 신뢰 가능한 배너 감지."""
    box_x, box_y, box_w, box_h = hint_banner_box(frame.shape)
    probe_pts = [
        (box_y + 2, box_x + box_w // 2),
        (box_y + box_h - 3, box_x + box_w // 2),
        (box_y + box_h // 2, box_x + 2),
        (box_y + box_h // 2, box_x + box_w - 3),
    ]
    h, w = frame.shape[0], frame.shape[1]
    hits = 0
    for py, px in probe_pts:
        if 0 <= py < h and 0 <= px < w:
            r, g, b = frame[py, px, 0], frame[py, px, 1], frame[py, px, 2]
            if (abs(int(r) - _HINT_BORDER_RGB[0]) < 20
                    and abs(int(g) - _HINT_BORDER_RGB[1]) < 20
                    and abs(int(b) - _HINT_BORDER_RGB[2]) < 20):
                hits += 1
    return hits >= 3


def read_hint_text(frame: np.ndarray) -> str:
    """힌트 배너가 떠 있을 때, 내부의 여러 줄 텍스트를 합쳐서 반환한다.
    "KEY HINT" 헤더 줄은 고정 문구이므로 제외한다.
    """
    box_x, box_y, box_w, box_h = hint_banner_box(frame.shape)
    bd = _HINT_BORDER_PX
    inner = frame[box_y + bd:box_y + box_h - bd, box_x + bd:box_x + box_w - bd, :]
    ink = _ink_mask(inner)
    glyphs = _build_glyph_set(_BANNER_FONT_SIZE)

    # 텍스트가 여러 줄로 나뉘므로, 잉크가 있는 row band를 줄 단위로 분리.
    row_has_ink = ink.any(axis=1)
    lines = []
    y = 0
    H = ink.shape[0]
    while y < H:
        if not row_has_ink[y]:
            y += 1
            continue
        y_start = y
        while y < H and row_has_ink[y]:
            y += 1
        y_end = y
        row_ink = ink[y_start:y_end, :]
        bounds = _ink_x_bounds(row_ink)
        if bounds is None:
            continue
        x0, x1 = bounds
        text = _read_line(row_ink, glyphs, x0, x1, y_off=0)
        if text:
            lines.append(text)

    # 첫 줄은 "KEY HINT" 헤더(고정 문구) — 있으면 제외.
    if lines and lines[0].upper().replace(" ", "") == "KEYHINT":
        lines = lines[1:]
    return " ".join(lines).strip()
