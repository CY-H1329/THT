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

# HUD처럼 자주(캐시 미스마다) 호출되는 경우의 기본 탐색 범위 — 이 정도
# 오차는 서브픽셀 렌더링 흔들림만 흡수하면 충분하고, HUD 문자열은 실측
# 검증 완료(dev_log.md).
_DX_SEARCH_NARROW = (0, -1, 1, -2, 2)
_DY_SEARCH_NARROW = (0, -1, 1)

# 참조 글리프는 알파벳 전체를 한 줄로 렌더링해 "전체에서 가장 위/왼쪽에
# 잉크가 있는 지점"에 맞춰 잘라둔 것이다(_build_glyph_set). 그런데 개별
# 글자마다 자기 자신의 실제 잉크가 그 공유 기준선에서 몇 px 안쪽에서
# 시작하는지는 글자마다 다르다(예: 'K'는 최대 3px, 't'는 1px 정도 —
# BANNER_FONT_SIZE=28에서 실측). 이 편차를 흡수하려면 좁은 범위로는
# 부족해서(진짜 글자는 스코어가 낮게 나오고, 우연히 배경(빈칸)만
# 일치하는 다른 폭의 엉뚱한 글자가 더 높은 점수로 선택되어 문자열이
# 통째로 깨짐 — 실측으로 발견) 힌트 배너 전용으로 훨씬 넓게 탐색한다.
# 힌트 배너는 떠 있을 때만, 그것도 텍스트가 바뀔 때만 읽으므로(호출
# 빈도 낮음) 이 폭을 HUD에도 적용하면(8배 조합 수) read_hud가 다시
# 143ms대로 느려진다 — 반드시 힌트 배너 경로에만 넘겨야 한다.
_DX_SEARCH_WIDE = (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5)
_DY_SEARCH_WIDE = (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5)


@dataclass
class _GlyphSet:
    font: ImageFont.FreeTypeFont
    chars: str
    masks: list  # list[np.ndarray[bool]], 문자별 잉크 마스크 (height, width)
    height: int
    width_groups: dict  # width -> (chars_str, stacked_masks (n, height, width) bool)
    # 아래 셋은 _match_at의 완전 벡터화용 — 알파벳 전체를 최대 너비로
    # 오른쪽 패딩해 하나로 쌓아둔 것. 각 글자의 진짜 너비(padded_widths)까지
    # 들고 있으면 "자기 폭까지만 세기"를 누적합 한 번으로 처리할 수 있다.
    padded: np.ndarray = None       # (n_chars, height, max_width) bool
    padded_widths: np.ndarray = None  # (n_chars,) int
    max_width: int = 0


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

    # 너비별로 묶어서 참조 마스크를 (n, h, w) 배열로 쌓아둔다 — _read_line이
    # 문자 하나하나를 파이썬 루프로 개별 비교하는 대신, 같은 너비의 문자
    # 묶음을 numpy로 한 번에 비교하게 하기 위함. 실측: 글자당 ~1200번의
    # 개별 numpy 호출(전체 알파벳 x dx x dy) 때문에 read_hud 한 번에
    # 143ms가 걸렸는데(dev_log.md), 이 배치 비교로 <10ms대로 떨어진다.
    # 묶음 안에서의 동점 처리(먼저 나온 문자가 이김)는 알파벳 등장 순서를
    # 그대로 보존해서 기존 매칭 결과와 완전히 동일하게 만든다.
    groups: dict = {}
    for ch, m in zip(_ALPHABET, masks):
        groups.setdefault(m.shape[1], ([], []))
        groups[m.shape[1]][0].append(ch)
        groups[m.shape[1]][1].append(m)
    width_groups = {
        w: ("".join(chs), np.stack(ms))
        for w, (chs, ms) in groups.items()
    }
    max_w = max(m.shape[1] for m in masks)
    padded = np.zeros((len(masks), y1 - y0, max_w), dtype=bool)
    for i, m in enumerate(masks):
        padded[i, :, :m.shape[1]] = m
    widths = np.array([m.shape[1] for m in masks], dtype=np.int64)
    return _GlyphSet(font=font, chars=_ALPHABET, masks=masks, height=y1 - y0,
                      width_groups=width_groups, padded=padded,
                      padded_widths=widths, max_width=max_w)


def _match_at_slow(ink: np.ndarray, glyphs: _GlyphSet, cursor: int, y_off: int,
                   dx_search, dy_search):
    """한 커서 위치에서 (점수, 글자, 너비, 시작열)을 고르는 기준 구현.

    같은 너비의 문자들만 (n,h,w)로 묶어 비교한다. 아래 _match_at이 이걸
    (dx,dy)까지 포함해 통째로 벡터화한 버전이고, 프레임 오른쪽 끝처럼
    벡터화 창이 안 잡히는 자리에서만 이 구현으로 되돌아온다.
    """
    gh = glyphs.height
    best = None  # (score, char, width, actual_cursor)
    for dx in dx_search:
        c0 = cursor + dx
        if c0 < 0:
            continue
        for dy in dy_search:
            yy = y_off + dy
            if yy < 0 or yy + gh > ink.shape[0]:
                continue
            for w, (chs, stacked) in glyphs.width_groups.items():
                if c0 + w > ink.shape[1]:
                    continue
                crop = ink[yy:yy + gh, c0:c0 + w]
                if crop.shape[0] != gh or crop.shape[1] != w:
                    continue
                scores = (stacked == crop[None, :, :]).mean(axis=(1, 2))
                idx = int(scores.argmax())
                s = float(scores[idx])
                if best is None or s > best[0]:
                    best = (s, chs[idx], w, c0)
    return best


def _match_at(ink: np.ndarray, glyphs: _GlyphSet, cursor: int, y_off: int,
              dx_search, dy_search):
    """_match_at_slow와 같은 결과를, 커서당 numpy 호출 몇 번으로 구한다.

    기존 구현은 커서 하나마다 (dx x dy x 너비그룹) = 150번쯤 되는 작은
    numpy 호출을 돌았고, 배열이 작아서 사실상 호출 오버헤드만 냈다 —
    HUD 한 줄에 ~62ms, 즉 이 에이전트에서 한 틱을 통틀어 가장 비싼 연산
    이었다(EN.detect 1.6ms와 비교). env가 벽시계로 적 AI를 돌리므로
    (README) 이 지연은 곧 전투 중 "더 맞는다"가 된다.

    여기서는 알파벳 전체를 최대 너비로 패딩해 쌓아둔 배열(_GlyphSet.padded)
    하나로 모든 (dx,dy) 후보 창을 한 번에 비교한다. 글자마다 너비가 다른
    건 열 방향 누적합에서 자기 너비 지점을 꺼내 쓰는 것으로 처리한다.
    동점 처리 우선순위(dx/dy 탐색 순서 -> 알파벳 등장 순서)는 기존과
    동일하게 유지한다 — 안 그러면 같은 점수의 다른 글자가 뽑혀 문자열이
    달라질 수 있다.
    """
    gh, mw = glyphs.height, glyphs.max_width
    H, W = ink.shape
    offs = [(cursor + dx, y_off + dy)
            for dx in dx_search for dy in dy_search
            if cursor + dx >= 0 and 0 <= y_off + dy and y_off + dy + gh <= H]
    # 창이 프레임 오른쪽 끝을 넘는 자리는 좁은 글자만 후보가 되므로
    # 패딩 비교로 못 다룬다 — 그런 자리는 기준 구현에 맡긴다.
    if not offs or any(c0 + mw > W for c0, _ in offs):
        return _match_at_slow(ink, glyphs, cursor, y_off, dx_search, dy_search)
    xs = np.array([c0 for c0, _ in offs])
    ys = np.array([yy for _, yy in offs])
    win = np.lib.stride_tricks.sliding_window_view(ink, (gh, mw))
    crops = win[ys, xs]                                  # (k, gh, mw)
    eq = glyphs.padded[:, None, :, :] == crops[None]     # (n, k, gh, mw)
    per_col = eq.sum(axis=2)                             # (n, k, mw)
    cum = np.cumsum(per_col, axis=2)
    widths = glyphs.padded_widths
    matched = cum[np.arange(len(widths)), :, widths - 1]  # (n, k) 자기 너비까지만
    scores = matched / (gh * widths)[:, None]
    top = float(scores.max())
    # 동점자 중에서 (dx,dy) 탐색 순서가 앞서는 것 -> 알파벳 순서가 앞서는 것.
    ci, ki = np.nonzero(scores >= top - 1e-12)
    order = np.lexsort((ci, ki))                          # ki 우선, 동률이면 ci
    c, k = int(ci[order[0]]), int(ki[order[0]])
    return (top, glyphs.chars[c], int(widths[c]), int(xs[k]))


def _read_line_segments(ink: np.ndarray, glyphs: _GlyphSet, x_start: int,
                        x_end: int, y_off: int, dx_search=_DX_SEARCH_NARROW,
                        dy_search=_DY_SEARCH_NARROW) -> list:
    """_read_line의 알맹이. 읽어낸 글자를 (글자, 다음_커서) 목록으로 준다.

    문자열만 필요하면 _read_line을 쓴다. 이 목록 형태는 read_hud의 증분
    재읽기(_hud_incremental)가 "픽셀이 안 바뀐 앞부분은 지난 프레임 결과를
    그대로 쓰고, 바뀐 지점부터만 다시 읽기" 위해 글자 경계 위치를 알아야
    해서 필요하다.

    ink: (height, width) bool 배열(잉크=True). [x_start, x_end) 구간을
    왼쪽부터 그리디하게 한 글자씩 읽어 문자열로 복원한다.

    y_off: 참조 글리프(높이 glyphs.height, 알파벳 전체의 잉크 상단에 맞춰
    잘라둔 것)를 ink의 몇 번째 행부터 맞줘 비교할지. 호출자가 "이 ink에서
    실제 글자가 시작하는 맨 윗 행"을 넘겨줘야 한다(가운데 정렬 추정이
    아니라 실측값). 작은 정렬 오차를 흡수하기 위해, 매 스텝마다 커서
    위치를 dx_search/dy_search 범위에서 같이 탐색해 가장 잘 맞는
    (글자, 시작위치) 조합을 고른다.

    dx_search/dy_search: 기본은 좁은 범위(HUD처럼 매 틱 호출될 수 있어
    빠름). 힌트 배너처럼 글자마다 기준선 오차가 더 크고(_DY_SEARCH_WIDE
    주석 참고) 호출 빈도가 낮은 경우에만 넓은 범위를 넘겨받는다 — 넓은
    범위를 HUD에도 쓰면 조합 수가 8배로 늘어 read_hud가 다시 143ms대로
    돌아간다(실측으로 발견, dev_log.md).
    """
    gh = glyphs.height
    out = []
    cursor = x_start
    stall_guard = 0
    while cursor < x_end and stall_guard < 500:
        stall_guard += 1
        best = _match_at(ink, glyphs, cursor, y_off, dx_search, dy_search)
        if best is None or best[0] < _MATCH_MIN_SCORE:
            break
        score, ch, w, actual_cursor = best
        out.append((ch, cursor))   # 다음 줄에서 실제 전진 위치로 덮어씀
        # dx가 크게 음수인 후보가 이기면 actual_cursor+w가 현재 cursor를
        # 못 넘어설 수 있다 — 그러면 다음 반복에서 똑같은 위치가 다시
        # 최고점을 받아 커서가 멈춘 채로 stall_guard까지 도는 무한
        # 루프가 된다(실측: 넓어진 dx 탐색 범위에서 발견). 최소 1px는
        # 항상 전진하도록 강제한다.
        cursor = max(cursor + 1, actual_cursor + w)
        out[-1] = (out[-1][0], cursor)
    return out


def _read_line(ink: np.ndarray, glyphs: _GlyphSet, x_start: int, x_end: int,
               y_off: int, dx_search=_DX_SEARCH_NARROW,
               dy_search=_DY_SEARCH_NARROW) -> str:
    """[x_start, x_end) 구간을 읽어 문자열로 돌려준다(_read_line_segments 래퍼)."""
    segs = _read_line_segments(ink, glyphs, x_start, x_end, y_off,
                               dx_search, dy_search)
    return "".join(ch for ch, _ in segs).rstrip()


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


# --- HUD 증분 재읽기 --------------------------------------------------
# HUD 한 줄 전체를 글자 단위로 매칭하면 프레임당 ~64ms다(실측). 이건
# 전투 중에 특히 비싸다 — 회전할 때마다 "dir NNN°"가 바뀌어서 픽셀 해시
# 캐시(explorer._read_hud_cached)가 매번 빗나가고, 결국 회전 한 틱이
# 64ms를 먹는다. env는 벽시계 기준으로 적 AI를 돌리므로(README) 이 지연은
# 그대로 "돌아보는 동안 더 맞는다"가 된다.
#
# 그런데 실제로 바뀌는 건 줄의 일부뿐이다(회전=방향 숫자, 1초마다=남은
# 시간, 피격=HP). 방 이름처럼 긴 앞부분은 그대로다. 그래서 지난 프레임의
# 잉크 마스크와 글자 경계를 들고 있다가, 픽셀이 처음 달라지는 열을 찾아
# 그 앞의 글자들은 지난 결과를 그대로 쓰고 거기서부터만 다시 읽는다.
_HUD_CACHE: dict = {"ink": None, "y_off": None, "x0": None, "segs": None}


def _hud_resume_margin(glyphs: _GlyphSet) -> int:
    """증분 재읽기를 시작해도 안전한, "바뀐 열"로부터의 여유 폭(px).

    _read_line_segments는 커서마다 dx만큼 뒤로 물러난 위치에서 모든 너비의
    글리프를 다 대보고 최고점을 고른다 — 즉 어떤 글자의 판정은 그 글자가
    차지한 폭보다 오른쪽 픽셀까지 본다. 재사용하는 앞부분이 "지난 프레임과
    픽셀이 같은 구간"만 보고 결정된 것이 되려면 그만큼 여유를 둬야 한다.
    """
    return max(glyphs.width_groups) + max(abs(d) for d in _DX_SEARCH_NARROW) + 1


def _read_hud_line(ink: np.ndarray, glyphs: _GlyphSet, x0: int, x1: int,
                   y_off: int) -> str:
    """HUD 한 줄을 읽되, 지난 프레임과 안 바뀐 앞부분은 재사용한다."""
    prev = _HUD_CACHE
    segs = None
    if (prev["segs"] is not None and prev["ink"] is not None
            and prev["ink"].shape == ink.shape
            and prev["y_off"] == y_off and prev["x0"] == x0):
        changed = np.flatnonzero((prev["ink"] != ink).any(axis=0))
        if len(changed) == 0:
            segs = prev["segs"]                    # HUD가 통째로 그대로
        else:
            safe_x = int(changed[0]) - _hud_resume_margin(glyphs)
            keep = [s for s in prev["segs"] if s[1] <= safe_x]
            if keep:
                cursor = keep[-1][1]
                segs = keep + _read_line_segments(ink, glyphs, cursor, x1, y_off)
    if segs is None:
        segs = _read_line_segments(ink, glyphs, x0, x1, y_off)
    _HUD_CACHE.update(ink=ink, y_off=y_off, x0=x0, segs=segs)
    return "".join(ch for ch, _ in segs).rstrip()


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
    text = _read_hud_line(ink, glyphs, x0, x1, y_off)
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


# env.py::_composite_hint_overlay가 힌트 줄을 26px 간격으로 그린다
# (text_top + i * 26). BANNER_FONT_SIZE=28의 참조 글리프 높이는 27px —
# 줄 간격(26)이 글자 높이(27)보다 작아서, 인접한 두 줄의 실제 잉크가
# 빈 행 하나 없이 맞닿거나 겹치는 경우가 흔하다(실측: "The key is
# hidden"/"behind the" 두 줄이 47행짜리 하나의 band로 합쳐짐). 그래서
# "빈 행이 나올 때까지가 한 줄"이라는 가정만으로는 못 나눈다 — band가
# 이 간격보다 눈에 띄게 크면 고정 pitch로 강제로 다시 쪼갠다.
_HINT_LINE_PITCH = 26


_HINT_Y_MARGIN = 6  # _DY_SEARCH가 -5까지 내려가므로 그만큼 위쪽 여유를 준다.


def _read_hint_band(ink: np.ndarray, glyphs: "_GlyphSet", y_start: int, y_bound: int) -> str:
    """ink: 배너 inner 영역 전체 잉크 마스크. [y_start, y_bound)를 "이 줄의
    실제 잉크가 있는 좁은 범위"로 보고 x 폭을 재되, _read_line에 넘기는
    배열 자체는 y_start보다 몇 행 위에서부터 잘라 y_off에 위쪽 여유를
    준다. 이유(실측으로 발견): y_start는 "이 줄 전체에서 가장 위에
    잉크가 있는 행"이라 보통 그 줄에서 키가 가장 큰 글자 기준이고, 다른
    글자는 자기 잉크가 그보다 몇 px 낮게(실측: 최대 3px, 헤더 'K') 시작
    한다. y_off=0으로 두면 정렬에 필요한 음수 dy가 yy=y_off+dy<0이 되어
    _read_line의 경계 체크에 걸려 항상 버려지고, 서로 다른 글자가 우연히
    배경(빈 칸) 위주로 맞아떨어지는 엉뚱한 후보가 선택되어 문자열이
    통째로 깨진다.
    """
    tight = ink[y_start:y_bound, :]
    bounds = _ink_x_bounds(tight)
    if bounds is None:
        return ""
    x0, x1 = bounds
    gh = glyphs.height
    top = max(0, y_start - _HINT_Y_MARGIN)
    y_off = y_start - top
    crop_bottom = min(ink.shape[0], top + y_off + gh + _HINT_Y_MARGIN)
    band = ink[top:crop_bottom, :]
    # 마지막 글자 앞 커닝 간격이 유난히 넓으면(실측: "HINT"의 N-T
    # 사이) 잉크 우측 끝에 딱 맞춘 x1에서 루프가 조기 종료돼 마지막
    # 글자를 놓친다 — 여유를 좀 더 준다(끝나면 그냥 공백만 더 매칭되고
    # rstrip으로 제거되니 안전).
    x1 = min(band.shape[1], x1 + 20)
    return _read_line(band, glyphs, x0, x1, y_off,
                       dx_search=_DX_SEARCH_WIDE, dy_search=_DY_SEARCH_WIDE)


# 노란 테두리 자체의 밝기(gray≈185, RGB (255,220,80)) — 이보다 확실히
# 밝아야 "테두리 위에 덧그려진 흰 글자"로 본다(실측: 본문 글자는
# gray≈235). env.py::_composite_hint_overlay 참고.
_HINT_BORDER_GRAY = 185
_HINT_TEXT_ON_BORDER_THRESHOLD = 210


def read_hint_text(frame: np.ndarray) -> str:
    """힌트 배너가 떠 있을 때, 내부의 여러 줄 텍스트를 합쳐서 반환한다.
    "KEY HINT" 헤더 줄은 고정 문구이므로 제외한다.
    """
    box_x, box_y, box_w, box_h = hint_banner_box(frame.shape)
    bd = _HINT_BORDER_PX
    # 안쪽 진남색 영역(box_h-bd까지)에 더해, 그 아래 테두리 두께(bd)만큼도
    # 함께 읽는다. 줄 간격이 촘촘해(_HINT_LINE_PITCH 주석 참고) 마지막
    # 줄 텍스트가 안쪽 진남색 영역 밑단을 넘어 노란 테두리 위까지 삐져
    # 나오는 경우가 실측으로 확인됨("...images on its east wall"의
    # "wall"이 통째로 잘림). 테두리 영역은 배경 자체가 밝은 노란색이라
    # _ink_mask의 검정-배경 기준 임계값을 그대로 쓰면 테두리 전체가
    # "잉크"로 오인되므로, 그 구간만 훨씬 밝은(흰 글자 전용) 임계값으로
    # 따로 마스킹해 이어붙인다.
    inner = frame[box_y + bd:box_y + box_h, box_x + bd:box_x + box_w - bd, :]
    dark_part = inner[: box_h - 2 * bd, :, :]
    border_part = inner[box_h - 2 * bd:, :, :]
    ink_dark = _ink_mask(dark_part)
    border_gray = border_part.astype(np.int32).sum(axis=-1) / 3.0
    ink_border = border_gray > _HINT_TEXT_ON_BORDER_THRESHOLD
    ink = np.concatenate([ink_dark, ink_border], axis=0)
    glyphs = _build_glyph_set(_BANNER_FONT_SIZE)

    # 텍스트가 여러 줄로 나뉘므로, 잉크가 있는 row band를 줄 단위로 분리.
    row_has_ink = ink.any(axis=1)
    lines = []
    y = 0
    H = ink.shape[0]
    single_line_max_h = glyphs.height + 4
    while y < H:
        if not row_has_ink[y]:
            y += 1
            continue
        y_start = y
        while y < H and row_has_ink[y]:
            y += 1
        y_end = y
        band_h = y_end - y_start
        if band_h > single_line_max_h:
            # 위 두 줄 이상이 맞닿아 하나의 band로 합쳐진 경우 — 고정
            # pitch로 다시 잘라 줄마다 따로 매칭한다.
            n_sub = max(1, round(band_h / _HINT_LINE_PITCH))
            for i in range(n_sub):
                sy0 = y_start + i * _HINT_LINE_PITCH
                sy1 = min(y_end, sy0 + _HINT_LINE_PITCH)
                if sy1 <= sy0:
                    continue
                text = _read_hint_band(ink, glyphs, sy0, sy1)
                if text:
                    lines.append(text)
        else:
            text = _read_hint_band(ink, glyphs, y_start, y_end)
            if text:
                lines.append(text)

    # 첫 줄은 "KEY HINT" 헤더(고정 문구) — 있으면 제외. 헤더 자체는
    # 항상 같은 문구라 정보가 없으므로, OCR이 마지막 T 하나를 놓쳐도
    # (커닝 간격이 넓어 흔히 발생 — 실측) 접두사만 맞으면 제외한다.
    if lines and lines[0].upper().replace(" ", "").startswith("KEYHIN"):
        lines = lines[1:]
    text = " ".join(lines).strip()

    # 실제 힌트 문구(HINT_TEMPLATES, doors.py)는 알파벳/숫자/공백만
    # 쓴다. 글자 다음의 여백이 넓으면(단어 끝 등) 낮은 점수의 문장부호
    # 후보가 몇 개 더 매칭되다 멈추는 경우가 있어(실측: "''''°''''"),
    # 끝에서부터 글자/숫자가 없는 토큰을 잘라낸다.
    tokens = text.split(" ")
    while tokens and not any(c.isalnum() for c in tokens[-1]):
        tokens.pop()
    return " ".join(tokens)
