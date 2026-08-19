"""Reading the HUD bar and hint-banner text from pixels, our own OCR.

Why not a general-purpose OCR like tesseract: the env draws the
HUD/hint-banner text with Pillow's bundled default font
(ImageFont.load_default), on a solid-color background with almost no
anti-aliasing (see hud.py, env.py). We can render reference glyphs with
that exact same font and size ourselves, so a greedy left-to-right
template match -- "which candidate character best matches the pixels at
this position" -- gives OCR that's essentially 100% accurate and runs
in under 1 ms. Room names and hint-text vocabulary are held out
between the dev and eval pools and unknown to us in advance, but since
this reads any string character by character, it doesn't need to know
specific words to work.

We don't import memory_fps_env.hud -- in keeping with the spirit of the
task ("read the HUD from pixels too"), the font-loading logic is
reimplemented independently here as well.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Font
# Same loading approach as hud.py/env.py (Pillow >=10.1's bundled
# scalable default font). The env's own code comments state that, given
# the same size, this renders byte-identically on every OS -- our
# reference rendering gets the same guarantee.
def _load_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


_HUD_FONT_SIZE = 16
_BANNER_FONT_SIZE = 28

# Every character that can actually appear in the HUD / hint banner / QA
# banner (upper/lowercase letters + digits + the relevant symbols, with
# a few extra included for margin).
_ALPHABET = (
    string.ascii_letters
    + string.digits
    + " .,:;'\"!?()-/°[]—·…"
)

_INK_THRESHOLD = 40  # brightness above the dark/navy background that counts as "character ink"
_MATCH_MIN_SCORE = 0.80  # below this score, treat this position as a failed match

# Default search window for HUD-style calls, which happen often (every
# cache miss) -- this much slack is enough to absorb sub-pixel
# rendering jitter, and the HUD string decoding has been validated by
# direct measurement.
_DX_SEARCH_NARROW = (0, -1, 1, -2, 2)
_DY_SEARCH_NARROW = (0, -1, 1)

# Reference glyphs are rendered as the whole alphabet on one line, then
# cropped to the topmost/leftmost ink across the whole set
# (_build_glyph_set). But how many pixels a given character's own ink
# actually starts inside that shared baseline varies per character
# (e.g. at BANNER_FONT_SIZE=28, measured: 'K' can be off by up to 3 px,
# 't' by about 1 px). The narrow search window isn't enough to absorb
# that (the real character then scores low, and some unrelated
# character of a different width that happens to line up with the
# background/blank space wins instead, breaking the whole decoded
# string -- found by direct measurement), so the hint banner gets a
# much wider search window of its own. The hint banner is only ever
# read while it's up, and only when the text changes (a low call
# frequency), so if this wide window were also used for the HUD (8x
# more combinations), read_hud would slow back down to ~143 ms -- it
# must only be passed in on the hint-banner path.
_DX_SEARCH_WIDE = (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5)
_DY_SEARCH_WIDE = (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5)


@dataclass
class _GlyphSet:
    font: ImageFont.FreeTypeFont
    chars: str
    masks: list  # list[np.ndarray[bool]], per-character ink mask (height, width)
    height: int
    width_groups: dict  # width -> (chars_str, stacked_masks (n, height, width) bool)


@lru_cache(maxsize=4)
def _build_glyph_set(size: int) -> _GlyphSet:
    """Render the whole alphabet in one shot and crop out a reference bitmap per character.

    Rendering it all at once means the same advance-width calculation
    the real HUD rendering uses applies here too, avoiding the bearing
    error that would show up if each character were drawn separately.
    """
    font = _load_font(size)
    # Render the whole alphabet on one line, on a generously sized canvas.
    pad_top = size  # margin for ascenders/descenders
    canvas_h = size * 2
    canvas_w = int(font.getlength(_ALPHABET)) + 8
    img = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(img)
    draw.text((0, pad_top // 2), _ALPHABET, fill=255, font=font)
    arr = np.array(img)

    # Each character's starting x is the cumulative advance width of the substring before it.
    xs = [0.0]
    for i in range(1, len(_ALPHABET) + 1):
        xs.append(font.getlength(_ALPHABET[:i]))

    # Crop tightly, vertically, to the row range where ink actually exists.
    ink_rows = np.where((arr > _INK_THRESHOLD).any(axis=1))[0]
    y0, y1 = (int(ink_rows.min()), int(ink_rows.max()) + 1) if len(ink_rows) else (0, canvas_h)

    masks = []
    for i, ch in enumerate(_ALPHABET):
        x0, x1 = int(round(xs[i])), max(int(round(xs[i])) + 1, int(round(xs[i + 1])))
        glyph = arr[y0:y1, x0:x1]
        masks.append(glyph > _INK_THRESHOLD)

    # Group reference masks by width into stacked (n, h, w) arrays, so
    # _read_line can compare a whole group of same-width characters with
    # numpy in one shot instead of looping over them individually in
    # Python. Measured: doing ~1200 individual numpy calls per character
    # (full alphabet x dx x dy) made a single read_hud call take 143 ms;
    # this batched comparison brings it under 10 ms. Ties within a group
    # go to whichever character comes first (preserving alphabet order),
    # keeping results identical to the unbatched version.
    groups: dict = {}
    for ch, m in zip(_ALPHABET, masks):
        groups.setdefault(m.shape[1], ([], []))
        groups[m.shape[1]][0].append(ch)
        groups[m.shape[1]][1].append(m)
    width_groups = {
        w: ("".join(chs), np.stack(ms))
        for w, (chs, ms) in groups.items()
    }
    return _GlyphSet(font=font, chars=_ALPHABET, masks=masks, height=y1 - y0,
                      width_groups=width_groups)


def _read_line(ink: np.ndarray, glyphs: _GlyphSet, x_start: int, x_end: int,
               y_off: int, dx_search=_DX_SEARCH_NARROW,
               dy_search=_DY_SEARCH_NARROW) -> str:
    """ink: (height, width) bool array (True = ink). Greedily reads one
    character at a time from x_start to x_end and reconstructs the string.

    y_off: which row of `ink` to line up with the top of the reference
    glyphs (whose height is glyphs.height, cropped to the topmost ink
    across the whole alphabet). The caller must pass in the actual
    topmost row where ink starts in this crop (a measured value, not a
    guess at centering). To absorb small alignment error, at every step
    we also search the cursor position across the dx_search/dy_search
    range and pick whichever (character, start position) combination
    scores best.

    dx_search/dy_search: default to the narrow range (fast, since the
    HUD can be called every tick). Only pass the wide range for cases
    like the hint banner, where per-character baseline error is bigger
    (see the _DY_SEARCH_WIDE comment) and call frequency is low -- using
    the wide range for the HUD too would multiply the combination count
    by 8x and bring read_hud back up to ~143 ms (found by measurement).
    """
    gh = glyphs.height
    out = []
    cursor = x_start
    stall_guard = 0
    while cursor < x_end and stall_guard < 500:
        stall_guard += 1
        best = None  # (score, char, width, actual_cursor)
        for dx in dx_search:
            c0 = cursor + dx
            if c0 < 0:
                continue
            for dy in dy_search:
                yy = y_off + dy
                if yy < 0 or yy + gh > ink.shape[0]:
                    continue
                # Compare all same-width characters as one (n,h,w) array
                # at once -- much faster than a separate numpy call per
                # character (see the _GlyphSet.width_groups comment).
                # Tie-breaking order is preserved to match alphabet order.
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
        if best is None or best[0] < _MATCH_MIN_SCORE:
            break
        score, ch, w, actual_cursor = best
        out.append(ch)
        # If the winning candidate has a large negative dx,
        # actual_cursor + w can land at or before the current cursor --
        # then the next iteration would score the exact same position
        # highest again and the cursor would stall there until
        # stall_guard runs out (an infinite loop, found with the wider
        # dx search window). Force at least 1 px of forward progress
        # every step.
        cursor = max(cursor + 1, actual_cursor + w)
    return "".join(out).rstrip()


def _ink_mask(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.astype(np.int32).sum(axis=-1) / 3.0
    return gray > _INK_THRESHOLD


def _ink_x_bounds(ink: np.ndarray) -> Optional[tuple]:
    cols = np.where(ink.any(axis=0))[0]
    if len(cols) == 0:
        return None
    return int(cols.min()), int(cols.max()) + 1


# HUD bar parsing

_HUD_BAR_HEIGHT = 28  # matches hud.py::_BAR_HEIGHT (confirmed against real frames)

_HUD_RE = re.compile(
    r"^(?P<room>.*?)\s*[·.]\s*T-(?P<secs>\d+)s\s*[·.]\s*HP\s*(?P<hp>\d+)/(?P<hpmax>\d+)"
    r"\s*[·.]\s*dir\s*(?P<heading>\d+).?(?:\s*[·.]\s*\[KEY\])?$"
    # The trailing '.?' stands in for the degree (°) symbol -- at 16 px
    # it's visually almost indistinguishable from an apostrophe ('), so
    # OCR reads it inconsistently. We only actually need the heading
    # number, so the symbol itself is never validated.
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
    """frame: the full (240,320,3) uint8 observation. Only the top HUD bar is read."""
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


# Hint banner parsing
# The same deterministic geometry as env.py::_composite_hint_overlay.
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
    """Just checks the color at a handful of deterministically computed
    border-pixel coordinates -- 100%-reliable banner detection with no
    CV/VLM needed."""
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


# env.py::_composite_hint_overlay draws hint lines 26 px apart
# (text_top + i * 26). The reference glyph height at BANNER_FONT_SIZE=28
# is 27 px -- since the line pitch (26) is smaller than the character
# height (27), two adjacent lines commonly touch or overlap with no
# blank row between them (measured: "The key is hidden" / "behind the"
# merging into one 47-row band). So "a line is whatever's between blank
# rows" isn't a safe assumption on its own -- if a band is noticeably
# taller than this pitch, we forcibly re-split it at the fixed pitch.
_HINT_LINE_PITCH = 26


_HINT_Y_MARGIN = 6  # _DY_SEARCH goes down to -5, so give that much headroom above.


def _read_hint_band(ink: np.ndarray, glyphs: "_GlyphSet", y_start: int, y_bound: int) -> str:
    """ink: the full ink mask of the banner's inner area. Treats
    [y_start, y_bound) as "the tight range where this line's ink
    actually is" for measuring its x-width, but the array actually
    passed to _read_line is cropped starting a few rows above y_start,
    giving y_off some headroom above. Why (found by measurement):
    y_start is "the topmost row with ink anywhere in this line," which
    is usually set by whichever character in that line is tallest;
    other characters start their own ink a few px lower than that
    (measured: up to 3 px, for the header's 'K'). With y_off=0, the
    negative dy needed to align them would put yy=y_off+dy below 0,
    which _read_line's bounds check always rejects -- so some unrelated
    character that happens to line up with the background (blank space)
    gets picked instead, and the whole decoded string breaks.
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
    # If the kerning gap before the last character is unusually wide
    # (measured: between the N and T of "HINT"), the loop can end early
    # right at x1 (fit tightly to the ink's right edge) and miss the
    # last character -- give it some extra room (harmless if there's
    # nothing there: it just matches blank space, which rstrip removes).
    x1 = min(band.shape[1], x1 + 20)
    return _read_line(band, glyphs, x0, x1, y_off,
                       dx_search=_DX_SEARCH_WIDE, dy_search=_DY_SEARCH_WIDE)


# Brightness of the yellow border itself (gray ~= 185, RGB (255,220,80))
# -- text has to be clearly brighter than this to count as "white text
# drawn on top of the border" (measured: body text is gray ~= 235). See
# env.py::_composite_hint_overlay.
_HINT_BORDER_GRAY = 185
_HINT_TEXT_ON_BORDER_THRESHOLD = 210


def read_hint_text(frame: np.ndarray) -> str:
    """While the hint banner is up, reads and joins its (possibly
    multi-line) text. The fixed "KEY HINT" header line is excluded.
    """
    box_x, box_y, box_w, box_h = hint_banner_box(frame.shape)
    bd = _HINT_BORDER_PX
    # In addition to the inner navy area (up to box_h-bd), also read
    # that much further down, into the border thickness (bd). Because
    # the line pitch is tight (see the _HINT_LINE_PITCH comment), the
    # last line's text was measured to sometimes spill past the bottom
    # of the inner navy area onto the yellow border itself (the "wall"
    # in "...images on its east wall" was cut off entirely). The border
    # region's background is itself bright yellow, so applying
    # _ink_mask's dark-background threshold there would wrongly treat
    # the whole border as "ink" -- so that strip gets its own,
    # much-brighter (white-text-only) threshold and is stitched onto
    # the rest.
    inner = frame[box_y + bd:box_y + box_h, box_x + bd:box_x + box_w - bd, :]
    dark_part = inner[: box_h - 2 * bd, :, :]
    border_part = inner[box_h - 2 * bd:, :, :]
    ink_dark = _ink_mask(dark_part)
    border_gray = border_part.astype(np.int32).sum(axis=-1) / 3.0
    ink_border = border_gray > _HINT_TEXT_ON_BORDER_THRESHOLD
    ink = np.concatenate([ink_dark, ink_border], axis=0)
    glyphs = _build_glyph_set(_BANNER_FONT_SIZE)

    # The text can span multiple lines, so split the ink into row bands, one band per line.
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
            # Two or more lines merged into a single band -- re-split at
            # the fixed pitch and match each line separately.
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

    # The first line is the fixed "KEY HINT" header -- drop it if
    # present. Since the header text is always the same fixed phrase and
    # carries no information, we drop it based on the prefix alone, even
    # if OCR misses the very last T (which happens often, due to the
    # wide kerning gap there -- confirmed by measurement).
    if lines and lines[0].upper().replace(" ", "").startswith("KEYHIN"):
        lines = lines[1:]
    text = " ".join(lines).strip()

    # The actual hint phrases (HINT_TEMPLATES, doors.py) only ever use
    # letters/digits/spaces. When the gap after a character is wide
    # (e.g. at the end of a word), a few low-scoring punctuation
    # candidates can keep matching before the loop stops (measured:
    # "''''°''''"), so trim tokens with no letters/digits off the end.
    tokens = text.split(" ")
    while tokens and not any(c.isalnum() for c in tokens[-1]):
        tokens.pop()
    return " ".join(tokens)
