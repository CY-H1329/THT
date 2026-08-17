# memory_fps_env/hud.py
"""HUD overlay compositor.

Renders a top bar with room name, step counter, and HP onto a Miniworld
RGB frame, in-place via PIL. Designed so the agent must extract these
values via vision (OCR or learned perception); they are never returned
in info or state.
"""

import math

from PIL import Image, ImageDraw, ImageFont
import numpy as np

_BAR_HEIGHT = 28
_PADDING = 8


def heading_degrees(dir_radians: float) -> int:
    """Convert an agent facing angle (radians) to an integer compass
    heading in ``[0, 360)``.

    Headings wrap, so a full turn maps back to 0 and negative angles map
    into the positive range. See the README "Observation" section for the
    heading-to-wall mapping (0° faces the room's east wall).
    """
    return int(round(math.degrees(dir_radians))) % 360

def _load_hud_font(size: int):
    """Load a platform-independent scalable font.

    Pillow >=10.1 ships a bundled scalable default font; using it makes the
    HUD render byte-identically on Windows, macOS, and Linux, which matters
    because the agent reads the HUD from pixels. Falls back to the legacy
    bitmap default on older Pillow.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow <10.1: load_default() takes no size arg.
        return ImageFont.load_default()


_FONT = _load_hud_font(16)


def _measure_width(text: str) -> int:
    """Return rendered width in pixels for ``text`` under the HUD font."""
    bbox = _FONT.getbbox(text)
    return bbox[2] - bbox[0]


def _truncate_to_fit(label: str, suffix: str, max_width: int) -> str:
    """Truncate ``label`` with a trailing ellipsis so ``label + suffix``
    fits within ``max_width`` rendered pixels. If the unmodified label
    already fits, it's returned unchanged.
    """
    if _measure_width(label + suffix) <= max_width:
        return label
    # Reserve room for the ellipsis itself.
    ellipsis = "…"
    suffix_w = _measure_width(suffix + ellipsis)
    budget = max_width - suffix_w
    truncated = label
    while truncated and _measure_width(truncated) > budget:
        truncated = truncated[:-1]
    return truncated.rstrip() + ellipsis


def draw_hud(
    frame: np.ndarray,
    room_name: str | None,
    seconds_remaining: int,
    hp: int,
    hp_max: int,
    has_key: bool = False,
    heading: int = 0,
) -> np.ndarray:
    """Return a copy of `frame` with the HUD bar composited on top.

    ``heading`` is the agent's facing direction in integer degrees
    ``[0, 360)``; it is rendered as a " · dir NNN°" field so the agent
    can recover its absolute orientation from pixels (needed to tell,
    e.g., a north wall from an east wall).

    When ``has_key`` is True, an extra " · [KEY]" suffix is appended
    to the HUD line so the agent can perceive its inventory state from
    pixels.

    Long room names are truncated with an ellipsis so the timer / HP /
    heading / key fields on the right are always fully visible. The
    suffix is treated as load-bearing and never truncated.
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    w, _ = img.size

    draw.rectangle([0, 0, w, _BAR_HEIGHT], fill=(0, 0, 0))
    draw.line([(0, _BAR_HEIGHT), (w, _BAR_HEIGHT)], fill=(80, 80, 80), width=1)

    label = room_name if room_name else "— corridor —"
    suffix = f" · T-{seconds_remaining}s · HP {hp}/{hp_max} · dir {heading}°"
    if has_key:
        suffix += " · [KEY]"

    max_width = w - 2 * _PADDING
    label = _truncate_to_fit(label, suffix, max_width)
    text = label + suffix
    draw.text((_PADDING, _PADDING - 2), text, fill=(255, 255, 255), font=_FONT)

    return np.array(img, dtype=np.uint8)
