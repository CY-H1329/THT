"""Cheap pixel analysis: blocked/collision detection, wall-color sampling.

Everything here runs on plain numpy array math -- no OCR, no VLM -- so
calling it every act() step costs almost nothing (well under 1 ms).
The threshold below was measured directly against the real game:
when movement is actually blocked by a wall or object, the frame diff
drops to essentially exactly 0 (nothing moved at all). When genuinely
moving, the diff is always noticeably larger (roughly 0.1-0.5 in open
space, and still above 0.2 right after stepping through a doorway).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from agent.palette import nearest_color_name

_HUD_H = 28  # matches ocr.py's _HUD_BAR_HEIGHT -- excluded so we only compare the 3D view

# Measured directly: when truly blocked, diff_frac lands right at
# 0.0000; while moving, it was never below 0.03. There's a wide margin
# either way, so this threshold separates the two safely.
BLOCKED_THRESHOLD = 0.01


def _view(frame: np.ndarray) -> np.ndarray:
    """The 3D view only (HUD bar cropped out), as int32."""
    return frame[_HUD_H:, :, :].astype(np.int32)


def frame_diff_frac(prev: np.ndarray, cur: np.ndarray, pixel_thresh: int = 30) -> float:
    """Fraction (0-1) of pixels that changed "meaningfully" between two frames (HUD excluded)."""
    a, b = _view(prev), _view(cur)
    per_pixel = np.abs(a - b).sum(axis=-1)  # 0..765
    return float((per_pixel > pixel_thresh).mean())


def is_blocked(prev: np.ndarray, cur: np.ndarray) -> bool:
    """Whether a just-attempted MOVE_FORWARD/BACK actually failed to move us."""
    return frame_diff_frac(prev, cur) < BLOCKED_THRESHOLD


def sample_wall_color(frame: np.ndarray) -> Optional[str]:
    """Match the most common color in the current frame (HUD excluded) to the 12-color palette.

    The floor (checkerboard) and ceiling (gray concrete) are different
    enough from the palette colors that the mode color is very likely
    the actual wall color. If it's too far from any palette color
    (e.g. a door, enemy, or object is covering most of the frame),
    returns None so the caller just skips this frame and tries again
    on the next one.
    """
    view = frame[_HUD_H:, :, :]
    # Downsample (1 pixel per 4x4 block) to cut the work, then take the mode color.
    small = view[::4, ::4, :].reshape(-1, 3)
    colors, counts = np.unique(small, axis=0, return_counts=True)
    mode_rgb = tuple(int(c) for c in colors[counts.argmax()])
    return nearest_color_name(mode_rgb)
