"""Enemy (humanoid mob) detection -- pulls bearing, distance, and confidence from pixels.

VLM-based aiming (agent.vlm.locate_enemy) took ~2 s per call, and we
kept getting hit during that wait (slow and floaty, confirmed by the
user). The earlier color-rule approach (agent.vision.enemy_bearing, now
removed) just compared against the room's stored wall color by simple
distance, and lighting made it mistake the wall itself for an enemy.
This module took its approach from a reference repo (CY-H1329/THT,
dev-mouvement branch -- an earlier branch of this same project) and
re-verified it against our env -- pure CV, no VLM, roughly 1-2 ms per
frame.

What a mob (6 miniworld boxes + a face quad) looks like:

* **Strongly saturated color.** The torso/pants/skin are solid,
  saturated colors, while the room-wall palette (the 12 colors in
  agent/palette.py) is much more muted.
* **Standing on the floor.** The key difference from a wall-hung
  picture frame: a frame floats in the air, so just above the floor
  boundary is still wall color, while a mob's own color runs all the
  way down to where it touches the floor.
* **Roughly human height (1.15-2.45 m).** Filters out small props
  (barrel/cone/duck) and large ones (a tree). Not perfect -- a tree is
  close enough in height that it only gets partially filtered this way
  (noted as a limitation in report.md).

Distance reuses the floor-contact row/distance that
agent.geometry.floor_boundary() already computed (same idea as a
perception module: compute it once per frame). Closer than 2.6 m
(FLOOR_MIN_RANGE), the floor-contact point is off-screen and can't be
measured, so we instead work distance backward from the top of the
silhouette and the known mob heights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from agent import geometry as geo

# Minimum saturation to count as a mob box. The reference repo used 88,
# but our WALL_PALETTE (agent/palette.py) includes near-primary wall
# colors like mustard(155)/rust(130)/brick(115)/terracotta(110), so
# reusing that value directly would be unsafe. Measured directly
# (tmp/verify_enemies.py): real rendered enemy pixels saturate at
# 101-213 (median 133), while wall background in the same frame tops
# out at 81 -- 95 leaves a safe margin.
MOB_SAT = 95
# Thickness (px) of the band read just above the floor-contact point.
FOOT_BAND = 10
MOB_HEIGHTS = (1.5, 2.0)      # short / tall
MIN_MOB_HEIGHT = 1.15
NEAR_TOP_MARGIN = 22       # allowed silhouette-top row (from the horizon) for a near-range candidate
NEAR_DEFAULT_DIST = 1.8    # fallback near-range distance (m) when height inversion doesn't converge
MAX_MOB_HEIGHT = 2.45
# Measured: a real mob's shirt/pants/skin quantize down to ~3 distinct
# colors, while a tree canopy's shading varies per face and quantizes
# to ~23. 8 leaves margin either way.
MAX_COLOR_DIVERSITY = 8


@dataclass
class Mob:
    bearing: float        # relative bearing from the current heading (deg, left is +)
    distance: float       # m (below 2.6 m this is a height-based estimate)
    height: float         # estimated silhouette height (m), -1.0 if it couldn't be cross-checked
    width_deg: float
    col_range: Tuple[int, int]
    resolved: bool        # whether distance came from an actual floor-contact measurement
    score: float          # 0-1, how mob-like this candidate is


def detect(frame: np.ndarray, n_cols: int = 64,
           wall_rgb: Optional[tuple] = None,
           sat: Optional[np.ndarray] = None,
           boundary: Optional[tuple] = None) -> List[Mob]:
    """Return the list of mob candidates in the frame, nearest first.

    All the column-wise looping is vectorized in numpy -- this runs
    every step (every tick during combat), so a plain Python loop would
    eat too much of the time budget.
    """
    if sat is None:
        sat = geo.saturation(frame)
    rows, dists = boundary if boundary is not None else geo.floor_boundary(
        frame, n_cols, sat)
    cw = geo.FRAME_W // n_cols
    # Saturation map bucketed by column (FRAME_H, n_cols). Within each
    # bucket we take the max so a thin mob doesn't get averaged away by
    # the background.
    satc = sat[:, :n_cols * cw].reshape(geo.FRAME_H, n_cols, cw).max(axis=2)

    # 1) Is the FOOT_BAND strip just above the floor-contact point
    #    strongly colored? -> a saturated object standing on the floor.
    y_hi = np.clip(rows.astype(int), int(geo.CY) + 1, geo.FRAME_H)
    offs = np.arange(FOOT_BAND)
    y_idx = np.clip(y_hi[:, None] - 1 - offs[None, :], int(geo.CY), geo.FRAME_H - 1)
    foot = satc[y_idx, np.arange(n_cols)[:, None]]        # (n_cols, FOOT_BAND)
    standing = np.median(foot, axis=1) >= MOB_SAT

    strong = satc >= MOB_SAT                               # (FRAME_H, n_cols)
    bearings = geo.column_bearings(n_cols)
    col_w = geo.FOV_X_DEG / n_cols

    # 2) Group into runs of consecutive columns.
    mobs: List[Mob] = []
    edges = np.flatnonzero(np.diff(np.r_[False, standing, False]))
    for i, j in zip(edges[::2], edges[1::2]):
        seg = slice(int(i) * cw, int(j) * cw)
        d_floor = float(np.min(dists[i:j]))
        resolved = d_floor > geo.FLOOR_MIN_RANGE + 0.01

        # 3) Silhouette top: the highest row where a strong color still
        #    runs across this column range.
        rows_strong = np.flatnonzero(strong[:, i:j].mean(axis=1) > 0.25)
        rows_strong = rows_strong[rows_strong >= geo.HUD_H]
        if len(rows_strong) == 0:
            continue
        y_top = float(rows_strong.min())

        # Tree false-positive guard (measured directly): a tree's
        # height (2.2 m) overlaps a mob's, so height alone doesn't
        # filter it out -- but its shape is different. A tree's base
        # (the trunk touching the floor) is narrow while its top (the
        # canopy) spreads much wider; a humanoid mob is the opposite --
        # its base (legs) is never narrower than its head. We compare
        # the width of the strong-color run at the silhouette's very
        # top row to the width of the base column range, and drop the
        # candidate if the top is much wider (a canopy shape).
        top_row = int(y_top)
        lo, hi = int(i), int(j)
        while lo > 0 and strong[top_row, lo - 1]:
            lo -= 1
        while hi < n_cols and strong[top_row, hi]:
            hi += 1
        base_cols = int(j) - int(i)
        canopy_cols = hi - lo
        if base_cols > 0 and canopy_cols / base_cols > 1.6:
            continue

        if resolved:
            distance = d_floor
            height = geo.height_at(y_top, distance)
        else:
            # Floor-contact point is off-screen (within 2.6 m), so we
            # can't cross-check height that way. Instead we gauge
            # plausibility by whether the silhouette reaches up near
            # the horizon.
            if y_top > geo.CY + NEAR_TOP_MARGIN:
                continue
            cand = [geo.dist_for_height(y_top, h) for h in MOB_HEIGHTS]
            cand = [d for d in cand if 0.4 <= d <= geo.FLOOR_MIN_RANGE + 0.6]
            distance = min(cand) if cand else NEAR_DEFAULT_DIST
            height = None                      # can't cross-check -> skip the height gate

        width_deg = abs(bearings[j - 1] - bearings[i]) + col_w
        score = _score(height, width_deg, distance, frame, seg, wall_rgb)
        if score > 0.0:
            mobs.append(Mob(
                bearing=(bearings[i] + bearings[j - 1]) / 2.0,
                distance=float(distance),
                height=float(height) if height is not None else -1.0,
                width_deg=float(width_deg), col_range=(int(i), int(j)),
                resolved=bool(resolved), score=float(score),
            ))

    mobs.sort(key=lambda m: m.distance)
    return mobs


def _score(height: Optional[float], width_deg: float, distance: float,
           frame: np.ndarray, seg: slice, wall_rgb) -> float:
    """How mob-like this candidate is, 0-1. 0 means drop it."""
    if height is not None and not (MIN_MOB_HEIGHT <= height <= MAX_MOB_HEIGHT):
        return 0.0                     # too short (barrel/cone/duck) or too tall (wall-like)
    max_w = math.degrees(2 * math.atan(1.3 / max(distance, 0.5))) + 6.0
    if width_deg > max_w:
        return 0.0                     # spread out too wide, like a flat wall surface
    if _color_diversity(frame, seg) > MAX_COLOR_DIVERSITY:
        return 0.0                     # measured: a tree's canopy shape can overlap a mob's
                                        # height (2.2 m), so height/width alone sometimes can't
                                        # filter it out. A mob's shirt/pants/skin are each a
                                        # solid color (quantizes to ~3 colors), while a tree's
                                        # shading varies per face and is far more diverse
                                        # (measured: ~23 for a tree vs ~3 for a mob).
    earned, possible = 0.0, 0.0
    if wall_rgb is not None:
        possible += 1.0
        if not geo.color_close(geo.region_color(frame, seg.start, seg.stop),
                                wall_rgb, tol=45):
            earned += 1.0          # clearly not the room's wall color
    possible += 1.0
    if _vertical_color_layers(frame, seg) >= 2:
        earned += 1.0              # shirt/pants/skin stacked vertically
    return 0.6 + 0.4 * (earned / possible if possible else 0.0)


def _vertical_color_layers(frame: np.ndarray, seg: slice,
                            tol: int = 45) -> int:
    """Scan this column range top-to-bottom and count how many
    "clearly distinct, saturated" color layers there are.

    A mob has shirt/pants/skin stacked vertically, giving 2-3 layers;
    a solid-color prop or a wall gives 1.
    """
    band = frame[int(geo.CY) - 60:geo.FRAME_H, seg]
    if band.size == 0:
        return 0
    rows = np.median(band.reshape(band.shape[0], -1, 3), axis=1)
    keep = rows.max(axis=1) - rows.min(axis=1) >= MOB_SAT
    rows = rows[keep]
    if len(rows) == 0:
        return 0
    if len(rows) == 1:
        return 1
    jumps = np.abs(np.diff(rows, axis=0)).max(axis=1) > tol
    return int(jumps.sum()) + 1


def _color_diversity(frame: np.ndarray, seg: slice, quant: int = 24) -> int:
    """Quantize the saturated pixels in this column range and count how many distinct colors remain.

    Measured (tmp/verify_enemies.py and related scripts): a real mob's
    shirt/pants/skin each render as a single flat color, so after
    quantizing there are only ~3 distinct colors, while a tree's canopy
    shades differently per face and produces up to ~23. Since a mob's
    height range (2.2 m) overlaps a tree's, this catches trees that the
    height/width gates alone can't filter out.
    """
    band = frame[int(geo.CY) - 60:geo.FRAME_H, seg]
    if band.size == 0:
        return 0
    pixels = band.reshape(-1, 3).astype(np.int32)
    sat = pixels.max(axis=1) - pixels.min(axis=1)
    colorful = pixels[sat > 30]
    if len(colorful) == 0:
        return 0
    quantized = (colorful // quant) * quant
    return int(len(np.unique(quantized, axis=0)))


# Close-range (within attack range) low-cost bearing estimate
# Discovered by measurement (real combat frames, seed 7): detect()'s
# floor-contact/silhouette-ratio check is tuned for an enemy that
# appears small and human-shaped in the middle-to-lower part of the
# frame -- mid-range. By the time we're in FLEE strike (meaning we've
# already been hit, so the enemy is 1.3-1.5 m away, right in front of
# us -- see the note at the top of explorer.py), its body fills almost
# the whole frame, the floor isn't visible, and the silhouette ratio no
# longer matches, so detect() structurally returns zero candidates.
# That meant every fight skipped CV aiming entirely and dropped straight
# into the 24-direction sweep (up to 72 ticks) -- the reason kills felt
# much slower than before. This function skips the humanoid-shape check
# entirely and just finds the bearing of the centroid of "a big blob
# that's neither floor nor this room's wall color" -- FLEE strike is
# only entered once HP has actually dropped (so we already know
# something is right in front of us), which makes false positives much
# less of a concern than during a general scan.
_CLOSE_RANGE_MIN_FRAC = 0.15   # ignore if "neither floor nor wall" covers less than this fraction
_CLOSE_RANGE_WALL_DIFF = 120   # minimum RGB-channel-sum difference from the wall color
# Discovered by measurement (seed 7): even after actually killing the
# enemy (HP 0, mesh removed from the render), a different room/hallway
# glimpsed through a doorway can still satisfy "neither floor nor wall,"
# so we kept "attacking" the dead target (all misses) and wastefully
# fell through to a sweep anyway. A genuinely close enemy has its feet
# right on the floor in front of us, so its body is guaranteed to
# extend down to the very bottom rows of the frame (measured: the frame
# right before a kill has candidate pixels across 29% of the bottom
# 20% band), while a scene glimpsed through a doorway has real floor in
# front of it and never reaches that far down (measured: 0%). This
# difference is what separates "a real body right in front of us" from
# "scenery glimpsed far off through a doorway."
_CLOSE_RANGE_BOTTOM_FRAC_MIN = 0.20  # minimum candidate-pixel fraction required in the bottom 20% band
_CLOSE_RANGE_BOTTOM_BAND = 0.20      # how much of the bottom of the frame counts as the "at our feet" band


def close_range_bearing(frame: np.ndarray, wall_rgb: Optional[Tuple[int, int, int]] = None,
                         min_frac: float = _CLOSE_RANGE_MIN_FRAC,
                         require_bottom_band: bool = True) -> Optional[float]:
    """Close-combat only: quickly estimate the bearing of a large, likely-enemy blob.

    Returns None if the caller should fall back to detect() (mid-range)
    or the sweep. Across the whole frame below the HUD, it converts the
    column-wise centroid of pixels that are "not the floor (achromatic
    checkerboard), and (if wall_rgb is given) not this room's wall
    color either" into a bearing. A blob that doesn't extend down to
    the very bottom of the frame (at our feet) is likely scenery seen
    through a doorway and gets excluded (see the comment above) -- but
    this bottom-band requirement is only enforced when
    require_bottom_band=True. A bug we found by measurement: always
    enforcing it meant that even the very first detection attempt of a
    fresh engagement (where there's no risk yet of attacking a corpse,
    since nothing has landed a hit) would miss a genuinely visible
    enemy seen at an angle or through a gap, and fall straight through
    to the slow 24-direction sweep (this is what made combat feel
    sluggish -- reproduced and confirmed on request: the same
    face-on frame is missed with require_bottom_band=True, but found
    correctly at bearing -0.85 deg with it False). The caller passes
    this flag based on whether this engagement has landed a hit yet.
    """
    body = frame[geo.HUD_H:, :, :]
    sat = geo.saturation(body)
    not_floor = sat >= geo._SAT_THRESHOLD
    if wall_rgb is not None:
        diff = np.abs(body.astype(np.int32) - np.array(wall_rgb, dtype=np.int32)).sum(axis=-1)
        mask = not_floor & (diff > _CLOSE_RANGE_WALL_DIFF)
    else:
        mask = not_floor
    if mask.mean() < min_frac:
        return None
    if require_bottom_band:
        bottom_start = int(mask.shape[0] * (1.0 - _CLOSE_RANGE_BOTTOM_BAND))
        if mask[bottom_start:].mean() < _CLOSE_RANGE_BOTTOM_FRAC_MIN:
            return None
    col_weight = mask.sum(axis=0).astype(np.float64)
    total = col_weight.sum()
    if total <= 0:
        return None
    cols = np.arange(len(col_weight))
    centroid_col = float((cols * col_weight).sum() / total)
    return -math.degrees(math.atan((centroid_col - geo.CX) / geo.FOCAL))
