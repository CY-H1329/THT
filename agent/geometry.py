"""Frame pixels -> free distance (depth) per direction, plus camera geometry constants.

Core idea: this env's floor is always a black-and-white checkerboard
(floor_tiles_bw), so its saturation is close to 0, while walls (from
WALL_PALETTE), 3D objects, and enemies are almost all saturated colors.
So scanning each pixel column from the bottom of the screen upward and
finding the first row where saturation crosses a threshold gives the
row where the floor ends in that direction -- i.e. the base of the
nearest obstacle (wall/object/enemy).

The camera is created with domain_rand=False, so its parameters are
fixed (confirmed in the env.py source: _MWWorld constructs MiniWorldEnv
with domain_rand=False). That means miniworld's DEFAULT_PARAMS apply as-is
(cam_height 1.5 m, fov_y 60 deg, pitch 0, forward_step 0.15 m, turn_step
15 deg) -- also confirmed by measurement (one MOVE_BACK step moved
exactly 0.150 m). So the row coordinate y of the floor's edge converts
exactly back to a distance:

    dz = cam_height * focal / (y - cy)      # distance along the optical axis
    d  = dz * sqrt(1 + u^2),  u = (x - cx)/focal

The idea for this formula and these constants came from a reference
repo (CY-H1329/THT, dev-mouvement branch -- an earlier branch of this
same project) and was independently re-verified by measurement against
this env. memory_fps_env itself is never imported -- these constants
are just miniworld's own defaults, re-confirmed from pixels; the agent
still only ever acts on the obs frame.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

# Camera/frame constants
FRAME_H, FRAME_W = 240, 320
HUD_H = 28                      # rows covered by the top HUD bar
FOV_Y_DEG = 60.0
CAM_HEIGHT = 1.5                # m
FOCAL = (FRAME_H / 2.0) / math.tan(math.radians(FOV_Y_DEG / 2.0))  # ~= 207.85 px
CX = (FRAME_W - 1) / 2.0
CY = (FRAME_H - 1) / 2.0
FOV_X_DEG = 2.0 * math.degrees(math.atan(CX / FOCAL))  # ~= 75 deg

# Movement/turn units (miniworld DEFAULT_PARAMS, fixed by domain_rand=False)
FORWARD_STEP = 0.15             # m per MOVE_FORWARD/MOVE_BACK
TURN_STEP = 15.0                # deg per TURN_LEFT/RIGHT
AGENT_RADIUS = 0.4              # m

# Sensing parameters
_SAT_THRESHOLD = 18             # below this saturation, treat the pixel as floor (achromatic checkerboard)
MAX_RANGE = 20.0                # anything past this is just reported as "open"
# Floor distance seen by the very bottom row of the frame. An obstacle
# closer than this hides the floor entirely, so we can't measure it and
# report NEAR_RANGE instead.
FLOOR_MIN_RANGE = CAM_HEIGHT * FOCAL / (FRAME_H - 1 - CY)   # ~= 2.6 m
NEAR_RANGE = 0.6


def wrap180(deg: float) -> float:
    """Wrap an angle into (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


@lru_cache(maxsize=8)
def column_bearings(n_cols: int) -> tuple:
    """Column index -> relative bearing (degrees) from the current heading.

    The right side of the screen is the direction where heading
    decreases (matching how TURN_RIGHT decreases heading -- see
    _heading_diff in agent/explorer.py), so we flip the sign so that
    "the absolute heading this column is looking at" = heading + bearing.
    """
    cw = FRAME_W / n_cols
    out = []
    for i in range(n_cols):
        px = (i + 0.5) * cw
        out.append(-math.degrees(math.atan((px - CX) / FOCAL)))
    return tuple(out)


def saturation(frame: np.ndarray) -> np.ndarray:
    """Per-pixel (max channel - min channel). Both floor detection and
    mob detection use this, so we compute it once per frame and reuse it."""
    return frame.max(axis=-1).astype(np.int16) - frame.min(axis=-1).astype(np.int16)


def floor_mask(frame: np.ndarray, sat: Optional[np.ndarray] = None) -> np.ndarray:
    """Mask of achromatic (floor-checkerboard) pixels."""
    if sat is None:
        sat = saturation(frame)
    return sat < _SAT_THRESHOLD


def floor_boundary(frame: np.ndarray, n_cols: int = 40,
                    sat: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Per column-bucket: (row where the floor ends, distance to that point in m).

    The row coordinate is also used (in enemies.py) to judge whether an
    obstacle is actually touching the floor. A wall-hung picture floats
    at 0.75-2.25 m, so just above the floor boundary is still wall
    color, whereas a mob or floor object's own color extends all the
    way down to the floor.
    """
    floor = floor_mask(frame, sat)
    y_lo = int(CY) + 2                      # start just below the horizon
    band = floor[y_lo:FRAME_H, :]           # (rows, W)
    cw = FRAME_W // n_cols

    cols = band[:, :n_cols * cw].reshape(band.shape[0], n_cols, cw).mean(axis=2) > 0.5
    rev = ~cols[::-1]                        # bottom-to-top "not floor" mask
    idx = rev.argmax(axis=0)                 # position of the first True (0 if none)
    has_obstacle = rev.any(axis=0)

    y_edge = (FRAME_H - 1) - idx             # row where non-floor starts
    dy = y_edge + 0.5 - CY
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = CAM_HEIGHT * FOCAL / np.maximum(dy, 1e-6)
    u = np.array([(i + 0.5) * cw - CX for i in range(n_cols)]) / FOCAL
    dist = dz * np.sqrt(1.0 + u * u)

    bottom_blocked = ~cols[-1]
    dist = np.where(bottom_blocked, NEAR_RANGE, dist)
    dist = np.where(has_obstacle, dist, MAX_RANGE)
    y_edge = np.where(has_obstacle, y_edge, FRAME_H - 1)
    return y_edge, np.clip(dist, NEAR_RANGE, MAX_RANGE)


def depth_profile(frame: np.ndarray, n_cols: int = 40,
                   sat: Optional[np.ndarray] = None) -> np.ndarray:
    """Array of free distance (m) per column. Index lines up with column_bearings(n_cols)."""
    return floor_boundary(frame, n_cols, sat)[1]


def height_at(row: float, dist: float) -> float:
    """Real-world height (m) an object at distance dist would have to be, to appear at pixel row `row`."""
    return CAM_HEIGHT - dist * (row - CY) / FOCAL


def dist_for_height(row: float, height: float) -> float:
    """Distance (m) at which an object of the given height would have its top appear at pixel row `row`.

    Below ~2.6 m the floor boundary is off-screen and we can't measure
    distance that way, but mobs have a known height (1.5 m / 2.0 m), so
    we work the distance backward from the row where their head appears.
    """
    dy = CY - row
    if dy <= 1e-6:
        return MAX_RANGE
    return max(0.3, FOCAL * (height - CAM_HEIGHT) / dy)


def cone_clearance(profile: np.ndarray, center_deg: float = 0.0,
                    half_width_deg: float = 12.0) -> float:
    """Minimum free distance within the [center +/- half] cone around the current heading."""
    bearings = column_bearings(len(profile))
    vals = [d for b, d in zip(bearings, profile)
            if abs(wrap180(b - center_deg)) <= half_width_deg]
    return float(min(vals)) if vals else MAX_RANGE


def frame_motion(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Mean absolute difference over the bottom half of two frames. Used
    to check whether a MOVE_FORWARD/BACK actually took effect."""
    a = frame_a[FRAME_H // 2:, :].astype(np.int16)
    b = frame_b[FRAME_H // 2:, :].astype(np.int16)
    return float(np.abs(a - b).mean())


BLOCKED_MOTION = 4.0   # below this, treat movement as "blocked"


def region_color(frame: np.ndarray, col_lo: int, col_hi: int) -> tuple:
    """Representative color (per-channel median) below the horizon, in the given column range."""
    lo = max(0, min(FRAME_W - 1, col_lo))
    hi = max(lo + 1, min(FRAME_W, col_hi))
    band = frame[int(CY):FRAME_H, lo:hi].reshape(-1, 3)
    if band.size == 0:
        return (0, 0, 0)
    med = np.median(band, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def color_close(a, b, tol: int = 40) -> bool:
    if a is None or b is None:
        return False
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))


# Locked-door padlock-panel recognition
# A real problem we hit: SEEK would keep treating a certain direction as
# just "physically blocked" and retry it over and over, when it turned
# out not to be a wall at all but a locked door (a padlock-icon panel)
# -- there's no way through without the key, so retrying was pure wasted
# time, and not knowing that cost us a lot of it. The locked door's
# padlock panel has a warm golden background (measured RGB roughly
# (143,130,104) to (196,180,100), R>G>B) with a dark black/gray padlock
# graphic on top -- we require both the golden background AND the dark
# blob inside it together, which is what separates it from an actual
# wall color like mustard (mustard has a similar background but no dark
# graphic inside it).
def _warm_gold(region: np.ndarray) -> np.ndarray:
    r = region[..., 0].astype(np.int32)
    g = region[..., 1].astype(np.int32)
    b = region[..., 2].astype(np.int32)
    return (r > g) & (g > b) & (r - b > 25) & (r > 80)


def lock_visible(frame: np.ndarray, min_warm_frac: float = 0.6,
                  min_dark_frac: float = 0.2) -> bool:
    """Whether the locked door's padlock panel takes up a large enough
    part of the frame (used by SEEK to check before giving up on a
    direction). Pairing this with the hint banner (which only appears
    once you're actually touching the locked door) is more reliable,
    but this is meant as a secondary signal that fires "from a distance,
    before touching it."

    Measured: the thresholds started lower, and that produced a false
    positive on a sand-toned wall combined with a dark wall image (e.g.
    a shark photo) -- warm_frac=0.52, dark_frac=0.16. A real locked door
    measured clearly higher (warm_frac=0.66, dark_frac=0.34), so both
    thresholds were raised to sit between the two. The sample size is
    still small, so this isn't treated as 100% reliable (a false
    positive would make SEEK wrongly skip a door that actually opens --
    noted as a limitation to write up in report.md)."""
    region = frame[HUD_H:FRAME_H, FRAME_W // 4: 3 * FRAME_W // 4]
    warm = _warm_gold(region)
    if float(warm.mean()) < min_warm_frac:
        return False
    gray = region.astype(np.int32).sum(axis=-1) / 3.0
    dark = gray < 60
    return float(dark.mean()) >= min_dark_frac
