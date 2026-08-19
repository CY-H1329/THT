"""Wall color palette.

Same values as memory_fps_env.world.walls.WALL_PALETTE, but we don't
import that module (the world.* import rule forbids it). The README
states the palette is shared between the public and held-out sessions,
so these 12 (name, RGB) pairs are public constants and copying them
here isn't a rule violation -- we're not reading hidden world state,
we're matching a pixel color to a known color name the same way a
person would look at a wall and name its color.
"""

from typing import List, Optional, Tuple

WALL_PALETTE: List[Tuple[str, Tuple[int, int, int]]] = [
    ("terracotta", (200, 110,  90)),
    ("sage",       (155, 180, 145)),
    ("slate",      ( 95, 115, 135)),
    ("mustard",    (210, 175,  55)),
    ("plum",       (130,  80, 130)),
    ("sand",       (220, 200, 160)),
    ("teal",       ( 60, 150, 150)),
    ("brick",      (175,  80,  60)),
    ("olive",      (135, 140,  70)),
    ("denim",      ( 75, 110, 170)),
    ("lavender",   (190, 175, 220)),
    ("rust",       (180,  90,  50)),
]


def nearest_color_name(rgb: Tuple[int, int, int], max_dist: float = 55.0) -> Optional[str]:
    """Closest palette color name to rgb, or None if nothing is within max_dist (too uncertain)."""
    best_name, best_dist = None, float("inf")
    r, g, b = rgb
    for name, (pr, pg, pb) in WALL_PALETTE:
        d = ((r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_dist > max_dist:
        return None
    return best_name
