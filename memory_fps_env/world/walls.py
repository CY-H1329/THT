"""Per-room wall color palette + solid-color PNG renderer.

Each room in the world gets its own wall color from this shared palette.
The palette is intentionally shared across public and heldout sessions:
forcing a heldout palette would only test color-name matching, not
vision.
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image

# (palette_name, RGB tuple). Names also serve as event-log identifiers.
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


def render_solid_png(rgb: Tuple[int, int, int], out_dir: Path, size: int = 64) -> Path:
    """Write a flat size×size PNG of solid `rgb` to out_dir.

    Filename is `solid_{r}_{g}_{b}.png` so repeat calls with the same
    color reuse the same file (Miniworld's texture loader caches by
    name, so reusing the path also reuses the GPU texture).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    r, g, b = rgb
    path = out_dir / f"solid_{r}_{g}_{b}.png"
    if not path.exists():
        Image.new("RGB", (size, size), (r, g, b)).save(path)
    return path
