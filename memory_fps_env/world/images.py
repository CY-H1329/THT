"""Wall-image sources: procedural (dev), public (shipped), heldout (eval-only).

Each source returns a list of ImageEntry objects with both the rendered pixels
(numpy RGB array, used by Miniworld as a texture) and the ground-truth metadata
(caption, salient features) that the event log records.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import json
import numpy as np
from PIL import Image, ImageDraw

from memory_fps_env.rng import EnvRNG

_PUBLIC_DIR = Path(__file__).resolve().parent.parent / "assets" / "images" / "public"


@dataclass
class ImageEntry:
    id: str
    category: str
    caption: str
    salient_features: List[str] = field(default_factory=list)
    rgb: np.ndarray = field(default=None, repr=False)


class ImageSource:
    def sample(self, rng: EnvRNG, k: int) -> List[ImageEntry]:
        raise NotImplementedError


class ProceduralImageSource(ImageSource):
    """Generates simple synthetic shape compositions with known captions.

    Categories are abstract — 'shapes_red_triangles', 'shapes_blue_circles',
    etc. — so the agent can still describe them in free text.
    """

    _SHAPES = ["triangles", "circles", "squares"]
    _COLORS = {
        "red": (220, 30, 30),
        "green": (30, 180, 30),
        "blue": (30, 60, 220),
        "yellow": (240, 210, 30),
        "purple": (160, 30, 200),
    }
    _COUNTS = [2, 3, 4, 5]

    def sample(self, rng: EnvRNG, k: int) -> List[ImageEntry]:
        # Build the full unique pool first (3 shapes × 5 colors × 4 counts = 60).
        # If k > pool size, entries are reused (with unique idx prefixes) so the
        # loop always terminates even when total_images grows with per-wall counts.
        pool = []
        for shape in self._SHAPES:
            for color in self._COLORS:
                for count in self._COUNTS:
                    rgb = self._render(shape, color, count, rng)
                    pool.append((shape, color, count, rgb))
        rng.shuffle(pool)
        out = []
        for idx in range(k):
            shape, color, count, rgb = pool[idx % len(pool)]
            # Copy rgb so cycled entries don't alias the same array (any future
            # in-place mutation by a caller would otherwise corrupt other rooms).
            entry = ImageEntry(
                id=f"proc_{idx}_{shape}_{color}_{count}",
                category=f"shapes_{color}_{shape}",
                caption=f"{count} {color} {shape} on a white background",
                salient_features=[color, shape, str(count), "white background"],
                rgb=rgb.copy(),
            )
            out.append(entry)
        return out

    def _render(self, shape: str, color: str, count: int, rng: EnvRNG) -> np.ndarray:
        size = 256
        img = Image.new("RGB", (size, size), "white")
        d = ImageDraw.Draw(img)
        rgb = self._COLORS[color]
        for _ in range(count):
            cx = rng.randint(40, size - 40)
            cy = rng.randint(40, size - 40)
            r = rng.randint(20, 50)
            if shape == "circles":
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
            elif shape == "squares":
                d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
            elif shape == "triangles":
                d.polygon(
                    [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)],
                    fill=rgb,
                )
        return np.array(img, dtype=np.uint8)


class PublicImageSource(ImageSource):
    """Loads pre-generated PNGs + sidecar JSON from assets/images/public/."""

    def __init__(self, path: Path | None = None):
        self.path = path or _PUBLIC_DIR

    def sample(self, rng: EnvRNG, k: int) -> List[ImageEntry]:
        entries = self._scan()
        if not entries:
            raise ValueError(f"No images found in {self.path}")
        if k > len(entries):
            raise ValueError(f"Requested {k} images, pool size {len(entries)}")
        picked = rng.sample(entries, k)
        for e in picked:
            e.rgb = np.array(Image.open(self.path / f"{e.id}.png").convert("RGB"))
        return picked

    def _scan(self) -> List[ImageEntry]:
        entries = []
        for jf in sorted(self.path.glob("*.json")):
            data = json.loads(jf.read_text(encoding="utf-8"))
            entries.append(
                ImageEntry(
                    id=jf.stem,
                    category=data.get("category", ""),
                    caption=data.get("caption", ""),
                    salient_features=data.get("salient_features", []),
                )
            )
        return entries


class HeldoutImageSource(PublicImageSource):
    """Same as PublicImageSource but rooted at a private path."""

    def __init__(self, path: Path):
        super().__init__(path=Path(path))
