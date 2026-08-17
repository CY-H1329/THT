"""Enemy face textures: public (shipped) + heldout (eval-only)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from memory_fps_env.rng import EnvRNG

_PUBLIC_DIR = (
    Path(__file__).resolve().parent.parent / "assets" / "enemy_faces" / "public"
)
_HELDOUT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "eval_harness" / "assets_private" / "enemy_faces" / "heldout"
)


@dataclass
class FaceEntry:
    id: str           # e.g. "face_03"
    path: Path        # absolute path to the PNG (for Miniworld's Texture.tex_paths)
    rgba: np.ndarray = field(default=None, repr=False)


class FaceSource:
    def sample(self, rng: EnvRNG, k: int) -> List[FaceEntry]:
        raise NotImplementedError

    def scan(self) -> List[FaceEntry]:
        raise NotImplementedError


class PublicFaceSource(FaceSource):
    def __init__(self, path: Path | None = None):
        self.path = path or _PUBLIC_DIR

    def scan(self) -> List[FaceEntry]:
        entries: List[FaceEntry] = []
        for png in sorted(self.path.glob("face_*.png")):
            rgba = np.array(Image.open(png).convert("RGBA"))
            entries.append(FaceEntry(id=png.stem, path=png, rgba=rgba))
        return entries

    def sample(self, rng: EnvRNG, k: int) -> List[FaceEntry]:
        entries = self.scan()
        if not entries:
            raise ValueError(f"No face PNGs found in {self.path}")
        # With replacement so episode enemy count (≤24) can exceed pool size (8).
        return [rng.choice(entries) for _ in range(k)]


class HeldoutFaceSource(PublicFaceSource):
    def __init__(self, path: Path | None = None):
        super().__init__(path=path or _HELDOUT_DIR)
