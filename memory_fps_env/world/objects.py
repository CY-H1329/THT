"""3D fixed-mesh object library for rooms.

Each room receives 1–2 objects at world-build time. Objects exercise a
different memory modality than wall images (3D, viewable from any angle).

Categories are split into disjoint **public** and **heldout** pools so
the candidate's agent cannot memorize specific object identities during
development. Every category is a real Miniworld mesh — no colored
primitives (Box / Ball / Cone). Trees, barrels, medkits, traffic cones, office chairs.

Sampling is **with replacement** because each pool (5 entries) is
smaller than the typical episode object count (≤ 24 = 12 rooms × 2),
so duplicates within an episode are expected.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from memory_fps_env.rng import EnvRNG


# Pool entry layout: (mesh_category, color_or_None, caption, height_meters).
#
#   mesh_category is the Miniworld mesh name. MeshEnt(mesh_name=category,
#   height=height) loads `<category>.obj` from miniworld/meshes/ directly.
#
#   color is reserved for sources that may include colored variants of a
#   category (e.g., a heldout pool with key_<color>.obj). The public pool
#   defined below uses color=None for every entry — colored keys were
#   removed when the gameplay-key mechanic was introduced.
#
#   height controls MeshEnt's vertical scaling; picked per-mesh so tall
#   meshes don't clip the 2.74 m ceiling and small props stay visible
#   from across a room.

_PoolEntry = Tuple[str, Optional[str], str, float]

_PUBLIC: List[_PoolEntry] = [
    ("barrel",       None,   "a wooden barrel",        1.0),
    ("cone",         None,   "a traffic cone",         0.6),
    ("duckie",       None,   "a yellow rubber duckie", 0.4),
    ("office_chair", None,   "an office chair",        1.0),
    ("tree",         None,   "a leafy tree",           2.2),
]

@dataclass
class ObjectEntry:
    id: str
    category: str
    color: Optional[str]
    caption: str
    height: float = 0.5  # consumed by build_world to scale MeshEnt


class ObjectSource:
    def sample(self, rng: EnvRNG, k: int) -> List[ObjectEntry]:
        raise NotImplementedError


class _PoolObjectSource(ObjectSource):
    """Base class for pool-backed sources. Samples with replacement so the
    5-entry pool can satisfy episodes that need up to ~24 objects."""

    _pool: List[_PoolEntry] = []

    def sample(self, rng: EnvRNG, k: int) -> List[ObjectEntry]:
        if not self._pool:
            raise ValueError("empty object pool")
        out: List[ObjectEntry] = []
        for i in range(k):
            cat, col, cap, h = rng.choice(self._pool)
            ident = f"obj_{i}_{cat}" + (f"_{col}" if col else "")
            out.append(ObjectEntry(
                id=ident, category=cat, color=col, caption=cap, height=h,
            ))
        return out


class PublicObjectSource(_PoolObjectSource):
    """Default object pool used during candidate development."""
    _pool = _PUBLIC

# Note: the heldout object pool lives outside the candidate package so
# the candidate cannot read the eval-only category list. The evaluator
# constructs its object source and passes the instance via the env's
# object_source kwarg.
