"""Locked-door mechanic: placement, key-room scoring, hint composition.

Adds one locked door per episode at the entrance to the BFS-deepest leaf
room from spawn. The key spawns in the most-uniquely-describable
candidate room when the agent first touches the door, paired with a
natural-language hint describing that room.

Pure functions only — this module operates on RoomGraph and returns
data; world-state mutation lives in layout.py / env.py.
"""

import math
from collections import deque
from pathlib import Path

from PIL import Image as _PILImage, ImageDraw as _PILDraw

from memory_fps_env.world.layout import RoomGraph


def find_locked_room(graph: RoomGraph) -> int:
    """Return the room id of the graph-leaf with maximum BFS depth from spawn.

    A *graph-leaf* is a room with exactly one neighbor counting the full
    graph (spanning tree + loop edges). This is stricter than
    "spanning-tree leaf" — and deliberately so: a spanning-tree leaf
    with a loop edge to another room has *two* paths to spawn, so
    locking its tree-edge wouldn't actually gate it. Requiring
    graph-degree-1 guarantees the door's single edge is the only
    connection to that room.

    Tie-break: lowest room id among the tied candidates.

    Raises ``RuntimeError`` if no non-spawn graph-leaf exists. In the
    current difficulty (`extra_edges = 2`, 8–12 rooms) this is
    statistically near-impossible — most graphs have ≥4 graph-leaves.
    If the failure ever fires, lower `extra_edges` for that difficulty
    or relax to spanning-tree leaves with an audit step.
    """
    spawn = graph.spawn_room_id
    depths = {spawn: 0}
    queue = deque([spawn])
    while queue:
        cur = queue.popleft()
        for nb in graph.neighbors(cur):
            if nb not in depths:
                depths[nb] = depths[cur] + 1
                queue.append(nb)

    leaves = [
        r.id for r in graph.rooms
        if r.id != spawn and len(graph.neighbors(r.id)) == 1
    ]
    if not leaves:
        raise RuntimeError(
            f"No non-spawn leaf room in graph (rooms={len(graph.rooms)}, "
            f"spawn={spawn}). Spanning-tree must have at least one leaf."
        )
    max_depth = max(depths[lid] for lid in leaves)
    deepest = sorted(lid for lid in leaves if depths[lid] == max_depth)
    return deepest[0]


from collections import Counter
from typing import List, Optional


# Hint template preference order. Earlier = more preferred (rank 1 best).
HINT_TEMPLATES: List[str] = [
    "color",              # "the room with sage walls"
    "image_count_wall",   # "the room with two images on its east wall"
    "image_category",     # "the room with two tiger images"
    "object",             # "the room with a barrel"
]


_WALL_NAMES = ["south", "east", "north", "west"]
_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


def compose_hint(room, template: str) -> Optional[str]:
    """Return the hint string for ``room`` under ``template``, or None
    if the template doesn't apply (e.g., "image_category" on a room
    with zero images).

    Templates pick the most prominent feature within the room: for
    image-count-wall, the wall with the highest count; for
    image-category, the most-repeated category; for object, the first
    object in the room.
    """
    if template == "color":
        if not room.wall_color:
            return None
        return f"the room with {room.wall_color} walls"

    if template == "image_count_wall":
        if not any(room.images_per_wall):
            return None
        # Pick the wall with the highest image count (tie: south wins, then east, north, west).
        best_idx = max(range(4), key=lambda i: (room.images_per_wall[i], -i))
        n = room.images_per_wall[best_idx]
        if n <= 0:
            return None
        word = _COUNT_WORDS.get(n, str(n))
        return f"the room with {word} images on its {_WALL_NAMES[best_idx]} wall"

    if template == "image_category":
        if not room.images:
            return None
        cats = Counter(img.category for img in room.images)
        cat, n = cats.most_common(1)[0]
        word = _COUNT_WORDS.get(n, str(n))
        noun = "image" if n == 1 else "images"
        return f"the room with {word} {cat} {noun}"

    if template == "object":
        if not room.objects:
            return None
        cat = room.objects[0].category
        return f"the room with a {cat}"

    raise ValueError(f"Unknown hint template: {template!r}")


from typing import Tuple


def count_matching_rooms(graph: RoomGraph, template: str, hint: str) -> int:
    """How many rooms in ``graph`` produce exactly ``hint`` under ``template``."""
    return sum(
        1 for r in graph.rooms
        if compose_hint(r, template) == hint
    )


def choose_key_room_and_hint(
    graph: RoomGraph, locked_room_id: int,
) -> Tuple[int, str, int]:
    """Pick the key's spawn room and compose its hint.

    Iterates over candidate rooms (every room except spawn and the locked
    room). For each candidate room R, tries each template T in
    HINT_TEMPLATES order; the first template whose hint matches exactly
    one room in the world *is* a unique single-template hint for R.

    Among rooms with a single-template unique hint, picks the room whose
    best-rank template is highest. Ties broken by lowest room id.

    If no single-template hint is unique for any candidate room, falls
    back to:
      - rank 5: compound hint ("the room with {color} walls and a {object}")
      - rank 6: room name ("the room called {name}")

    Returns ``(room_id, hint_text, rank)``.
    """
    spawn = graph.spawn_room_id
    candidates = [
        r.id for r in graph.rooms
        if r.id != spawn and r.id != locked_room_id
    ]

    # Pass 1: single-template unique hints.
    best: Tuple[int, str, int] | None = None  # (rank_idx, hint, room_id)
    for room_id in candidates:
        room = graph.rooms[room_id]
        for rank_idx, template in enumerate(HINT_TEMPLATES, start=1):
            hint = compose_hint(room, template)
            if hint is None:
                continue
            if count_matching_rooms(graph, template, hint) == 1:
                cmp_key = (rank_idx, room_id)
                if best is None or cmp_key < (best[0], best[2]):
                    best = (rank_idx, hint, room_id)
                break  # this room's best template found
    if best is not None:
        rank_idx, hint, room_id = best
        return room_id, hint, rank_idx

    # Pass 2: rank-5 compound hint (color + object).
    for room_id in candidates:
        room = graph.rooms[room_id]
        if not room.wall_color or not room.objects:
            continue
        color_part = f"{room.wall_color} walls"
        obj_part = f"a {room.objects[0].category}"
        compound_hint = f"the room with {color_part} and {obj_part}"
        # Match condition for compound: both clauses must apply to the
        # same room. Compose each clause per-room and AND them.
        wall_color = room.wall_color
        obj_category = room.objects[0].category

        def _matches(r, _wc=wall_color, _oc=obj_category):
            return (
                r.wall_color == _wc
                and r.objects
                and r.objects[0].category == _oc
            )
        if sum(1 for r in graph.rooms if _matches(r)) == 1:
            return room_id, compound_hint, 5

    # Pass 3: hard fallback — room name.
    for room_id in candidates:
        room = graph.rooms[room_id]
        return room_id, f"the room called {room.name}", 6

    raise RuntimeError(
        f"No candidate room for key (spawn={spawn}, locked={locked_room_id}, "
        f"total rooms={len(graph.rooms)})."
    )


_WALL_MARGIN = 0.5     # m — keep key away from walls so agent can reach
_OBJECT_MARGIN = 1.0   # m — keep key away from object slots


def pick_key_position_in_room(room, doors_rng) -> Tuple[float, float]:
    """Sample a key position inside ``room`` honoring wall + object margins.

    Tries up to 50 rejection samples. If none satisfy the margins (very
    small room with many objects), returns the room center as a
    last-resort fallback.
    """
    ox, oz = room.origin
    w, d = room.size
    # Object slots match layout._object_entity placement: 30% / 70%.
    slots = []
    for i, _obj in enumerate(room.objects[:2]):
        if i == 0:
            slots.append((ox + w * 0.3, oz + d * 0.3))
        else:
            slots.append((ox + w * 0.7, oz + d * 0.7))

    for _ in range(50):
        px = doors_rng.uniform(ox + _WALL_MARGIN, ox + w - _WALL_MARGIN)
        pz = doors_rng.uniform(oz + _WALL_MARGIN, oz + d - _WALL_MARGIN)
        if all(math.hypot(px - sx, pz - sz) >= _OBJECT_MARGIN for sx, sz in slots):
            return (px, pz)

    # Fallback: room center. This only fires for very crowded rooms
    # (1 m × 1 m would be too small, but 6 m × 6 m always has room).
    return (ox + w / 2.0, oz + d / 2.0)


def render_lock_png(out_dir: Path, size: int = 256) -> Path:
    """Write a lock-icon PNG to ``out_dir/lock.png`` and return its path.

    Idempotent — if the file already exists, reuse it (Miniworld's
    texture loader caches by name).

    The icon is a dark grey shackle + body on a soft yellow background.
    Background color is chosen to contrast with every entry in
    WALL_PALETTE so the door reads as "door" against any room wall.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lock.png"
    if path.exists():
        return path

    img = _PILImage.new("RGB", (size, size), (235, 215, 120))  # warm yellow
    d = _PILDraw.Draw(img)
    cx, cy = size // 2, size // 2
    body_w = int(size * 0.45)
    body_h = int(size * 0.35)
    # Lock body (rounded rect approximated with rectangle + corners).
    d.rectangle(
        [cx - body_w // 2, cy, cx + body_w // 2, cy + body_h],
        fill=(35, 35, 35),
    )
    # Shackle (arc).
    shackle_top = cy - int(size * 0.18)
    shackle_w = int(size * 0.35)
    d.arc(
        [cx - shackle_w // 2, shackle_top,
         cx + shackle_w // 2, cy + int(size * 0.05)],
        start=180, end=360, fill=(35, 35, 35), width=int(size * 0.06),
    )
    # Keyhole.
    kh_r = int(size * 0.04)
    d.ellipse(
        [cx - kh_r, cy + body_h // 4 - kh_r,
         cx + kh_r, cy + body_h // 4 + kh_r],
        fill=(235, 215, 120),
    )
    img.save(path)
    return path
