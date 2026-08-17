"""Procedural world generation.

Two-step pipeline:
    1. generate_graph(...)  -> RoomGraph  (abstract topology + assignments)
    2. build_world(graph, env) -> None     (instantiates Miniworld geometry)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Set, FrozenSet

from memory_fps_env.config import DifficultyConfig
from memory_fps_env.rng import EnvRNG
from memory_fps_env.world.images import ImageEntry, ImageSource
from memory_fps_env.world.names import NameSource
from memory_fps_env.world.objects import ObjectEntry, ObjectSource


@dataclass
class Room:
    id: int
    name: str
    size: Tuple[int, int]  # (width, depth) in world units
    images: List[ImageEntry] = field(default_factory=list)
    images_per_wall: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    objects: List["ObjectEntry"] = field(default_factory=list)
    # Geometry fields filled by build_world:
    origin: Tuple[float, float] = (0.0, 0.0)
    wall_color: str = ""


@dataclass
class RoomGraph:
    rooms: List[Room]
    edges: Set[FrozenSet[int]]
    spawn_room_id: int

    def neighbors(self, room_id: int) -> List[int]:
        out = []
        for e in self.edges:
            if room_id in e:
                other = next(iter(e - {room_id}))
                out.append(other)
        return sorted(out)

    def is_connected(self) -> bool:
        if not self.rooms:
            return True
        seen = {0}
        stack = [0]
        while stack:
            cur = stack.pop()
            for n in self.neighbors(cur):
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        return len(seen) == len(self.rooms)


def generate_graph(
    rng: EnvRNG,
    cfg: DifficultyConfig,
    name_source: NameSource,
    image_source: ImageSource,
    object_source: ObjectSource,
) -> RoomGraph:
    layout_rng = rng.substream("layout")
    names_rng = rng.substream("names")
    images_rng = rng.substream("images")
    objects_rng = rng.substream("objects")

    # 1. Decide room count.
    n_rooms = layout_rng.randint(cfg.room_count_min, cfg.room_count_max)

    # 2. Build spanning tree by attaching each new node to a random existing
    #    one. We cap each room's tree-degree at 4 so the tree is always
    #    embeddable on a square grid (one cardinal slot per neighbor). Without
    #    this cap, ~68% of seeds produced a tree that forced the geometry
    #    builder into a non-adjacent spiral fallback, which then silently
    #    dropped the spanning-tree edge in the corridor pass and left rooms
    #    physically unreachable.
    rooms: List[Room] = []
    edges: Set[FrozenSet[int]] = set()
    tree_degree = [0] * n_rooms  # tree-only degree, not counting extra edges
    for i in range(n_rooms):
        width = layout_rng.randint(cfg.room_size_min, cfg.room_size_max)
        depth = layout_rng.randint(cfg.room_size_min, cfg.room_size_max)
        rooms.append(Room(id=i, name="", size=(width, depth)))
        if i > 0:
            # Pick a parent that still has < 4 tree-edges. With i ≥ 1 prior
            # rooms and total tree-degree so far = 2*(i-1), the average is
            # < 4, so at least one prior room is guaranteed to have degree
            # < 4 and the candidate list is never empty.
            candidates = [j for j in range(i) if tree_degree[j] < 4]
            parent = candidates[layout_rng.randint(0, len(candidates) - 1)]
            edges.add(frozenset({parent, i}))
            tree_degree[parent] += 1
            tree_degree[i] += 1

    # 3. Add extra edges for loops (avoid duplicates and self-loops).
    #    Reject edges that would push either endpoint's total degree above 4
    #    so the graph as a whole stays grid-embeddable. With degree ≤ 4 the
    #    BFS placement in ``_place_rooms_on_grid`` always finds a free
    #    cardinal slot for each spanning-tree edge.
    def _total_degree(node: int) -> int:
        return sum(1 for ee in edges if node in ee)

    added = 0
    attempts = 0
    while added < cfg.extra_edges and attempts < 100:
        attempts += 1
        a = layout_rng.randint(0, n_rooms - 1)
        b = layout_rng.randint(0, n_rooms - 1)
        if a == b:
            continue
        e = frozenset({a, b})
        if e in edges:
            continue
        if _total_degree(a) >= 4 or _total_degree(b) >= 4:
            continue
        edges.add(e)
        added += 1

    # 4. Assign unique names.
    names = name_source.sample(names_rng, k=n_rooms)
    for room, nm in zip(rooms, names):
        room.name = nm

    # 5. Assign per-wall image counts and sample the total pool.
    # Each wall draws its image count from a weighted distribution over
    # [0, len(weights)-1] (default [0, 1, 2] with weights [0.2, 0.6, 0.2]).
    counts = list(range(len(cfg.images_per_wall_weights)))
    per_wall_counts = [
        layout_rng.choices(counts, weights=cfg.images_per_wall_weights, k=4)
        for _ in rooms
    ]
    total_images = sum(sum(walls) for walls in per_wall_counts)
    pool = image_source.sample(images_rng, k=total_images)
    cursor = 0
    for room, walls in zip(rooms, per_wall_counts):
        n = sum(walls)
        room.images = pool[cursor:cursor + n]
        room.images_per_wall = list(walls)
        cursor += n

    # 6. Assign 1-2 objects per room (without replacement across rooms).
    obj_counts = [objects_rng.randint(1, 2) for _ in rooms]
    obj_pool = object_source.sample(objects_rng, k=sum(obj_counts))
    cursor = 0
    for room, count in zip(rooms, obj_counts):
        room.objects = obj_pool[cursor:cursor + count]
        cursor += count

    # 7. Pick spawn room.
    spawn_room_id = layout_rng.randint(0, n_rooms - 1)

    return RoomGraph(rooms=rooms, edges=edges, spawn_room_id=spawn_room_id)


# ---------------------------------------------------------------------------
# Miniworld geometry build
# ---------------------------------------------------------------------------

from miniworld.entity import ImageFrame  # textured quad
from memory_fps_env.world.walls import WALL_PALETTE, render_solid_png


class _DynamicImageFrame(ImageFrame):
    """ImageFrame whose is_static is False.

    Miniworld bakes is_static=True entities into a one-shot OpenGL display
    list at reset() time, so they can't be moved or removed mid-episode.
    Wall images legitimately want that (they never move or disappear), but
    enemy face quads must (a) translate with the chasing mob and (b) vanish
    when the enemy is killed. Setting is_static=False routes the face quad
    through the per-frame dynamic-render loop instead.
    """

    @property
    def is_static(self):
        return False


_PORTAL_HALFWIDTH = 1.0  # half-width of door opening in world units
_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _embed_on_grid(graph: RoomGraph):
    """Find an integer-grid embedding of ``graph`` rooted at the spawn room.

    Each room is assigned a unique cell ``(gx, gz)`` such that every
    spanning-tree edge connects cells that differ by exactly one unit in
    one cardinal direction. (Loop edges may end up between non-adjacent
    cells; those simply don't materialize as portals in ``build_world``.)

    Returns ``{room_id: (gx, gz)}`` on success or ``None`` if no embedding
    exists within the recursion budget.

    Uses DFS with backtracking over (child, cardinal-direction) choices.
    This is exponential in the worst case but degree ≤ 4 + ~12 rooms is
    tractable. We don't bother bounding nodes-explored because empirically
    every graph in our seed range solves in < 1 ms.
    """
    spawn = graph.spawn_room_id

    # Resolve a deterministic spanning tree by BFS from spawn (consistent
    # with how the graph was built). Loop edges that aren't in this tree
    # impose no placement constraint — they're allowed to span non-adjacent
    # cells.
    parent = {spawn: None}
    bfs_order = [spawn]
    queue = [spawn]
    while queue:
        cur = queue.pop(0)
        for nb in graph.neighbors(cur):
            if nb not in parent:
                parent[nb] = cur
                bfs_order.append(nb)
                queue.append(nb)
    # Sanity: every room must be in `parent` because the graph is connected.
    if len(parent) != len(graph.rooms):
        return None

    grid_pos = {spawn: (0, 0)}
    used = {(0, 0): spawn}
    order = bfs_order[1:]  # children of spawn first, then their children, etc.

    def recurse(idx: int) -> bool:
        if idx == len(order):
            return True
        room_id = order[idx]
        p = parent[room_id]
        pgx, pgz = grid_pos[p]
        for dx, dz in _DIRECTIONS:
            slot = (pgx + dx, pgz + dz)
            if slot in used:
                continue
            grid_pos[room_id] = slot
            used[slot] = room_id
            if recurse(idx + 1):
                return True
            del grid_pos[room_id]
            del used[slot]
        return False

    if recurse(0):
        return grid_pos
    return None


def _place_rooms_on_grid(graph: RoomGraph) -> None:
    """Lay rooms out on a uniform integer grid via DFS-with-backtracking.

    Each room is placed in a unique grid cell of size ``cell × cell`` where
    ``cell`` is the maximum of any room's sampled dimensions. Within each
    cell, the room's footprint is *inflated* to fill the cell, so adjacent
    rooms share a wall flush (and Miniworld's ``connect_rooms`` joins them
    with a portal rather than a corridor Room — which is what keeps
    ``len(env.rooms) == len(graph.rooms)``).

    Why uniform cells: when rooms keep their variable sampled sizes, a child
    placed flush against its parent can still be blocked on all four
    cardinal sides by *cousins* whose larger footprints extend past the
    parent. Uniform cells eliminate this geometric source of failure.

    Why DFS with backtracking: even with the degree cap, a greedy BFS that
    picks the first free cardinal cell for each child can deadlock when a
    sibling/cousin happens to occupy a cell a later child needs. Trees with
    max-degree ≤ 4 always have a grid embedding (this is a classical
    result), but finding one requires global choice — hence backtracking
    over the spanning tree determined by the same BFS the graph generator
    uses. Loop edges are not constrained to grid-adjacency.

    Mutates ``room.origin`` and ``room.size`` in-place.
    """
    grid_pos = _embed_on_grid(graph)
    if grid_pos is None:
        raise RuntimeError(
            f"Could not embed graph on integer grid; spanning tree of "
            f"{len(graph.rooms)} rooms is not grid-embeddable. This should "
            f"not happen with the degree cap enforced by generate_graph."
        )

    cell = max(max(r.size) for r in graph.rooms)
    for rid, (gx, gz) in grid_pos.items():
        room = graph.rooms[rid]
        room.origin = (gx * cell, gz * cell)
        room.size = (cell, cell)


def _door_center_between(room_a, room_b):
    """Return (door_center_xz, normal_axis) for the touching wall between
    two adjacent grid-aligned rooms.

    normal_axis is "x" if the rooms touch on a vertical wall (varying z
    along the wall), or "z" if they touch on a horizontal wall.
    """
    ax0, az0 = room_a.origin
    ax1, az1 = ax0 + room_a.size[0], az0 + room_a.size[1]
    bx0, bz0 = room_b.origin
    bx1, bz1 = bx0 + room_b.size[0], bz0 + room_b.size[1]
    eps = 0.01

    if abs(ax1 - bx0) < eps:
        z_mid = (max(az0, bz0) + min(az1, bz1)) / 2.0
        return (ax1, z_mid), "x"
    if abs(ax0 - bx1) < eps:
        z_mid = (max(az0, bz0) + min(az1, bz1)) / 2.0
        return (ax0, z_mid), "x"
    if abs(az1 - bz0) < eps:
        x_mid = (max(ax0, bx0) + min(ax1, bx1)) / 2.0
        return (x_mid, az1), "z"
    if abs(az0 - bz1) < eps:
        x_mid = (max(ax0, bx0) + min(ax1, bx1)) / 2.0
        return (x_mid, az0), "z"
    raise RuntimeError(
        f"Rooms {room_a.id} and {room_b.id} do not share a touching wall."
    )


def build_world(env, graph: RoomGraph, enemies=None,
                face_source=None, face_rng=None,
                doors_rng=None,
                door_collision_thickness: float = 0.1) -> None:
    """Instantiate Miniworld geometry from a RoomGraph.

    Must be called from inside a MiniWorldEnv._gen_world() override.
    ``face_source`` / ``face_rng`` supply face textures for the humanoid
    enemy heads; pass ``None`` to render enemies with blank heads (used by
    geometry-only test fixtures that don't pass enemies either).
    """
    _place_rooms_on_grid(graph)

    # 1. Rooms. Each gets its own solid wall color from a shared palette;
    # floor and ceiling stay constant across all rooms.
    import random as _rnd
    import tempfile
    from pathlib import Path
    from miniworld.opengl import Texture

    # Shared per-episode tempdir (used for both wall-color PNGs here and
    # wall-image PNGs further below). MemoryFPSEnv.close() cleans it up.
    tmp_dir = Path(tempfile.mkdtemp(prefix="mfpsenv_imgs_"))
    env._tmp_image_dir = str(tmp_dir)

    # Deterministic RNG for wall color choice. Seeded by spawn_room_id and
    # room count, both seed-derived by generate_graph, so the color choice
    # is reproducible without threading EnvRNG into build_world.
    color_rng = _rnd.Random(f"walls:{graph.spawn_room_id}:{len(graph.rooms)}")

    mw_rooms = {}
    for room in graph.rooms:
        w, d = room.size
        ox, oz = room.origin
        name, rgb = color_rng.choice(WALL_PALETTE)
        png_path = render_solid_png(rgb, out_dir=tmp_dir)
        tex_name = f"_mfe_wall_{room.id}_{name}"
        Texture.tex_paths[tex_name] = [str(png_path)]
        mw_room = env.add_rect_room(
            min_x=ox, max_x=ox + w,
            min_z=oz, max_z=oz + d,
            wall_tex=tex_name,
            floor_tex="floor_tiles_bw",
            ceil_tex="concrete",
        )
        mw_rooms[room.id] = mw_room
        room.wall_color = name

    # 2. Doorways between rooms whose walls touch.
    #
    # Miniworld's connect_rooms requires the two portals to live on
    # facing, touching walls. If portal start/end positions coincide on
    # both rooms (which happens when the walls share the same coordinate
    # and the portal extents overlap), connect_rooms returns without
    # adding a corridor Room — keeping len(env.rooms) == len(graph.rooms).
    #
    # We also record, per (room, wall_idx), the t-intervals occupied by
    # doorways. Wall image placement (section 3) uses this to skip any
    # image position that would overlap a doorway — otherwise the image's
    # collision center (radius 0 but contributing to the agent's 0.4 m
    # check) would block the agent from passing through.
    eps = 0.01
    materialized_edges: Set[FrozenSet[int]] = set()
    # room_id → {wall_idx (0..3): [(t_min, t_max), ...]} in wall-local
    # coordinates (t is along the wall axis, relative to room.origin).
    doorways_per_room: dict = {r.id: {0: [], 1: [], 2: [], 3: []} for r in graph.rooms}
    for edge in graph.edges:
        a, b = sorted(edge)
        ra, rb = graph.rooms[a], graph.rooms[b]
        ax0, az0 = ra.origin
        ax1, az1 = ax0 + ra.size[0], az0 + ra.size[1]
        bx0, bz0 = rb.origin
        bx1, bz1 = bx0 + rb.size[0], bz0 + rb.size[1]

        # East-west touch: ra east wall meets rb west wall.
        if abs(ax1 - bx0) < eps and min(az1, bz1) - max(az0, bz0) > 2 * _PORTAL_HALFWIDTH:
            z_mid = (max(az0, bz0) + min(az1, bz1)) / 2.0
            env.connect_rooms(
                mw_rooms[ra.id], mw_rooms[rb.id],
                min_z=z_mid - _PORTAL_HALFWIDTH,
                max_z=z_mid + _PORTAL_HALFWIDTH,
            )
            # ra wall 1 (east, z-axis): t = z - az0
            doorways_per_room[ra.id][1].append(
                (z_mid - _PORTAL_HALFWIDTH - az0, z_mid + _PORTAL_HALFWIDTH - az0))
            # rb wall 3 (west, z-axis): t = z - bz0
            doorways_per_room[rb.id][3].append(
                (z_mid - _PORTAL_HALFWIDTH - bz0, z_mid + _PORTAL_HALFWIDTH - bz0))
            materialized_edges.add(edge)
        # West-east touch: ra west wall meets rb east wall.
        elif abs(ax0 - bx1) < eps and min(az1, bz1) - max(az0, bz0) > 2 * _PORTAL_HALFWIDTH:
            z_mid = (max(az0, bz0) + min(az1, bz1)) / 2.0
            env.connect_rooms(
                mw_rooms[rb.id], mw_rooms[ra.id],
                min_z=z_mid - _PORTAL_HALFWIDTH,
                max_z=z_mid + _PORTAL_HALFWIDTH,
            )
            doorways_per_room[ra.id][3].append(
                (z_mid - _PORTAL_HALFWIDTH - az0, z_mid + _PORTAL_HALFWIDTH - az0))
            doorways_per_room[rb.id][1].append(
                (z_mid - _PORTAL_HALFWIDTH - bz0, z_mid + _PORTAL_HALFWIDTH - bz0))
            materialized_edges.add(edge)
        # North-south touch: ra north wall meets rb south wall.
        elif abs(az1 - bz0) < eps and min(ax1, bx1) - max(ax0, bx0) > 2 * _PORTAL_HALFWIDTH:
            x_mid = (max(ax0, bx0) + min(ax1, bx1)) / 2.0
            env.connect_rooms(
                mw_rooms[ra.id], mw_rooms[rb.id],
                min_x=x_mid - _PORTAL_HALFWIDTH,
                max_x=x_mid + _PORTAL_HALFWIDTH,
            )
            # ra wall 2 (high-z, x-axis): t = x - ax0
            doorways_per_room[ra.id][2].append(
                (x_mid - _PORTAL_HALFWIDTH - ax0, x_mid + _PORTAL_HALFWIDTH - ax0))
            # rb wall 0 (low-z, x-axis): t = x - bx0
            doorways_per_room[rb.id][0].append(
                (x_mid - _PORTAL_HALFWIDTH - bx0, x_mid + _PORTAL_HALFWIDTH - bx0))
            materialized_edges.add(edge)
        # South-north touch: ra south wall meets rb north wall.
        elif abs(az0 - bz1) < eps and min(ax1, bx1) - max(ax0, bx0) > 2 * _PORTAL_HALFWIDTH:
            x_mid = (max(ax0, bx0) + min(ax1, bx1)) / 2.0
            env.connect_rooms(
                mw_rooms[rb.id], mw_rooms[ra.id],
                min_x=x_mid - _PORTAL_HALFWIDTH,
                max_x=x_mid + _PORTAL_HALFWIDTH,
            )
            doorways_per_room[ra.id][0].append(
                (x_mid - _PORTAL_HALFWIDTH - ax0, x_mid + _PORTAL_HALFWIDTH - ax0))
            doorways_per_room[rb.id][2].append(
                (x_mid - _PORTAL_HALFWIDTH - bx0, x_mid + _PORTAL_HALFWIDTH - bx0))
            materialized_edges.add(edge)
        # Otherwise the two rooms aren't grid-adjacent (extra loop edge
        # whose endpoints happened not to be placed adjacent). The
        # abstract edge stays in the graph for QA reasoning; no physical
        # corridor is rendered. Loop edges are not load-bearing — the
        # spanning-tree subset is what reachability depends on, and the
        # post-pass below asserts every room is reachable.
    # Expose materialized edges + doorway-interval map for tests/diagnostics.
    env._materialized_edges = materialized_edges
    env._doorways_per_room = doorways_per_room

    # 2b. Reachability invariant: every room must be reachable from spawn
    # via materialized corridors. The degree cap in generate_graph + flush
    # BFS placement guarantee this for the spanning tree; we verify rather
    # than trust.
    reachable = {graph.spawn_room_id}
    stack = [graph.spawn_room_id]
    while stack:
        cur = stack.pop()
        for e in materialized_edges:
            if cur in e:
                other = next(iter(e - {cur}))
                if other not in reachable:
                    reachable.add(other)
                    stack.append(other)
    if len(reachable) != len(graph.rooms):
        missing = sorted(set(range(len(graph.rooms))) - reachable)
        raise RuntimeError(
            f"Reachability invariant violated: rooms {missing} are not "
            f"reachable from spawn {graph.spawn_room_id} via materialized "
            f"corridors. Graph edges: {len(graph.edges)}, materialized: "
            f"{len(materialized_edges)}."
        )

    # 3. Wall image quads (one ImageFrame entity per image).
    from PIL import Image as PILImage
    for room in graph.rooms:
        ox, oz = room.origin
        w, d = room.size
        # Per-wall layout: each wall picks a list of along-wall t-positions
        # (in [t_margin, length - t_margin]) and an inward-facing direction.
        # ImageFrame's textured face is at local +X; after glRotatef(dir, +Y)
        # the face direction equals (cos(dir), 0, -sin(dir)). So to make
        # the face point INTO the room (perpendicular to the wall), use:
        #   wall 0 (low-z):  face +Z  → dir = 270°
        #   wall 1 (high-x): face -X  → dir = 180°
        #   wall 2 (high-z): face -Z  → dir =  90°
        #   wall 3 (low-x):  face +X  → dir =   0°
        wall_specs = [
            # (wall_index, length, fixed_yaw_deg, t_to_xz)
            # Offset 0.25 from the room edge keeps each room's frames strictly
            # inside its own bounding box (tolerance 0.2) and out of adjacent
            # rooms' boxes — so the geometry test can classify frames per room.
            (0, w, 270, lambda t: (ox + t, oz + 0.25)),
            (1, d, 180, lambda t: (ox + w - 0.25, oz + t)),
            (2, w,  90, lambda t: (ox + t, oz + d - 0.25)),
            (3, d,   0, lambda t: (ox + 0.25, oz + t)),
        ]
        cursor = 0
        image_width = 1.5
        # Corner margin keeps each image's outer edge clear of the adjacent
        # wall. Without it, an image whose edge sits at the corner gets
        # z-fought / occluded by the perpendicular wall's surface, so the
        # image renders partially or fully invisible from inside the room.
        corner_margin = 0.5
        for wall_idx, length, yaw_deg, to_xz in wall_specs:
            n = room.images_per_wall[wall_idx]
            if n <= 0:
                continue
            # Clamp n down if the wall (minus both corner margins) can't fit
            # n images at image_width spacing.
            usable_for_n = max(0.0, length - 2 * corner_margin)
            max_fit = max(1, int(usable_for_n // image_width))
            n = min(n, max_fit)
            # Even spacing of n centers along the wall, with corner margin
            # on each end.
            #   n == 1 → centered at length/2
            #   n  > 1 → first center at image_width/2 + corner_margin,
            #            last at length - image_width/2 - corner_margin,
            #            others evenly between.
            if n == 1:
                t_positions = [length / 2.0]
            else:
                first = image_width / 2.0 + corner_margin
                last = length - image_width / 2.0 - corner_margin
                step = (last - first) / (n - 1) if n > 1 else 0.0
                t_positions = [first + k * step for k in range(n)]
            # Filter out positions whose image footprint overlaps a doorway —
            # otherwise the image's collision center blocks the agent from
            # crossing through. The check uses image_width/2 on each side so
            # the image quad itself doesn't intrude into the door opening.
            doors = doorways_per_room[room.id][wall_idx]
            half = image_width / 2.0
            kept = []
            for t in t_positions:
                img_min, img_max = t - half, t + half
                if any(img_min < dmax and dmin < img_max for dmin, dmax in doors):
                    continue
                kept.append(t)
            t_positions = kept
            room.images_per_wall[wall_idx] = len(t_positions)
            for t in t_positions:
                img_entry = room.images[cursor]
                cursor += 1
                px, pz = to_xz(t)
                png_path = tmp_dir / f"{img_entry.id}.png"
                if not png_path.exists():
                    PILImage.fromarray(img_entry.rgb).save(str(png_path))
                tex_name = f"_mfe_img_{img_entry.id}"
                Texture.tex_paths[tex_name] = [str(png_path)]
                env.place_entity(
                    ImageFrame(
                        pos=[px, 1.5, pz],
                        dir=yaw_deg * 3.14159 / 180,
                        width=image_width,
                        tex_name=tex_name,
                    ),
                    pos=[px, 1.5, pz],
                    dir=yaw_deg * 3.14159 / 180,
                )
        # After all 4 walls processed: trim room.images down to the actually-placed count.
        room.images = room.images[:cursor]

    # 3b. 3D objects per room — each ObjectEntry maps to a Miniworld mesh.
    # Box still used by the enemy humanoid further down; COLOR_NAMES used by
    # _color_name below to validate enemy shirt/pants/skin colors.
    from miniworld.entity import Box, COLOR_NAMES, Key, MeshEnt

    # Enemy appearance colors are rolled directly from MiniWorld's six-color
    # render palette (see enemies.py _SHIRT/_PANTS/_SKIN_COLORS), so the
    # logged appearance equals the on-screen pixels with no remapping. This
    # guard only catches an unexpected or missing value.
    def _color_name(c):
        return c if c in COLOR_NAMES else "grey"

    def _object_entity(entry):
        # Keys use Miniworld's Key class which loads key_<color>.obj.
        # Every other category is a raw .obj in miniworld/meshes/
        # loaded directly by MeshEnt.
        if entry.category == "key":
            return Key(color=entry.color)
        return MeshEnt(
            mesh_name=entry.category, height=entry.height, static=True,
        )

    for room in graph.rooms:
        ox, oz = room.origin
        w, d = room.size
        # Deterministic stagger across two interior slots. Both are offset
        # away from the room center so the agent's spawn point (which is the
        # spawn-room center) does not overlap an object — that overlap makes
        # MiniWorld's collision check silently reject MOVE_FORWARD/MOVE_BACK.
        slots = [(ox + w * 0.3, oz + d * 0.3), (ox + w * 0.7, oz + d * 0.7)]
        for entry, (px, pz) in zip(room.objects, slots):
            ent = _object_entity(entry)
            env.place_entity(ent, pos=[px, 0, pz], dir=0)
            entry._mw_entity = ent

    # 4. Spawn agent in spawn room.
    spawn = graph.rooms[graph.spawn_room_id]
    sx = spawn.origin[0] + spawn.size[0] / 2
    sz = spawn.origin[1] + spawn.size[1] / 2
    env.place_agent(pos=[sx, 0, sz], dir=0)

    # 5. Place enemies as humanoid mobs (6 cubes + face quad).
    if enemies:
        from miniworld.opengl import Texture
        face_entries: List = []
        if face_source is not None:
            try:
                living_count = sum(1 for e in enemies if e.is_alive())
                if living_count > 0:
                    face_entries = face_source.sample(face_rng, k=living_count)
            except ValueError:
                face_entries = []  # missing assets → run with blank heads
        face_iter = iter(face_entries)

        for e in enemies:
            if not e.is_alive():
                continue
            room = graph.rooms[e.room_id]
            ox, oz = room.origin
            w, d = room.size
            jx = ((e.id * 0.37) % 1.0 - 0.5) * 1.0
            jz = ((e.id * 0.61) % 1.0 - 0.5) * 1.0
            px = ox + w * 0.5 + jx
            pz = oz + d * 0.5 + jz
            e.pos = (px, pz)

            shirt = _color_name(e.appearance.get("shirt_color"))
            pants = _color_name(e.appearance.get("pants_color"))
            skin = _color_name(e.appearance.get("skin_color"))
            scale = 0.75 if e.appearance.get("body_shape") == "short" else 1.0

            # Body cubes. Heights are stacked so feet=0, head ends at ~2*scale.
            torso = Box(color=shirt, size=[0.50 * scale, 0.75 * scale, 0.25 * scale])
            head  = Box(color=skin,  size=[0.50 * scale, 0.50 * scale, 0.50 * scale])
            l_arm = Box(color=shirt, size=[0.25 * scale, 0.75 * scale, 0.25 * scale])
            r_arm = Box(color=shirt, size=[0.25 * scale, 0.75 * scale, 0.25 * scale])
            l_leg = Box(color=pants, size=[0.25 * scale, 0.75 * scale, 0.25 * scale])
            r_leg = Box(color=pants, size=[0.25 * scale, 0.75 * scale, 0.25 * scale])

            # Place each cube with explicit pos / dir (Miniworld's place_entity
            # would otherwise randomize the location). Each part records its
            # offset from the enemy's logical XZ so step()'s AI-tick sync can
            # translate the whole mob when the enemy chases or moves.
            #
            # _mw_entity_parts holds (entity, x_offset, z_offset) tuples.
            # Used by:
            #   - env.step() to sync rendered position to enemy.pos each tick
            #   - env._resolve_agent_attack() to remove the corpse on death
            e._mw_entity_parts = []
            env.place_entity(l_leg, pos=[px - 0.125 * scale, 0.0,         pz], dir=0)
            e._mw_entity_parts.append((l_leg, -0.125 * scale, 0.0))
            env.place_entity(r_leg, pos=[px + 0.125 * scale, 0.0,         pz], dir=0)
            e._mw_entity_parts.append((r_leg, +0.125 * scale, 0.0))
            env.place_entity(torso, pos=[px,                 0.75 * scale, pz], dir=0)
            e._mw_entity_parts.append((torso, 0.0, 0.0))
            env.place_entity(l_arm, pos=[px - 0.375 * scale, 0.75 * scale, pz], dir=0)
            e._mw_entity_parts.append((l_arm, -0.375 * scale, 0.0))
            env.place_entity(r_arm, pos=[px + 0.375 * scale, 0.75 * scale, pz], dir=0)
            e._mw_entity_parts.append((r_arm, +0.375 * scale, 0.0))
            env.place_entity(head,  pos=[px,                 1.50 * scale, pz], dir=0)
            e._mw_entity_parts.append((head, 0.0, 0.0))
            e._mw_entity = torso  # hitbox / debug back-pointer

            # Face quad — only if we have a face entry.
            try:
                face = next(face_iter)
            except StopIteration:
                face = None
            e.appearance["face_id"] = face.id if face is not None else "none"
            if face is not None:
                tex_name = f"_mfe_face_{e.id}_{face.id}"
                Texture.tex_paths[tex_name] = [str(face.path)]
                head_center_y = 1.75 * scale  # head center y (head bottom at 1.5, height 0.5 → center 1.75)
                face_quad = _DynamicImageFrame(
                    pos=[px, head_center_y, pz + 0.13 * scale],
                    dir=3.14159,  # facing −z
                    width=0.45 * scale,
                    tex_name=tex_name,
                )
                env.place_entity(
                    face_quad,
                    pos=[px, head_center_y, pz + 0.13 * scale],
                    dir=3.14159,
                )
                e._mw_entity_parts.append((face_quad, 0.0, 0.13 * scale))

    # 6. Locked door at the entrance to the BFS-deepest leaf room.
    # Local import to avoid circular dependency: doors.py imports RoomGraph from layout.py.
    from memory_fps_env.world.doors import (
        find_locked_room,
        choose_key_room_and_hint,
        pick_key_position_in_room,
        render_lock_png,
    )

    if doors_rng is None:
        # Standalone test usage (e.g., _ProbeEnv in test_layout_geometry)
        # may not pass doors_rng. Synthesize a deterministic shim so
        # build_world remains usable without an EnvRNG.
        import random as _rnd_doors
        class _ShimRng:
            def __init__(self, seed):
                self._r = _rnd_doors.Random(seed)
            def uniform(self, a, b):
                return self._r.uniform(a, b)
        doors_rng = _ShimRng(f"doors:{graph.spawn_room_id}:{len(graph.rooms)}")

    locked_room_id = find_locked_room(graph)
    key_room_id, hint_text, hint_rank = choose_key_room_and_hint(
        graph, locked_room_id=locked_room_id,
    )
    key_pos = pick_key_position_in_room(graph.rooms[key_room_id], doors_rng)

    # The leaf has exactly one neighbor; that's its parent in the tree.
    locked_neighbors = graph.neighbors(locked_room_id)
    assert len(locked_neighbors) == 1, (
        f"locked_room {locked_room_id} should be a leaf (degree 1), "
        f"got {len(locked_neighbors)} neighbors"
    )
    parent_room_id = locked_neighbors[0]

    locked_room = graph.rooms[locked_room_id]
    parent_room = graph.rooms[parent_room_id]
    door_center, door_normal_axis = _door_center_between(locked_room, parent_room)

    # Lock-icon texture (rendered into the per-episode tempdir).
    from miniworld.opengl import Texture as _Tex
    lock_png = render_lock_png(out_dir=Path(env._tmp_image_dir))
    lock_tex_name = "_mfe_lock"
    _Tex.tex_paths[lock_tex_name] = [str(lock_png)]

    # Collision: thin static Box across the doorway.
    door_thickness = door_collision_thickness
    if door_normal_axis == "x":
        # Door spans the z-axis, blocks crossing in x.
        box_size = [door_thickness, 2.5, 2 * _PORTAL_HALFWIDTH]
    else:
        box_size = [2 * _PORTAL_HALFWIDTH, 2.5, door_thickness]
    door_box = Box(color="grey", size=box_size)
    env.place_entity(
        door_box,
        pos=[door_center[0], 0.0, door_center[1]],
        dir=0,
    )

    # Two lock-icon ImageFrames back-to-back, one facing each adjacent room.
    # Each plane is offset *outside* the collision Box by (thickness/2 + eps)
    # so the Box's opaque face doesn't occlude the plane from the agent's
    # view. Width fills the full portal opening (2 m) so there's no see-
    # through gap on the sides of the icon.
    # _is_door_entity=True marks these so geometry tests can distinguish them
    # from room wall-image frames (which are always placed 0.25 m from a wall,
    # never at the exact wall boundary where door centers land).
    plane_width = 2.0
    plane_offset = door_thickness / 2 + 0.02
    if door_normal_axis == "x":
        # face_a: textured side faces +x → visible from the +x-side room.
        face_a_pos = [door_center[0] + plane_offset, 1.5, door_center[1]]
        face_a = _DynamicImageFrame(
            pos=face_a_pos, dir=0.0, width=plane_width, tex_name=lock_tex_name,
        )
        face_a._is_door_entity = True
        env.place_entity(face_a, pos=face_a_pos, dir=0.0)
        # face_b: textured side faces -x → visible from the -x-side room.
        face_b_pos = [door_center[0] - plane_offset, 1.5, door_center[1]]
        face_b = _DynamicImageFrame(
            pos=face_b_pos, dir=3.14159, width=plane_width, tex_name=lock_tex_name,
        )
        face_b._is_door_entity = True
        env.place_entity(face_b, pos=face_b_pos, dir=3.14159)
    else:
        # face_a: textured side faces -z → visible from the -z-side room.
        face_a_pos = [door_center[0], 1.5, door_center[1] - plane_offset]
        face_a = _DynamicImageFrame(
            pos=face_a_pos, dir=3.14159 / 2, width=plane_width, tex_name=lock_tex_name,
        )
        face_a._is_door_entity = True
        env.place_entity(face_a, pos=face_a_pos, dir=3.14159 / 2)
        # face_b: textured side faces +z → visible from the +z-side room.
        face_b_pos = [door_center[0], 1.5, door_center[1] + plane_offset]
        face_b = _DynamicImageFrame(
            pos=face_b_pos, dir=3 * 3.14159 / 2, width=plane_width, tex_name=lock_tex_name,
        )
        face_b._is_door_entity = True
        env.place_entity(face_b, pos=face_b_pos, dir=3 * 3.14159 / 2)

    # Record on env for the runtime state machine to consume in later tasks.
    env._door_center = (float(door_center[0]), float(door_center[1]))
    env._door_entities = [door_box, face_a, face_b]
    env._locked_room_id = locked_room_id
    env._door_parent_room_id = parent_room_id
    env._key_room_id = key_room_id
    env._key_position = (float(key_pos[0]), float(key_pos[1]))
    env._hint_text = hint_text
    env._hint_rank = hint_rank
