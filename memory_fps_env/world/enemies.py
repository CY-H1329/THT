"""Enemy entity, appearance, spawn rules, and AI state machine."""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from memory_fps_env.config import DifficultyConfig
from memory_fps_env.rng import EnvRNG
from memory_fps_env.world.layout import RoomGraph

# Enemy box colors are rolled directly from MiniWorld's six-color render
# palette (red/green/blue/yellow/purple/grey) so the logged appearance
# matches exactly what the vision-only agent sees on screen — no lossy
# remapping happens at render time.
_SHIRT_COLORS = ["red", "blue", "green", "yellow", "purple"]
_PANTS_COLORS = ["red", "blue", "green", "yellow", "purple", "grey"]
_SKIN_COLORS = ["yellow", "green", "purple", "grey"]
_BODY_SHAPES = ["short", "tall"]


@dataclass
class Enemy:
    id: int
    room_id: int
    hp: int
    damage: int
    cooldown: float
    appearance: Dict[str, str]
    pos: Tuple[float, float]
    state: str = "idle"     # set by tick_ai
    last_attack_t: float = -1e9
    last_seen_pos: Optional[Tuple[float, float]] = None
    last_seen_t: float = -1e9

    def is_alive(self) -> bool:
        return self.hp > 0

    def to_log_dict(self) -> dict:
        return {
            "id": self.id,
            "room": self.room_id,
            "hp": self.hp,
            "damage": self.damage,
            "attack_cooldown": self.cooldown,
            "appearance": dict(self.appearance),
        }


def _roll_appearance(rng: EnvRNG) -> Dict[str, str]:
    return {
        "shirt_color": rng.choice(_SHIRT_COLORS),
        "pants_color": rng.choice(_PANTS_COLORS),
        "skin_color": rng.choice(_SKIN_COLORS),
        "body_shape": rng.choice(_BODY_SHAPES),
    }


def spawn_for_graph(
    rng: EnvRNG,
    cfg: DifficultyConfig,
    graph: RoomGraph,
) -> List[Enemy]:
    """Decide which rooms get enemies and roll their stats."""
    enemies: List[Enemy] = []
    eid = 0
    for room in graph.rooms:
        if room.id == graph.spawn_room_id:
            continue
        r = rng.random()
        if r < cfg.enemy_room_prob_one:
            n = 1
        elif r < cfg.enemy_room_prob_one + cfg.enemy_room_prob_two:
            n = 2
        else:
            n = 0
        ox, oz = room.origin
        w, d = room.size
        for _ in range(n):
            # Keep enemies away from walls and image quads (1m margin).
            px = ox + rng.uniform(1.5, max(1.6, w - 1.5))
            pz = oz + rng.uniform(1.5, max(1.6, d - 1.5))
            enemies.append(
                Enemy(
                    id=eid,
                    room_id=room.id,
                    hp=rng.randint(cfg.enemy_hp_min, cfg.enemy_hp_max),
                    damage=rng.randint(cfg.enemy_damage_min, cfg.enemy_damage_max),
                    cooldown=rng.uniform(cfg.enemy_cooldown_min, cfg.enemy_cooldown_max),
                    appearance=_roll_appearance(rng),
                    pos=(px, pz),
                )
            )
            eid += 1
    return enemies


# ---------------------------------------------------------------------------
# AI state machine
# ---------------------------------------------------------------------------

_ENEMY_SPEED = 1.5      # world-units per second
_MEMORY_DURATION = 5.0  # seconds the enemy keeps walking to last-seen pos
_WANDER_PERIOD = 3.0    # seconds before re-rolling wander direction
_WANDER_DISTANCE = 5.0  # forward distance of the wander target waypoint
_DEFAULT_RADIUS = 0.3   # fallback enemy collision radius when no _mw_entity


def _wander_target(enemy_id: int, pos: Tuple[float, float], t_now: float) -> Tuple[float, float]:
    """Deterministic random direction that re-rolls every _WANDER_PERIOD.

    Seeded by (enemy_id, time_bucket) so the wander is reproducible per
    seed/time and independent of any global RNG state.
    """
    bucket = int(t_now / _WANDER_PERIOD)
    rng = random.Random(f"wander:{enemy_id}:{bucket}")
    angle = rng.uniform(0, 2 * math.pi)
    return (
        pos[0] + _WANDER_DISTANCE * math.cos(angle),
        pos[1] + _WANDER_DISTANCE * math.sin(angle),
    )


def _wall_blocks(world, x: float, z: float, radius: float) -> bool:
    """True if a circle of `radius` at (x, z) intersects any world wall.

    Uses Miniworld's intersect_circle_segs against the static wall set
    built during _gen_static_data. Skips entity collisions intentionally
    so two enemies (or an enemy and an object) can co-occupy a cell.
    """
    if world is None:
        return False
    segs = getattr(world, "wall_segs", None)
    if segs is None or len(segs) == 0:
        return False
    try:
        import numpy as np
        from miniworld.math import intersect_circle_segs
    except ImportError:
        return False
    return bool(intersect_circle_segs(
        np.array([x, 0.0, z]), radius, segs,
    ))


def tick_ai(
    enemy: Enemy,
    agent_pos: Tuple[float, float],
    visibility_range: float,
    attack_range: float,
    t_now: float,
    dt: float,
    world=None,
) -> None:
    """Advance enemy state by one sub-tick.

    State machine:
      - dead:               no-op.
      - attack:             agent within ``attack_range``. No movement.
      - chase:              agent within ``visibility_range``. Walk toward
                            agent. Refreshes last_seen_{pos,t}.
      - pursue_last_known:  agent left visibility within the last
                            ``_MEMORY_DURATION`` seconds. Walk toward the
                            last-known agent position.
      - wander:             no recent sighting. Walk in a random direction
                            that re-rolls every ``_WANDER_PERIOD`` seconds.

    Walls block motion (when ``world`` is provided). When the diagonal
    step into ``(nx, nz)`` is blocked, the enemy tries axis-aligned
    sliding — first x-only, then z-only — so it can hug walls until it
    finds a doorway. If both axes are blocked, the enemy stays put for
    this sub-tick.

    Note: room-membership is intentionally NOT a visibility gate. With
    wall collision in place, the enemy already cannot physically clip
    through walls; allowing it to "see" the agent across a wall just
    lets it walk toward the doorway and pursue once the path opens.
    """
    if not enemy.is_alive():
        return

    dx = agent_pos[0] - enemy.pos[0]
    dz = agent_pos[1] - enemy.pos[1]
    dist = math.hypot(dx, dz)

    if dist <= visibility_range:
        enemy.last_seen_pos = agent_pos
        enemy.last_seen_t = t_now
        if dist <= attack_range:
            enemy.state = "attack"
            return
        enemy.state = "chase"
        target = agent_pos
    elif (enemy.last_seen_pos is not None
          and t_now - enemy.last_seen_t < _MEMORY_DURATION):
        enemy.state = "pursue_last_known"
        target = enemy.last_seen_pos
    else:
        enemy.state = "wander"
        target = _wander_target(enemy.id, enemy.pos, t_now)

    # Step toward target.
    tdx = target[0] - enemy.pos[0]
    tdz = target[1] - enemy.pos[1]
    tdist = math.hypot(tdx, tdz)
    if tdist < 1e-3:
        return
    step = min(_ENEMY_SPEED * dt, tdist)
    nx = enemy.pos[0] + (tdx / tdist) * step
    nz = enemy.pos[1] + (tdz / tdist) * step

    if world is None:
        enemy.pos = (nx, nz)
        return

    radius = getattr(getattr(enemy, "_mw_entity", None), "radius", _DEFAULT_RADIUS)
    if not _wall_blocks(world, nx, nz, radius):
        enemy.pos = (nx, nz)
    elif not _wall_blocks(world, nx, enemy.pos[1], radius):
        enemy.pos = (nx, enemy.pos[1])
    elif not _wall_blocks(world, enemy.pos[0], nz, radius):
        enemy.pos = (enemy.pos[0], nz)
    # else: pinned against geometry, no motion this sub-tick.
