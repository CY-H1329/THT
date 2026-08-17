"""MemoryFPSEnv — Gymnasium env wrapping miniworld with HUD + procgen layout.

Provides walking + HUD, procedural room layout, enemies with HP-death
termination, and the locked-door / key mechanic.
"""

import math
import time
from enum import IntEnum
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from miniworld.miniworld import MiniWorldEnv
from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont

from memory_fps_env.config import load_difficulty
from memory_fps_env.hud import draw_hud, heading_degrees
from memory_fps_env.rng import EnvRNG
from memory_fps_env.world.images import (
    ImageSource,
    ProceduralImageSource,
    PublicImageSource,
)
from memory_fps_env.world.enemies import spawn_for_graph, tick_ai
from memory_fps_env.world.faces import (
    FaceSource,
    PublicFaceSource,
    HeldoutFaceSource,
)
from memory_fps_env.world.layout import build_world, generate_graph, RoomGraph
from memory_fps_env.world.names import (
    NameSource,
    ProceduralNameSource,
    PublicNameSource,
)
from memory_fps_env.world.objects import (
    ObjectSource,
    PublicObjectSource,
)


class Action(IntEnum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    MOVE_FORWARD = 2
    MOVE_BACK = 3
    ATTACK = 4
    NO_OP = 5
    END_EPISODE = 6


_OBS_W, _OBS_H = 320, 240

# Combat constants
_ATTACK_RANGE = 3.0
_ATTACK_CONE_HALF_RAD = math.radians(20)


def _load_hud_font(size: int):
    """Load a platform-independent scalable font.

    Pillow >=10.1 ships a bundled scalable default font; using it makes the
    HUD render byte-identically on Windows, macOS, and Linux, which matters
    because the agent reads the HUD from pixels. Falls back to the legacy
    bitmap default on older Pillow.
    """
    try:
        return _PILFont.load_default(size=size)
    except TypeError:
        # Pillow <10.1: load_default() takes no size arg.
        return _PILFont.load_default()


_BANNER_FONT = _load_hud_font(28)


def _per_wall_images_payload(room):
    """Slice room.images by room.images_per_wall into 4 wall-indexed lists.

    Output: list of 4 lists, each with the image dicts that landed on that
    wall (south=0, east=1, north=2, west=3). Order within a wall is the
    order images were placed (even spacing along the wall).
    """
    out = []
    cursor = 0
    for n in room.images_per_wall:
        wall = []
        for img in room.images[cursor:cursor + n]:
            wall.append({
                "file": img.id,
                "category": img.category,
                "caption": img.caption,
                "salient_features": list(img.salient_features),
            })
        out.append(wall)
        cursor += n
    return out


def _build_qa_banner(width: int, height: int) -> np.ndarray:
    img = _PILImage.new("RGB", (width, height), (10, 10, 10))
    d = _PILDraw.Draw(img)
    text = "QA PHASE"
    sub = "answer the upcoming questions"
    d.text((width // 2 - 80, height // 2 - 40), text,
           fill=(255, 220, 80), font=_BANNER_FONT)
    d.text((width // 2 - 130, height // 2 + 5), sub, fill=(220, 220, 220))
    return np.array(img, dtype=np.uint8)


_HINT_OVERLAY_BORDER_PX = 8
_HINT_OVERLAY_WIDTH_FRAC = 0.82
_HINT_OVERLAY_HEIGHT_FRAC = 0.55


def _composite_hint_overlay(frame: np.ndarray, hint_text: str) -> np.ndarray:
    """Composite a centered hint overlay onto the frame (in-place style).

    The overlay is a dark inner box with a thick yellow border, sized to
    ~82% width × 55% height of the frame, centered. The 3D view and the
    HUD bar remain visible around the overlay so the agent doesn't lose
    its sense of place while reading the hint.
    """
    h, w = frame.shape[:2]
    box_w = int(w * _HINT_OVERLAY_WIDTH_FRAC)
    box_h = int(h * _HINT_OVERLAY_HEIGHT_FRAC)
    box_x = (w - box_w) // 2
    box_y = (h - box_h) // 2

    img = _PILImage.fromarray(frame)
    d = _PILDraw.Draw(img)
    bd = _HINT_OVERLAY_BORDER_PX

    # Yellow outer border + dark inner fill drawn together via two rects.
    d.rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        fill=(255, 220, 80),
    )
    d.rectangle(
        [box_x + bd, box_y + bd, box_x + box_w - bd, box_y + box_h - bd],
        fill=(15, 15, 25),
    )

    # "KEY HINT" header (yellow), centered horizontally near the top of
    # the inner box.
    d.text(
        (box_x + box_w // 2 - 60, box_y + bd + 6),
        "KEY HINT", fill=(255, 220, 80), font=_BANNER_FONT,
    )

    # Hint text wrapped to ~18 chars/line so it fits inside the inner box
    # at the BANNER_FONT size without overflowing the right border.
    words = hint_text.split()
    lines, cur = [], ""
    for word in words:
        candidate = (cur + " " + word).strip()
        if len(candidate) > 18:
            lines.append(cur)
            cur = word
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    text_left = box_x + bd + 12
    text_top = box_y + bd + 44
    for i, line in enumerate(lines):
        d.text(
            (text_left, text_top + i * 26),
            line, fill=(235, 235, 235), font=_BANNER_FONT,
        )

    return np.array(img, dtype=np.uint8)


class MemoryFPSEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        seed: int = 0,
        difficulty: str = "medium",
        name_source: "str | NameSource" = "procedural",
        image_source: "str | ImageSource" = "public",
        object_source: "str | ObjectSource" = "public",
        enemy_face_source: "str | FaceSource" = "public",
        event_sink=None,
        time_source: Optional[Callable[[], float]] = None,
    ):
        super().__init__()
        self.seed_value = int(seed)
        self.cfg = load_difficulty(difficulty)
        self.name_source = self._resolve_name_source(name_source)
        self.image_source = self._resolve_image_source(image_source)
        self.object_source = self._resolve_object_source(object_source)
        self.enemy_face_source = self._resolve_enemy_face_source(enemy_face_source)
        # Event sink is duck-typed: must expose .write(event_dict, step=int).
        # Lives outside the candidate package — the caller owns
        # the sink's lifecycle and is responsible for closing it. None means
        # all event writes are no-ops, which is the candidate-facing default.
        self._log = event_sink
        self._last_room_id: int = -1
        # Wall-clock source. Defaults to monotonic; tests inject a fake
        # callable to drive sim_time deterministically.
        self._time_source: Callable[[], float] = time_source or time.monotonic

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(_OBS_H, _OBS_W, 3), dtype=np.uint8
        )

        self._world: Optional[_MWWorld] = None
        self._graph: Optional[RoomGraph] = None
        self._enemies: list = []
        self._step_count: int = 0
        self._sim_time: float = 0.0     # accumulated simulated seconds
        self._last_real_time: float = 0.0   # wall-clock of previous step()
        self._phase: str = "play"
        self._terminated: bool = False
        self._door_touched_once: bool = False
        self._door_in_contact: bool = False
        self._banner_until_sim_t: float = -1.0
        self._held_key: bool = False
        self._key_entity = None
        self._door_entities: list = []
        self._door_center: tuple = (0.0, 0.0)
        self._locked_room_id: int = -1
        self._door_parent_room_id: int = -1
        self._key_room_id: int = -1
        self._key_position: tuple = (0.0, 0.0)
        self._hint_text: str = ""
        self._hint_rank: int = 0

    # --- API ----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed_value = int(seed)
        # Clean up any per-episode temp image dir from a prior reset()
        # before we drop the old _world. _gen_world (called from _MWWorld
        # constructor below) will write a fresh dir onto the new world.
        if self._world is not None:
            prev_tmp = getattr(self._world, "_tmp_image_dir", None)
            if prev_tmp:
                import shutil
                shutil.rmtree(prev_tmp, ignore_errors=True)
        rng = EnvRNG(self.seed_value)
        self._graph = generate_graph(
            rng, self.cfg,
            self.name_source, self.image_source, self.object_source,
        )
        self._enemies = spawn_for_graph(
            rng.substream("enemies"), self.cfg, self._graph
        )
        self._world = _MWWorld(
            self._graph, self._enemies,
            face_source=self.enemy_face_source,
            face_rng=rng.substream("faces"),
            doors_rng=rng.substream("doors"),
            door_collision_thickness=self.cfg.door_collision_thickness,
            obs_w=_OBS_W, obs_h=_OBS_H,
        )
        self._world.reset(seed=self.seed_value)
        # Roll agent HP from config range. Must happen AFTER _world.reset so
        # the values stick (reset internally calls _gen_world again).
        hp_max = rng.substream("agent_hp").randint(
            self.cfg.agent_hp_min, self.cfg.agent_hp_max
        )
        self._world.agent_hp_max = hp_max
        self._world.agent_hp = hp_max
        self._step_count = 0
        self._sim_time = 0.0
        self._last_real_time = self._time_source()
        self._phase = "play"
        self._terminated = False
        self._door_touched_once = False
        self._door_in_contact = False
        self._banner_until_sim_t = -1.0
        self._held_key = False
        self._key_entity = None
        # build_world sets these on _world (the _MWWorld instance that calls it
        # via _gen_world). Copy them onto self so the runtime state machine and
        # tests can access them directly from the MemoryFPSEnv.
        self._door_entities = getattr(self._world, "_door_entities", [])
        self._door_center = getattr(self._world, "_door_center", (0.0, 0.0))
        self._locked_room_id = getattr(self._world, "_locked_room_id", -1)
        self._door_parent_room_id = getattr(self._world, "_door_parent_room_id", -1)
        self._key_room_id = getattr(self._world, "_key_room_id", -1)
        self._key_position = getattr(self._world, "_key_position", (0.0, 0.0))
        self._hint_text = getattr(self._world, "_hint_text", "")
        self._hint_rank = getattr(self._world, "_hint_rank", 0)

        # Dump the full ground-truth world via the eval-side sink. The sink
        # is None for candidate-facing runs (writes become no-ops via the
        # guard below).
        if self._log is not None:
            self._log.write({
                "type": "world_built",
                "spawn_room": self._graph.spawn_room_id,
                "rooms": [
                    {
                        "id": r.id,
                        "name": r.name,
                        "wall_color": r.wall_color,
                        "neighbors": self._graph.neighbors(r.id),
                        # Per-wall image breakdown. Wall index k → list of
                        # images on wall k (south=0, east=1, north=2, west=3).
                        # Lets QA ask "which wall of the kitchen had the
                        # lion image?" without losing the flat-list info
                        # (which is just sum(images_per_wall, [])).
                        "images_per_wall": _per_wall_images_payload(r),
                        "objects": [
                            {"id": o.id, "category": o.category,
                             "color": o.color, "caption": o.caption}
                            for o in r.objects
                        ],
                    }
                    for r in self._graph.rooms
                ],
                "enemies": [e.to_log_dict() for e in self._enemies],
            }, step=0)
        if self._log is not None and self._locked_room_id >= 0:
            self._log.write({
                "type": "door_placed",
                "locked_room_id": self._locked_room_id,
                "door_edge": [self._locked_room_id, self._door_parent_room_id],
                "door_center": list(self._door_center),
            }, step=0)
            self._log.write({
                "type": "key_spawn_planned",
                "key_room_id": self._key_room_id,
                "key_pos": list(self._key_position),
                "hint_text": self._hint_text,
                "hint_template_rank": self._hint_rank,
            }, step=0)
        cur = self._world.current_room_id(self._graph)
        self._last_room_id = cur if cur is not None else -1
        return self._build_obs(), {"phase": "play"}

    def close(self):
        """Release per-episode resources (temp PNG dir).

        The event sink (if any) is owned by the caller — env never closes it.
        Safe to call multiple times and before any reset().
        """
        tmp_dir = (
            getattr(self._world, "_tmp_image_dir", None)
            if self._world is not None
            else None
        )
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            try:
                self._world._tmp_image_dir = None
            except Exception:
                pass

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("step() called after termination")
        self._step_count += 1
        terminated, truncated = False, False

        if action == Action.END_EPISODE:
            terminated = True
        elif action == Action.NO_OP:
            pass
        elif action == Action.ATTACK:
            self._resolve_agent_attack()
        else:
            self._world.apply_action(int(action))

        # Log room-transition events as soon as the agent crosses a portal.
        cur_room_for_log = self._world.current_room_id(self._graph)
        cur_room_for_log = -1 if cur_room_for_log is None else cur_room_for_log
        if (self._log is not None
                and cur_room_for_log != self._last_room_id
                and cur_room_for_log >= 0):
            self._log.write({
                "type": "entered_room",
                "room_id": cur_room_for_log,
                "room_name": self._graph.rooms[cur_room_for_log].name,
            }, step=self._step_count)
        self._last_room_id = cur_room_for_log

        # Wall-clock dt since the previous step(). The env tracks real
        # time so enemies advance whether or not the agent took a meaningful
        # action — calling step(NO_OP) once a second moves an enemy ~1.5 m
        # closer; calling 100 times in a millisecond does next to nothing.
        # The clamp prevents a paused debugger or long model-inference stall
        # from pumping the AI for an entire episode in a single call.
        now = self._time_source()
        dt_real = max(0.0, min(now - self._last_real_time,
                               self.cfg.max_dt_per_step_seconds))
        self._last_real_time = now

        # Sub-tick the AI in ai_dt_seconds slices so enemy motion stays
        # smooth and attack cooldowns are checked at the right granularity.
        ax, _, az = self._world.agent.pos
        ai_dt = self.cfg.ai_dt_seconds
        n_subticks = max(1, int(round(dt_real / ai_dt))) if dt_real > 0 else 0
        sub_dt = dt_real / n_subticks if n_subticks > 0 else 0.0
        for _sub in range(n_subticks):
            self._sim_time += sub_dt
            for e in self._enemies:
                tick_ai(
                    e,
                    agent_pos=(ax, az),
                    visibility_range=self.cfg.enemy_visibility_range,
                    attack_range=self.cfg.enemy_attack_range,
                    t_now=self._sim_time,
                    dt=sub_dt,
                    world=self._world,
                )
                # Sync rendered body parts to the new logical XZ so the
                # visible mob moves while it chases the agent.
                if e.is_alive():
                    for ent, ox_off, oz_off in getattr(e, "_mw_entity_parts", ()):
                        cur_pos = ent.pos
                        ent.pos = (e.pos[0] + ox_off, cur_pos[1], e.pos[1] + oz_off)
            # Apply enemy attacks per sub-tick so cooldowns are honored.
            self._apply_enemy_attacks()

        # --- Locked door touch detection -------------------------------
        # Distance from agent's XZ to door center. We do the check after
        # AI ticks so the per-step physics order is: movement → AI →
        # door touch → HP check. Re-touching the door requires a
        # walk-away (contact transition).
        if self._locked_room_id >= 0:
            ax, _, az = self._world.agent.pos
            dx, dz = self._door_center
            dist_door = math.hypot(ax - dx, az - dz)
            cfg_radius = self.cfg.door_touch_radius
            in_radius = dist_door <= cfg_radius
            if in_radius and not self._door_in_contact:
                self._door_in_contact = True
                self._on_door_touch_transition()
            elif not in_radius and self._door_in_contact:
                self._door_in_contact = False

        # --- Key pickup detection -------------------------------------
        if self._key_entity is not None and not self._held_key:
            ax, _, az = self._world.agent.pos
            kx, kz = self._key_position
            if math.hypot(ax - kx, az - kz) <= self.cfg.key_pickup_radius:
                if self._key_entity in self._world.entities:
                    self._world.entities.remove(self._key_entity)
                self._key_entity = None
                self._held_key = True
                if self._log is not None:
                    self._log.write({
                        "type": "key_picked_up",
                        "pos": [ax, az],
                    }, step=self._step_count)

        # HP-death termination.
        if self._world.agent_hp <= 0:
            terminated = True

        if self._sim_time >= self.cfg.max_seconds and not terminated:
            truncated = True

        if terminated or truncated:
            self._phase = "qa"
            self._terminated = True
            if self._log is not None:
                if action == Action.END_EPISODE:
                    self._log.write({"type": "agent_ended"},
                                    step=self._step_count)
                    termination = "agent_ended"
                elif self._world.agent_hp <= 0:
                    self._log.write({
                        "type": "agent_died",
                        "by_enemy_id": (-1 if self._log.killer_enemy is None
                                        else self._log.killer_enemy.get("_id", -1)),
                        "appearance": (
                            {k: v for k, v in self._log.killer_enemy.items()
                             if k != "_id"}
                            if self._log.killer_enemy is not None else None
                        ),
                    }, step=self._step_count)
                    termination = "agent_died"
                else:
                    self._log.write({"type": "time_limit"},
                                    step=self._step_count)
                    termination = "time_limit"
                killer_appearance = None
                if self._log.killer_enemy is not None:
                    killer_appearance = {
                        k: v for k, v in self._log.killer_enemy.items()
                        if k != "_id"
                    }
                self._log.write({
                    "type": "episode_end",
                    "termination": termination,
                    "final_hp": self._world.agent_hp,
                    "sim_time": self._sim_time,
                    "enemy_attack_order": list(self._log.enemy_attack_order),
                    "agent_attack_order": list(self._log.agent_attack_order),
                    "enemy_kill_order": list(self._log.enemy_kill_order),
                    "killer_enemy": killer_appearance,
                }, step=self._step_count)
                # Sink lifecycle is owned by the caller. The
                # env never closes a sink it didn't create.
            return (
                _build_qa_banner(_OBS_W, _OBS_H),
                0.0,
                terminated,
                truncated,
                {"phase": "qa"},
            )

        return self._build_obs(), 0.0, terminated, truncated, {"phase": self._phase}

    # --- combat -------------------------------------------------------
    def _resolve_agent_attack(self) -> dict | None:
        """Forward raycast from the agent. Damages the nearest enemy inside
        ``_ATTACK_RANGE`` whose bearing is within ``_ATTACK_CONE_HALF_RAD``.
        Returns a hit-info dict (or ``None`` on miss). Decrements enemy HP
        exactly once per call.
        """
        ax, _, az = self._world.agent.pos
        adir = self._world.agent.dir
        # Miniworld convention: agent.dir_vec = (cos(dir), 0, -sin(dir)), so
        # the visual forward in (x, z) is (cos(adir), -sin(adir)).
        fwd_x = math.cos(adir)
        fwd_z = -math.sin(adir)
        cone_cos = math.cos(_ATTACK_CONE_HALF_RAD)
        best = None
        best_dist = float("inf")
        for e in self._enemies:
            if not e.is_alive():
                continue
            dx = e.pos[0] - ax
            dz = e.pos[1] - az
            dist = math.hypot(dx, dz)
            if dist > _ATTACK_RANGE or dist < 1e-6:
                continue
            # Cosine of the angle between forward and (dx, dz)/dist.
            dot = (fwd_x * dx + fwd_z * dz) / dist
            if dot < cone_cos:
                continue
            if dist < best_dist:
                best_dist = dist
                best = e
        if best is None:
            if self._log is not None:
                self._log.write({"type": "agent_attack", "hit": False},
                                step=self._step_count)
            return None

        # Log "first-time agent attacked this enemy" before the decrement.
        if self._log is not None and best.id not in self._log.agent_attack_order:
            self._log.agent_attack_order.append(best.id)
            self._log.write({
                "type": "agent_first_attacked_enemy",
                "enemy_id": best.id, "room_id": best.room_id,
                "appearance": dict(best.appearance),
                "order": len(self._log.agent_attack_order),
            }, step=self._step_count)

        # Decrement HP exactly once.
        best.hp -= 1

        # If the kill blow landed, remove the rendered body parts so the
        # corpse disappears immediately. (Logical state — AI tick, raycast,
        # enemy attacks — already skips dead enemies via is_alive().)
        if not best.is_alive():
            for ent, _, _ in getattr(best, "_mw_entity_parts", ()):
                if ent in self._world.entities:
                    self._world.entities.remove(ent)
            best._mw_entity_parts = []
            best._mw_entity = None

        if self._log is not None:
            self._log.write({
                "type": "agent_attack", "hit": True,
                "enemy_id": best.id, "enemy_remaining_hp": best.hp,
            }, step=self._step_count)
            if not best.is_alive():
                self._log.enemy_kill_order.append(best.id)
                self._log.write({
                    "type": "enemy_killed", "enemy_id": best.id,
                    "room_id": best.room_id, "appearance": dict(best.appearance),
                    "order": len(self._log.enemy_kill_order),
                }, step=self._step_count)
        return {"enemy_id": best.id, "remaining_hp": best.hp}

    def _apply_enemy_attacks(self) -> int:
        """Apply damage from every enemy currently in 'attack' state, gated by
        per-enemy cooldown. Returns total damage applied this sub-tick.
        """
        t_now = self._sim_time
        total = 0
        for e in self._enemies:
            if not e.is_alive() or e.state != "attack":
                continue
            if t_now - e.last_attack_t < e.cooldown:
                continue
            e.last_attack_t = t_now
            self._world.agent_hp -= e.damage
            total += e.damage

            if self._log is not None:
                if e.id not in self._log.enemy_attack_order:
                    self._log.enemy_attack_order.append(e.id)
                    self._log.write({
                        "type": "enemy_first_attacked_agent",
                        "enemy_id": e.id, "room_id": e.room_id,
                        "appearance": dict(e.appearance),
                        "order": len(self._log.enemy_attack_order),
                    }, step=self._step_count)
                self._log.write({
                    "type": "enemy_attack", "enemy_id": e.id,
                    "damage": e.damage,
                    "agent_hp_after": self._world.agent_hp,
                }, step=self._step_count)
                # Record the killer (the first enemy to bring HP to <= 0).
                if (self._world.agent_hp <= 0
                        and self._log.killer_enemy is None):
                    killer = dict(e.appearance)
                    killer["_id"] = e.id
                    self._log.killer_enemy = killer
        return total

    def _on_door_touch_transition(self) -> None:
        """Called once per (out-of-contact → in-contact) transition.

        Deferred key spawn fires on the first touch only. Banner is shown
        on every touch (first or re-touch) while agent doesn't hold the key.
        """
        if self._held_key:
            self._unlock_door()
            return
        # First-time touch: spawn key.
        if not self._door_touched_once:
            if self._log is not None:
                self._log.write(
                    {"type": "door_first_touched"}, step=self._step_count
                )
            self._spawn_key()
            self._door_touched_once = True
        # Every touch (first or repeat): start the banner.
        self._banner_until_sim_t = (
            self._sim_time + self.cfg.banner_duration_seconds
        )

    def _spawn_key(self) -> None:
        """Instantiate the Miniworld Key entity in the chosen key room.

        Position came from pick_key_position_in_room at build time and is
        cached on env._key_position. Color is fixed (yellow) so the agent
        always knows what shape to look for.

        We set ``key.radius = 0.0`` so the key does not participate in
        Miniworld's dynamic-entity collision. Otherwise the agent's 0.4 m
        radius plus the key mesh's ~0.43 m radius would stop the agent at
        ~0.83 m center-to-center — *farther* than ``key_pickup_radius``,
        meaning the agent could never actually trigger pickup in real
        gameplay. Disabling collision lets the agent walk through the key;
        the per-step proximity check in ``step()`` then triggers pickup
        the moment the centers come within ``key_pickup_radius``.
        """
        from miniworld.entity import Key
        key = Key(color="yellow")  # Miniworld Key colors: red/green/blue/yellow
        key.radius = 0.0  # walk-through; pickup is by step()-level proximity check
        kx, kz = self._key_position
        self._world.place_entity(key, pos=[kx, 0.0, kz], dir=0)
        self._key_entity = key
        if self._log is not None:
            self._log.write({
                "type": "key_spawned",
                "room_id": self._key_room_id,
                "pos": list(self._key_position),
            }, step=self._step_count)

    def _unlock_door(self) -> None:
        """Remove door entities + collision; consume key."""
        for ent in self._door_entities:
            if ent in self._world.entities:
                self._world.entities.remove(ent)
        self._door_entities = []
        self._held_key = False
        if self._log is not None:
            self._log.write({"type": "door_unlocked"}, step=self._step_count)

    # --- internals ----------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        frame = self._world.render_frame()
        room_name = self._world.current_room_name(self._graph)
        # Whole-second countdown (rounds down) of remaining simulated time.
        seconds_remaining = max(0, int(self.cfg.max_seconds - self._sim_time))
        # Read live HP from the world (set on reset, decremented by enemy attacks).
        hp = getattr(self._world, "agent_hp", 10)
        hp_max = getattr(self._world, "agent_hp_max", 10)
        # Agent facing as an integer compass heading, so absolute-direction
        # questions (e.g. "the east wall") are answerable from pixels.
        heading = heading_degrees(self._world.agent.dir)
        obs = draw_hud(frame, room_name, seconds_remaining, hp, hp_max,
                       has_key=self._held_key, heading=heading)
        # Composite the hint overlay on top of the normal 3D+HUD frame
        # when the banner timer is active.
        if self._sim_time < self._banner_until_sim_t:
            obs = _composite_hint_overlay(obs, self._hint_text)
        return obs

    @staticmethod
    def _resolve_name_source(spec) -> NameSource:
        if isinstance(spec, NameSource):
            return spec
        if spec == "procedural":
            return ProceduralNameSource()
        if spec == "public":
            return PublicNameSource()
        raise ValueError(f"Unknown name_source: {spec}")

    @staticmethod
    def _resolve_image_source(spec) -> ImageSource:
        import warnings
        if isinstance(spec, ImageSource):
            # Empty PublicImageSource → fall back to procedural with a warning.
            if isinstance(spec, PublicImageSource):
                try:
                    if not spec._scan():
                        warnings.warn(
                            f"Public image assets directory {spec.path} is empty; "
                            "falling back to ProceduralImageSource. Run "
                            "scripts/generate_images.py to populate it.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        return ProceduralImageSource()
                except Exception:
                    pass
            return spec
        if spec == "procedural":
            return ProceduralImageSource()
        if spec == "public":
            src = PublicImageSource()
            if not src._scan():
                warnings.warn(
                    f"Public image assets directory {src.path} is empty; "
                    "falling back to ProceduralImageSource. Run "
                    "scripts/generate_images.py to populate it.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return ProceduralImageSource()
            return src
        raise ValueError(f"Unknown image_source: {spec}")

    @staticmethod
    def _resolve_object_source(spec) -> ObjectSource:
        if isinstance(spec, ObjectSource):
            return spec
        if spec == "public":
            return PublicObjectSource()
        # "heldout" is intentionally NOT a string spec in the candidate
        # package — the heldout pool lives outside it. The evaluator
        # imports HeldoutObjectSource and passes it as an instance.
        raise ValueError(f"Unknown object_source: {spec!r}")

    @staticmethod
    def _resolve_enemy_face_source(spec) -> FaceSource:
        if isinstance(spec, FaceSource):
            return spec
        if spec == "public":
            return PublicFaceSource()
        if spec == "heldout":
            return HeldoutFaceSource()
        raise ValueError(f"Unknown enemy_face_source: {spec}")


# Action-to-Miniworld-action mapping.
_MW_TURN_LEFT = 0
_MW_TURN_RIGHT = 1
_MW_MOVE_FORWARD = 2
_MW_MOVE_BACK = 3


class _MWWorld(MiniWorldEnv):
    """Internal Miniworld instance, owned by MemoryFPSEnv."""

    def __init__(self, graph, enemies, face_source, face_rng, obs_w, obs_h,
                 doors_rng=None, door_collision_thickness: float = 0.1):
        self._graph = graph
        self._enemies = enemies
        self._face_source = face_source
        self._face_rng = face_rng
        self._doors_rng = doors_rng
        self._door_collision_thickness = door_collision_thickness
        super().__init__(
            max_episode_steps=10_000_000,  # we manage termination ourselves
            obs_width=obs_w, obs_height=obs_h, domain_rand=False,
        )

    def _gen_world(self):
        build_world(
            self, self._graph,
            enemies=self._enemies,
            face_source=self._face_source,
            face_rng=self._face_rng,
            doors_rng=self._doors_rng,
            door_collision_thickness=self._door_collision_thickness,
        )

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        # Default HP fallback; outer MemoryFPSEnv overrides with rolled values
        # after this reset returns.
        self.agent_hp_max = getattr(self, "agent_hp_max", 10)
        self.agent_hp = self.agent_hp_max
        return out

    def apply_action(self, action: int) -> None:
        # Map our Action enum (0..3 share semantics) onto Miniworld's enum.
        mw_action = {
            0: _MW_TURN_LEFT,
            1: _MW_TURN_RIGHT,
            2: _MW_MOVE_FORWARD,
            3: _MW_MOVE_BACK,
        }.get(action)
        if mw_action is not None:
            self.step(mw_action)  # MiniWorldEnv.step

    def render_frame(self) -> np.ndarray:
        return self.render_obs()

    def current_room_id(self, graph: RoomGraph) -> int | None:
        ax, _, az = self.agent.pos
        for r in graph.rooms:
            ox, oz = r.origin
            w, d = r.size
            if ox <= ax <= ox + w and oz <= az <= oz + d:
                return r.id
        return None

    def current_room_name(self, graph: RoomGraph) -> str | None:
        ax, _, az = self.agent.pos
        for r in graph.rooms:
            ox, oz = r.origin
            w, d = r.size
            if ox <= ax <= ox + w and oz <= az <= oz + d:
                return r.name
        return None
