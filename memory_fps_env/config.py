from dataclasses import dataclass
from pathlib import Path
from typing import List
import yaml

DIFFICULTY_VERSION = 1
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "difficulty.yaml"


@dataclass
class DifficultyConfig:
    room_count_min: int
    room_count_max: int
    room_size_min: int
    room_size_max: int
    extra_edges: int
    max_seconds: float
    ai_dt_seconds: float
    max_dt_per_step_seconds: float
    agent_hp_min: int
    agent_hp_max: int
    enemy_hp_min: int
    enemy_hp_max: int
    enemy_damage_min: int
    enemy_damage_max: int
    enemy_cooldown_min: float
    enemy_cooldown_max: float
    enemy_visibility_range: float
    enemy_attack_range: float
    enemy_room_prob_one: float
    enemy_room_prob_two: float
    images_per_wall_weights: List[float]
    door_touch_radius: float
    key_pickup_radius: float
    banner_duration_seconds: float
    door_collision_thickness: float


def load_difficulty(name: str, path: Path | None = None) -> DifficultyConfig:
    path = path or _DEFAULT_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data.get("version") != DIFFICULTY_VERSION:
        raise RuntimeError(
            f"difficulty.yaml version {data.get('version')} != "
            f"DIFFICULTY_VERSION {DIFFICULTY_VERSION}"
        )

    if name not in data:
        raise KeyError(f"Unknown difficulty: {name}")

    payload = data[name]
    # Migration guards for the per-wall-image-count knob.
    legacy_keys = {
        "images_per_room_min", "images_per_room_max",
        "images_per_wall_min", "images_per_wall_max",
    } & set(payload)
    if legacy_keys:
        raise ValueError(
            f"difficulty.yaml '{name}' uses legacy keys {sorted(legacy_keys)}. "
            f"These were replaced by images_per_wall_weights "
            f"(list of per-count weights, e.g. [0.2, 0.6, 0.2] for "
            f"P(n=0)=0.2, P(n=1)=0.6, P(n=2)=0.2)."
        )
    # Migration guard for the wall-clock time switch (was step-based).
    if "max_steps" in payload:
        raise ValueError(
            f"difficulty.yaml '{name}' uses the old key 'max_steps'. The env "
            f"now terminates by wall-clock; replace it with "
            f"'max_seconds: <int>' (total simulated seconds before truncation), "
            f"and add 'ai_dt_seconds' (AI sub-tick granularity, default 0.1) and "
            f"'max_dt_per_step_seconds' (clamp for long pauses, default 5.0)."
        )
    # The legacy survival/exploration/qa "weights" block was removed when
    # grading moved to (QA × 0.6 + Report × 0.4) — see README. The yaml
    # no longer accepts a `weights:` key; warn if someone tries to use it.
    if "weights" in payload:
        raise ValueError(
            f"difficulty.yaml '{name}' has a `weights:` block but the "
            f"survival/exploration/qa weighting was retired. Grading is now "
            f"60% QA + 40% Report (see README §Evaluation). Remove the "
            f"`weights:` section."
        )
    cfg = DifficultyConfig(**payload)
    if not cfg.images_per_wall_weights or any(w < 0 for w in cfg.images_per_wall_weights):
        raise ValueError(
            f"images_per_wall_weights must be a non-empty list of non-negative "
            f"floats; got {cfg.images_per_wall_weights}"
        )
    return cfg
