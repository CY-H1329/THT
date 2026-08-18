"""agent.enemies.detect()가 보고하는 bearing/distance가 실제(ground truth)와
맞는지 검증한다. seed=0, 가짜 시계로 결정론적 재현."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import enemies as EN
from agent import geometry as geo
from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv


def make_fake_clock(dt=0.05):
    state = {"t": 0.0}

    def clock():
        state["t"] += dt
        return state["t"]
    return clock


SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STOP_AT = int(sys.argv[2]) if len(sys.argv) > 2 else 189

env = MemoryFPSEnv(seed=SEED, time_source=make_fake_clock())
obs, _info = env.reset()
policy = ExplorerPolicy()

for i in range(1, STOP_AT + 1):
    action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        print("ended early")
        raise SystemExit

hud = read_hud(obs)
x, _, z = env._world.agent.pos
heading = float(hud.heading)
room_id = env._world.current_room_id(env._graph)
print(f"step={STOP_AT} agent_pos=({x:.2f},{z:.2f}) heading={heading} room={room_id}")

for e in env._enemies:
    if not e.is_alive():
        continue
    dx, dz = e.pos[0] - x, e.pos[1] - z
    true_dist = math.hypot(dx, dz)
    abs_bearing = math.degrees(math.atan2(-dz, dx)) % 360.0
    rel = geo.wrap180(abs_bearing - heading)
    print(f"  enemy id={e.id} room={e.room_id} pos=({e.pos[0]:.2f},{e.pos[1]:.2f}) "
          f"true_dist={true_dist:.2f} true_rel_bearing={rel:.1f}")

wall_rgb = policy._current_wall_rgb(hud)
mobs = EN.detect(obs, wall_rgb=wall_rgb)
print("detect() output:")
for m in mobs:
    print(f"  bearing={m.bearing:.1f} distance={m.distance:.2f} score={m.score:.2f} "
          f"col_range={m.col_range} height={m.height:.2f} resolved={m.resolved}")

from PIL import Image
Image.fromarray(obs).save("/tmp/bearing_check_frame.png")
print("saved /tmp/bearing_check_frame.png")
