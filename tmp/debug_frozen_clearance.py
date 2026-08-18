"""clearance가 고정된 채 MOVE_FORWARD만 반복되는 지점에서 실제 위치가
안 움직이는지, 근처에 뭐가 있는지(적/오브젝트) 확인한다. 가짜 시계로
결정론적 재현."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

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
START = int(sys.argv[2]) if len(sys.argv) > 2 else 745
END = int(sys.argv[3]) if len(sys.argv) > 3 else 755

env = MemoryFPSEnv(seed=SEED, time_source=make_fake_clock())
obs, _info = env.reset()
policy = ExplorerPolicy()

for i in range(1, START + 1):
    action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        print("ended early")
        raise SystemExit

for i in range(START, END + 1):
    x, _, z = env._world.agent.pos
    hud = read_hud(obs)
    room_id = env._world.current_room_id(env._graph)
    nearby = []
    for e in env._enemies:
        if not e.is_alive():
            continue
        d = ((e.pos[0] - x) ** 2 + (e.pos[1] - z) ** 2) ** 0.5
        if d < 3.0:
            nearby.append((e.id, e.room_id, round(d, 2), e.appearance))
    print(f"i={i} pos=({x:.3f},{z:.3f}) heading={hud.heading} room={room_id} nearby_enemies={nearby}")
    action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        break

Image.fromarray(obs).save("/tmp/frozen_frame.png")
sat = geo.saturation(obs)
print("saved /tmp/frozen_frame.png")
print("center column sat profile (bottom 100 rows, every 10th row):")
cx = obs.shape[1] // 2
for row in range(140, 240, 10):
    print(f"  row={row} sat_around_center={sat[row, cx-5:cx+5]}")
