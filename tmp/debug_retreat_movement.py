"""retreat(MOVE_BACK) 단계에서 실제로 플레이어 위치가 움직이는지, 적과의
거리가 벌어지는지 좁혀지는지를 env 내부 좌표로 직접 확인한다(디버깅
전용 — env._world는 agent 코드에서 절대 안 씀).
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400

env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()
policy = ExplorerPolicy()


def agent_pos():
    x, _, z = env._world.agent.pos
    return (float(x), float(z))


def nearest_enemy_dist():
    ax, az = agent_pos()
    best = None
    for e in env._enemies:
        if not e.is_alive():
            continue
        d = math.hypot(e.pos[0] - ax, e.pos[1] - az)
        if best is None or d < best:
            best = d
    return best


term = trunc = False
last_pos = agent_pos()
i = 0
for i in range(1, MAX_STEPS + 1):
    phase = policy.flee_phase
    action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)
    hud = read_hud(obs) if not (term or trunc) else None

    if phase == "retreat":
        pos = agent_pos()
        moved = math.hypot(pos[0] - last_pos[0], pos[1] - last_pos[1])
        dist = nearest_enemy_dist()
        print(f"step={i:4d} pos={pos[0]:6.2f},{pos[1]:6.2f} moved={moved:.3f} "
              f"enemy_dist={dist if dist is None else round(dist,2)} "
              f"hp={hud.hp if hud and hud.ok else '?'}")
        last_pos = pos

    if term or trunc:
        break

env.close()
print(f"\n=== ended step={i} terminated={term} truncated={trunc} ===")
