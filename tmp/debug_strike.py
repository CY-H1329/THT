"""strike 단계에서 실제로 어떤 액션이 나가는지, enemy_bearing이 뭐라고 읽히는지
스텝별로 로그를 찍어서 왜 HP가 계속 깎이는지 확인한다.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from agent.vision import enemy_bearing
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300

env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()
policy = ExplorerPolicy()

action_names = {v: k for k, v in policy._Action.__members__.items()}

term = trunc = False
i = 0
for i in range(1, MAX_STEPS + 1):
    action = policy.step(obs)
    phase = policy.flee_phase
    state = policy.state
    wall_color = policy._current_wall_color(read_hud(obs)) if state == "FLEE" else None
    bearing_before = enemy_bearing(obs, wall_color) if state == "FLEE" else None

    obs, _r, term, trunc, _info = env.step(action)
    hud = read_hud(obs) if not (term or trunc) else None

    if state == "FLEE":
        print(f"step={i:4d} phase={phase!s:10s} bearing={bearing_before!s:6s} "
              f"action={action_names.get(int(action), action)!s:14s} "
              f"hp={hud.hp if hud and hud.ok else '?'}")

    if term or trunc:
        break

env.close()
print(f"\n=== ended step={i} terminated={term} truncated={trunc} ===")
