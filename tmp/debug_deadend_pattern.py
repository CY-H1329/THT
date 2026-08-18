"""막다른 방에서 벽치기/오락가락/모서리충돌사 패턴을 재현하고, 죽기 직전
N틱의 전체 상태를 덤프한다. 가짜 시계로 결정론적 재현."""

import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import geometry as geo
from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
TAIL = int(sys.argv[3]) if len(sys.argv) > 3 else 80


def make_fake_clock(dt=0.05):
    state = {"t": 0.0}

    def clock():
        state["t"] += dt
        return state["t"]
    return clock


env = MemoryFPSEnv(seed=SEED, time_source=make_fake_clock())
obs, _info = env.reset()
policy = ExplorerPolicy()
action_names = {v: k for k, v in policy._Action.__members__.items()}

hist = deque(maxlen=TAIL)
term = trunc = False
i = 0
for i in range(1, MAX_STEPS + 1):
    hud = read_hud(obs)
    x, _, z = env._world.agent.pos
    action = policy.step(obs)
    hist.append((
        i, policy.state, hud.room_name, hud.heading, policy.target_heading,
        round(policy._nav_bias_deg, 1), policy.recover_attempts,
        round(x, 2), round(z, 2),
        action_names.get(int(action), action),
    ))
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        break

env.close()
print(f"=== ended step={i} terminated={term} truncated={trunc} ===")
print(f"last {len(hist)} ticks:")
for h in hist:
    print(h)
print(f"\nrooms discovered: {list(policy.scene.nodes.keys())}")
for name, node in policy.scene.nodes.items():
    print(f"  {name}: wall_color={node.wall_color} done={node.done} "
          f"exits={node.exits} leads_to={node.exit_leads_to}")
