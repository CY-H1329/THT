"""env의 event_sink 훅(디버그 전용, agent 코드에는 없음)으로 공격 hit/miss,
적 킬 순서, 방마다 적 수를 그대로 받아서 왜 강타(strike) 단계에서 계속
맞기만 하는지 정확히 확인한다.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400


class ListSink:
    def __init__(self):
        self.events = []
        self.enemy_attack_order = []
        self.agent_attack_order = []
        self.enemy_kill_order = []
        self.killer_enemy = None

    def write(self, event, step=None):
        self.events.append((step, event))


sink = ListSink()
env = MemoryFPSEnv(seed=SEED, event_sink=sink)
obs, _info = env.reset()
policy = ExplorerPolicy()

term = trunc = False
i = 0
for i in range(1, MAX_STEPS + 1):
    action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        break

env.close()
print(f"=== ended step={i} terminated={term} truncated={trunc} ===\n")

for step, ev in sink.events:
    t = ev.get("type")
    if t in ("agent_attack", "enemy_attack", "agent_first_attacked_enemy",
              "enemy_first_attacked_agent", "enemy_killed", "episode_end",
              "agent_died"):
        print(f"step={step:4} {ev}")
