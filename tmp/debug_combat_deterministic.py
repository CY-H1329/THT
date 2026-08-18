"""가짜 시계로 재현 가능한 전투 로그. event_sink로 hit/kill 진실을 같이 본다."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import enemies as EN
from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05


def make_fake_clock(dt):
    state = {"t": 0.0}

    def clock():
        state["t"] += dt
        return state["t"]
    return clock


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
env = MemoryFPSEnv(seed=SEED, event_sink=sink, time_source=make_fake_clock(DT))
obs, _info = env.reset()
policy = ExplorerPolicy()
action_names = {v: k for k, v in policy._Action.__members__.items()}

term = trunc = False
i = 0
in_flee_prev = False
for i in range(1, MAX_STEPS + 1):
    hud = read_hud(obs)
    action = policy.step(obs)
    in_flee = policy.state == "FLEE"
    if in_flee or in_flee_prev:
        wall_rgb = policy._current_wall_rgb(hud)
        mobs = EN.detect(obs, wall_rgb=wall_rgb)
        mob_desc = [(round(m.bearing, 1), round(m.distance, 2), round(m.score, 2),
                     m.col_range) for m in mobs[:2]]
        print(f"i={i:4d} state={policy.state:6s} sweep={policy.flee_sweep_active!s:5} "
              f"heading={hud.heading!s:4s} mobs={mob_desc} "
              f"action={action_names.get(int(action), action)}")
    in_flee_prev = in_flee
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        break

env.close()
print(f"\n=== ended step={i} terminated={term} truncated={trunc} ===")
for step, ev in sink.events:
    if ev.get("type") in ("agent_attack", "enemy_attack", "enemy_killed", "agent_died", "episode_end"):
        print(f"step={step} {ev}")
