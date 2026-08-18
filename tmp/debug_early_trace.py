import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400


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

for i in range(1, MAX_STEPS + 1):
    hud = read_hud(obs)
    action = policy.step(obs)
    print(f"i={i:4d} state={policy.state:9s} room={hud.room_name!s:22s} "
          f"heading={hud.heading!s:4s} target={policy.target_heading!s:4s} "
          f"nav_fail={policy._nav_fail_count} action={action_names.get(int(action), action)}")
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        break

print(f"\nrooms: {list(policy.scene.nodes.keys())}")
for name, node in policy.scene.nodes.items():
    print(f"  {name}: exits={node.exits}")
