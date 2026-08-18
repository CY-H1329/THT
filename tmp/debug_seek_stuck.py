import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import geometry as geo
from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
START = int(sys.argv[2]) if len(sys.argv) > 2 else 300
END = int(sys.argv[3]) if len(sys.argv) > 3 else 340

env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()
policy = ExplorerPolicy()
action_names = {v: k for k, v in policy._Action.__members__.items()}

for i in range(1, END + 1):
    hud = read_hud(obs)
    clearance = geo.cone_clearance(geo.depth_profile(obs), 0.0, 15.0)
    action = policy.step(obs)
    if i >= START:
        print(f"i={i:4d} state={policy.state:6s} heading={hud.heading!s:4s} "
              f"target={policy.target_heading!s:4s} recover={policy.recover_attempts:2d} "
              f"clearance={clearance:5.2f} action={action_names.get(int(action), action)}")
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        print("episode ended")
        break
