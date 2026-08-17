import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv, Action

env = MemoryFPSEnv(seed=0)
obs, _info = env.reset()
policy = ExplorerPolicy()

for i in range(1, 61):
    hud = read_hud(obs)
    action = policy.step(obs)
    print(f"step={i:3d} state={policy.state:8s} heading={hud.heading} "
          f"target={policy.target_heading} queue={policy.survey_queue} "
          f"need_align={policy._need_align} action={Action(action).name} hp={hud.hp}")
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        print("ended")
        break
env.close()
