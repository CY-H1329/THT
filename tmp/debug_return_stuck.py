import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from agent.vision import is_blocked
from memory_fps_env.env import MemoryFPSEnv, Action

env = MemoryFPSEnv(seed=0)
obs, _info = env.reset()
policy = ExplorerPolicy()

for i in range(1, 750):
    hud = read_hud(obs)
    prev_state = policy.state
    prev_frame_before = policy.prev_frame
    action = policy.step(obs)
    if prev_state == "RETURN" and 650 <= i <= 720:
        blocked = is_blocked(prev_frame_before, obs) if prev_frame_before is not None else None
        print(f"step={i:4d} heading={hud.heading:3d} target={policy.target_heading} "
              f"return_to={policy.return_target_room!r} room={hud.room_name!r} "
              f"need_align={policy._need_align} last_fwd={policy._last_action_was_forward} "
              f"recover={policy.recover_attempts} blocked={blocked} action={Action(action).name}")
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        print("ended at", i)
        break
env.close()
