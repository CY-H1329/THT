import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from agent.vision import is_blocked, sample_wall_color
from memory_fps_env.env import Action, MemoryFPSEnv

FRAMES = ROOT / "tmp" / "debug_frames"

# 1) 벽 색 샘플링: 저장된 프레임으로 육안 확인한 색과 맞는지 (Ebon Annex는
#    붉은 테라코타 계열, Topaz Court도 비슷한 붉은 벽돌 계열로 보였음).
for name in ["seed0_step010.png", "seed42_step020.png"]:
    p = FRAMES / name
    if p.exists():
        frame = np.array(Image.open(p).convert("RGB"))
        print(f"{name}: wall_color={sample_wall_color(frame)}")

# 2) is_blocked: 실제로 90도 돌려서 벽으로 걸어가며 언제 True가 되는지.
env = MemoryFPSEnv(seed=0)
obs, _ = env.reset()
for _ in range(6):
    obs, _r, _t, _tr, _i = env.step(int(Action.TURN_RIGHT))
prev = obs
for i in range(1, 32):
    obs, _r, term, trunc, _info = env.step(int(Action.MOVE_FORWARD))
    blocked = is_blocked(prev, obs)
    if i in (1, 10, 20, 27, 28, 29, 30, 31) or blocked:
        print(f"step={i:2d} is_blocked={blocked}")
    prev = obs
    if blocked:
        break
env.close()
