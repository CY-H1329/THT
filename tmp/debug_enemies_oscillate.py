import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from agent import enemies as EN
from agent import geometry as geo
from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
START = int(sys.argv[2]) if len(sys.argv) > 2 else 320
END = int(sys.argv[3]) if len(sys.argv) > 3 else 342

env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()
policy = ExplorerPolicy()
action_names = {v: k for k, v in policy._Action.__members__.items()}

for i in range(1, END + 1):
    hud = read_hud(obs)
    wall_rgb = policy._current_wall_rgb(hud) if hud.ok else None
    mobs = EN.detect(obs, wall_rgb=wall_rgb)
    action = policy.step(obs)
    show = (i >= START) or (mobs and mobs[0].distance >= 19.9)
    if show:
        mob_desc = [(round(m.bearing, 1), round(m.distance, 2), round(m.score, 2),
                     m.col_range, m.resolved) for m in mobs[:3]]
        print(f"i={i:4d} heading={hud.heading!s:4s} state={policy.state:6s} "
              f"mobs={mob_desc} action={action_names.get(int(action), action)}")
        if mobs and mobs[0].distance >= 19.9:
            lo, hi = mobs[0].col_range
            sat = geo.saturation(obs)
            rows, dists = geo.floor_boundary(obs, 64, sat)
            print(f"    col_range=({lo},{hi}) rows={rows[lo:hi]} dists={np.round(dists[lo:hi],2)}")
            cw = 320 // 64
            band = sat[:, lo * cw:hi * cw]
            print(f"    sat col-band: min={band.min()} max={band.max()} "
                  f"bottom_row_sat={sat[-1, lo*cw:hi*cw]}")
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        print("episode ended")
        break
