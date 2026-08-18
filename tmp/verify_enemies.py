"""agent/enemies.py의 MOB_SAT 임계값을 우리 env 실측으로 검증한다.

우리 WALL_PALETTE 자체가 mustard(155)/rust(130)/brick(115)/terracotta(110)
처럼 채도가 높은 색을 포함하고 있어서(참고 레포가 측정한 "벽 ≤75"와
다름), 임계값을 그대로 가져다 쓰면 안 된다 — 실제 렌더된 벽 픽셀과 실제
렌더된 적 픽셀의 채도 분포를 직접 뽑아서 비교한다.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from agent import geometry as geo
from agent.ocr import read_hud
from memory_fps_env.env import Action, MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()

wall_sats = []
mob_sats = []

for i in range(1, MAX_STEPS + 1):
    hud = read_hud(obs)
    x, _, z = env._world.agent.pos
    heading = float(hud.heading) if hud.ok and hud.heading is not None else None
    room_id = env._world.current_room_id(env._graph)

    sat = geo.saturation(obs)
    rows, dists = geo.floor_boundary(obs, 64, sat)

    if heading is not None and room_id is not None:
        import math
        for e in env._enemies:
            if not e.is_alive() or e.room_id != room_id:
                continue
            dx, dz = e.pos[0] - x, e.pos[1] - z
            true_dist = math.hypot(dx, dz)
            abs_bearing = math.degrees(math.atan2(-dz, dx)) % 360.0
            rel = geo.wrap180(abs_bearing - heading)
            if abs(rel) <= 25.0 and true_dist <= 6.0:
                # 이 방위 근처 열들에서 바닥 접점 바로 위 밴드의 채도를 수집.
                bearings = geo.column_bearings(64)
                cols = [ci for ci, b in enumerate(bearings) if abs(geo.wrap180(b - rel)) <= 8]
                for ci in cols:
                    y = int(rows[ci])
                    band = sat[max(geo.HUD_H, y - 12):y, ci * (320 // 64):(ci + 1) * (320 // 64)]
                    if band.size:
                        mob_sats.append(float(np.median(band)))
            elif true_dist > 8.0 or e.room_id != room_id:
                pass

    # 방 벽(먼 배경, 문 아닌 방향) 채도도 같이 수집: 화면에서 바닥경계가
    # "열림(20m)"이 아니면서 3m 이상인 열들 = 십중팔구 벽.
    for ci in range(0, 64, 8):
        d = dists[ci]
        if 3.0 <= d <= 15.0:
            y = int(rows[ci])
            band = sat[max(geo.HUD_H, y - 12):y, ci * (320 // 64):(ci + 1) * (320 // 64)]
            if band.size:
                wall_sats.append(float(np.median(band)))

    action = [Action.MOVE_FORWARD, Action.TURN_RIGHT][i % 7 == 0]
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        break

env.close()

import numpy as np
wall_sats = np.array(wall_sats)
mob_sats = np.array(mob_sats)
print(f"wall samples: {len(wall_sats)}, mob samples: {len(mob_sats)}")
if len(wall_sats):
    print(f"wall sat: p50={np.percentile(wall_sats,50):.1f} p90={np.percentile(wall_sats,90):.1f} "
          f"p99={np.percentile(wall_sats,99):.1f} max={wall_sats.max():.1f}")
if len(mob_sats):
    print(f"mob  sat: p10={np.percentile(mob_sats,10):.1f} p50={np.percentile(mob_sats,50):.1f} "
          f"min={mob_sats.min():.1f}")
