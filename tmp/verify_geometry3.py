"""geometry.py의 거리 공식을 방 경계(axis-aligned box, env._world.rooms)로
검증한다 — 이동/충돌 미끄러짐 없이 순수 기하로 정답을 계산하는 가장 깨끗한
방법. 에이전트를 이동시키지 않고 제자리에서 회전만 하며 여러 각도를
샘플링한다.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import geometry as geo
from agent.ocr import read_hud
from memory_fps_env.env import Action, MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def ray_box_distance(ax, az, fwd_x, fwd_z, min_x, max_x, min_z, max_z):
    """정확한 정답: (ax,az)에서 (fwd_x,fwd_z) 방향으로 쐈을 때 방 경계까지
    거리(문/오브젝트는 무시 — 순수 벽 상자 기준)."""
    ts = []
    if abs(fwd_x) > 1e-9:
        for xb in (min_x, max_x):
            t = (xb - ax) / fwd_x
            if t > 0.05:
                z_at = az + t * fwd_z
                if min_z - 0.05 <= z_at <= max_z + 0.05:
                    ts.append(t)
    if abs(fwd_z) > 1e-9:
        for zb in (min_z, max_z):
            t = (zb - az) / fwd_z
            if t > 0.05:
                x_at = ax + t * fwd_x
                if min_x - 0.05 <= x_at <= max_x + 0.05:
                    ts.append(t)
    return min(ts) if ts else None


env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()

results = []
for trial in range(24):
    hud = read_hud(obs)
    x, _, z = env._world.agent.pos
    heading = float(hud.heading) if hud.ok and hud.heading is not None else None
    room_id = env._world.current_room_id(env._graph)

    if heading is not None and room_id is not None:
        room = env._world.rooms[room_id]
        r = math.radians(heading)
        fwd_x, fwd_z = math.cos(r), -math.sin(r)
        true_d = ray_box_distance(x, z, fwd_x, fwd_z,
                                   room.min_x, room.max_x, room.min_z, room.max_z)
        if true_d is not None and true_d <= geo.MAX_RANGE:
            profile = geo.depth_profile(obs, n_cols=60)
            pred = geo.cone_clearance(profile, center_deg=0.0, half_width_deg=3.0)
            results.append((true_d, pred, pred - true_d, heading, room_id))
            print(f"trial {trial}: room={room_id} heading={heading:5.1f} "
                  f"true={true_d:5.2f}m pred={pred:5.2f}m err={pred-true_d:+5.2f}m")

    for _ in range(2):  # 15도 x 2 = 30도씩 회전하며 샘플
        obs, _r, term, trunc, _info = env.step(Action.TURN_RIGHT)
        if term or trunc:
            break
    if term or trunc:
        break

env.close()

if results:
    errs = [abs(r[2]) for r in results]
    print(f"\n=== {len(results)}개, 평균절대오차={sum(errs)/len(errs):.3f}m "
          f"최대오차={max(errs):.3f}m ===")
else:
    print("샘플 없음")
