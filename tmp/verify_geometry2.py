"""geometry.py의 거리 공식을 적(enemy) ground-truth 좌표로 검증한다.

벽 검증(verify_geometry.py)은 MOVE_FORWARD가 벽을 따라 미끄러지는
경우가 있어 오차가 컸다. 적은 단일 점 목표라 이 문제가 없다 — 에이전트
좌표와 적 좌표(둘 다 env._world/env._enemies에서 디버깅 전용으로 읽음)로
실제 직선거리와 방위각을 계산하고, floor_boundary가 그 방위각의 열에서
예측한 거리와 비교한다. 적이 시야 안(±35도, 20m 이내)에 있을 때만 비교.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import geometry as geo
from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import Action, MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
policy = ExplorerPolicy()


def agent_xz():
    x, _, z = env._world.agent.pos
    return float(x), float(z)


env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()

results = []
for i in range(1, MAX_STEPS + 1):
    hud = read_hud(obs)
    ax, az = agent_xz()
    heading = float(hud.heading) if hud.ok and hud.heading is not None else None

    agent_room = env._world.current_room_id(env._graph)
    same_room_enemy = heading is not None and any(
        e.is_alive() and e.room_id == agent_room for e in env._enemies)

    if heading is not None:
        for e in env._enemies:
            if not e.is_alive() or e.room_id != agent_room:
                continue  # 다른 방이면 벽에 가려질 수 있어 비교 제외
            dx = e.pos[0] - ax
            dz = e.pos[1] - az
            true_dist = math.hypot(dx, dz)
            # world (dx,dz) -> heading 절대각 (geometry.heading_to_vec의 역).
            abs_bearing = math.degrees(math.atan2(-dz, dx)) % 360.0
            rel = geo.wrap180(abs_bearing - heading)
            if abs(rel) <= 35.0 and true_dist <= 18.0:
                profile = geo.depth_profile(obs, n_cols=60)
                pred = geo.cone_clearance(profile, center_deg=rel, half_width_deg=4.0)
                results.append((true_dist, pred, pred - true_dist))

    if same_room_enemy:
        # 같은 방에 살아있는 적이 있으면, 검증 샘플을 늘리려고 정책 대신
        # 제자리 회전으로 훑는다(순수 검증용 오버라이드).
        action = Action.TURN_RIGHT
    else:
        action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        break

env.close()

if results:
    errs = [abs(r[2]) for r in results]
    print(f"샘플 {len(results)}개")
    for t, p, e in results[:20]:
        print(f"  true={t:5.2f}m pred={p:5.2f}m err={e:+5.2f}m")
    print(f"평균 절대오차={sum(errs)/len(errs):.3f}m 최대오차={max(errs):.3f}m "
          f"(단, pred가 적이 아니라 그 뒤 벽까지 거리를 잰 경우 오차가 커질 수 있음 "
          f"— 적이 바닥 경계까지 안 닿아있으면 floor_boundary가 적을 못 봄)")
else:
    print("적을 시야 안에서 못 잡음 — 더 오래 돌려보세요.")
