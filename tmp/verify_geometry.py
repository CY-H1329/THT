"""agent/geometry.py의 floor_boundary() 거리 역산이 실제와 맞는지 검증한다.

방법: 여러 지점에서 정면(bearing≈0) 거리 추정치를 구하고, 그 방향으로
MOVE_FORWARD를 계속 반복해서 실제로 막힐 때까지 이동한 거리(env 내부
좌표로 실측, 디버깅 전용)와 비교한다. 벽이든 오브젝트든 상관없이 "정면에
뭔가 있다고 예측한 거리"와 "실제로 거기까지 가서 막혔는지"를 직접 대조.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import geometry as geo
from memory_fps_env.env import Action, MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 15


def agent_xz():
    x, _, z = env._world.agent.pos
    return float(x), float(z)


env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()

results = []
trial = 0
step_budget = 4000
steps_used = 0

while trial < N_TRIALS and steps_used < step_budget:
    profile = geo.depth_profile(obs, n_cols=40)
    predicted = geo.cone_clearance(profile, center_deg=0.0, half_width_deg=6.0)
    if predicted >= geo.MAX_RANGE:
        # 정면이 활짝 열려 있으면(20m 넘게) 이 회로에서는 스킵하고 회전.
        obs, _r, term, trunc, _info = env.step(Action.TURN_RIGHT)
        steps_used += 1
        if term or trunc:
            break
        continue

    start = agent_xz()
    moved_total = 0.0
    blocked = False
    for _ in range(200):
        before = agent_xz()
        obs, _r, term, trunc, _info = env.step(Action.MOVE_FORWARD)
        steps_used += 1
        after = agent_xz()
        d = math.hypot(after[0] - before[0], after[1] - before[1])
        moved_total += d
        if term or trunc:
            blocked = True
            break
        if d < 0.02:  # 실제로 거의 안 움직임 = 막힘(ground truth)
            blocked = True
            break
        if moved_total > predicted + 3.0:  # 예측보다 한참 더 갔으면 이상함 방지
            break

    actual = math.hypot(agent_xz()[0] - start[0], agent_xz()[1] - start[1])
    err = actual - predicted
    results.append((predicted, actual, err, blocked))
    print(f"trial {trial}: predicted={predicted:5.2f}m actual_moved={actual:5.2f}m "
          f"error={err:+5.2f}m blocked={blocked}")
    trial += 1

    # 다음 트라이얼을 위해 회전해서 다른 방향을 본다.
    for _ in range(6):
        obs, _r, term, trunc, _info = env.step(Action.TURN_RIGHT)
        steps_used += 1
        if term or trunc:
            break
    if term or trunc:
        break

env.close()

if results:
    errs = [abs(r[2]) for r in results]
    print(f"\n=== {len(results)}개 트라이얼, 평균 절대오차={sum(errs)/len(errs):.3f}m, "
          f"최대오차={max(errs):.3f}m ===")
else:
    print("트라이얼을 못 채움(스텝 예산 소진 또는 에피소드 조기 종료).")
