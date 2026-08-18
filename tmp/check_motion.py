"""움직임 기반 적 탐지가 기대는 사실들을 직접 검증한다 (개발용).

agent/motion.py의 설계는 네 가지 측정 사실 위에 서 있다. 이 스크립트는
그 네 가지를 매번 다시 재서 PASS/FAIL로 찍는다. 다른 GPU/드라이버로
옮겼을 때 가장 먼저 돌려봐야 하는 스크립트다 — 특히 1번(렌더러 결정성)이
깨지면 탐지기의 전제가 통째로 무너진다.

    python tmp/check_motion.py
    python tmp/check_motion.py --seed 7

시뮬 시계를 가짜 시간원으로 돌리므로 실제로 기다리지 않는다(몇 초면 끝).
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import motion as MOT                      # noqa: E402
from memory_fps_env.env import Action, MemoryFPSEnv  # noqa: E402

DT = 0.012          # 실측 스텝당 시뮬 시간(초)
results = []


class Clock:
    """가짜 단조 시계. env는 step() 사이 실제 경과 시간으로 시뮬을 굴린다."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def check(name, ok, detail):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")


def diff_px(a, b, thresh=MOT.DIFF_THRESH):
    """HUD 아래 영역에서 채널 최대차가 thresh 이상인 픽셀 수."""
    from agent import geometry as geo
    d = np.abs(a[geo.HUD_H:].astype(int) - b[geo.HUD_H:].astype(int)).max(axis=2)
    return int((d >= thresh).sum()), int(d.max())


def face(env, e, dist):
    """적에게서 dist만큼 동쪽에 서서 적을 바라보게 한다."""
    w = env._world
    w.agent.pos = np.array([e.pos[0] + dist, 0.0, e.pos[1]])
    w.agent.dir = math.pi
    return w.render_frame()


def test_determinism(seed):
    """1. 살아 있는 적이 없으면 프레임이 바이트 단위로 같아야 한다."""
    clk = Clock()
    env = MemoryFPSEnv(seed=seed, time_source=clk)
    obs, _ = env.reset()
    for e in env._enemies:
        e.hp = 0                       # 월드를 완전히 정지시킨다
    base = env.step(Action.NO_OP)[0].copy()
    for _ in range(300):
        clk.t += DT
        obs, *_ = env.step(Action.NO_OP)
    n, mx = diff_px(base, obs)
    check("렌더러 결정성 (적 없음, 3.6 시뮬초)", n == 0 and mx == 0,
          f"changed_px={n}, max_delta={mx}  (기대: 0, 0)")
    env.close()


def test_attack_is_stationary(seed):
    """2. ATTACK은 에이전트를 움직이지 않아야 한다(정지 쌍의 근거)."""
    clk = Clock()
    env = MemoryFPSEnv(seed=seed, time_source=clk)
    env.reset()
    for e in env._enemies:
        e.hp = 0
    p0 = np.array(env._world.agent.pos, dtype=float)
    d0 = float(env._world.agent.dir)
    env.step(Action.NO_OP)
    base = env.step(Action.ATTACK)[0].copy()
    for _ in range(50):
        clk.t += DT
        obs, *_ = env.step(Action.ATTACK)
    moved = float(np.linalg.norm(np.array(env._world.agent.pos, dtype=float) - p0))
    turned = abs(float(env._world.agent.dir) - d0)
    n, _ = diff_px(base, obs)
    check("ATTACK 50회가 위치·시야를 안 바꾼다",
          moved < 1e-9 and turned < 1e-9 and n == 0,
          f"moved={moved:.3f}m, turned={turned:.3f}rad, changed_px={n}")
    env.close()


def test_sensitivity(seed):
    """3. 짧게 멈추기만 해도 움직이는 적이 잡혀야 한다."""
    rows = []
    ok = True
    for dist in (3, 4, 6, 8):
        clk = Clock()
        env = MemoryFPSEnv(seed=seed, time_source=clk)
        env.reset()
        e = env._enemies[0]
        face(env, e, dist)
        base = env.step(Action.NO_OP)[0].copy()
        for _ in range(4):             # 4스텝 = 0.05 시뮬초
            clk.t += DT
            obs, *_ = env.step(Action.NO_OP)
        n, _ = diff_px(base, obs)
        rows.append(f"{dist}m:{n}px")
        if n < MOT.MIN_PIXELS:
            ok = False
        env.close()
    check("4스텝(0.05 시뮬초) 유지로 3~8m 적 검출", ok, "  ".join(rows))


def test_freeze_blindspot(seed):
    """4. 1.5m 안의 적은 멈춘다 — 알려진 사각지대가 실제로 존재하는지."""
    clk = Clock()
    env = MemoryFPSEnv(seed=seed, time_source=clk)
    env.reset()
    e = env._enemies[0]
    face(env, e, 1.2)
    env.step(Action.NO_OP)
    p0 = e.pos
    for _ in range(200):
        clk.t += DT
        env.step(Action.NO_OP)
    moved = math.hypot(e.pos[0] - p0[0], e.pos[1] - p0[1])
    check("1.5m 안의 적은 attack 상태로 정지 (알려진 사각지대)",
          e.state == "attack" and moved < 1e-6,
          f"state={e.state}, moved={moved:.3f}m — 이 구간은 트랙 위치와 "
          f"HP 감소 신호가 대신 메운다")
    env.close()


def test_detector_unit():
    """5. MotionDetector가 포즈가 바뀌면 차분을 거부하는지 (합성 프레임)."""
    d = MOT.MotionDetector()
    blank = np.zeros((240, 320, 3), np.uint8)
    moving = blank.copy()
    moving[100:150, 150:175] = 200
    d.observe(blank, (0.0, 0.0, 0.0), True, False)
    got = d.observe(moving, (0.0, 0.0, 0.0), True, False)
    same_pose_ok = len(got) == 1 and d.paired
    d2 = MOT.MotionDetector()
    d2.observe(blank, (0.0, 0.0, 0.0), True, False)
    moved_pose = d2.observe(moving, (1.0, 0.0, 0.0), True, False)
    d3 = MOT.MotionDetector()
    d3.observe(blank, (0.0, 0.0, 0.0), True, False)
    turning = d3.observe(moving, (0.0, 0.0, 0.0), False, False)
    check("포즈 동일할 때만 차분한다",
          same_pose_ok and not moved_pose and not d2.paired and not turning,
          f"same_pose={len(got)}블롭, moved_pose={len(moved_pose)}블롭, "
          f"non_stationary={len(turning)}블롭")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(f"움직임 탐지 전제 검증 (seed {args.seed})\n")
    test_determinism(args.seed)
    test_attack_is_stationary(args.seed)
    test_sensitivity(args.seed)
    test_freeze_blindspot(args.seed)
    test_detector_unit()
    bad = results.count(False)
    print(f"\n{len(results) - bad}/{len(results)} PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
