"""공격 없이 랜덤하게만 움직였을 때 평균 생존시간(시뮬레이션 초) 측정.

DFS 탐험 정책을 만들기 전에 baseline을 잡기 위한 스크립트. ATTACK,
END_EPISODE는 액션 후보에서 아예 빼서 "순수 방치 생존시간"을 잰다.
생존시간은 env 내부 상태를 직접 읽지 않고, HUD의 "T-Ns" 남은시간을
OCR로 읽어서(600 - 마지막으로 읽은 남은시간) 역산한다 — act()가 볼 수
있는 정보만 쓴다는 규칙을 그대로 지킴.

terminated=True / truncated=False 조합은 END_EPISODE를 절대 안 쓰므로
곧 "HP 0 사망"과 동일하고, truncated=True는 600초를 다 채운 생존을 뜻함.
"""

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.ocr import read_hud
from memory_fps_env.env import Action, MemoryFPSEnv

ACTIONS = [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT,
           Action.MOVE_BACK, Action.NO_OP]
WEIGHTS = [6, 2, 2, 1, 1]


def run_episode(seed: int, max_steps: int = 60_000, verbose_every: int = 0):
    rng = random.Random(seed)
    env = MemoryFPSEnv(seed=seed)
    obs, _info = env.reset()
    last_secs_remaining = 600
    steps = 0
    t0 = time.monotonic()
    terminated = truncated = False
    for i in range(1, max_steps + 1):
        steps = i
        a = rng.choices(ACTIONS, weights=WEIGHTS)[0]
        obs, _r, terminated, truncated, _info = env.step(int(a))
        if not terminated and not truncated:
            hud = read_hud(obs)
            if hud.ok and hud.seconds_remaining is not None:
                last_secs_remaining = hud.seconds_remaining
        if verbose_every and i % verbose_every == 0:
            print(f"    ...step={i} secs_remaining~{last_secs_remaining} "
                  f"wall_clock={time.monotonic()-t0:.1f}s")
        if terminated or truncated:
            break
    env.close()
    wall_clock = time.monotonic() - t0
    survived_seconds = 600 - last_secs_remaining
    return {
        "seed": seed, "steps": steps, "wall_clock_s": round(wall_clock, 1),
        "survived_seconds": survived_seconds,
        "died": bool(terminated), "full_survival": bool(truncated),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-steps", type=int, default=60_000)
    p.add_argument("--verbose-every", type=int, default=500)
    args = p.parse_args()

    results = []
    for seed in args.seeds:
        print(f"--- seed={seed} ---")
        r = run_episode(seed, max_steps=args.max_steps, verbose_every=args.verbose_every)
        print(f"  seed={r['seed']} steps={r['steps']} wall_clock={r['wall_clock_s']}s "
              f"survived={r['survived_seconds']}s died={r['died']} "
              f"full_survival={r['full_survival']}")
        results.append(r)

    if len(results) > 1:
        avg_survived = sum(r["survived_seconds"] for r in results) / len(results)
        death_rate = sum(r["died"] for r in results) / len(results)
        print(f"\n=== {len(results)} episodes: avg survived={avg_survived:.1f}s, "
              f"death_rate={death_rate:.0%} ===")
