"""적 처치/집계 성능을 공개 시드 전체에서 재는 하네스 (개발용).

에이전트가 주장한 킬 수를 env 정답(`_enemies[i].is_alive()`)과 대조한다.
QA에서 "적을 몇 마리 죽였나"를 묻기 때문에 실제 처치 수만큼이나 **집계
정확도**가 중요하다.

    python tmp/bench_enemies.py                     # 공개 시드 10개, 각 40초
    python tmp/bench_enemies.py --limit 60          # 더 길게
    python tmp/bench_enemies.py --seeds 0 42 99
    python tmp/bench_enemies.py --compare <경로>    # 다른 agent/ 버전과 비교

`--compare`에는 `agent/` 패키지를 담고 있는 디렉터리를 준다. 예를 들어
커밋 전 버전과 비교하려면:

    mkdir -p /tmp/base && git archive HEAD agent | tar -x -C /tmp/base
    python tmp/bench_enemies.py --compare /tmp/base

주의 1: 시드마다 **별도 프로세스**로 돌린다. README가 권하는 격리 방식이고,
Windows에서는 프로세스당 env 하나가 사실상 강제다.

주의 2: env의 시뮬 시계는 step() 사이 실제 경과 시간으로 흐른다. 그래서
여러 판을 **동시에** 돌리면 CPU 경합 때문에 스텝당 시뮬 시간이 늘어나
적이 더 많이 움직인다 — 결과가 왜곡되므로 순차로만 돌린다.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_SEEDS = [0, 1, 7, 42, 99, 137, 256, 512, 1024, 1337]

# 자식 프로세스로 실행되는 본체. AGENT_ROOT를 sys.path 앞에 꽂아 비교
# 대상 버전을 갈아 끼운다.
CHILD = r'''
import json, os, sys, time
sys.path.insert(0, os.environ["AGENT_ROOT"])
sys.path.append(os.environ["ENV_ROOT"])
from agent import Agent
from memory_fps_env.env import MemoryFPSEnv

seed, limit = int(sys.argv[1]), float(sys.argv[2])
env = MemoryFPSEnv(seed=seed); obs, _ = env.reset(); ag = Agent()
graph = env._graph; visited = set()
t0 = time.time(); steps = 0; done = False
while not done:
    rid = env._world.current_room_id(graph)
    if rid is not None: visited.add(rid)
    obs, _r, term, trunc, _i = env.step(int(ag.act(obs)))
    steps += 1; done = term or trunc
    if time.time() - t0 > limit: break
ex = ag.explorer
sys.stdout.write("@@RESULT@@" + json.dumps({
    "seed": seed, "steps": steps,
    "rooms_visited": len(visited), "rooms_total": len(graph.rooms),
    "hp": env._world.agent_hp, "hp_max": env._world.agent_hp_max,
    "died": env._world.agent_hp <= 0,
    "kills_truth": sum(1 for e in env._enemies if not e.is_alive()),
    "enemies_total": len(env._enemies),
    "kills_claimed": ex.stats.get("kills", 0),
    "tracks": len(getattr(ex.map, "tracks", [])),
    "attacks": ex.stats["attacks"],
}) + "\n")
'''


def run_seed(agent_root: Path, seed: int, limit: float) -> dict | None:
    # 부모 환경을 그대로 물려준다. DISPLAY/XAUTHORITY 같은 GL 관련 변수를
    # 하나라도 빠뜨리면 pyglet이 컨텍스트를 못 열고 조용히 죽는다.
    env = dict(os.environ)
    env["AGENT_ROOT"] = str(agent_root)
    env["ENV_ROOT"] = str(ROOT)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", CHILD, str(seed), str(limit)],
            capture_output=True, text=True, env=env, timeout=limit + 180)
    except subprocess.TimeoutExpired:
        print("       (시간 초과)")
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    for t in tail:
        print(f"       ! {t}")
    return None


def bench(agent_root: Path, seeds, limit, label) -> dict:
    rows = {}
    for s in seeds:
        r = run_seed(agent_root, s, limit)
        rows[s] = r
        mark = "timeout/err" if r is None else (
            f"kills {r['kills_claimed']}/{r['kills_truth']} "
            f"hp {r['hp']}/{r['hp_max']}")
        print(f"  [{label}] seed {s:<5} {mark}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=PUBLIC_SEEDS)
    ap.add_argument("--limit", type=float, default=40.0,
                    help="에피소드당 최대 실행 시간(실시간 초)")
    ap.add_argument("--compare", type=str, default="",
                    help="비교할 agent/ 패키지를 담은 디렉터리")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    print(f"현재 버전 ({ROOT})")
    cur = bench(ROOT, args.seeds, args.limit, "current")
    base = {}
    if args.compare:
        print(f"\n비교 버전 ({args.compare})")
        base = bench(Path(args.compare), args.seeds, args.limit, "compare")

    print("\n" + "=" * 74)
    hdr = f"{'seed':>5} | {'rooms':>7} | {'hp':>9} | {'died':>5} | {'real kills':>11} | {'claimed':>7}"
    print(hdr + "\n" + "-" * 74)
    agg = {"died": 0, "truth": 0, "claim": 0, "err": 0, "rooms": 0, "n": 0}
    bagg = {"died": 0, "truth": 0, "rooms": 0, "n": 0}
    for s in args.seeds:
        C = cur.get(s)
        if C is None:
            print(f"{s:>5} | (실행 실패)")
            continue
        agg["n"] += 1
        agg["died"] += C["died"]; agg["truth"] += C["kills_truth"]
        agg["claim"] += C["kills_claimed"]; agg["rooms"] += C["rooms_visited"]
        agg["err"] += abs(C["kills_claimed"] - C["kills_truth"])
        B = base.get(s)
        if B:
            bagg["n"] += 1; bagg["died"] += B["died"]
            bagg["truth"] += B["kills_truth"]; bagg["rooms"] += B["rooms_visited"]
        print(f"{s:>5} | {C['rooms_visited']:>3}/{C['rooms_total']:<3} |"
              f" {C['hp']:>4}/{C['hp_max']:<4} | {str(C['died']):>5} |"
              f" {C['kills_truth']:>4}/{C['enemies_total']:<6} |"
              f" {C['kills_claimed']:>7}")
    print("-" * 74)
    print(f"현재:  죽음 {agg['died']}/{agg['n']} | 방문한 방 {agg['rooms']} | "
          f"실제 처치 {agg['truth']} | 주장 {agg['claim']} | 집계 오차 합 {agg['err']}")
    if base:
        print(f"비교:  죽음 {bagg['died']}/{bagg['n']} | 방문한 방 {bagg['rooms']} | "
              f"실제 처치 {bagg['truth']}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"current": cur, "compare": base}, indent=2),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
