"""탐색 성능 측정용 개발 하네스 (agent/ 밖, 평가에는 포함되지 않음).

에이전트를 실제 env에 물려 돌리면서 env 내부 정답(ground truth)과 비교한다:
방문한 방 수 / 에이전트가 만든 지도의 정확도 / 충돌·정체 횟수 / 생존.

    python tmp/run_explore.py 0 42          # 시드 지정
    python tmp/run_explore.py --limit 60 0  # 60초(시뮬)만 돌려 빠르게 확인
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import Agent                       # noqa: E402
from agent import mapper as M                 # noqa: E402
from memory_fps_env.env import MemoryFPSEnv   # noqa: E402


def run(seed: int, limit_seconds: float, verbose: bool = False) -> dict:
    env = MemoryFPSEnv(seed=seed)
    obs, _ = env.reset()
    agent = Agent()
    graph = env._graph
    truth_names = {r.id: r.name for r in graph.rooms}
    visited_truth = set()
    t0 = time.time()
    steps = 0
    done = False
    info = {}
    while not done:
        rid = env._world.current_room_id(graph)
        if rid is not None:
            visited_truth.add(rid)
        action = agent.act(obs)
        obs, _r, term, trunc, info = env.step(int(action))
        steps += 1
        done = term or trunc
        if time.time() - t0 > limit_seconds:
            break

    ex = agent.explorer
    # 에이전트 지도 ↔ 정답 대조
    name_to_id = {n: i for i, n in truth_names.items()}
    mapped, correct_edges, wrong_edges = {}, 0, 0
    for cell, room in ex.map.rooms.items():
        rid = next((i for n, i in name_to_id.items()
                    if M.names_match(n, room.key)), None)
        mapped[room.key] = rid
    for cell, room in ex.map.rooms.items():
        for d in room.open_dirs():
            dgx, dgz = M.DIR_DELTA[d]
            nb = ex.map.rooms.get((cell[0] + dgx, cell[1] + dgz))
            if nb is None:
                continue
            a, b = mapped.get(room.key), mapped.get(nb.key)
            if a is None or b is None:
                continue
            if frozenset({a, b}) in env._world._materialized_edges:
                correct_edges += 1
            else:
                wrong_edges += 1

    out = {
        "seed": seed,
        "rooms_total": len(graph.rooms),
        "rooms_visited_truth": len(visited_truth),
        "rooms_in_agent_map": len(ex.map.rooms),
        "rooms_scanned": sum(1 for r in ex.map.rooms.values() if r.scanned),
        "map_named_ok": sum(1 for v in mapped.values() if v is not None),
        "edges_correct": correct_edges // 2,
        "edges_wrong": wrong_edges // 2,
        "cell_size": ex.map.cell_size,
        "true_cell_size": graph.rooms[0].size[0],
        "steps": steps,
        "wallclock_s": round(time.time() - t0, 1),
        "sim_seconds": round(env._sim_time, 1),
        "hp": f"{env._world.agent_hp}/{env._world.agent_hp_max}",
        "phase": info.get("phase"),
        "blocked": ex.stats["blocked"],
        "attacks": ex.stats["attacks"],
        "scans": ex.stats["scans"],
        "transitions": ex.stats["transitions"],
        "state": ex.state,
        "state_steps": dict(sorted(ex.state_steps.items(), key=lambda kv: -kv[1])),
    }
    if verbose:
        out["map"] = ex.map.summary()
        out["truth"] = {
            "spawn": graph.spawn_room_id,
            "rooms": {r.id: {"name": r.name, "origin": list(r.origin),
                             "neighbors": graph.neighbors(r.id)}
                      for r in graph.rooms},
        }
    env.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--limit", type=float, default=120.0,
                    help="에피소드당 최대 실행 시간(초, 실시간)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    results = []
    for seed in (args.seeds or [0]):
        r = run(seed, args.limit, args.verbose)
        results.append(r)
        print(f"seed {r['seed']:5d} | visited {r['rooms_visited_truth']}/{r['rooms_total']}"
              f" | mapped {r['rooms_in_agent_map']} (scanned {r['rooms_scanned']},"
              f" named_ok {r['map_named_ok']})"
              f" | edges {r['edges_correct']}✓/{r['edges_wrong']}✗"
              f" | cell {r['cell_size']}/{r['true_cell_size']}"
              f" | steps {r['steps']} blocked {r['blocked']}"
              f" | hp {r['hp']} | {r['wallclock_s']}s\n"
              f"            states {r['state_steps']} attacks {r['attacks']} scans {r['scans']}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
