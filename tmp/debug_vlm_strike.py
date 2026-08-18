"""strike 단계에서 VLM(locate_enemy)이 실제로 뭐라고 응답하는지, 그게 어떤
액션으로 매핑되는지 스텝별로 로그를 찍는다.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import explorer as explorer_mod
from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400

_orig_locate = explorer_mod.locate_enemy
_calls = []


def _traced_locate(frame, *a, **kw):
    r = _orig_locate(frame, *a, **kw)
    _calls.append(r)
    return r


explorer_mod.locate_enemy = _traced_locate

env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()
policy = ExplorerPolicy()
action_names = {v: k for k, v in policy._Action.__members__.items()}

term = trunc = False
i = 0
for i in range(1, MAX_STEPS + 1):
    n_before = len(_calls)
    state = policy.state
    phase = policy.flee_phase
    action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)
    hud = read_hud(obs) if not (term or trunc) else None

    if len(_calls) > n_before:
        r = _calls[-1]
        print(f"step={i:4d} phase={phase!s:10s} vlm_ok={r.ok!s:5} data={r.data} "
              f"err={r.error} action={action_names.get(int(action), action)!s:14s} "
              f"hp={hud.hp if hud and hud.ok else '?'}")
    elif state == "FLEE":
        print(f"step={i:4d} phase={phase!s:10s} (no vlm call) "
              f"action={action_names.get(int(action), action)!s:14s} "
              f"hp={hud.hp if hud and hud.ok else '?'}")

    if term or trunc:
        break

env.close()
print(f"\n=== ended step={i} terminated={term} truncated={trunc} total_vlm_calls={len(_calls)} ===")
