import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import explorer as explorer_mod
from agent.explorer import ExplorerPolicy
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

_orig = explorer_mod.locate_door
calls = []


def _traced(frame, *a, **kw):
    r = _orig(frame, *a, **kw)
    calls.append(r)
    print(f"  [locate_door call #{len(calls)}] ok={r.ok} data={r.data} err={r.error}")
    return r


explorer_mod.locate_door = _traced

env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()
policy = ExplorerPolicy()

term = trunc = False
i = 0
for i in range(1, MAX_STEPS + 1):
    action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        break

env.close()
print(f"\n=== ended step={i} terminated={term} truncated={trunc} door_checks={len(calls)} ===")
print(f"rooms discovered: {list(policy.scene.nodes.keys())}")
