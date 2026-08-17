import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()
policy = ExplorerPolicy()

for i in range(1, MAX_STEPS + 1):
    action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)
    if i % 100 == 0:
        hud = read_hud(obs)
        print(f"step={i:4d} state={policy.state:8s} rooms_known={len(policy.scene.nodes)} "
              f"stack_depth={len(policy.scene.stack)} hp={hud.hp}/{hud.hp_max} "
              f"room={hud.room_name!r} T-{hud.seconds_remaining}s")
    if term or trunc:
        print(f"episode ended at step {i} (terminated={term}, truncated={trunc})")
        break
else:
    print(f"hit MAX_STEPS={MAX_STEPS} without ending")

env.close()
print(f"\n=== seed={SEED} summary ===")
print(f"rooms discovered: {len(policy.scene.nodes)}")
for name, node in policy.scene.nodes.items():
    print(f"  {name!r}: wall_color={node.wall_color} done={node.done} exits={node.exits}")
