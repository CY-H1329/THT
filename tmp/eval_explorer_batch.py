import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEEDS = [0, 1, 7, 42]
MAX_STEPS = 1500

results = []
for seed in SEEDS:
    t0 = time.monotonic()
    env = MemoryFPSEnv(seed=seed)
    obs, _info = env.reset()
    policy = ExplorerPolicy()
    last_secs = 600
    i = 0
    term = trunc = False
    for i in range(1, MAX_STEPS + 1):
        action = policy.step(obs)
        obs, _r, term, trunc, _info = env.step(action)
        if not (term or trunc):
            hud = read_hud(obs)
            if hud.ok and hud.seconds_remaining is not None:
                last_secs = hud.seconds_remaining
        if term or trunc:
            break
    env.close()
    survived = 600 - last_secs
    wall = time.monotonic() - t0
    r = {"seed": seed, "steps": i, "survived_s": survived, "died": bool(term),
         "rooms": len(policy.scene.nodes), "wall_s": round(wall, 1)}
    results.append(r)
    print(r, flush=True)

print("\n=== summary ===")
avg_survived = sum(r["survived_s"] for r in results) / len(results)
avg_rooms = sum(r["rooms"] for r in results) / len(results)
death_rate = sum(r["died"] for r in results) / len(results)
print(f"avg_survived={avg_survived:.1f}s avg_rooms={avg_rooms:.1f} death_rate={death_rate:.0%}")
