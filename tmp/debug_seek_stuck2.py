import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import geometry as geo
from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 900
STUCK_AFTER = int(sys.argv[3]) if len(sys.argv) > 3 else 150

def make_fake_clock(dt=0.05):
    state = {"t": 0.0}

    def clock():
        state["t"] += dt
        return state["t"]
    return clock


env = MemoryFPSEnv(seed=SEED, time_source=make_fake_clock())
obs, _info = env.reset()
policy = ExplorerPolicy()
action_names = {v: k for k, v in policy._Action.__members__.items()}

hist = deque(maxlen=40)
last_room = None
same_room_streak = 0
dumped = False

for i in range(1, MAX_STEPS + 1):
    hud = read_hud(obs)
    clearance = geo.cone_clearance(geo.depth_profile(obs), 0.0, 15.0)
    action = policy.step(obs)
    hist.append((i, policy.state, hud.room_name, hud.heading, policy.target_heading,
                 policy.recover_attempts, round(clearance, 2),
                 action_names.get(int(action), action)))

    if hud.room_name == last_room:
        same_room_streak += 1
    else:
        same_room_streak = 0
        last_room = hud.room_name

    if same_room_streak > STUCK_AFTER and not dumped:
        print(f"=== STUCK detected at i={i}, room={hud.room_name!r} unchanged for "
              f"{same_room_streak} steps — last {len(hist)} entries: ===")
        for h in hist:
            print(h)
        dumped = True
    if dumped:
        x, _, z = env._world.agent.pos
        room_id = env._world.current_room_id(env._graph)
        nearby = []
        for e in env._enemies:
            if not e.is_alive():
                continue
            d = ((e.pos[0] - x) ** 2 + (e.pos[1] - z) ** 2) ** 0.5
            if d < 3.0:
                nearby.append((e.id, round(d, 2), e.appearance.get("skin_color"),
                               e.appearance.get("shirt_color")))
        print(hist[-1], f"pos=({x:.3f},{z:.3f}) nearby={nearby}")
        if i % 30 == 0:
            from PIL import Image
            Image.fromarray(obs).save(f"/tmp/stuck_frame_{i}.png")

    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        print(f"episode ended at i={i}")
        break

if not dumped:
    print("no stuck window detected in this run")
