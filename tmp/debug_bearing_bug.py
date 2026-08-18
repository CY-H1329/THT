"""seed=0에서 첫 FLEE(strike) 틱까지 재생한 뒤, enemy_bearing()의 각 밴드가
실제로 어떤 RGB를 샘플링하고 room_wall_color와 어떻게 비교되는지 원본
값으로 찍어본다. 조명 차이로 인한 오탐(false "center") 가설 검증용.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from agent.palette import nearest_color_name, WALL_PALETTE
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()
policy = ExplorerPolicy()

i = 0
while True:
    i += 1
    action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)
    if policy.state == "FLEE" and policy.flee_phase == "strike":
        break
    if term or trunc or i > 400:
        print("no strike reached")
        raise SystemExit

hud = read_hud(obs)
node = policy.scene.nodes.get(policy._canonicalize(hud.room_name))
room_wall_color = node.wall_color if node else None
print(f"step={i} room={hud.room_name!r} stored_wall_color={room_wall_color}")

h, w = obs.shape[0], obs.shape[1]
y0, y1 = int(h * 0.35), int(h * 0.75)
bands = {"left": (0.05, 0.35), "center": (0.35, 0.65), "right": (0.65, 0.95)}
for label, (fx0, fx1) in bands.items():
    x0, x1 = int(w * fx0), int(w * fx1)
    crop = obs[y0:y1, x0:x1, :]
    small = crop[::3, ::3, :].reshape(-1, 3)
    colors, counts = np.unique(small, axis=0, return_counts=True)
    mode_rgb = tuple(int(c) for c in colors[counts.argmax()])
    name = nearest_color_name(mode_rgb)
    # 저장된 wall_color의 팔레트 RGB와 raw 유클리드 거리도 같이.
    stored_rgb = dict(WALL_PALETTE).get(room_wall_color) if room_wall_color else None
    dist = None
    if stored_rgb is not None:
        dist = float(np.linalg.norm(np.array(mode_rgb) - np.array(stored_rgb)))
    print(f"  band={label:6s} mode_rgb={mode_rgb} nearest_name={name!s:12s} "
          f"dist_to_stored={dist}")

from PIL import Image
OUT = ROOT / "tmp" / "debug_frames" / f"bearing_bug_seed{SEED}_step{i}.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(obs).save(OUT)
print(f"saved frame: {OUT}")
