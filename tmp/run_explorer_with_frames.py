"""탐험 정책을 실제로 돌리면서, 50스텝마다 + HP 변화 시점 + 종료 시점의
프레임을 tmp/debug_frames/run_*.png 로 저장한다. 죽었는지(HP 0) /
시간초과로 끝났는지(600초)를 명확히 구분해서 출력한다.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

OUT_DIR = ROOT / "tmp" / "debug_frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()
policy = ExplorerPolicy()
Image.fromarray(obs).save(OUT_DIR / f"run_seed{SEED}_step000.png")

last_hp = None
term = trunc = False
i = 0
for i in range(1, MAX_STEPS + 1):
    state_before = policy.state
    phase_before = policy.flee_phase
    action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)
    hud = read_hud(obs) if not (term or trunc) else None

    if term or trunc:
        Image.fromarray(obs).save(OUT_DIR / f"run_seed{SEED}_step{i:04d}_END.png")
        break

    if hud.ok and hud.hp is not None:
        if last_hp is not None and hud.hp < last_hp:
            Image.fromarray(obs).save(OUT_DIR / f"run_seed{SEED}_step{i:04d}_hit_hp{hud.hp}.png")
        last_hp = hud.hp

    # FLEE 상태 진행 전체를 프레임 단위로 저장 (retreat/turnaround/strike 시퀀스 검증용)
    if state_before == "FLEE" or policy.state == "FLEE":
        phase = phase_before or policy.flee_phase or "none"
        Image.fromarray(obs).save(
            OUT_DIR / f"run_seed{SEED}_step{i:04d}_FLEE_{phase}_hp{hud.hp}.png"
        )

    if i % 50 == 0:
        Image.fromarray(obs).save(OUT_DIR / f"run_seed{SEED}_step{i:04d}.png")
        print(f"step={i:4d} state={policy.state:8s} rooms={len(policy.scene.nodes)} "
              f"hp={hud.hp}/{hud.hp_max} room={hud.room_name!r} T-{hud.seconds_remaining}s")

env.close()
print(f"\n=== 종료: step={i}, terminated={term}, truncated={trunc}, last_hp={last_hp} ===")
if term:
    print("-> HP가 0이 되어 사망 (END_EPISODE는 이 정책이 안 씀, truncated도 아니므로 사망 확정)")
elif trunc:
    print("-> 600초 시간 다 채우고 생존 (truncated=시간초과)")
print(f"discovered rooms: {list(policy.scene.nodes.keys())}")
