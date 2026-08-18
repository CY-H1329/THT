"""막힌 순간의 실제 프레임을 저장해서 "진짜 벽인지" 눈으로 확인하고,
그 시점에 center_clear/사이드 오프셋 판정이 프레임과 맞는지 로그로 남긴다.
가짜 시계로 결정론적 재현."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from agent import geometry as geo
from agent.explorer import ExplorerPolicy, _SEEK_CONE_HALF_DEG, _SEEK_SAFE_CLEARANCE, \
    _SIDESTEP_OFFSETS_DEG, _SIDESTEP_CONE_HALF_DEG
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
SAVE_DIR = ROOT / "tmp" / "debug_frames"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


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

saved = 0
for i in range(1, MAX_STEPS + 1):
    hud = read_hud(obs)
    ra_before = policy.recover_attempts
    action = policy.step(obs)
    ra_after = policy.recover_attempts

    if ra_after > 0 and saved < 20:
        profile = geo.depth_profile(obs)
        center = geo.cone_clearance(profile, 0.0, _SEEK_CONE_HALF_DEG)
        sides = {off: round(geo.cone_clearance(profile, off, _SIDESTEP_CONE_HALF_DEG), 2)
                 for off in _SIDESTEP_OFFSETS_DEG}
        print(f"i={i:4d} room={hud.room_name!r} heading={hud.heading} "
              f"target={policy.target_heading} bias={policy._nav_bias_deg:.1f} "
              f"recover={ra_before}->{ra_after} center_clear={round(center,2)} "
              f"(need>={_SEEK_SAFE_CLEARANCE}) sides={sides} action={action_names.get(int(action), action)}")
        Image.fromarray(obs).save(SAVE_DIR / f"wallcheck_seed{SEED}_step{i:04d}.png")
        saved += 1

    obs, _r, term, trunc, _info = env.step(action)
    if term or trunc:
        print(f"episode ended at {i}")
        break
    if saved >= 20 and ra_after == 0:
        # 이미 20장 모았고 지금은 안 막혔으면 종료(추가로 다른 막힘 지점까지 안 기다림).
        pass

print(f"saved {saved} frames to {SAVE_DIR}")
