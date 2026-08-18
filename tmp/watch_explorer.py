"""ExplorerPolicy가 실제로 플레이하는 걸 화면으로 보는 도구(디버깅/시연용).

manual_play.py와 같은 cv2 창을 쓰되, 키보드 대신 ExplorerPolicy.step()이
액션을 낸다. 창을 보면서 진행 상황을 실시간으로 확인할 수 있다.

사용법: python tmp/watch_explorer.py [seed] [max_steps] [ms_per_frame]
  ESC: 중단
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import cv2  # type: ignore
except ImportError:
    print("opencv-python 필요: pip install -e .[play]", file=sys.stderr)
    sys.exit(1)

from agent.explorer import ExplorerPolicy
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
MS_PER_FRAME = int(sys.argv[3]) if len(sys.argv) > 3 else 30  # 낮을수록 빨리 재생

_WINDOW = "Memory-FPS (explorer watch)"

env = MemoryFPSEnv(seed=SEED)
obs, _info = env.reset()
policy = ExplorerPolicy()
action_names = {v: k for k, v in policy._Action.__members__.items()}

cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(_WINDOW, 640, 480)

print(f"seed={SEED} 시작. ESC로 중단.")

i = 0
term = trunc = False
prev_state = policy.state
hint_reported = False
had_key = False
for i in range(1, MAX_STEPS + 1):
    action = policy.step(obs)
    obs, _r, term, trunc, _info = env.step(action)

    if policy.state != prev_state:
        if policy.state == "HINT_CAPTURE":
            print(f"step={i:4d} >>> 잠긴 문 발견, 힌트 배너 확인하러 접근 중...")
        elif prev_state == "HINT_CAPTURE":
            print(f"step={i:4d} <<< 힌트 캡처 종료, SEEK 복귀")
        prev_state = policy.state
    if policy.scene.key_hint_text and not hint_reported:
        print(f"step={i:4d} *** 힌트 텍스트 확보: {policy.scene.key_hint_text!r} "
              f"(방={policy.scene.key_hint_room}, heading={policy.scene.key_hint_heading})")
        hint_reported = True

    # 열쇠 헌팅 로직 없이도 자연스러운 탐험 동선만으로 열쇠를 줍는지
    # 눈으로 확인하기 위한 계측(사용자 요청) — HUD의 [KEY] 표시가
    # False->True로 바뀌는 순간을 잡는다.
    hud_key_check = read_hud(obs)
    if hud_key_check.ok and hud_key_check.has_key and not had_key:
        print(f"step={i:4d} $$$ 열쇠 자연 습득! room={hud_key_check.room_name!r}")
        had_key = True

    bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    cv2.imshow(_WINDOW, bgr)
    key = cv2.waitKey(MS_PER_FRAME) & 0xFF
    if key == 27:  # ESC
        print("사용자가 중단함.")
        break

    if i % 20 == 0 and not (term or trunc):
        hud = read_hud(obs)
        print(f"step={i:4d} state={policy.state:9s} phase={policy.flee_phase!s:8s} "
              f"rooms={len(policy.scene.nodes)} hp={hud.hp}/{hud.hp_max} "
              f"room={hud.room_name!r} T-{hud.seconds_remaining}s "
              f"action={action_names.get(int(action), action)}")

    if term or trunc:
        print(f"\n=== 종료: step={i}, terminated={term}, truncated={trunc} ===")
        cv2.imshow(_WINDOW, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1500)
        break

env.close()
cv2.destroyAllWindows()
print(f"discovered rooms: {list(policy.scene.nodes.keys())}")
for name, node in policy.scene.nodes.items():
    print(f"  {name}: exits={node.exits} vlm_hint={node.vlm_door_hint}")
if policy.scene.key_hint_text:
    print(f"key hint: {policy.scene.key_hint_text!r} "
          f"(room={policy.scene.key_hint_room}, heading={policy.scene.key_hint_heading})")
