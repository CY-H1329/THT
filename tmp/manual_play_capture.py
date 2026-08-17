"""examples/manual_play.py를 최대한 그대로 두고, 프레임 저장 한 줄만 추가.

사용법:
    python tmp/manual_play_capture.py [seed]

조작: w/s 전진/후진, a/d 좌/우 회전, SPACE 공격, n 무행동, q 종료(QA 시작),
      ESC 중단.

주의: macOS 입력 소스가 한글로 돼 있으면 IME가 키 입력을 변형시켜서
WASD가 안 먹힐 수 있습니다 — 실행 전 입력 소스를 영문(ABC/U.S.)으로
바꿔주세요.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore

from PIL import Image

from memory_fps_env.env import Action, MemoryFPSEnv


KEY_MAP = {
    ord("w"): Action.MOVE_FORWARD,
    ord("s"): Action.MOVE_BACK,
    ord("a"): Action.TURN_LEFT,
    ord("d"): Action.TURN_RIGHT,
    ord(" "): Action.ATTACK,
    ord("n"): Action.NO_OP,
    ord("q"): Action.END_EPISODE,
}

_WINDOW = "Memory-FPS (manual play)"
OUT_DIR = ROOT / "tmp" / "vlm_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _require_cv2() -> None:
    if cv2 is None:
        print("opencv-python is required for manual_play. "
              "Install with: pip install -e .[play]", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    _require_cv2()

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = MemoryFPSEnv(seed=seed)
    obs, info = env.reset()

    print("Controls: WASD move/turn, SPACE attack, N no-op, Q end, ESC abort")
    print("(입력 소스가 한글이면 안 먹힙니다 — 영문(ABC/U.S.)으로 바꿔주세요)")

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WINDOW, 640, 480)

    steps = 0
    saved = 0
    aborted = False
    terminated = False
    truncated = False

    _TICK_MS = 50
    try:
        while not (terminated or truncated):
            cv2.imshow(_WINDOW, obs[..., ::-1])
            key = cv2.waitKey(_TICK_MS) & 0xFF

            if key == 27:  # ESC
                aborted = True
                break

            action = KEY_MAP.get(key, Action.NO_OP)
            obs, _reward, terminated, truncated, info = env.step(int(action))
            steps += 1

            # 원본과의 차이: 매 action(스텝)마다 결과 프레임을 저장.
            saved += 1
            Image.fromarray(obs).save(OUT_DIR / f"seed{seed}_step{saved:05d}.png")
    finally:
        cv2.destroyAllWindows()
        env.close()

    if aborted:
        print(f"aborted by user after {steps} steps, phase={info.get('phase')}")
    else:
        print(f"finished in {steps} steps, terminated={terminated}, "
              f"truncated={truncated}, phase={info.get('phase')}")
    print(f"{saved} frames saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
