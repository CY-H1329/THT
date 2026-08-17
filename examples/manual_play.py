# examples/manual_play.py
"""Keyboard control for debugging the Memory-FPS env.

Opens an OpenCV window showing the env's RGB observation and ticks the env
~20 times per second (50 ms per tick), so enemies keep moving and the
screen keeps refreshing even when no key is pressed. Every tick is one
env.step(): a real action if you held a key down during the window, or
NO_OP otherwise.

Controls:
    w / s / a / d : MOVE_FORWARD / MOVE_BACK / TURN_LEFT / TURN_RIGHT
    SPACE         : ATTACK
    n             : NO_OP
    q             : END_EPISODE
    ESC           : abort and close the window

Install the optional `play` extras to get OpenCV:
    pip install -e .[play]

Then run:
    python examples/manual_play.py
"""

from __future__ import annotations

import sys

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - import-time guard
    cv2 = None  # type: ignore

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


def _require_cv2() -> None:
    if cv2 is None:
        print(
            "opencv-python is required for manual_play. "
            "Install with: pip install -e .[play]",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> int:
    _require_cv2()

    env = MemoryFPSEnv(seed=0)
    obs, info = env.reset()

    print("Controls: WASD move/turn, SPACE attack, N no-op, Q end, ESC abort")

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WINDOW, 640, 480)

    steps = 0
    aborted = False
    terminated = False
    truncated = False

    # Tick the env at ~20 Hz so enemies keep moving (and the screen keeps
    # refreshing) even when no key is pressed. cv2.waitKey returns the
    # pressed key if one arrives within the window, else -1 (which masks
    # to 255 after & 0xFF). On no-key iterations we send NO_OP so the
    # env's wall-clock dt is consumed exactly once per tick.
    _TICK_MS = 50  # ~20 ticks/sec → ~1.5 m of chase per real second
    try:
        while not (terminated or truncated):
            # cv2 expects BGR; env obs is RGB.
            cv2.imshow(_WINDOW, obs[..., ::-1])
            key = cv2.waitKey(_TICK_MS) & 0xFF

            if key == 27:  # ESC
                aborted = True
                break

            action = KEY_MAP.get(key, Action.NO_OP)
            obs, _reward, terminated, truncated, info = env.step(int(action))
            steps += 1
    finally:
        cv2.destroyAllWindows()
        env.close()

    if aborted:
        print(f"aborted by user after {steps} steps, phase={info.get('phase')}")
    else:
        print(
            f"finished in {steps} steps, "
            f"terminated={terminated}, truncated={truncated}, "
            f"phase={info.get('phase')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
