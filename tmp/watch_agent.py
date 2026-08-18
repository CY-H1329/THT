"""Agent(act+answer) 전체를 cv2 창으로 보면서 돌리고, 끝나면 그 자리에서
질문을 직접 쳐볼 수 있는 도구.

tmp/watch_explorer.py와 같은 창을 쓰되 ExplorerPolicy가 아니라 평가
하네스와 동일한 진입점(agent.Agent)을 구동한다 — 즉 화면에 보이는 것과
기억에 쌓이는 것을 동시에 눈으로 확인할 수 있다.

사용법: python tmp/watch_agent.py [seed] [max_steps] [ms_per_frame]
  ESC : 플레이 중단하고 곧장 QA로
  q   : QA 프롬프트 종료
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import cv2  # type: ignore
except ImportError:
    print("opencv-python 필요: pip install -e .[play]", file=sys.stderr)
    sys.exit(1)

from agent import Agent
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
MS_PER_FRAME = int(sys.argv[3]) if len(sys.argv) > 3 else 30  # 낮을수록 빨리 재생

_WINDOW = "Memory-FPS (agent watch)"


def play(agent: Agent) -> None:
    env = MemoryFPSEnv(seed=SEED)
    obs, _info = env.reset()
    action_names = {v: k for k, v in agent.policy._Action.__members__.items()}

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WINDOW, 640, 480)
    print(f"seed={SEED} 시작. ESC로 중단하고 QA로 넘어감.")

    prev_state, hint_reported, had_key = agent.policy.state, False, False
    worst_act, term, trunc = 0.0, False, False
    for i in range(1, MAX_STEPS + 1):
        t = time.monotonic()
        action = agent.act(obs)
        worst_act = max(worst_act, time.monotonic() - t)
        obs, _r, term, trunc, _info = env.step(action)

        policy = agent.policy
        if policy.state != prev_state:
            print(f"step={i:4d} state {prev_state} -> {policy.state}")
            prev_state = policy.state
        if policy.scene.key_hint_text and not hint_reported:
            print(f"step={i:4d} *** 힌트 확보: {policy.scene.key_hint_text!r}")
            hint_reported = True
        hud = read_hud(obs)
        if hud.ok and hud.has_key and not had_key:
            print(f"step={i:4d} $$$ 열쇠 획득! room={hud.room_name!r}")
            had_key = True

        cv2.imshow(_WINDOW, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        if (cv2.waitKey(MS_PER_FRAME) & 0xFF) == 27:  # ESC
            print("사용자가 중단함 — QA로 넘어감.")
            break

        if i % 20 == 0 and not (term or trunc):
            print(f"step={i:4d} state={policy.state:9s} rooms={len(policy.scene.nodes)} "
                  f"hp={hud.hp}/{hud.hp_max} room={hud.room_name!r} "
                  f"T-{hud.seconds_remaining}s vlm={agent.ingest.calls_used} "
                  f"action={action_names.get(int(action), action)}")

        if term or trunc:
            print(f"\n=== 종료: step={i}, terminated={term}, truncated={trunc} ===")
            cv2.waitKey(1500)
            break

    env.close()
    cv2.destroyAllWindows()
    print(f"worst_act={worst_act:.2f}s rooms={list(agent.policy.scene.nodes)}")


def qa(agent: Agent) -> None:
    print("\n=== QA 단계 (첫 질문에서 기억이 동결됨) — 빈 줄이나 q로 종료 ===")
    while True:
        try:
            question = input("Q> ").strip()
        except EOFError:
            break
        if not question or question.lower() in ("q", "quit", "exit"):
            break
        t = time.monotonic()
        print(f"A> {agent.answer(question)}   [{time.monotonic() - t:.2f}s]")
    # 질문을 하나도 안 던진 채(예: stdin이 없는 백그라운드 실행) 끝나면
    # answer()가 한 번도 안 불려서 기억이 동결되지 않은 채로 덤프된다 —
    # 평가와 같은 상태를 보려면 여기서 명시적으로 동결한다(이미 동결됐으면
    # 아무 일도 안 일어남).
    agent.ingest.finalize(agent.policy)
    print("\n=== 동결된 기억 ===")
    print(json.dumps(agent.ingest.content.to_dict(), indent=1, ensure_ascii=False))


def main() -> None:
    agent = Agent()
    play(agent)
    qa(agent)


if __name__ == "__main__":
    main()
