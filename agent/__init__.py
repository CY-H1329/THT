"""평가 하네스가 임포트하는 진입점 — `from agent import Agent`.

역할 분담:
  - act(obs)   -> agent.explorer.ExplorerPolicy.step(obs)  (탐험/전투 FSM)
                  + agent.content_ingest.ContentIngest.on_frame(...)  (기억 적재)
  - answer(q)  -> 첫 호출 때 ContentIngest.finalize()로 기억을 동결한 뒤
                  agent.qa.answer_question(...)으로 순수 파이썬 질의.

README 규칙에 맞춘 설계 요점:
  - import 시점에 네트워크/무거운 로딩이 없다(무거운 모듈은 __init__ 안에서
    임포트하고, VLM 호출은 항상 백그라운드 스레드에서만 일어난다).
  - act()는 절대 VLM 응답을 기다리지 않는다(5초 예산). 방을 나갈 때
    백그라운드 잡만 제출한다.
  - 기억은 QA 시작 시점(=answer 첫 호출)에 한 번만 동결되고, 이후
    answer()는 추론 없이 이미 동결된 구조만 읽는다(10초 예산, 재현성).
  - act()/answer() 안의 어떤 예외도 에피소드를 죽이면 안 된다 — 최악의
    경우에도 유효한 액션/안전한 문장을 반환한다.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Agent"]


class Agent:
    def __init__(self) -> None:
        from agent.content_ingest import ContentIngest
        from agent.explorer import ExplorerPolicy

        self.policy = ExplorerPolicy()
        self.ingest = ContentIngest()
        # 탐험 FSM이 "힌트가 가리키는 방"을 판단하려면 관측 DB가 필요하다
        # (GOTO_HINT). 둘을 여기서만 연결해 두 모듈의 독립성은 유지한다.
        self.policy.content_db = self.ingest.content
        self._finalized = False
        self._frozen = None  # finalize() 후의 (ContentDB, SceneGraph)

    # --- Play phase ---------------------------------------------------
    def act(self, obs: np.ndarray) -> int:
        try:
            action = int(self.policy.step(obs))
        except Exception:
            # 정책이 어떤 이유로 터져도 에피소드는 계속돼야 한다 — 전진이
            # 아무것도 안 하는 것보다 항상 낫다(방을 더 볼 가능성).
            action = 2  # Action.MOVE_FORWARD
        try:
            self.ingest.on_frame(obs, self.policy)
        except Exception:
            pass
        return action

    # --- QA phase -----------------------------------------------------
    def answer(self, question: str) -> str:
        from agent.qa import answer_question

        if not self._finalized:
            self._finalized = True
            try:
                self.ingest.finalize(self.policy)
            except Exception:
                pass
            self._frozen = (self.ingest.content, self.policy.scene)
        try:
            content, scene = self._frozen
            return answer_question(question, content, scene)
        except Exception:
            return "I did not pay attention to that."
