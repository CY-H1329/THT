"""Entry point the evaluation harness imports -- `from agent import Agent`.

Division of labor:
  - act(obs)   -> agent.explorer.ExplorerPolicy.step(obs)  (explore/combat FSM)
                  + agent.content_ingest.ContentIngest.on_frame(...)  (memory ingest)
  - answer(q)  -> on the first call, ContentIngest.finalize() freezes memory,
                  then agent.qa.answer_question(...) does a pure-Python lookup.

Design points driven by the README rules:
  - No network calls or heavy loading at import time (heavy modules are
    imported inside __init__, and VLM calls only ever happen on a
    background thread).
  - act() never waits on a VLM response (5 s budget) -- it only submits
    a background job when leaving a room.
  - Memory is frozen exactly once, at the start of QA (the first answer()
    call); every answer() after that just reads the already-frozen
    structure with no further inference (10 s budget, reproducibility).
  - No exception inside act()/answer() should ever kill the episode --
    worst case, we still return a legal action or a safe sentence.
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
        # The explorer FSM needs the observation DB to figure out which
        # room a hint is pointing at (GOTO_HINT). We wire the two
        # together only here, so each module stays independent otherwise.
        self.policy.content_db = self.ingest.content
        self._finalized = False
        self._frozen = None  # (ContentDB, SceneGraph) after finalize()

    # Play phase
    def act(self, obs: np.ndarray) -> int:
        try:
            action = int(self.policy.step(obs))
        except Exception:
            # Whatever breaks in the policy, the episode has to keep
            # going -- moving forward is always better than doing
            # nothing (it's still a chance to see more of the map).
            action = 2  # Action.MOVE_FORWARD
        try:
            self.ingest.on_frame(obs, self.policy)
        except Exception:
            pass
        return action

    # QA phase
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
