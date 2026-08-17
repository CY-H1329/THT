# examples/agent_skeleton.py
"""Template Agent class. Candidates fill in act() and answer().

Your agent receives:
  - obs: (240, 320, 3) uint8 RGB frame, with a HUD bar at the top
         showing room name, time remaining, and HP. You must extract
         these from pixels — they are NOT provided through info or any
         other channel.

The eval harness calls act(obs) once per step during the play phase and
answer(question) once per question during the QA phase. The phase is not
passed to your Agent — it is implied by which method is being called, so
you do not need to detect it yourself.
"""

import numpy as np


class Agent:
    def __init__(self):
        # Initialize your memory store here.
        pass

    def act(self, obs: np.ndarray) -> int:
        """Return one of Action.* (integer 0..6)."""
        from memory_fps_env.env import Action
        return Action.NO_OP

    def answer(self, question: str) -> str:
        """Return a free-text answer to the given question."""
        return ""
