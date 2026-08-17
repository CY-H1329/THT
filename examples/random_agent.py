# examples/random_agent.py
"""Minimal random-action agent."""

import random
from memory_fps_env.env import MemoryFPSEnv, Action


class RandomAgent:
    def __init__(self, end_after: int = 1500, seed: int = 0):
        self.rng = random.Random(seed)
        self.end_after = end_after
        self.t = 0

    def act(self, obs):
        self.t += 1
        if self.t >= self.end_after:
            return Action.END_EPISODE
        # Skew toward movement; rare attack; rare no_op.
        return self.rng.choices(
            [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT,
             Action.MOVE_BACK, Action.ATTACK, Action.NO_OP],
            weights=[5, 2, 2, 1, 1, 1],
        )[0]

    def answer(self, question: str) -> str:
        return "I did not pay attention to that."


if __name__ == "__main__":
    env = MemoryFPSEnv(seed=0)
    agent = RandomAgent()
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action = int(agent.act(obs))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    print(f"finished in {steps} steps")
