"""Agent(act+answer) 전체를 seed 하나에 대해 돌리고, 동결된 기억과
대표 질문 답변을 출력한다. 사용: python tmp/eval_agent.py [seed] [max_steps]"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import Agent
from agent.ocr import read_hud
from memory_fps_env.env import MemoryFPSEnv

QUESTIONS = [
    "Did you visit {room}?",
    "What wall image was in {room}?",
    "What 3D objects were in {room}?",
    "What color were the walls in {room}?",
    "Were there any enemies in {room}?",
    "How many enemies did you kill?",
    "What did the key hint say?",
    "Which room contained the key?",
    "Did you find the key and unlock the door?",
    "What was in the room behind the locked door?",
    "How many rooms did you visit?",
    "How did the episode end?",
    "Did you visit Nonexistent Hall?",
]


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

    env = MemoryFPSEnv(seed=seed)
    obs, _info = env.reset()
    agent = Agent()
    worst_act, last_secs, i = 0.0, 600, 0
    t0 = time.monotonic()
    term = trunc = False
    for i in range(1, max_steps + 1):
        t = time.monotonic()
        action = agent.act(obs)
        worst_act = max(worst_act, time.monotonic() - t)
        obs, _r, term, trunc, _info = env.step(action)
        if term or trunc:
            break
        hud = read_hud(obs)
        if hud.ok and hud.seconds_remaining is not None:
            last_secs = hud.seconds_remaining
    env.close()

    print(f"seed={seed} steps={i} died={term} survived_s={600 - last_secs} "
          f"worst_act={worst_act:.2f}s wall={time.monotonic() - t0:.0f}s")
    print(f"rooms_explored={list(agent.policy.scene.nodes)}")

    worst_answer = 0.0
    rooms = list(agent.policy.scene.nodes) or ["the first room"]
    for template in QUESTIONS:
        q = template.format(room=rooms[0] if "{room}" in template else "")
        t = time.monotonic()
        a = agent.answer(q)
        dt = time.monotonic() - t
        worst_answer = max(worst_answer, dt)
        print(f"  Q: {q}\n  A: {a}   [{dt:.2f}s]")
    if len(rooms) > 1:
        q = f"What wall image was in {rooms[1]}?"
        print(f"  Q: {q}\n  A: {agent.answer(q)}")

    print(f"worst_answer={worst_answer:.2f}s vlm_calls={agent.ingest.calls_used} "
          f"cost=${agent.ingest.cost_usd:.4f}")
    print(json.dumps(agent.ingest.content.to_dict(), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
