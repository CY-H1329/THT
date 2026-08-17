"""평가 하네스가 `from agent import Agent`로 불러가는 진입점.

현재 구현 범위: 자율 주행/탐색(explorer)과 픽셀 파싱(ocr, geometry).
QA 응답은 탐색 중 쌓아둔 Explorer.memory를 그대로 조회하는 최소 구현이며,
장면 인식(벽 사진/오브젝트 분류) 기반 응답은 다음 단계에서 붙인다.
"""

from agent.explorer import Explorer


class Agent:
    def __init__(self, trace: bool = False):
        self.explorer = Explorer(trace=trace)

    def act(self, obs) -> int:
        return self.explorer.act(obs)

    def answer(self, question: str) -> str:
        # TODO: 메모리 질의 계층. 지금은 탐색 결과만 갖고 있다.
        return "I did not pay attention to that."


__all__ = ["Agent", "Explorer"]
