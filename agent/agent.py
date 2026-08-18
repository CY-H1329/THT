"""제출용 Agent 클래스 (README의 Agent contract 구현).

act(obs)는 순수 픽셀 관찰만 받아 agent.explorer.ExplorerPolicy(규칙기반
탐험 FSM)에 그대로 위임한다. 그 위에서 이 클래스가 하는 일은 딱 하나:
SURVEY 중인 방의 프레임을 버퍼링해뒀다가, 그 방이 DFS상 "done"(4방향
확인 끝)이 되는 순간 agent.vlm.update_room_db()를 그 방 1개분만 호출해
agent.content_db.ContentDB(이미지/오브젝트/적 콘텐츠)를 채운다 — 방
하나씩 분산 처리라 개별 act() 호출은 여전히 ~5초 예산 안에 들고,
QA 시작 시점엔 이미 다 채워져 있어 answer()는 조회만 하면 된다
(agent/vlm.py 상단 주석 참고).

answer(question)은 agent.qa.answer()에 SceneGraph(탐험 상태) + ContentDB
(관찰 콘텐츠)를 넘겨 조회만 한다 — 새로운 추론 상태를 쌓지 않는다
(README: "Memory is frozen at the start of the QA phase").
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

from agent.config import VLM_MAX_CALLS_PER_EPISODE
from agent.content_db import ContentDB
from agent.explorer import ExplorerPolicy
from agent.qa import answer as qa_answer
from agent.vlm import update_room_db

_MAX_FRAMES_PER_ROOM = 4  # SURVEY가 4방향(0/90/180/270)을 도니 딱 그만큼


class Agent:
    def __init__(self) -> None:
        self.policy = ExplorerPolicy()
        self.content_db = ContentDB()
        self._frames_by_room: Dict[str, List[np.ndarray]] = {}
        self._db_built_rooms: Set[str] = set()
        self._was_flee = False  # 전투 종료 순간을 감지해 킬 카운트를 근사한다

    def act(self, obs: np.ndarray) -> int:
        vlm_calls_before = self.policy.vlm_calls_used
        action = self.policy.step(obs)
        self._maybe_count_kill()
        self._maybe_buffer_frame(obs)
        # policy.step() 안에서 이미 이 틱에 VLM을 한 번 불렀다면(SURVEY의
        # classify_heading, 전투 sweep의 locate_enemy, VLM_RECOVER 등)
        # DB 구축용 update_room_db()는 다음 틱으로 미룬다 — 같은 act()
        # 호출 안에서 네트워크 호출이 두 번 겹치면 README의 "act 호출당
        # ~5초" 예산을 넘길 위험이 있다(각 호출이 최대 VLM_TIMEOUT_S=4초
        # 까지 기다릴 수 있음). node.done은 계속 True로 남아있으니 한 틱
        # 늦게 처리해도 데이터 유실은 없다.
        if self.policy.vlm_calls_used == vlm_calls_before:
            self._maybe_build_room_db()
        return int(action)

    # 실제 "죽었다"는 신호가 obs에 없어 근사치다: 전투(FLEE)에 들어가
    # 최소 한 번 공격을 낸 뒤 전투 상태를 빠져나오면 죽였거나(대부분)
    # 시간 초과로 포기한 것 — 둘을 구분할 지면 신호가 없다는 한계를
    # report.md에 명시한다.
    def _maybe_count_kill(self) -> None:
        now_flee = self.policy.state == "FLEE"
        if self._was_flee and not now_flee and self.policy.flee_ever_attacked:
            self.content_db.killed_enemies += 1
        self._was_flee = now_flee

    def answer(self, question: str) -> str:
        try:
            return qa_answer(question, self.policy.scene, self.content_db)
        except Exception:
            return "I don't remember that."

    # --- 내부 -----------------------------------------------------------
    def _maybe_buffer_frame(self, obs: np.ndarray) -> None:
        if self.policy.state != "SURVEY":
            return
        room = self.policy.current_room
        if room is None:
            return
        buf = self._frames_by_room.setdefault(room, [])
        if len(buf) < _MAX_FRAMES_PER_ROOM:
            buf.append(obs.copy())

    def _maybe_build_room_db(self) -> None:
        # act() 호출 하나당 update_room_db()는 최대 1번만 쏜다(첫 방을
        # 처리하면 즉시 리턴) — 여러 방이 한꺼번에 done이 돼도 한 틱에
        # 네트워크 호출이 여러 번 겹쳐 시간 예산을 넘기지 않도록.
        for name, node in self.policy.scene.nodes.items():
            if not node.done or name in self._db_built_rooms:
                continue
            self._db_built_rooms.add(name)
            frames = self._frames_by_room.pop(name, [])
            if not frames:
                continue
            if self.policy.vlm_calls_used >= VLM_MAX_CALLS_PER_EPISODE:
                return
            self.policy.vlm_calls_used += 1
            current = self.content_db.to_dict()["rooms"].get(name, {})
            result = update_room_db(frames, current, room_hint=name)
            if result.ok:
                self.content_db.apply_update(name, result.data)
            return
