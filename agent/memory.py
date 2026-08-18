"""씬 그래프(scene graph) 형태의 메모리 저장소.

방 = 노드, 문 = 엣지. world.RoomGraph와 구조적으로 대응되지만, 우리는
world.*를 못 보므로 오직 관찰(OCR 방이름, 벽색, 나중엔 VLM 결과)로만
직접 구성한다.

지금 단계(순수 이동)에서는 탐험(DFS)에 필요한 필드만 채운다. images/
objects/enemies_seen은 나중에 VLM 연동 단계에서 채워질 자리만 미리
마련해둔다(구조를 두 번 안 바꾸려고).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# 문은 오직 0/90/180/270(동/남/서/북)에만 존재한다 (layout.py의 격자
# 임베딩 구조상 대각선 방향은 있을 수 없음 — README의 heading<->wall 표와
# 일치).
CARDINAL_HEADINGS = (0, 90, 180, 270)


@dataclass
class RoomNode:
    name: str
    wall_color: Optional[str] = None
    entry_heading: Optional[int] = None  # 처음 이 방에 들어올 때 보고 있던 방향
    # exits[heading] = "unknown" | "wall" | "open" | "locked"
    # ("locked": agent.geometry.lock_visible()로 자물쇠 판을 실측 확인한
    # 방향 — 문은 있지만 열쇠 없인 못 지나감. unknown_headings()가 자동
    # 제외하므로 재시도 후보에서 빠짐.)
    exits: Dict[int, str] = field(default_factory=lambda: {h: "unknown" for h in CARDINAL_HEADINGS})
    exit_leads_to: Dict[int, str] = field(default_factory=dict)  # heading -> room name (open인 경우만)
    # SURVEY 중 VLM이 "이 방향은 문/통로로 보인다"고 답한 방향 집합
    # (실제로 걸어가서 확인하기 전의 힌트일 뿐 — exits는 그대로
    # "unknown"으로 남는다). SEEK가 후보 우선순위를 정할 때만 쓴다.
    vlm_door_hint: Set[int] = field(default_factory=set)
    done: bool = False  # 4방향 다 확인 끝났으면 True (DFS에서 backtrack 대상)
    # 아래는 VLM 연동 단계에서 채움.
    images: List[dict] = field(default_factory=list)
    objects: List[dict] = field(default_factory=list)
    enemies_seen: List[dict] = field(default_factory=list)

    def unknown_headings(self) -> List[int]:
        return [h for h, s in self.exits.items() if s == "unknown"]


class SceneGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, RoomNode] = {}
        self.edges: Set[frozenset] = set()
        # DFS 되돌아가기용 스택: (room_name, entry_heading)
        self.stack: List[tuple] = []
        self.visited_order: List[str] = []
        # 잠긴 문을 터치해서 얻은 힌트 배너 텍스트 — 특정 방에 대한 정보가
        # 아니라 "열쇠가 있는 방"을 설명하는 전역 사실이라 방 노드가 아닌
        # 여기(그래프 레벨)에 둔다. key_hint_room/heading은 그 힌트를 얻은
        # 잠긴 문의 위치(어느 방의 어느 방향)를 가리킨다 — 사람이 "잠긴
        # 문 뒤에 뭐가 있어?" 같은 질문을 할 수도 있으므로.
        self.key_hint_text: Optional[str] = None
        self.key_hint_room: Optional[str] = None
        self.key_hint_heading: Optional[int] = None
        # 힌트 텍스트를 방 기록(wall_color 등)과 대조해 추정한 "열쇠가
        # 있을 것 같은 방" — 아직 실제로 못 찾았으면 None.
        self.key_target_room: Optional[str] = None

    def get_or_create(self, name: str) -> RoomNode:
        if name not in self.nodes:
            self.nodes[name] = RoomNode(name=name)
            self.visited_order.append(name)
        return self.nodes[name]

    def add_edge(self, a: str, b: str) -> None:
        self.edges.add(frozenset({a, b}))

    def find_path(self, start: str, target: str) -> Optional[List[tuple]]:
        """start에서 target까지, 실제로 확인된 문(exit_leads_to)만 타고
        가는 최단 경로를 [(현재 방 이름, 나갈 heading), ...] 순서로 반환.
        각 홉에서 그 heading으로 나가면 다음 방(또는 마지막이면 target)에
        도착한다. 경로가 없으면 None, start==target이면 빈 리스트."""
        if start == target:
            return []
        if start not in self.nodes:
            return None
        from collections import deque
        visited = {start}
        queue = deque([(start, [])])
        while queue:
            room, path = queue.popleft()
            node = self.nodes.get(room)
            if node is None:
                continue
            for heading, dest in node.exit_leads_to.items():
                if dest in visited:
                    continue
                new_path = path + [(room, heading)]
                if dest == target:
                    return new_path
                visited.add(dest)
                queue.append((dest, new_path))
        return None

    @property
    def current(self) -> Optional[RoomNode]:
        if not self.stack:
            return None
        name, _entry = self.stack[-1]
        return self.nodes.get(name)
