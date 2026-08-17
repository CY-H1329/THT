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
    # exits[heading] = "unknown" | "wall" | "open"
    exits: Dict[int, str] = field(default_factory=lambda: {h: "unknown" for h in CARDINAL_HEADINGS})
    exit_leads_to: Dict[int, str] = field(default_factory=dict)  # heading -> room name (open인 경우만)
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

    def get_or_create(self, name: str) -> RoomNode:
        if name not in self.nodes:
            self.nodes[name] = RoomNode(name=name)
            self.visited_order.append(name)
        return self.nodes[name]

    def add_edge(self, a: str, b: str) -> None:
        self.edges.add(frozenset({a, b}))

    @property
    def current(self) -> Optional[RoomNode]:
        if not self.stack:
            return None
        name, _entry = self.stack[-1]
        return self.nodes.get(name)
