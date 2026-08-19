"""Scene-graph memory store.

Rooms are nodes, doors are edges. This structurally mirrors
world.RoomGraph, but we never see world.* -- it's built purely from
observation (HUD-OCR room names, wall color, later VLM results).

At this stage (pure movement) we only fill the fields exploration (DFS)
needs. images/objects/enemies_seen are reserved for the VLM-integration
stage so the schema doesn't have to change twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# Doors only ever exist at 0/90/180/270 (east/south/west/north) -- the
# grid embedding in layout.py rules out diagonal doorways, and this
# matches the README's heading<->wall table.
CARDINAL_HEADINGS = (0, 90, 180, 270)


@dataclass
class RoomNode:
    name: str
    wall_color: Optional[str] = None
    entry_heading: Optional[int] = None  # heading we were facing the first time we entered this room
    # exits[heading] = "unknown" | "wall" | "open" | "locked"
    # ("locked" means agent.geometry.lock_visible() actually confirmed a
    # padlock panel in that direction -- there's a door, but no way
    # through without the key. unknown_headings() excludes it
    # automatically, so it drops out of the retry candidates.)
    exits: Dict[int, str] = field(default_factory=lambda: {h: "unknown" for h in CARDINAL_HEADINGS})
    exit_leads_to: Dict[int, str] = field(default_factory=dict)  # heading -> room name (only for "open" exits)
    # Headings the VLM said "looks like a door/passage" during SURVEY
    # (just a hint before we've actually walked over and checked --
    # exits stays "unknown" either way). SEEK only uses this to order
    # which candidate to try first.
    vlm_door_hint: Set[int] = field(default_factory=set)
    done: bool = False  # True once all 4 headings are resolved (ready for DFS backtrack)
    # Filled in during the VLM-integration stage.
    images: List[dict] = field(default_factory=list)
    objects: List[dict] = field(default_factory=list)
    enemies_seen: List[dict] = field(default_factory=list)

    def unknown_headings(self) -> List[int]:
        return [h for h, s in self.exits.items() if s == "unknown"]


class SceneGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, RoomNode] = {}
        self.edges: Set[frozenset] = set()
        # DFS backtrack stack: (room_name, entry_heading)
        self.stack: List[tuple] = []
        self.visited_order: List[str] = []
        # Hint-banner text read by touching the locked door. This is a
        # global fact describing "the room with the key," not something
        # tied to one room, so it lives here at the graph level rather
        # than on a room node. key_hint_room/heading point at the
        # locked door itself (which room, which heading) in case
        # someone asks "what was behind the locked door?"
        self.key_hint_text: Optional[str] = None
        self.key_hint_room: Optional[str] = None
        self.key_hint_heading: Optional[int] = None

    def get_or_create(self, name: str) -> RoomNode:
        if name not in self.nodes:
            self.nodes[name] = RoomNode(name=name)
            self.visited_order.append(name)
        return self.nodes[name]

    def add_edge(self, a: str, b: str) -> None:
        self.edges.add(frozenset({a, b}))

    def shortest_path(self, src: str, dst: str) -> Optional[List[int]]:
        """BFS for the list of "exit headings" to walk from src to dst.

        Traverses nodes[*].exit_leads_to (adjacency tagged with the
        heading you leave through), not the undirected edges set --
        actually walking there requires knowing which way to exit each
        room. A return value of [0, 270] means "leave src heading east,
        then leave that room heading north, and you're at dst."
        Returns [] if src == dst, None if there's no path.

        exit_leads_to is only populated for doors we've actually walked
        through (explorer's _on_seek_arrival/_enter_room), so any path
        this returns is made entirely of doors we've already confirmed
        -- GOTO_HINT never gets sent to try an unknown door.
        """
        if src == dst:
            return []
        if src not in self.nodes or dst not in self.nodes:
            return None
        # (room name, heading list to get here)
        queue: List[tuple] = [(src, [])]
        seen = {src}
        while queue:
            name, path = queue.pop(0)
            node = self.nodes.get(name)
            if node is None:
                continue
            for heading, nxt in node.exit_leads_to.items():
                if nxt in seen:
                    continue
                if nxt == dst:
                    return path + [heading]
                seen.add(nxt)
                queue.append((nxt, path + [heading]))
        return None

    @property
    def current(self) -> Optional[RoomNode]:
        if not self.stack:
            return None
        name, _entry = self.stack[-1]
        return self.nodes.get(name)
