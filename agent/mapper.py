"""방 단위 위상 지도 + 격자 기하.

이 월드의 기하는 제약이 아주 강하다(개발 시드 관찰 + README 기술):

* 모든 방은 한 에피소드 안에서 같은 크기의 **정사각 셀**이고, 셀 격자에
  딱 맞춰 배치된다. 즉 방 원점은 항상 (gx*cell, gz*cell)이다.
* 이웃한 두 방은 벽을 공유하고, 문은 **공유 벽의 정중앙**에 폭 2m로
  뚫린다. 복도 방은 존재하지 않는다.
* 방 이름은 에피소드 내에서 유일하다(HUD에서 읽힌다).

그래서 "셀 크기 하나"만 알아내면 문 위치·방 경계가 전부 예측 가능해지고,
탐색은 (방 그래프 DFS) + (셀 안에서의 웨이포인트 주행)으로 환원된다.
이 모듈은 그 지도와 좌표 변환만 담당한다. 실제 주행/행동 결정은
explorer.py에 있다.

좌표계는 miniworld 월드 좌표를 그대로 쓴다: heading 0° = +x(동),
90° = -z(남), 180° = -x(서), 270° = +z(북). 스폰 방을 셀 (0,0)으로
두므로 우리 좌표는 실제 월드 좌표와 평행이동만큼 다르다(문제 없음).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agent import geometry as geo

# 벽 방향 라벨 → (heading, 격자 이웃 오프셋)
DIR_HEADING = {"E": 0.0, "S": 90.0, "W": 180.0, "N": 270.0}
DIR_DELTA = {"E": (1, 0), "S": (0, -1), "W": (-1, 0), "N": (0, 1)}
DIRS = ("E", "S", "W", "N")

# 벽 상태
UNKNOWN, WALL, DOOR, LOCKED = "unknown", "wall", "door", "locked"

CELL_MIN, CELL_MAX = 6.0, 10.0   # difficulty.yaml의 room_size 범위(정수)


def normalize_name(name: Optional[str]) -> Optional[str]:
    """HUD 이름 정규화. 긴 이름은 HUD 폭에 맞춰 '…'로 잘려 들어오는데,
    잘리는 길이는 옆에 붙는 T-초 자릿수에 따라 프레임마다 달라진다.
    말줄임표만 떼고, 같은 방인지는 prefix 비교로 판단한다(names_match).
    """
    if not name:
        return None
    return name.rstrip().rstrip("…").rstrip(".").rstrip()


def names_match(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return a.startswith(b) or b.startswith(a)


# 적 속도 1.5 m/s × 스텝당 시뮬시간(실측 ~12ms). 트랙 연결 반경을 경과
# 스텝에 비례해 넓히는 데 쓴다.
ENEMY_DRIFT_PER_STEP = 1.5 * 0.012
# 몹 폭이 1m이고 역투영 위치 오차가 ±0.5m쯤이라, 1m 반경으로는 같은 적이
# 두 트랙으로 갈렸다(실측: 적 5마리에 트랙 21개). 두 적이 1.6m 안에 붙어
# 있으면 하나로 세는 손해가 있지만, 갈라지는 쪽이 훨씬 흔하고 더 나쁘다.
TRACK_BASE_RADIUS = 1.6
TRACK_MAX_RADIUS = 4.0
# phantom_until이 이 값을 넘으면 영구 배제로 본다.
RETIRE_MARK = 10 ** 8


@dataclass
class EnemyTrack:
    """움직임으로 확인된 적 하나. 외형이 아니라 '움직였다'가 근거다.

    motion.py의 차분 블롭을 월드 좌표로 역투영해 쌓는다. 같은 적을 여러
    번 보면 위치를 갱신하고, 새 위치가 기존 트랙에서 너무 멀면 새 적으로
    친다. 죽은 뒤에도 리스트에 남겨 둔다 — QA에서 "적을 몇 마리 죽였나 /
    어느 방에 있었나"를 답하려면 사후 기록이 필요하다.
    """
    x: float
    z: float
    room_key: str
    first_step: int
    last_step: int
    sightings: int = 1
    attacked: int = 0              # 마지막 목격 이후 겨냥해 때린 횟수
    attacked_total: int = 0        # 누적. 목격으로 리셋되지 않는다 — 유령
                                   # 트랙이 목격 한 번으로 부활해 다시 24대를
                                   # 맞는 순환을 끊는 유일한 근거다.
    last_attack_step: int = -999
    colors: List[str] = field(default_factory=list)   # 팔레트 색 표본(QA용)
    alive: bool = True
    killed_step: int = -1
    revived: int = 0               # 죽었다고 봤는데 다시 움직인 횟수(오판)
    # 사거리 안에서 충분히 때렸는데도 사라지지도, 다시 움직이지도 않는
    # 트랙. 역투영 위치가 틀렸다는 뜻이라 조준 대상에서 뺀다. 다만 **영구
    # 배제는 하지 않는다** — 한동안만 쉰다. 영구로 뒀더니 방 안 트랙이
    # 전부 유령으로 강등된 뒤 에이전트가 목표를 완전히 잃고 맞아 죽었다
    # (시드 42: 793스텝 만에 HP 0).
    phantom_until: int = -1

    def age(self, step: int) -> int:
        return step - self.last_step

    def is_phantom(self, step: int) -> bool:
        return step < self.phantom_until

    def is_retired(self) -> bool:
        """영구 배제. 목격이 거의 없는데 누적 공격만 쌓인 트랙."""
        return self.phantom_until >= RETIRE_MARK

    def aimable(self, step: int, fresh: int, min_sightings: int) -> bool:
        """조준 대상으로 삼아도 되는가.

        두 조건이 핵심이다.

        * **최근에 봤어야 한다.** 적은 1.5 m/s로 걷기 때문에 오래된 트랙의
          저장 위치는 이미 적이 없는 자리다. 기억용 보존 기간과 조준용
          유효 기간은 완전히 다른 값이어야 한다.
        * **한 번만 본 블롭은 안 된다.** 지평선 근처에서는 1픽셀 행이
          1m가 넘어서, 멀리 있는 적이 엉뚱하게 가까운 좌표로 투영된다.
          진짜로 다가오는 적은 목격이 금방 쌓인다(실측 12회).
        """
        return (self.alive and not self.is_phantom(step)
                and self.sightings >= min_sightings
                and self.age(step) <= fresh)


@dataclass
class RoomNode:
    key: str                       # 지금까지 본 것 중 가장 긴 이름 (정규화됨)
    cell: Tuple[int, int]
    walls: Dict[str, str] = field(default_factory=lambda: {d: UNKNOWN for d in DIRS})
    wall_color: Optional[Tuple[int, int, int]] = None
    scanned: bool = False
    visits: int = 0
    first_seen_step: int = -1
    # 스캔에서 얻은 정적 장애물 추정 위치(월드 좌표). 적/오브젝트 구분에 쓴다.
    static_blobs: List[Tuple[float, float]] = field(default_factory=list)
    # 몸으로 부딪혀 확인한 벽. 스캔 판정만으로 벽이라 한 곳과 구분해서,
    # 탐색이 끝난 뒤 남는 시간에 다시 확인할 대상을 고른다.
    verified: set = field(default_factory=set)

    def open_dirs(self) -> List[str]:
        return [d for d in DIRS if self.walls[d] == DOOR]

    def unknown_dirs(self) -> List[str]:
        return [d for d in DIRS if self.walls[d] == UNKNOWN]

    def unverified_walls(self) -> List[str]:
        """스캔만으로 '벽'이라 본, 아직 몸으로 확인 안 한 방향."""
        return [d for d in DIRS if self.walls[d] == WALL and d not in self.verified]


class WorldMap:
    """셀 격자 위의 방 그래프."""

    def __init__(self, cell_size: float = 8.0):
        self.cell_size = cell_size
        self._cell_votes: List[float] = []
        self._cell_scores: Dict[float, float] = {}
        self.rooms: Dict[Tuple[int, int], RoomNode] = {}
        # 적 트랙은 방 노드가 아니라 지도 전역에 둔다. 적은 방을 넘어
        # 쫓아오므로(문을 통과한다) 방 단위로 묶으면 같은 적이 두 트랙으로
        # 갈린다. 대신 트랙마다 처음 본 방 이름을 들고 있는다.
        self.tracks: List[EnemyTrack] = []

    # --- 적 트랙 -------------------------------------------------------
    def note_motion(self, x: float, z: float, room_key: str,
                    step: int) -> Tuple[EnemyTrack, bool]:
        """움직임이 관측된 월드 좌표를 트랙에 반영. (트랙, 되살아났는지).

        연결 반경은 경과 스텝에 비례해 넓힌다. 적은 1.5 m/s로 걷기 때문에
        오래 못 본 트랙일수록 실제 위치가 멀리 가 있다. 고정 반경을 쓰면
        방을 한 바퀴 돌고 온 사이에 같은 적이 새 트랙으로 갈린다.

        **죽었다고 표시한 트랙도 후보에 넣는다.** 죽은 자리에서 다시 움직임이
        나온다는 건 사망 판정이 틀렸다는 뜻이므로, 새 트랙을 만드는 대신
        그 트랙을 되살린다. 이걸 안 했더니 "판정 → 재검출 → 새 트랙 →
        재판정"이 반복되며 킬 수가 실제 3에 대해 19까지 부풀었다.
        """
        best, best_d = None, None
        for t in self.tracks:
            r = min(TRACK_MAX_RADIUS,
                    TRACK_BASE_RADIUS + ENEMY_DRIFT_PER_STEP * t.age(step))
            d = math.hypot(x - t.x, z - t.z)
            if d <= r and (best_d is None or d < best_d):
                best, best_d = t, d
        if best is None:
            t = EnemyTrack(x=x, z=z, room_key=room_key,
                           first_step=step, last_step=step)
            self.tracks.append(t)
            return t, False
        revived = not best.alive
        if revived:
            best.alive = True
            best.killed_step = -1
            best.revived += 1
        # 새로 움직임이 보였다 = 살아서 거기 있다. 롤링 카운터만 리셋해서,
        # 긴 교전에서 공격이 쌓여 진짜 적이 유령으로 강등되는 일을 막는다.
        # attacked_total은 건드리지 않는다 — 이걸 같이 리셋했더니 유령
        # 트랙이 목격 한 번마다 되살아나 24대씩 영원히 얻어맞았다
        # (실측 시드 7: 목격 1회짜리 트랙 하나가 조준 공격 168회를 먹었다).
        if not best.is_retired():
            best.phantom_until = -1
        best.attacked = 0
        best.x, best.z = x, z
        best.last_step = step
        best.sightings += 1
        return best, revived

    def track_near(self, x: float, z: float, radius: float, step: int,
                   max_age: int, min_sightings: int = 1) -> Optional[EnemyTrack]:
        """(x, z) 근처의 조준 가능한 최신 트랙. 없으면 None."""
        best, best_d = None, None
        for t in self.tracks:
            if not t.aimable(step, max_age, min_sightings):
                continue
            d = math.hypot(x - t.x, z - t.z)
            if d <= radius and (best_d is None or d < best_d):
                best, best_d = t, d
        return best

    def live_tracks(self, step: int, max_age: int,
                    min_sightings: int = 1) -> List[EnemyTrack]:
        return [t for t in self.tracks
                if t.aimable(step, max_age, min_sightings)]

    def killed_tracks(self) -> List[EnemyTrack]:
        return [t for t in self.tracks if not t.alive]

    def kill_count(self) -> int:
        return len(self.killed_tracks())

    # --- 셀 크기 -------------------------------------------------------
    def vote_cell_size(self, scores: Dict[float, float]) -> float:
        """방 하나의 스캔 정합 점수(셀 크기별)를 누적해 셀 크기를 정한다.

        셀 크기는 에피소드 내내 하나뿐이라 방마다 나온 점수를 더하면
        빠르게 확정된다. 방 하나만 보고 정하면 가구가 많은 방에서 한 칸
        큰/작은 값이 이기는 경우가 있었다(개발 시드 28개 중 3개).
        """
        for c, sc in scores.items():
            self._cell_scores[c] = self._cell_scores.get(c, 0.0) + sc
        self._cell_votes.append(1.0)
        self.cell_size = max(self._cell_scores.items(), key=lambda kv: kv[1])[0]
        return self.cell_size

    @property
    def cell_locked(self) -> bool:
        """방 3개를 본 뒤에는 셀 크기를 더 흔들지 않는다."""
        return len(self._cell_votes) >= 3

    # --- 방 등록/조회 ---------------------------------------------------
    def get(self, cell: Tuple[int, int]) -> Optional[RoomNode]:
        return self.rooms.get(cell)

    def find_by_name(self, name: Optional[str]) -> Optional[RoomNode]:
        n = normalize_name(name)
        if not n:
            return None
        for room in self.rooms.values():
            if names_match(room.key, n):
                return room
        return None

    def ensure_room(self, cell: Tuple[int, int], name: Optional[str],
                    step: int) -> RoomNode:
        n = normalize_name(name)
        room = self.rooms.get(cell)
        if room is None:
            room = RoomNode(key=(n or f"?{cell}"), cell=cell, first_seen_step=step)
            self.rooms[cell] = room
        elif n and len(n) > len(room.key):
            room.key = n        # 덜 잘린 이름을 봤으면 갱신
        return room

    def link(self, cell: Tuple[int, int], direction: str, state: str) -> None:
        """벽 상태를 기록하고, 문이면 반대편 방의 벽도 같은 상태로 맞춘다."""
        room = self.rooms.get(cell)
        if room is None:
            return
        room.walls[direction] = state
        if state in (DOOR, LOCKED):
            dgx, dgz = DIR_DELTA[direction]
            other = (cell[0] + dgx, cell[1] + dgz)
            if other in self.rooms:
                self.rooms[other].walls[OPPOSITE[direction]] = state

    # --- 기하 ----------------------------------------------------------
    def room_origin(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        c = self.cell_size
        return cell[0] * c, cell[1] * c

    def cell_of(self, x: float, z: float) -> Tuple[int, int]:
        c = self.cell_size
        return int(math.floor(x / c)), int(math.floor(z / c))

    def room_center(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        ox, oz = self.room_origin(cell)
        c = self.cell_size
        return ox + c / 2.0, oz + c / 2.0

    def door_point(self, cell: Tuple[int, int], direction: str) -> Tuple[float, float]:
        """방 `cell`의 `direction` 벽 정중앙(=문 중심) 월드 좌표."""
        ox, oz = self.room_origin(cell)
        c = self.cell_size
        return {
            "E": (ox + c, oz + c / 2.0),
            "W": (ox, oz + c / 2.0),
            "N": (ox + c / 2.0, oz + c),
            "S": (ox + c / 2.0, oz),
        }[direction]

    def approach_point(self, cell: Tuple[int, int], direction: str,
                       back: float = 1.4) -> Tuple[float, float]:
        """문 앞 `back` m 지점(방 안쪽). 문을 정면으로 마주 보게 하는 웨이포인트."""
        dx, dz = self.door_point(cell, direction)
        hx, hz = _dir_vec(direction)
        return dx - hx * back, dz - hz * back

    def exit_point(self, cell: Tuple[int, int], direction: str,
                   ahead: float = 1.6) -> Tuple[float, float]:
        """문을 통과해 이웃 방으로 `ahead` m 들어간 지점."""
        dx, dz = self.door_point(cell, direction)
        hx, hz = _dir_vec(direction)
        return dx + hx * ahead, dz + hz * ahead

    def wall_distance(self, x: float, z: float, cell: Tuple[int, int],
                      direction: str) -> float:
        """방 안 한 점에서 해당 벽면까지의 수직 거리."""
        ox, oz = self.room_origin(cell)
        c = self.cell_size
        return {
            "E": ox + c - x,
            "W": x - ox,
            "N": oz + c - z,
            "S": z - oz,
        }[direction]

    def predicted_range(self, x: float, z: float, heading: float,
                        max_hops: int = 2) -> float:
        """지도만으로 예상되는 전방 자유 거리.

        깊이 센서는 2.6m보다 가까운 물체를 거리로 분해하지 못하고 그냥
        "가깝다"만 알려준다(바닥이 통째로 가려지기 때문). 그래서 벽에
        다가가는 정상 주행과 예기치 못한 장애물을 센서만으로는 구별할 수
        없다. 지도가 예측한 거리와 비교해야 그 둘이 갈린다.

        알려진 문(벽 정중앙 2m 구멍)은 통과 가능한 것으로 보고 다음 방까지
        이어서 계산한다.
        """
        hx, hz = geo.heading_to_vec(heading)
        cell = self.cell_of(x, z)
        px, pz = x, z
        total = 0.0
        for _ in range(max_hops):
            t, wall = self._exit(px, pz, hx, hz, cell)
            if wall is None:
                return total + geo.MAX_RANGE
            cx, cz = px + hx * t, pz + hz * t
            room = self.rooms.get(cell)
            mid_ok = _near_midpoint(cx, cz, self.room_origin(cell),
                                    self.cell_size, wall)
            if room is not None and room.walls.get(wall) == DOOR and mid_ok:
                total += t
                dgx, dgz = DIR_DELTA[wall]
                cell = (cell[0] + dgx, cell[1] + dgz)
                px, pz = cx + hx * 0.05, cz + hz * 0.05
                continue
            return total + t
        return total + geo.MAX_RANGE

    def _exit(self, x: float, z: float, hx: float, hz: float,
              cell: Tuple[int, int]):
        ox, oz = self.room_origin(cell)
        c = self.cell_size
        best_t, best_wall = float("inf"), None
        for wall, num, den in (
            ("E", ox + c - x, hx), ("W", ox - x, hx),
            ("N", oz + c - z, hz), ("S", oz - z, hz),
        ):
            if abs(den) < 1e-9:
                continue
            t = num / den
            if 1e-3 < t < best_t:
                best_t, best_wall = t, wall
        return best_t, best_wall

    # --- 탐색 계획 ------------------------------------------------------
    def route(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[str]]:
        """알려진 문만 따라 start→goal로 가는 방향 시퀀스(BFS 최단)."""
        if start == goal:
            return []
        from collections import deque
        prev: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
        q = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            room = self.rooms.get(cur)
            if room is None:
                continue
            for d in room.open_dirs():
                dgx, dgz = DIR_DELTA[d]
                nxt = (cur[0] + dgx, cur[1] + dgz)
                if nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = (cur, d)
                if nxt == goal:
                    path = []
                    node = goal
                    while node != start:
                        node, d0 = prev[node]
                        path.append(d0)
                    return list(reversed(path))
                q.append(nxt)
        return None

    def frontier_rooms(self, include_unverified: bool = False) -> List[Tuple[int, int]]:
        """아직 할 일이 남은 방들: 미스캔이거나, 상태 미상인 벽이 있는 방.

        include_unverified=True면 "스캔으로만 벽이라 판정한" 방향이 남은
        방까지 포함한다(탐색이 끝난 뒤 재확인용).
        """
        out = []
        for cell, room in self.rooms.items():
            if not room.scanned or room.unknown_dirs():
                out.append(cell)
            elif include_unverified and room.unverified_walls():
                out.append(cell)
        return out

    def summary(self) -> dict:
        return {
            "cell_size": self.cell_size,
            "rooms": {
                room.key: {
                    "cell": list(cell),
                    "walls": dict(room.walls),
                    "wall_color": room.wall_color,
                    "scanned": room.scanned,
                    "visits": room.visits,
                }
                for cell, room in sorted(self.rooms.items())
            },
        }


OPPOSITE = {"E": "W", "W": "E", "N": "S", "S": "N"}


def _dir_vec(direction: str) -> Tuple[float, float]:
    """방향 라벨 → 월드 (x, z) 단위 벡터."""
    return {
        "E": (1.0, 0.0),
        "W": (-1.0, 0.0),
        "N": (0.0, 1.0),
        "S": (0.0, -1.0),
    }[direction]


def _near_midpoint(cx: float, cz: float, origin: Tuple[float, float],
                   cell: float, wall: str, half: float = 0.9) -> bool:
    """벽면 위 점이 문 개구부(정중앙 2m) 안쪽인지."""
    ox, oz = origin
    if wall in ("E", "W"):
        return abs(cz - (oz + cell / 2.0)) <= half
    return abs(cx - (ox + cell / 2.0)) <= half
