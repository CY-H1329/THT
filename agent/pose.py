"""자기 위치 추정 (dead reckoning + 스캔 기반 보정).

이 env에서 오도메트리가 거의 공짜인 이유:

* heading은 HUD에 정수 도 단위로 그대로 찍힌다(오차 0). 회전은 충돌
  판정이 없어 항상 성공하므로 액션만으로도 적분 가능하다.
* 전진은 정확히 0.15m이고 miniworld의 충돌 처리는 "전부 이동 or 아예
  이동 안 함"이라 부분 이동이 없다. 그리고 이동 성공 여부는 화면 하단
  변화량으로 사실상 100% 판정된다(geometry.frame_motion 참고).

남는 오차원은 "막혔는데 이동했다고 잘못 센 경우"와 문을 넘나들 때의
누적 오차뿐이라, 방에 들어갈 때마다 360° 스캔을 방 사각형에 정합해
절대 위치를 다시 잡는다(localize.match_scan).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from agent import geometry as geo

# 액션 상수 (memory_fps_env를 import하지 않기 위해 자체 정의)
TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, MOVE_BACK, ATTACK, NO_OP, END_EPISODE = range(7)


class PoseTracker:
    def __init__(self, x: float = 0.0, z: float = 0.0, heading: float = 0.0):
        self.x, self.z = x, z
        self.heading = heading
        self.blocked_count = 0      # 누적 충돌 횟수 (진단용)
        self.travelled = 0.0

    def update(self, action: int, moved: bool, hud_heading: Optional[float]) -> None:
        if action == TURN_LEFT:
            self.heading = (self.heading + geo.TURN_STEP) % 360.0
        elif action == TURN_RIGHT:
            self.heading = (self.heading - geo.TURN_STEP) % 360.0
        elif action in (MOVE_FORWARD, MOVE_BACK):
            if moved:
                sign = 1.0 if action == MOVE_FORWARD else -1.0
                hx, hz = geo.heading_to_vec(self.heading)
                self.x += sign * geo.FORWARD_STEP * hx
                self.z += sign * geo.FORWARD_STEP * hz
                self.travelled += geo.FORWARD_STEP
            else:
                self.blocked_count += 1
        # HUD heading은 정답이므로 읽혔으면 무조건 그걸로 맞춘다.
        if hud_heading is not None:
            self.heading = float(hud_heading) % 360.0

    @property
    def xz(self) -> Tuple[float, float]:
        return self.x, self.z

    def ahead(self, dist: float) -> Tuple[float, float]:
        hx, hz = geo.heading_to_vec(self.heading)
        return self.x + hx * dist, self.z + hz * dist

    def bearing_to(self, tx: float, tz: float) -> float:
        """목표점을 향하려면 heading을 얼마나 돌려야 하는지(deg, 좌회전 +)."""
        return geo.wrap180(geo.vec_to_heading(tx - self.x, tz - self.z) - self.heading)

    def distance_to(self, tx: float, tz: float) -> float:
        return math.hypot(tx - self.x, tz - self.z)
