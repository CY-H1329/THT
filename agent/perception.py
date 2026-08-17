"""한 프레임에서 뽑아낼 수 있는 걸 전부 모아 Percept 하나로 만든다.

비용 관리가 핵심이다. HUD OCR은 글리프 템플릿 매칭이라 프레임당 ~20ms로
가장 비싼데, HUD 바 픽셀이 바뀌지 않으면 결과도 절대 바뀌지 않는다.
그래서 바 영역 바이트를 해시해서 캐시하고, 바뀐 프레임에서만 다시 읽는다
(타이머가 1초에 한 번 바뀌므로 실측 호출 빈도는 초당 1~2회 수준).

깊이 프로필/이동 판정은 numpy 벡터 연산이라 프레임당 0.2ms 남짓이므로
매 스텝 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from agent import geometry as geo
from agent.ocr import HudReading, hint_banner_active, read_hint_text, read_hud


@dataclass
class Percept:
    frame: np.ndarray
    profile: Optional[np.ndarray]      # 방향별 자유 거리(m). 배너 중에는 None.
    motion: float                      # 직전 프레임 대비 화면 하단 변화량
    moved: bool                        # 직전 이동 액션이 실제로 먹혔는지
    sat: Optional[np.ndarray] = None   # 픽셀별 채도 (바닥/몹 판정이 공유)
    boundary: Optional[tuple] = None   # (바닥 경계 행, 거리) — 몹 탐지기가 재사용
    banner: bool = False               # 힌트 배너가 떠 있는지
    hint_text: str = ""
    hud_fresh: bool = False            # 이번 스텝에 OCR을 새로 돌렸는지
    heading: Optional[float] = None
    room_name: Optional[str] = None
    is_corridor: bool = False
    hp: Optional[int] = None
    hp_max: Optional[int] = None
    seconds_remaining: Optional[int] = None
    has_key: bool = False
    wall_color: Optional[tuple] = None


class Perception:
    def __init__(self, n_cols: int = 40):
        self.n_cols = n_cols
        self._prev_frame: Optional[np.ndarray] = None
        self._hud_key: Optional[int] = None
        self._hud: Optional[HudReading] = None
        self._steps_since_hud = 0

    def observe(self, frame: np.ndarray, force_hud: bool = False,
                want_wall_color: bool = False) -> Percept:
        banner = hint_banner_active(frame)
        motion = (geo.frame_motion(self._prev_frame, frame)
                  if self._prev_frame is not None else 0.0)
        self._prev_frame = frame

        # 배너가 화면 중앙 55%를 덮으면 바닥 경계가 가려져 깊이가 무의미해진다.
        sat = None if banner else geo.saturation(frame)
        boundary = None if banner else geo.floor_boundary(frame, self.n_cols, sat)
        profile = None if banner else boundary[1]

        p = Percept(
            frame=frame,
            profile=profile,
            sat=sat,
            boundary=boundary,
            motion=motion,
            moved=motion >= geo.BLOCKED_MOTION,
            banner=banner,
        )
        if banner:
            p.hint_text = read_hint_text(frame)

        # --- HUD (캐시) ---
        # 바 픽셀이 그대로면 OCR 결과도 그대로다. 다만 heading 필드가 회전
        # 때마다 바뀌므로 "바뀌면 무조건 다시 읽기"는 회전 중에 초당 수십
        # 번 OCR을 부른다. heading은 액션으로 정확히 적분되니 재독은
        # 호출자가 정한 주기(force_hud)에만 하고, 바가 그대로면 그마저도
        # 건너뛴다.
        bar = frame[:28, :, :]
        key = hash(bar.tobytes())
        self._steps_since_hud += 1
        if force_hud and (key != self._hud_key or self._hud is None):
            self._hud = read_hud(frame)
            self._hud_key = key
            self._steps_since_hud = 0
            p.hud_fresh = True
        hud = self._hud
        if hud is not None and hud.ok:
            p.heading = float(hud.heading)
            p.room_name = hud.room_name
            p.is_corridor = hud.is_corridor
            p.hp, p.hp_max = hud.hp, hud.hp_max
            p.seconds_remaining = hud.seconds_remaining
            p.has_key = hud.has_key

        if want_wall_color and not banner:
            p.wall_color = geo.wall_color_estimate(frame)
        return p
