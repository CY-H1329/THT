"""프레임 픽셀 → 방향별 자유 거리(depth) 추출 + 카메라 기하 상수.

핵심 아이디어: 이 env의 바닥은 항상 흑백 체커보드(floor_tiles_bw)라
채도(saturation)가 0에 가깝고, 벽은 WALL_PALETTE의 유채색, 3D 오브젝트/
적도 대부분 유채색이다. 그래서 각 픽셀 열(column)을 화면 아래에서 위로
훑다가 "채도가 임계값을 넘는 첫 행"을 찾으면 그게 그 방향에서 바닥이
끝나는 지점 = 가장 가까운 장애물(벽/오브젝트/적)의 밑동이다.

카메라는 domain_rand=False라 파라미터가 고정이다(env.py 소스 확인:
_MWWorld가 MiniWorldEnv를 domain_rand=False로 생성). miniworld
DEFAULT_PARAMS 기본값(cam_height 1.5m, fov_y 60°, pitch 0, forward_step
0.15m, turn_step 15°)이 그대로 적용된다 — 실측(tmp/debug_retreat_movement.py:
MOVE_BACK 한 스텝당 0.150m 정확히 일치)으로도 확인됨. 따라서 바닥 밑동의
행 좌표 y는 거리로 정확히 역산된다:

    dz = cam_height * focal / (y - cy)      # 광축 방향 거리
    d  = dz * sqrt(1 + u^2),  u = (x - cx)/focal

이 공식과 상수들은 참고 레포(CY-H1329/THT, dev-mouvement 브랜치)의
agent/geometry.py에서 아이디어를 얻었고, 실측(tmp/verify_geometry.py)으로
우리 env에서 직접 재검증했다. memory_fps_env는 import하지 않는다 —
상수는 miniworld 기본값을 픽셀에서 재확인한 값이며, 에이전트는 obs
프레임만으로 동작한다.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

# --- 카메라/프레임 상수 -------------------------------------------------
FRAME_H, FRAME_W = 240, 320
HUD_H = 28                      # 상단 HUD 바가 덮는 행 수
FOV_Y_DEG = 60.0
CAM_HEIGHT = 1.5                # m
FOCAL = (FRAME_H / 2.0) / math.tan(math.radians(FOV_Y_DEG / 2.0))  # ≈207.85 px
CX = (FRAME_W - 1) / 2.0
CY = (FRAME_H - 1) / 2.0
FOV_X_DEG = 2.0 * math.degrees(math.atan(CX / FOCAL))  # ≈75°

# 이동/회전 단위 (miniworld DEFAULT_PARAMS, domain_rand=False로 고정)
FORWARD_STEP = 0.15             # m per MOVE_FORWARD/MOVE_BACK
TURN_STEP = 15.0                # deg per TURN_LEFT/RIGHT
AGENT_RADIUS = 0.4              # m

# --- 센서 파라미터 ------------------------------------------------------
_SAT_THRESHOLD = 18             # 이 채도 미만이면 "바닥(무채색 체커보드)"
MAX_RANGE = 20.0                # 이 이상은 "열려 있음"으로 뭉뚱그림
# 프레임 맨 아랫행이 보는 바닥 거리. 이보다 가까운 장애물은 바닥을 전부
# 가려서 거리를 못 재므로 NEAR_RANGE로 보고한다.
FLOOR_MIN_RANGE = CAM_HEIGHT * FOCAL / (FRAME_H - 1 - CY)   # ≈2.6 m
NEAR_RANGE = 0.6


def wrap180(deg: float) -> float:
    """각도를 (-180, 180] 범위로 접는다."""
    return (deg + 180.0) % 360.0 - 180.0


@lru_cache(maxsize=8)
def column_bearings(n_cols: int) -> tuple:
    """열 인덱스 → 현재 heading 기준 상대 방위각(도).

    화면 오른쪽 = heading이 감소하는 쪽(TURN_RIGHT가 heading을 줄이는
    방향과 일치, agent/explorer.py의 _heading_diff 참고)이므로 부호를
    뒤집어 "이 열이 바라보는 절대 heading = heading + bearing"이 되게 한다.
    """
    cw = FRAME_W / n_cols
    out = []
    for i in range(n_cols):
        px = (i + 0.5) * cw
        out.append(-math.degrees(math.atan((px - CX) / FOCAL)))
    return tuple(out)


def saturation(frame: np.ndarray) -> np.ndarray:
    """픽셀별 (최대채널 - 최소채널). 바닥 판정과 몹 판정이 둘 다 이 값을
    쓰므로, 프레임당 한 번만 구해서 돌려쓴다."""
    return frame.max(axis=-1).astype(np.int16) - frame.min(axis=-1).astype(np.int16)


def floor_mask(frame: np.ndarray, sat: Optional[np.ndarray] = None) -> np.ndarray:
    """무채색(=바닥 체커보드) 픽셀 마스크."""
    if sat is None:
        sat = saturation(frame)
    return sat < _SAT_THRESHOLD


def floor_boundary(frame: np.ndarray, n_cols: int = 40,
                    sat: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """열 묶음별 (바닥이 끝나는 행, 그 지점까지의 거리 m).

    행 좌표는 깊이뿐 아니라 "그 장애물이 바닥에 닿아 있는가"를 판정하는
    데도 쓰인다(enemies.py). 벽에 걸린 사진 액자는 높이 0.75~2.25m에 떠
    있어서 바닥 경계선 바로 위가 벽색인 반면, 몹과 오브젝트는 자기 색이
    바닥까지 이어진다.
    """
    floor = floor_mask(frame, sat)
    y_lo = int(CY) + 2                      # 지평선 바로 아래부터
    band = floor[y_lo:FRAME_H, :]           # (rows, W)
    cw = FRAME_W // n_cols

    cols = band[:, :n_cols * cw].reshape(band.shape[0], n_cols, cw).mean(axis=2) > 0.5
    rev = ~cols[::-1]                        # 아래→위 순서의 "비바닥" 마스크
    idx = rev.argmax(axis=0)                 # 첫 True 위치 (없으면 0)
    has_obstacle = rev.any(axis=0)

    y_edge = (FRAME_H - 1) - idx             # 비바닥이 시작되는 행
    dy = y_edge + 0.5 - CY
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = CAM_HEIGHT * FOCAL / np.maximum(dy, 1e-6)
    u = np.array([(i + 0.5) * cw - CX for i in range(n_cols)]) / FOCAL
    dist = dz * np.sqrt(1.0 + u * u)

    bottom_blocked = ~cols[-1]
    dist = np.where(bottom_blocked, NEAR_RANGE, dist)
    dist = np.where(has_obstacle, dist, MAX_RANGE)
    y_edge = np.where(has_obstacle, y_edge, FRAME_H - 1)
    return y_edge, np.clip(dist, NEAR_RANGE, MAX_RANGE)


def depth_profile(frame: np.ndarray, n_cols: int = 40,
                   sat: Optional[np.ndarray] = None) -> np.ndarray:
    """열 방향별 자유 거리(m) 배열. 인덱스는 column_bearings(n_cols)와 대응."""
    return floor_boundary(frame, n_cols, sat)[1]


def height_at(row: float, dist: float) -> float:
    """거리 dist에 있는 물체의 픽셀 행 row가 대응하는 실제 높이(m)."""
    return CAM_HEIGHT - dist * (row - CY) / FOCAL


def dist_for_height(row: float, height: float) -> float:
    """높이 height인 물체의 꼭대기가 행 row에 보일 때의 거리(m).

    2.6m보다 가까우면 바닥 경계가 화면 밖이라 거리를 못 재는데, 몹은 키가
    알려져 있으므로(1.5m/2.0m) 머리 꼭대기 행으로 거꾸로 거리를 추정한다.
    """
    dy = CY - row
    if dy <= 1e-6:
        return MAX_RANGE
    return max(0.3, FOCAL * (height - CAM_HEIGHT) / dy)


def cone_clearance(profile: np.ndarray, center_deg: float = 0.0,
                    half_width_deg: float = 12.0) -> float:
    """heading 기준 [center±half] 부채꼴 안의 최소 자유 거리."""
    bearings = column_bearings(len(profile))
    vals = [d for b, d in zip(bearings, profile)
            if abs(wrap180(b - center_deg)) <= half_width_deg]
    return float(min(vals)) if vals else MAX_RANGE


def frame_motion(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """두 프레임의 화면 아래 절반 평균 절대차. MOVE_FORWARD/BACK이 실제로
    먹혔는지 판정하는 데 쓴다."""
    a = frame_a[FRAME_H // 2:, :].astype(np.int16)
    b = frame_b[FRAME_H // 2:, :].astype(np.int16)
    return float(np.abs(a - b).mean())


BLOCKED_MOTION = 4.0   # 이 미만이면 "이동이 막혔다"


def region_color(frame: np.ndarray, col_lo: int, col_hi: int) -> tuple:
    """지정한 열 구간의 지평선 아래 대표색(채널별 중앙값)."""
    lo = max(0, min(FRAME_W - 1, col_lo))
    hi = max(lo + 1, min(FRAME_W, col_hi))
    band = frame[int(CY):FRAME_H, lo:hi].reshape(-1, 3)
    if band.size == 0:
        return (0, 0, 0)
    med = np.median(band, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def color_close(a, b, tol: int = 40) -> bool:
    if a is None or b is None:
        return False
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))
