"""프레임 픽셀 → 방향별 자유 거리(depth) 추출 + 카메라 기하 상수.

핵심 아이디어: 이 env의 바닥은 항상 흑백 체커보드(floor_tiles_bw)라
채도(saturation)가 0에 가깝고, 벽은 WALL_PALETTE의 유채색, 3D 오브젝트도
대부분 유채색이다. 그래서 각 픽셀 열(column)을 화면 아래에서 위로
훑다가 "채도가 임계값을 넘는 첫 행"을 찾으면 그게 그 방향에서 바닥이
끝나는 지점 = 가장 가까운 장애물(벽/오브젝트/적)의 밑동이다.

카메라는 domain_rand=False라 파라미터가 고정이다(cam_height 1.5m,
fov_y 60°, pitch 0, 320x240). 따라서 바닥 밑동의 행 좌표 y는 거리로
정확히 역산된다:

    dz = cam_height * focal / (y - cy)      # 광축 방향 거리
    d  = dz * sqrt(1 + u^2),  u = (x - cx)/focal

실측 검증: 벽까지 거리 오차 중앙값 ±0.1m, 최대 0.5m (2~15m 구간).

memory_fps_env는 import하지 않는다 — 상수는 miniworld 기본값을 픽셀에서
재확인한 값이며, 에이전트는 obs 프레임만으로 동작한다.
"""

from __future__ import annotations

import math
from functools import lru_cache

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

# 이동/회전 단위 (miniworld DEFAULT_PARAMS, domain_rand=False)
FORWARD_STEP = 0.15             # m per MOVE_FORWARD
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


def heading_to_vec(heading_deg: float) -> tuple:
    """HUD heading(도) → 월드 (x, z) 전진 단위벡터.

    miniworld 규약: dir_vec = (cos θ, 0, -sin θ). README의 벽 대응표
    (0°=east, 90°=south, 180°=west, 270°=north)와 일치한다.
    """
    r = math.radians(heading_deg)
    return math.cos(r), -math.sin(r)


def vec_to_heading(dx: float, dz: float) -> float:
    """월드 변위 (dx, dz)를 바라보는 heading(도, [0,360))."""
    return math.degrees(math.atan2(-dz, dx)) % 360.0


@lru_cache(maxsize=8)
def column_bearings(n_cols: int) -> tuple:
    """열 인덱스 → 현재 heading 기준 상대 방위각(도).

    화면 오른쪽 = heading이 감소하는 쪽(시계 방향)이므로 부호를 뒤집어
    "이 열이 바라보는 절대 heading = heading + bearing"이 되게 한다.
    """
    cw = FRAME_W / n_cols
    out = []
    for i in range(n_cols):
        px = (i + 0.5) * cw
        out.append(-math.degrees(math.atan((px - CX) / FOCAL)))
    return tuple(out)


def saturation(frame: np.ndarray) -> np.ndarray:
    """픽셀별 (최대채널 - 최소채널). uint8 그대로 계산해 복사를 아낀다.

    바닥 판정(무채색)과 몹 판정(원색)이 둘 다 이 값을 쓰므로, 프레임당
    한 번만 구해서 돌려쓴다(perception.Percept.sat).
    """
    return frame.max(axis=-1) - frame.min(axis=-1)


def floor_mask(frame: np.ndarray, sat: np.ndarray | None = None) -> np.ndarray:
    """무채색(=바닥 체커보드) 픽셀 마스크."""
    if sat is None:
        sat = saturation(frame)
    return sat < _SAT_THRESHOLD


def floor_boundary(frame: np.ndarray, n_cols: int = 40,
                   sat: np.ndarray | None = None):
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

    # 열 묶음마다 "다수결 바닥" 프로필로 줄이고, 아래에서 위로 첫 비바닥 행을 찾는다.
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

    # 맨 아랫행부터 비바닥 = 바닥이 통째로 가려짐 → 근접 장애물.
    bottom_blocked = ~cols[-1]
    dist = np.where(bottom_blocked, NEAR_RANGE, dist)
    # 끝까지 바닥만 보였다 = 지평선까지 열림.
    dist = np.where(has_obstacle, dist, MAX_RANGE)
    y_edge = np.where(has_obstacle, y_edge, FRAME_H - 1)
    return y_edge, np.clip(dist, NEAR_RANGE, MAX_RANGE)


def depth_profile(frame: np.ndarray, n_cols: int = 40,
                  sat: np.ndarray | None = None) -> np.ndarray:
    """열 방향별 자유 거리(m) 배열. 인덱스는 column_bearings(n_cols)와 대응.

    지평선(CY) 위쪽은 아예 보지 않는다 — 천장(concrete, 무채색)까지
    올라가서 거짓 "열림"을 만드는 걸 원천 차단한다.
    """
    return floor_boundary(frame, n_cols, sat)[1]


def height_at(row: float, dist: float) -> float:
    """거리 dist에 있는 물체의 픽셀 행 row가 대응하는 실제 높이(m)."""
    return CAM_HEIGHT - dist * (row - CY) / FOCAL


def dist_for_height(row: float, height: float) -> float:
    """높이 height인 물체의 꼭대기가 행 row에 보일 때의 거리(m).

    2.6m보다 가까우면 바닥 경계가 화면 밖이라 거리를 못 재는데, 몹은 키가
    1.5m(short) 또는 2.0m(tall)로 알려져 있으므로 머리 꼭대기 행으로
    거꾸로 거리를 추정할 수 있다.
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
    """두 프레임의 화면 아래 절반 평균 절대차.

    MOVE_FORWARD가 실제로 먹혔는지 판정하는 데 쓴다. 실측상 이동 시
    31 이상, 벽에 막혔을 때 0.03 이하로 분리가 극단적이라 임계값 선택이
    거의 자유롭다(BLOCKED_MOTION 참고). 바닥 체커보드가 강한 텍스처라서
    가능한 판정이다.
    """
    a = frame_a[FRAME_H // 2:, :].astype(np.int16)
    b = frame_b[FRAME_H // 2:, :].astype(np.int16)
    return float(np.abs(a - b).mean())


BLOCKED_MOTION = 4.0   # 이 미만이면 "전진이 막혔다"


# --- 색 기반 단서 -------------------------------------------------------
# 자물쇠 판 텍스처: 따뜻한 노랑 바탕(235,215,120) 위에 진회색 자물쇠 그림.
# 렌더링 조명 때문에 실제 픽셀은 원본의 0.6~0.85배로 어두워지고 거리마다
# 달라지므로(실측 (153,140,78)~(196,180,100)), 절대 RGB가 아니라 채널 비율로
# 판정한다. 다만 olive 벽색(135,140,70)이 색상만으로는 거의 구분이 안 돼서,
# "노란 면적 + 그 안의 진한 자물쇠 그림"을 함께 요구한다.
def _warm_yellow(region: np.ndarray) -> np.ndarray:
    r = region[..., 0].astype(np.int32)
    g = region[..., 1].astype(np.int32)
    b = region[..., 2].astype(np.int32)
    mx = np.maximum(r, g)
    return (
        (mx > 60)
        & (np.abs(r - g) <= 0.14 * np.maximum(mx, 1))
        & (b * 100 < mx * 62)
        & (b * 100 > mx * 35)
    )


def lock_visible(frame: np.ndarray, min_frac: float = 0.12) -> bool:
    """잠긴 문의 자물쇠 판이 화면 중앙부를 채우고 있는지.

    보조 단서다. 잠김의 1차 신호는 힌트 배너(잠긴 문에 닿을 때만 뜬다)와
    "문 한복판에서 막힘"이며, 이 색 판정은 그 둘을 보강한다.
    """
    region = frame[HUD_H:FRAME_H, FRAME_W // 4: 3 * FRAME_W // 4]
    warm = _warm_yellow(region)
    if float(warm.mean()) < min_frac:
        return False
    gray = region.astype(np.int32).sum(axis=-1) / 3.0
    dark = (gray < 70) & (region.max(axis=-1) - region.min(axis=-1) < 40)
    return float(dark.mean()) >= 0.01


def region_color(frame: np.ndarray, col_lo: int, col_hi: int) -> tuple:
    """지정한 열 구간의 지평선 아래 대표색(채널별 중앙값).

    근접 장애물이 '벽'인지 '몹/오브젝트'인지 가르는 데 쓴다. 방 벽은 팔레트
    단색이고 그 색은 스캔 때 기록해 두므로, 이 색과 같으면 벽이다. 깊이
    센서는 2.6m 안쪽에서 거리 분해가 안 되기 때문에(FLOOR_MIN_RANGE),
    가까운 것의 정체는 색으로 판단하는 편이 훨씬 정확하다.
    """
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


def wall_color_estimate(frame: np.ndarray) -> tuple | None:
    """화면에서 가장 넓은 유채색 면의 대표 RGB = 현재 방의 벽 색.

    방 식별은 HUD 방 이름으로 충분하지만, 벽 색은 QA(“sage walls” 힌트
    템플릿)와 문 건너편 방을 미리 알아보는 데 쓸 수 있어 같이 뽑아둔다.
    """
    band = frame[HUD_H:int(CY), :].reshape(-1, 3).astype(np.int16)
    sat = band.max(axis=1) - band.min(axis=1)
    colored = band[sat >= _SAT_THRESHOLD]
    if len(colored) < 200:
        return None
    # 양자화 후 최빈 색 (조명 그라데이션 흡수용으로 16단계로 뭉갠다)
    q = (colored // 16) * 16
    keys, counts = np.unique(q, axis=0, return_counts=True)
    return tuple(int(v) for v in keys[counts.argmax()])
