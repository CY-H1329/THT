"""저비용 픽셀 분석: 막힘(충돌) 판정, 모션(적) 감지, 벽 색 샘플링.

전부 OCR/VLM 없이 numpy 배열 연산만으로 동작 — act() 매 스텝 호출해도
비용이 거의 없다(<1ms급). 임계값은 tmp/calibrate_*.py로 실제 게임에서
측정해서 정한 값이다(dev_log.md 참고):
  - 벽/물건에 막히면 프레임 diff가 정확히 0에 가깝게 떨어진다(움직임이
    전혀 없으니까). 반면 실제로 이동 중일 때는 항상 diff가 꽤 크다
    (열린 공간 0.1~0.5대, 방금 막 문 통과한 직후에도 0.2 이상).
  - 적이 공격 사거리 안에 들어오면 제자리에 멈춰서 공격하므로(움직임 X),
    모션 diff로는 "이미 붙어서 때리는 적"을 못 잡는다 — 그건 HP 하락으로
    잡아야 한다(ocr.py의 HudReading.hp를 스텝마다 비교).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from agent.palette import nearest_color_name

_HUD_H = 28  # ocr.py의 _HUD_BAR_HEIGHT와 동일 — 3D 뷰만 비교하려고 제외

# tmp/calibrate_blocked.py, calibrate_blocked_objects.py로 실측: 진짜 막히면
# diff_frac이 0.0000으로 딱 떨어지고, 움직이는 중엔 최소 0.03 이상이었음.
# 여유를 크게 둬도 안전하게 구분됨.
BLOCKED_THRESHOLD = 0.01


def _view(frame: np.ndarray) -> np.ndarray:
    """HUD 바를 뺀 3D 뷰 영역만 int32로."""
    return frame[_HUD_H:, :, :].astype(np.int32)


def frame_diff_frac(prev: np.ndarray, cur: np.ndarray, pixel_thresh: int = 30) -> float:
    """두 프레임(HUD 제외) 사이에서 '유의미하게 바뀐' 픽셀의 비율(0~1)."""
    a, b = _view(prev), _view(cur)
    per_pixel = np.abs(a - b).sum(axis=-1)  # 0..765
    return float((per_pixel > pixel_thresh).mean())


def is_blocked(prev: np.ndarray, cur: np.ndarray) -> bool:
    """직전에 MOVE_FORWARD/BACK을 시도했는데 실제로는 못 움직였는지."""
    return frame_diff_frac(prev, cur) < BLOCKED_THRESHOLD


@dataclass
class MotionReading:
    detected: bool
    direction: Optional[str] = None  # "left" | "center" | "right"
    changed_frac: float = 0.0


# 벽/물건처럼 '완전히 안 움직임'과 구분하기 위한 최소 기준. 카메라가 완전히
# 정지해 있을 때(NO_OP 연속)는 배경(벽/바닥/천장)이 전부 고정이므로, 조금이라도
# 바뀐 영역이 있으면 그건 스스로 움직이는 무언가(=적)일 수밖에 없다.
MOTION_MIN_FRAC = 0.003


def detect_motion(prev: np.ndarray, cur: np.ndarray) -> MotionReading:
    """카메라가 정지해 있던 두 프레임(NO_OP 연속) 사이의 변화를 본다.

    벽/바닥/천장/오브젝트/이미지는 전부 고정이므로, 카메라가 안 움직였는데
    바뀐 영역이 있다면 그건 스스로 움직이는 것(=적)이다. 바뀐 영역의 x좌표
    분포로 대략 왼쪽/가운데/오른쪽 중 어디인지도 같이 준다.
    """
    a, b = _view(prev), _view(cur)
    per_pixel = np.abs(a - b).sum(axis=-1)
    changed = per_pixel > 30
    frac = float(changed.mean())
    if frac < MOTION_MIN_FRAC:
        return MotionReading(detected=False, changed_frac=frac)

    cols = np.where(changed.any(axis=0))[0]
    width = cur.shape[1]
    center_x = float(cols.mean())
    if center_x < width / 3:
        direction = "left"
    elif center_x > 2 * width / 3:
        direction = "right"
    else:
        direction = "center"
    return MotionReading(detected=True, direction=direction, changed_frac=frac)


def sample_wall_color(frame: np.ndarray) -> Optional[str]:
    """현재 프레임(HUD 제외)에서 가장 흔한 색을 12색 팔레트에 매칭.

    바닥(체커무늬)/천장(회색 콘크리트)은 팔레트 색과 충분히 다르므로,
    최빈값이 실제로 벽 색일 확률이 높다. 팔레트와 거리가 너무 멀면(문·
    적·오브젝트가 화면을 가리는 경우) None을 반환해 호출자가 이번 프레임은
    무시하고 다음 프레임에 다시 시도하게 한다.
    """
    view = frame[_HUD_H:, :, :]
    # 연산량을 줄이기 위해 다운샘플링(4픽셀당 1개) 후 최빈 색 계산.
    small = view[::4, ::4, :].reshape(-1, 3)
    colors, counts = np.unique(small, axis=0, return_counts=True)
    mode_rgb = tuple(int(c) for c in colors[counts.argmax()])
    return nearest_color_name(mode_rgb)


def wall_color_fraction(frame: np.ndarray, target_color: str) -> float:
    """frame(HUD 제외)에서 팔레트 색 target_color에 해당하는 픽셀의 비율(0~1).

    "이 프레임이 이 방 벽으로 충분히 둘러싸여 있는가"(=문틈으로 다른 방이
    크게 안 보이는가)를 판단하는 용도. VLM에 보낼 프레임을 고를 때, 이
    비율이 낮으면(예: <0.5) 문 앞이라 다른 방이 섞여 보일 위험이 크다고
    보고 아예 후보에서 제외한다 — 프롬프트만으로는 VLM이 이 실수를 계속
    해서(실측 확인) 코드 단에서 미리 걸러내는 안전망.
    """
    from agent.palette import WALL_PALETTE
    target_rgb = dict(WALL_PALETTE).get(target_color)
    if target_rgb is None:
        return 0.0
    # 조명(음영)에 따라 같은 벽인데도 순수 팔레트 RGB보다 최대 0.65배까지
    # 어둡게 렌더될 수 있음(실측 확인: (49,71,111) vs 팔레트 denim
    # (75,110,170) — R/G/B 비율이 셋 다 거의 정확히 0.65로 균일하게
    # 어두워짐, 즉 색조는 그대로고 밝기만 스케일됨). 그래서 밝기를 빼고
    # "색조(방향)"만 코사인 유사도로 비교한다 — 절대 RGB 거리로는 이런
    # 조명 차이를 못 버텨서(실측: 거리 75, 임계값 55로도 못 잡음).
    view = frame[_HUD_H:, :, :].astype(np.float64)
    target = np.array(target_rgb, dtype=np.float64)

    # 채도 낮은(회색에 가까운) 픽셀은 색조 방향이 불안정해서, 균형잡힌
    # 팔레트 색(예: sage)과 우연히 코사인 유사도가 높게 나와 오탐을 낸다
    # (실측 확인: 천장 회색이 sage로 오판됨). 채도(최대-최소 채널 차)가
    # 낮은 픽셀은 아예 분모에서도 제외한다 — "몰라서 카운트 안 함"이지
    # "달라서 카운트 안 함"이 아니게.
    saturation = view.max(axis=-1) - view.min(axis=-1)
    colorful = saturation > 20

    pixel_norm = np.linalg.norm(view, axis=-1, keepdims=True)
    pixel_norm = np.where(pixel_norm < 1e-6, 1.0, pixel_norm)
    view_dir = view / pixel_norm
    target_dir = target / np.linalg.norm(target)
    cos_sim = (view_dir * target_dir).sum(axis=-1)

    if not colorful.any():
        return 0.0
    return float((cos_sim[colorful] > 0.99).mean())


def something_in_front(frame: np.ndarray, room_wall_color: Optional[str]) -> bool:
    """화면 중앙-중간높이 밴드가 이 방의 원래 벽 색과 다르면 True.

    전투 중 "지금 보고 있는 방향에 뭔가(적일 가능성) 있는가"를 판단하는
    용도. 정확히 뭔지는 모르지만(적/오브젝트/문 구분 안 함), 방금 공격을
    했는데 화면에 벽이 아닌 뭔가가 있다면 거기에 계속 공격을 퍼붓는 게
    아무것도 안 보고 그냥 다음 방향으로 넘어가는 것보다 낫다.
    바닥(체커무늬)·천장(회색)이 안 섞이도록 화면 중앙 부분만 본다.
    """
    h, w = frame.shape[0], frame.shape[1]
    y0, y1 = int(h * 0.35), int(h * 0.75)
    x0, x1 = int(w * 0.3), int(w * 0.7)
    band = frame[y0:y1, x0:x1, :]
    small = band[::3, ::3, :].reshape(-1, 3)
    if small.size == 0:
        return False
    colors, counts = np.unique(small, axis=0, return_counts=True)
    mode_rgb = tuple(int(c) for c in colors[counts.argmax()])
    name = nearest_color_name(mode_rgb)
    if room_wall_color is None:
        return name is None  # 방 색을 아직 모르면, 팔레트에 안 맞는 것만 후보로
    return name != room_wall_color
