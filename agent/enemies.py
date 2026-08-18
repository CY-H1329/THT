"""적(사람형 몹) 탐지 — 픽셀에서 방위·거리·신뢰도를 뽑는다.

VLM 기반 조준(agent.vlm.locate_enemy)은 호출당 ~2초가 걸려서 그동안
계속 맞았고("느리고 붕뜬다" — 사용자 실측), 색상 규칙기반(예전
agent.vision.enemy_bearing)은 저장된 방 벽색과의 단순 거리 비교라 조명
때문에 벽 자체를 적으로 오판했다(dev_log.md). 이 모듈은 참고 레포
(CY-H1329/THT, dev-mouvement 브랜치)의 agent/enemies.py에서 아이디어를
얻어 우리 env로 재검증한 순수 CV 방식이다 — VLM 없이 프레임당 ~1~2ms.

몹의 생김새(miniworld Box 6개 + 얼굴 쿼드) 특징:

* 색이 **진하다(채도 높음)**. 몸통/바지/피부는 원색 계열이라 채도가
  높게 찍히고, 방 벽 팔레트(agent/palette.py의 12색)는 훨씬 탁하다.
* **바닥에 서 있다**. 벽에 걸린 사진 액자와 갈리는 결정적 차이 — 액자는
  공중에 떠 있어서 바닥 경계선 바로 위가 벽색이지만, 몹은 자기 색이
  바닥 접점까지 이어진다.
* **키가 대략 사람 크기(1.15~2.45m)**. 통·콘·오리 같은 작은 소품과
  나무 같은 큰 소품을 걸러낸다(완벽하진 않음 — 나무는 키가 비슷해서
  일부만 걸러짐, 한계로 report.md에 기록).

거리는 agent.geometry.floor_boundary()가 이미 계산해 둔 바닥 접점
행/거리를 그대로 재사용한다(agent.perception 스타일 — 프레임당 한 번만
계산). 2.6m(FLOOR_MIN_RANGE)보다 가까우면 바닥 접점이 화면 밖이라 못
재므로, 대신 실루엣 꼭대기 행과 알려진 키로 거리를 거꾸로 추정한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from agent import geometry as geo

# 몹 박스로 볼 최소 채도. 참고 레포는 88을 썼지만 우리 WALL_PALETTE는
# mustard(155)/rust(130)/brick(115)/terracotta(110)처럼 원색에 가까운
# 벽색도 포함하고 있어서(agent/palette.py) 그대로 쓰면 위험하다.
# tmp/verify_enemies.py로 실측: 실제 렌더된 적 픽셀 채도는 101~213
# (중앙값 133), 같은 프레임의 벽 배경은 최대 81 — 여유를 두고 95로.
MOB_SAT = 95
# 바닥 접점 바로 위에서 색을 읽을 띠의 두께(px).
FOOT_BAND = 10
MOB_HEIGHTS = (1.5, 2.0)      # short / tall
MIN_MOB_HEIGHT = 1.15
NEAR_TOP_MARGIN = 22       # 근거리 후보의 실루엣 꼭대기 허용 행(지평선 기준)
NEAR_DEFAULT_DIST = 1.8    # 키 역산이 발산할 때 쓸 근거리 대표값(m)
MAX_MOB_HEIGHT = 2.45
# 실측(dev_log.md): 실제 몹 셔츠/바지/피부는 양자화 후 고유색 ~3개,
# 나무 캐노피는 면마다 음영이 달라 ~23개. 여유를 두고 8로.
MAX_COLOR_DIVERSITY = 8


@dataclass
class Mob:
    bearing: float        # 현재 heading 기준 상대 방위(도, 좌 +)
    distance: float       # m (2.6m 안쪽은 키 기반 추정치)
    height: float         # 추정 실루엣 높이(m), 검산 불가 시 -1.0
    width_deg: float
    col_range: Tuple[int, int]
    resolved: bool        # 바닥 접점으로 거리를 실측했는지
    score: float          # 0~1, 몹다움


def detect(frame: np.ndarray, n_cols: int = 64,
           wall_rgb: Optional[tuple] = None,
           sat: Optional[np.ndarray] = None,
           boundary: Optional[tuple] = None) -> List[Mob]:
    """프레임에서 몹 후보 목록을 반환한다 (가까운 순).

    열 단위 루프는 전부 numpy로 접었다 — 매 스텝(전투 중 매 틱) 부르는
    함수라 파이썬 루프면 예산을 많이 먹는다.
    """
    if sat is None:
        sat = geo.saturation(frame)
    rows, dists = boundary if boundary is not None else geo.floor_boundary(
        frame, n_cols, sat)
    cw = geo.FRAME_W // n_cols
    # 열 묶음 단위 채도맵 (FRAME_H, n_cols). 묶음 안에서는 상위값을 취해
    # 가느다란 몹이 배경에 묻히지 않게 한다.
    satc = sat[:, :n_cols * cw].reshape(geo.FRAME_H, n_cols, cw).max(axis=2)

    # 1) 바닥 접점 바로 위 FOOT_BAND 픽셀이 진한 색인가 = 바닥에 선 유채색 물체.
    y_hi = np.clip(rows.astype(int), int(geo.CY) + 1, geo.FRAME_H)
    offs = np.arange(FOOT_BAND)
    y_idx = np.clip(y_hi[:, None] - 1 - offs[None, :], int(geo.CY), geo.FRAME_H - 1)
    foot = satc[y_idx, np.arange(n_cols)[:, None]]        # (n_cols, FOOT_BAND)
    standing = np.median(foot, axis=1) >= MOB_SAT

    strong = satc >= MOB_SAT                               # (FRAME_H, n_cols)
    bearings = geo.column_bearings(n_cols)
    col_w = geo.FOV_X_DEG / n_cols

    # 2) 연속된 열 묶음으로 그룹화.
    mobs: List[Mob] = []
    edges = np.flatnonzero(np.diff(np.r_[False, standing, False]))
    for i, j in zip(edges[::2], edges[1::2]):
        seg = slice(int(i) * cw, int(j) * cw)
        d_floor = float(np.min(dists[i:j]))
        resolved = d_floor > geo.FLOOR_MIN_RANGE + 0.01

        # 3) 실루엣 꼭대기: 이 열 구간에서 진한 색이 이어지는 가장 윗 행.
        rows_strong = np.flatnonzero(strong[:, i:j].mean(axis=1) > 0.25)
        rows_strong = rows_strong[rows_strong >= geo.HUD_H]
        if len(rows_strong) == 0:
            continue
        y_top = float(rows_strong.min())

        # 나무 오탐 방지(실측 확인, dev_log.md): 나무는 키가 몹과 겹쳐서
        # (2.2m) 키만으로는 안 걸러지지만, 모양이 다르다 — 발밑(바닥에 닿은
        # 줄기)은 좁고 꼭대기(캐노피)는 훨씬 넓게 퍼진다. 사람형 몹은
        # 반대로 발밑(다리)이 머리보다 좁지 않다. 실루엣 맨 윗행에서
        # 좌우로 이어지는 진한색 폭을, 발밑 열 구간 폭과 비교해서 훨씬
        # 넓으면(캐노피 패턴) 후보에서 제외한다.
        top_row = int(y_top)
        lo, hi = int(i), int(j)
        while lo > 0 and strong[top_row, lo - 1]:
            lo -= 1
        while hi < n_cols and strong[top_row, hi]:
            hi += 1
        base_cols = int(j) - int(i)
        canopy_cols = hi - lo
        if base_cols > 0 and canopy_cols / base_cols > 1.6:
            continue

        if resolved:
            distance = d_floor
            height = geo.height_at(y_top, distance)
        else:
            # 바닥 접점이 화면 밖(2.6m 안쪽). 키 검산이 불가능하므로 대신
            # "실루엣이 지평선 근처까지 올라오는가"로 키를 가늠한다.
            if y_top > geo.CY + NEAR_TOP_MARGIN:
                continue
            cand = [geo.dist_for_height(y_top, h) for h in MOB_HEIGHTS]
            cand = [d for d in cand if 0.4 <= d <= geo.FLOOR_MIN_RANGE + 0.6]
            distance = min(cand) if cand else NEAR_DEFAULT_DIST
            height = None                      # 검산 불가 → 키 게이트 생략

        width_deg = abs(bearings[j - 1] - bearings[i]) + col_w
        score = _score(height, width_deg, distance, frame, seg, wall_rgb)
        if score > 0.0:
            mobs.append(Mob(
                bearing=(bearings[i] + bearings[j - 1]) / 2.0,
                distance=float(distance),
                height=float(height) if height is not None else -1.0,
                width_deg=float(width_deg), col_range=(int(i), int(j)),
                resolved=bool(resolved), score=float(score),
            ))

    mobs.sort(key=lambda m: m.distance)
    return mobs


def _score(height: Optional[float], width_deg: float, distance: float,
           frame: np.ndarray, seg: slice, wall_rgb) -> float:
    """몹다움 0~1. 0이면 후보에서 제외."""
    if height is not None and not (MIN_MOB_HEIGHT <= height <= MAX_MOB_HEIGHT):
        return 0.0                     # 통/콘/오리처럼 낮거나, 벽처럼 높다
    max_w = math.degrees(2 * math.atan(1.3 / max(distance, 0.5))) + 6.0
    if width_deg > max_w:
        return 0.0                     # 벽면처럼 넓게 퍼져 있다
    if _color_diversity(frame, seg) > MAX_COLOR_DIVERSITY:
        return 0.0                     # 실측 확인: 나무는 캐노피 모양이 몹 키(2.2m)와
                                        # 겹쳐서 키/너비만으론 안 걸러지는 경우가 있다
                                        # (dev_log.md). 몹은 셔츠/바지/피부가 각각 단색
                                        # (양자화 후 고유색 ~3개)인데, 나무는 면마다 음영이
                                        # 달라 색이 훨씬 다양하다(실측: 나무 23개 vs 몹 3개).
    earned, possible = 0.0, 0.0
    if wall_rgb is not None:
        possible += 1.0
        if not geo.color_close(geo.region_color(frame, seg.start, seg.stop),
                                wall_rgb, tol=45):
            earned += 1.0          # 방 벽색과 뚜렷이 다르다
    possible += 1.0
    if _vertical_color_layers(frame, seg) >= 2:
        earned += 1.0              # 셔츠/바지/피부가 세로로 쌓여 있다
    return 0.6 + 0.4 * (earned / possible if possible else 0.0)


def _vertical_color_layers(frame: np.ndarray, seg: slice,
                            tol: int = 45) -> int:
    """열 구간을 세로로 훑어 '뚜렷이 다른 진한 색' 층이 몇 개인지 센다.

    몹은 셔츠/바지/피부가 세로로 쌓여 2~3층이 나오고, 단색 소품이나 벽은
    1층이다.
    """
    band = frame[int(geo.CY) - 60:geo.FRAME_H, seg]
    if band.size == 0:
        return 0
    rows = np.median(band.reshape(band.shape[0], -1, 3), axis=1)
    keep = rows.max(axis=1) - rows.min(axis=1) >= MOB_SAT
    rows = rows[keep]
    if len(rows) == 0:
        return 0
    if len(rows) == 1:
        return 1
    jumps = np.abs(np.diff(rows, axis=0)).max(axis=1) > tol
    return int(jumps.sum()) + 1


def _color_diversity(frame: np.ndarray, seg: slice, quant: int = 24) -> int:
    """열 구간의 유채색 픽셀을 양자화해서 서로 다른 색이 몇 종류인지 센다.

    실측(tmp/verify_enemies.py 계열): 실제 몹은 셔츠/바지/피부가 각각
    단색 렌더링이라 양자화 후 고유색이 ~3개인데, 나무는 캐노피가 면마다
    다르게 음영져서 ~23개까지 나온다. 몹 키(2.2m)가 나무 키와 겹쳐서
    키/너비 게이트만으론 못 거르는 나무를 이걸로 추가로 거른다.
    """
    band = frame[int(geo.CY) - 60:geo.FRAME_H, seg]
    if band.size == 0:
        return 0
    pixels = band.reshape(-1, 3).astype(np.int32)
    sat = pixels.max(axis=1) - pixels.min(axis=1)
    colorful = pixels[sat > 30]
    if len(colorful) == 0:
        return 0
    quantized = (colorful // quant) * quant
    return int(len(np.unique(quantized, axis=0)))


# --- 근접(공격 사거리 이내) 전용 저비용 방위 추정 --------------------------
# 실측으로 발견(dev_log.md, seed=7 실제 전투 프레임): detect()의 바닥
# 접점/사람 비율 판정은 "적이 화면 중앙~하단에 사람 모양으로 작게 보이는"
# 중간 거리 기준이라, FLEE strike 진입 시점(=이미 맞아서 적이 1.3~1.5m
# 코앞에 있음, 상단 주석 참고)에는 몸통이 화면을 거의 다 채워 바닥이 안
# 보이고 실루엣 비율도 안 맞아 구조적으로 0개 탐지된다 — 그 결과 매
# 전투가 CV를 한 번도 못 쓰고 곧장 24방향 sweep(최대 72틱)으로 빠져서
# "예전처럼 즉각적으로 안 죽인다"는 원인이었다. 이 함수는 사람형 판정을
# 아예 안 하고 "바닥도 아니고 이 방의 벽색도 아닌 큰 덩어리"의 무게중심
# 방위만 구한다 — FLEE strike는 HP가 실제로 깎여서 진입한 상태라(=바로
# 앞에 뭔가 있다는 게 이미 확정됨) 일반 스캔과 달리 오탐 걱정이 적다.
_CLOSE_RANGE_MIN_FRAC = 0.15   # 이 이하로 "벽도 바닥도 아닌" 영역이면 무시
_CLOSE_RANGE_WALL_DIFF = 120   # RGB 채널합 기준 벽색과의 최소 차이
# 실측으로 발견(dev_log.md, seed=7): 적을 실제로 죽인 뒤에도(HP 0, 몸체가
# 렌더에서 제거됨) 문틈으로 보이는 다른 방/복도 풍경이 "바닥도 벽도 아닌
# 큰 덩어리" 조건을 계속 만족해서, 죽은 상대를 계속 "공격"(전부 빗나감)
# 하며 sweep까지 이어지는 낭비가 있었다. 진짜 근접 적은 발이 바로 앞
# 바닥에 닿아 있어서 몸체가 화면 맨 아래 몇 행까지 반드시 이어지는데
# (실측: 킬 직전 프레임은 맨 아래 20%의 29%가 후보 픽셀), 문틈 너머
# 풍경은 그 앞에 진짜 바닥이 있어서 맨 아래까지 안 내려온다(실측: 0%).
# 이 차이로 "코앞의 진짜 몸체"와 "멀리 문틈 너머 풍경"을 가른다.
_CLOSE_RANGE_BOTTOM_FRAC_MIN = 0.20  # 맨 아래 20% 구간에서 요구하는 최소 후보 비율
_CLOSE_RANGE_BOTTOM_BAND = 0.20      # 프레임 하단 몇 %를 "발밑" 구간으로 볼지
# 발밑 구간에서 "여기가 몸통이다"로 볼 열의 최소 세로 채움 비율. 몸통이
# 지나가는 열은 발밑 띠를 거의 다 채우고, 배경이 걸친 열은 드문드문하다.
_CLOSE_RANGE_RUN_COVER = 0.5


def close_range_bearing(frame: np.ndarray, wall_rgb: Optional[Tuple[int, int, int]] = None,
                         min_frac: float = _CLOSE_RANGE_MIN_FRAC) -> Optional[float]:
    """근접 전투 전용: 적일 가능성이 높은 큰 덩어리의 방위(도)만 빠르게 추정.

    None을 반환하면 호출자가 detect()(중간 거리용)나 sweep으로 넘어가면
    된다. HUD 아래 전체 프레임에서 "바닥(무채색 체커보드)도 아니고
    (wall_rgb가 있으면) 이 방 벽색과도 다른" 픽셀의 열(column) 방향
    무게중심을 방위로 환산한다. 화면 맨 아래(발밑)까지 이어지지 않는
    덩어리는 문틈 너머 풍경일 가능성이 높아 제외한다(위 주석 참고).
    """
    body = frame[geo.HUD_H:, :, :]
    sat = geo.saturation(body)
    not_floor = sat >= geo._SAT_THRESHOLD
    if wall_rgb is not None:
        diff = np.abs(body.astype(np.int32) - np.array(wall_rgb, dtype=np.int32)).sum(axis=-1)
        mask = not_floor & (diff > _CLOSE_RANGE_WALL_DIFF)
    else:
        mask = not_floor
    if mask.mean() < min_frac:
        return None
    bottom_start = int(mask.shape[0] * (1.0 - _CLOSE_RANGE_BOTTOM_BAND))
    bottom = mask[bottom_start:]
    if bottom.mean() < _CLOSE_RANGE_BOTTOM_FRAC_MIN:
        return None
    # 방위는 "발밑 띠에서 세로로 꽉 찬 열이 연속으로 이어지는 가장 넓은
    # 구간"의 중심으로 잡는다. 예전엔 후보 픽셀 전체의 열 무게중심을 썼는데,
    # 실측(close_frames 595장, env 내부 진짜 방위와 대조)으로 그게 심하게
    # 편향된다는 게 확인됐다 — 적이 화면 가장자리에 있어도 배경 잡음이
    # 무게중심을 화면 중앙으로 끌어당겨서, 진짜 적이 -40도에 있는데 -4도
    # (=공격 콘 안)라고 답하고 계속 헛스윙을 했다. 실측 비교:
    #   무게중심   중앙값 오차 42.1도, 공격 명중률 10%
    #   최대연속구간 중앙값 오차 10.7도, 공격 명중률 36%
    # 몸통은 발밑 띠를 세로로 꽉 채우며 가로로 이어지는 반면 배경/문틈은
    # 드문드문해서, 이 "가장 넓은 연속 구간"이 곧 몸통의 가로 위치가 된다.
    cover = bottom.mean(axis=0) >= _CLOSE_RANGE_RUN_COVER
    edges = np.flatnonzero(np.diff(np.r_[False, cover, False]))
    if len(edges) == 0:
        return None
    starts, ends = edges[::2], edges[1::2]
    widest = int(np.argmax(ends - starts))
    center_col = (int(starts[widest]) + int(ends[widest]) - 1) / 2.0
    return -math.degrees(math.atan((center_col - geo.CX) / geo.FOCAL))
