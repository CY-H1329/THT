"""적(사람형 몹) 탐지 — 픽셀에서 방위·거리·신뢰도를 뽑는다.

기하 휴리스틱("지도가 예측한 것보다 가까운 뭔가")만으로는 벽·문틀·통·
나무가 다 걸려서, 엉뚱한 걸 때리다 진짜 적에게 맞아 죽는 일이 잦았다.
몹은 렌더 방식이 명확히 달라서 그 특징을 직접 보는 편이 훨씬 정확하다.

몹의 생김새(miniworld Box 6개 + 얼굴 쿼드):

* 색이 **진하다**. 몸통/바지/피부는 miniworld의 원색 팔레트(red/green/
  blue/yellow/purple/grey)라 채도가 100~255로 찍힌다. 방 벽 팔레트는
  전부 탁한 색이라 같은 조명에서 75 이하다(실측).
* **바닥에 서 있다**. 이게 벽 사진 액자와 갈리는 결정적 차이다. 액자는
  높이 0.75~2.25m에 떠 있어서 바닥 경계선 바로 위가 벽색이지만, 몹은
  자기 색이 바닥 접점까지 이어진다. 사진은 실사라 채도가 높을 수 있어
  채도만으로는 절대 구분되지 않는다.
* **키가 1.5m(short) 또는 2.0m(tall)**. 실루엣 꼭대기 행으로 높이를
  검산하면 통(1.0m)·콘(0.6m)·오리(0.4m) 같은 소품이 걸러진다. 나무
  (2.2m)는 키가 비슷해서 이 단계에서는 안 걸러지고, 방을 스캔할 때
  기록해 둔 정적 장애물 목록(mapper.RoomNode.static_blobs)이 처리한다.

거리는 두 경로로 낸다. 2.6m 밖이면 바닥 접점 행에서 역투영(IPM)으로
직접 재고, 그보다 가까우면 바닥이 화면 밖이라 못 재므로 머리 꼭대기 행과
알려진 키로 거꾸로 추정한다. 어차피 2.6m 안쪽은 공격 사거리(3m) 안이라
정확한 거리보다 방위가 중요하다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from agent import geometry as geo

# 몹 박스로 볼 최소 채도. 벽 팔레트는 조명 아래에서 최대 75 근처, 몹의
# 원색 박스는 101~255로 찍힌다(개발 시드 실측).
MOB_SAT = 88
# 바닥 접점 바로 위에서 색을 읽을 띠의 두께(px).
FOOT_BAND = 10
MOB_HEIGHTS = (1.5, 2.0)      # short / tall
MIN_MOB_HEIGHT = 1.15
NEAR_TOP_MARGIN = 22       # 근거리 후보의 실루엣 꼭대기 허용 행(지평선 기준)
NEAR_DEFAULT_DIST = 1.8    # 키 역산이 발산할 때 쓸 근거리 대표값(m)
MAX_MOB_HEIGHT = 2.45


# miniworld가 Box 엔티티를 칠할 때 쓰는 6색 팔레트(miniworld.entity.COLORS).
# 몹의 셔츠/바지/피부는 이 팔레트에서만 뽑히고(world/enemies.py의
# _SHIRT/_PANTS/_SKIN_COLORS), 그 목록은 env **코드**에 박혀 있어서 평가
# 때 교체되는 에셋이 아니다. 반면 3D 오브젝트는 텍스처를 입힌 .obj 메시라
# 이 방향들에 잘 걸리지 않는다.
#
# 실측(개발 시드, 벽 픽셀 제외): 팔레트 방향과 6° 안에 드는 유채색 픽셀의
# 비율이 적 0.81~0.94, 공개 오브젝트 0.00~0.03. 예외는 duckie(0.79)인데
# 키 0.4m라 높이 게이트에서 이미 걸린다. 조명은 색을 밝기만 바꾸므로
# 절대 RGB가 아니라 **RGB 벡터의 방향**으로 판정한다.
MW_PALETTE = {
    "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
    "purple": (0.44, 0.15, 0.76), "yellow": (1.0, 1.0, 0.0),
    "grey": (0.39, 0.39, 0.39),
}
_PAL_NAMES = tuple(MW_PALETTE)
_PAL_VECS = np.array([np.array(v) / np.linalg.norm(v)
                      for v in MW_PALETTE.values()])
PAL_TOL_DEG = 8.0
_PAL_MIN_SAT = 55


def palette_pixels(frame: np.ndarray, col_lo: int, col_hi: int) -> int:
    """열 구간에서 몹 팔레트 방향에 들어맞는 유채색 픽셀 수.

    '실루엣이 아직 거기 있는가'를 세는 용도다(motion.py의 사망 판정).
    벽 팔레트는 탁해서 _PAL_MIN_SAT를 넘지 못하고, 3D 오브젝트 텍스처는
    팔레트 방향에 거의 걸리지 않는다(실측 0.00~0.03).
    """
    lo = max(0, min(geo.FRAME_W - 1, int(col_lo)))
    hi = max(lo + 1, min(geo.FRAME_W, int(col_hi)))
    band = frame[geo.HUD_H:geo.FRAME_H, lo:hi].astype(np.float32)
    sat = band.max(axis=2) - band.min(axis=2)
    ok = (sat >= _PAL_MIN_SAT) & (band.max(axis=2) >= 30)
    if not ok.any():
        return 0
    px = band[ok]
    v = px / (np.linalg.norm(px, axis=1, keepdims=True) + 1e-6)
    err = np.degrees(np.arccos(np.clip(v @ _PAL_VECS.T, -1.0, 1.0))).min(axis=1)
    return int((err <= PAL_TOL_DEG).sum())


def palette_sample(frame: np.ndarray, col_lo: int, col_hi: int) -> List[str]:
    """열 구간에서 읽어낸 몹 팔레트 색 이름들(위→아래, 중복 제거).

    QA용이다("적이 무슨 색이었나"). 몹은 위에서부터 피부/셔츠/바지 순으로
    쌓여 있으므로 순서를 유지해서 돌려준다. 탐지 판정에는 쓰지 않는다 —
    탐지는 motion.py의 움직임 근거가 맡는다.
    """
    lo = max(0, min(geo.FRAME_W - 1, int(col_lo)))
    hi = max(lo + 1, min(geo.FRAME_W, int(col_hi)))
    band = frame[geo.HUD_H:geo.FRAME_H, lo:hi].astype(np.float32)
    sat = band.max(axis=2) - band.min(axis=2)
    ok = (sat >= _PAL_MIN_SAT) & (band.max(axis=2) >= 30)
    rows = np.flatnonzero(ok.any(axis=1))
    out: List[str] = []
    for y in rows:
        px = band[y][ok[y]]
        med = np.median(px, axis=0)
        n = np.linalg.norm(med)
        if n < 1e-6:
            continue
        err = np.degrees(np.arccos(np.clip(_PAL_VECS @ (med / n), -1.0, 1.0)))
        k = int(err.argmin())
        if err[k] > PAL_TOL_DEG:
            continue
        name = _PAL_NAMES[k]
        if not out or out[-1] != name:
            out.append(name)
    # 같은 색이 떨어져서 두 번 나오면(팔에 가려진 셔츠 등) 한 번만 남긴다.
    seen: List[str] = []
    for n in out:
        if n not in seen:
            seen.append(n)
    return seen


@dataclass
class Mob:
    bearing: float        # 현재 heading 기준 상대 방위(도, 좌 +)
    distance: float       # m (2.6m 안쪽은 키 기반 추정치)
    height: float         # 추정 실루엣 높이(m)
    width_deg: float
    col_range: Tuple[int, int]
    resolved: bool        # 바닥 접점으로 거리를 실측했는지
    score: float          # 0~1, 몹다움


def detect(frame: np.ndarray, n_cols: int = 64,
           wall_rgb: Optional[tuple] = None,
           sat: Optional[np.ndarray] = None,
           boundary: Optional[tuple] = None) -> List[Mob]:
    """프레임에서 몹 후보 목록을 반환한다 (가까운 순).

    열 단위 루프는 전부 numpy로 접었다(프레임당 ~1.5ms). 매 스텝 돌리는
    함수라 파이썬 루프로 두면 에이전트 예산의 절반을 먹는다.
    """
    if sat is None:
        sat = geo.saturation(frame)
    rows, dists = boundary if boundary is not None else geo.floor_boundary(
        frame, n_cols, sat)
    cw = geo.FRAME_W // n_cols
    # 열 묶음 단위 채도맵 (FRAME_H, n_cols). 묶음 안에서는 상위 30%를 취해
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
        resolved = d_floor > geo.NEAR_RANGE + 0.01

        # 3) 실루엣 꼭대기: 이 열 구간에서 진한 색이 이어지는 가장 윗 행.
        rows_strong = np.flatnonzero(strong[:, i:j].mean(axis=1) > 0.25)
        rows_strong = rows_strong[rows_strong >= geo.HUD_H]
        if len(rows_strong) == 0:
            continue
        y_top = float(rows_strong.min())

        if resolved:
            distance = d_floor
            height = geo.height_at(y_top, distance)
        else:
            # 바닥 접점이 화면 밖(2.6m 안쪽). 키 검산이 불가능하므로 대신
            # "실루엣이 지평선 근처까지 올라오는가"로 키를 가늠한다. 통
            # (1.0m)·콘(0.6m)·오리(0.4m)는 이 거리에서 꼭대기가 지평선보다
            # 한참 아래에 찍힌다.
            if y_top > geo.CY + NEAR_TOP_MARGIN:
                continue
            cand = [geo.dist_for_height(y_top, h) for h in MOB_HEIGHTS]
            cand = [d for d in cand if 0.4 <= d <= geo.FLOOR_MIN_RANGE + 0.6]
            # 키가 카메라 높이(1.5m)와 겹치면 역산이 발산한다(short 몹의
            # 머리 꼭대기가 정확히 지평선). 그럴 땐 근거리 대표값을 쓴다.
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
    # 폭: 팔 벌린 몹이 1.2m를 넘지 않는다. 그 거리에서의 각폭 + 여유.
    max_w = math.degrees(2 * math.atan(1.3 / max(distance, 0.5))) + 6.0
    if width_deg > max_w:
        return 0.0                     # 벽면처럼 넓게 퍼져 있다
    # 점수는 "확인할 수 있었던 단서 중 몇 개가 통과했는가"로 낸다. 아직
    # 스캔하지 않은 방에서는 벽색을 모르는데, 예전엔 그 가점을 못 받아
    # 점수가 0.8에서 막혔다. 그래서 화면을 가득 채운 1.8m 앞의 몹을
    # "확신 부족"으로 버리고 130스텝을 헛돌다 죽은 판이 있었다.
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
