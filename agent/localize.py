"""360° 스캔 ↔ 방 사각형 정합(scan matching)으로 방 안 절대 위치를 찾는다.

방은 한 변이 `cell`인 정사각형이고 벽은 축에 나란하다. 따라서 방 안의
어떤 위치 후보 (x, z)를 가정하면, 스캔의 각 광선이 "벽까지 몇 m여야
하는지"가 닫힌 형태로 계산된다. 실제 측정과 맞는 광선 수(inlier)를
최대로 만드는 후보가 우리 위치다.

중앙값 같은 통계 대신 이 방식을 쓰는 이유: 문 앞에 서 있으면 그 벽
방향 광선이 통째로 문을 통과해 버려서 "그 벽까지 거리"의 중앙값 자체가
틀린다. 정합은 그런 광선을 그냥 outlier로 흘리고 나머지 세 벽으로
위치를 잡는다. 오브젝트·적에 가려 짧게 찍힌 광선도 마찬가지다.

비용: 0.25m 격자(방 10m 기준 후보 1600개) × 광선 320개를 numpy로 한 번에
계산해 ~10ms. 방에 들어갈 때 한 번만 돌린다.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from agent import geometry as geo

INLIER_TOL = 0.4        # 이 안이면 벽을 맞힌 광선으로 인정
_GRID = 0.25            # 후보 격자 간격 (m)
_MARGIN = 0.45          # 벽에 몸이 붙을 수 없는 최소 거리 (agent radius)


def _boundary_distance(x: np.ndarray, z: np.ndarray,
                       hx: np.ndarray, hz: np.ndarray,
                       cell: float) -> np.ndarray:
    """후보 위치 (C,) × 광선 방향 (R,) → 방 경계까지 거리 (C, R)."""
    x = x[:, None]
    z = z[:, None]
    hx = hx[None, :]
    hz = hz[None, :]
    big = np.float64(1e9)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(hx > 1e-9, (cell - x) / hx,
                      np.where(hx < -1e-9, (0.0 - x) / hx, big))
        tz = np.where(hz > 1e-9, (cell - z) / hz,
                      np.where(hz < -1e-9, (0.0 - z) / hz, big))
    return np.minimum(tx, tz)


def match_scan(rays: Sequence[Tuple[float, float]], cell: float,
               stride: int = 3) -> Tuple[float, float, float]:
    """(방 안 x, 방 안 z, 인라이어 비율). rays는 (절대 heading, 거리) 목록."""
    if not rays:
        return cell / 2.0, cell / 2.0, 0.0
    arr = np.asarray(rays, dtype=np.float64)[::stride]
    ang = np.radians(arr[:, 0])
    d = arr[:, 1]
    # 지평선까지 뚫린 광선(MAX_RANGE)은 거리 정보가 없으므로 뺀다.
    keep = d < geo.MAX_RANGE - 0.01
    ang, d = ang[keep], d[keep]
    if len(d) < 20:
        return cell / 2.0, cell / 2.0, 0.0
    hx, hz = np.cos(ang), -np.sin(ang)

    lo, hi = _MARGIN, cell - _MARGIN
    axis = np.arange(lo, hi + 1e-6, _GRID)
    gx, gz = np.meshgrid(axis, axis, indexing="ij")
    cand_x, cand_z = gx.ravel(), gz.ravel()

    t = _boundary_distance(cand_x, cand_z, hx, hz, cell)
    resid = d[None, :] - t
    inliers = (np.abs(resid) <= INLIER_TOL).sum(axis=1)
    best = int(inliers.argmax())
    return float(cand_x[best]), float(cand_z[best]), float(inliers[best]) / len(d)


def match_all(rays: Sequence[Tuple[float, float]],
              cell_candidates: Sequence[float],
              stride: int = 3) -> dict:
    """셀 크기 후보별 정합 결과 {cell: (x, z, score)}."""
    return {c: match_scan(rays, c, stride=stride) for c in cell_candidates}


def wall_openings(rays: Sequence[Tuple[float, float]], x: float, z: float,
                  cell: float, min_votes: int = 3, min_hits: int = 3,
                  door_half: float = 1.25) -> dict:
    """벽별 개구부(문) 판정. 판단 근거가 없으면 아예 답하지 않는다.

    문은 항상 벽 정중앙 폭 2m이므로, **그 구간을 지나는 광선만** 본다.

    * 그중 min_votes개 이상이 벽면을 지나쳐서 끝났다 → 문
    * 지나친 게 없고 min_hits개 이상이 정확히 벽면에서 끝났다 → 벽
    * 둘 다 아니면(오브젝트에 가려 문 구간을 못 봤다 등) 판정 보류

    보류를 남기는 게 중요하다. 예전엔 "표가 0이면 벽"으로 단정했는데,
    가구에 가려 문 앞을 못 본 방에서 진짜 문을 벽으로 확정해 버려서 지도
    절반이 통째로 잘려 나갔다(개발 시드에서 9개 방 중 3~5개만 탐색).
    보류로 두면 PROBE가 직접 밀어보고 확정한다.
    """
    votes = {"E": 0, "W": 0, "N": 0, "S": 0}
    hits = {"E": 0, "W": 0, "N": 0, "S": 0}
    for h, d in rays:
        hx, hz = geo.heading_to_vec(h)
        t, wall = _exit(x, z, hx, hz, cell)
        if wall is None:
            continue
        cx, cz = x + hx * t, z + hz * t
        mid = cz if wall in ("E", "W") else cx
        if abs(mid - cell / 2.0) > door_half:
            continue                      # 문이 있을 수 있는 구간이 아니다
        if d > t + 0.75:
            votes[wall] += 1
        elif abs(d - t) <= 0.5:
            hits[wall] += 1
    out = {}
    for w in votes:
        if votes[w] >= min_votes:
            out[w] = "door"
        elif votes[w] == 0 and hits[w] >= min_hits:
            out[w] = "wall"
    return out


def obstacle_points(rays: Sequence[Tuple[float, float]], x: float, z: float,
                    cell: float, margin: float = 0.6) -> List[Tuple[float, float]]:
    """방 경계보다 확실히 앞에서 끝난 광선 = 방 안 장애물(오브젝트/적) 밑동."""
    pts = set()
    for h, d in rays:
        if d >= geo.MAX_RANGE - 0.01:
            continue
        hx, hz = geo.heading_to_vec(h)
        t, wall = _exit(x, z, hx, hz, cell)
        if wall is None or d >= t - margin:
            continue
        pts.add((round((x + hx * d) * 2) / 2, round((z + hz * d) * 2) / 2))
    return sorted(pts)


def _exit(x: float, z: float, hx: float, hz: float,
          cell: float) -> Tuple[float, Optional[str]]:
    """방 사각형(0..cell)에서 광선이 나가는 거리와 그 벽."""
    best_t, best_wall = float("inf"), None
    for wall, num, den in (
        ("E", cell - x, hx),
        ("W", 0.0 - x, hx),
        ("N", cell - z, hz),
        ("S", 0.0 - z, hz),
    ):
        if abs(den) < 1e-9:
            continue
        t = num / den
        if 0.05 < t < best_t:
            best_t, best_wall = t, wall
    return best_t, best_wall
