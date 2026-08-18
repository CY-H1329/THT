"""열쇠(노란 Key 메시) 탐지.

열쇠는 `Key(color="yellow")`로 생성된다(env.py::_spawn_key). 색이 env
**코드**에 박혀 있어서 평가 때 교체되는 에셋이 아니다 — 벽 사진이나 3D
오브젝트와 달리 "노란 열쇠"라는 사실은 held-out 시드에서도 그대로다.

네 가지를 함께 봐야 한다. 하나만 쓰면 전부 오검출로 무너진다:

* **순수 노랑**. miniworld 팔레트의 yellow는 (1,1,0)이라 파랑 채널이
  바닥에 붙는다. 반면 벽 팔레트의 노란 계열(sand 220,200,160 /
  mustard 210,175,55)은 파랑이 살아 있다. 실측에서 시드 7의 열쇠 방이
  바로 sand 벽이라, 밝기만 보는 필터는 벽 전체를 열쇠로 잡았다.
* **지평선 아래**. 열쇠는 높이 0.35m라 언제나 지평선 밑에 찍힌다.
* **작다**. 거리 d에서 열쇠의 픽셀 크기는 대략 FOCAL*0.35/d(세로)로
  정해져 있다. 나무 잎사귀 덩어리는 이 상한을 크게 넘는다 — 실측 시드 0
  에서 잎이 2699px를 차지해 그 안에 열쇠가 통째로 묻혔다.
* **밑동이 가지런하다**. 바닥에 놓인 물체는 열마다 가장 아랫 노랑 행이
  비슷하다. 공중에 뜬 잎사귀는 들쭉날쭉하다. 이걸로 잎과 열쇠를 가른다.

탐지는 어디까지나 **가속기**다. 놓쳐도 탐색 쪽에서 방을 훑는 커버리지
주행이 열쇠를 주워 준다(획득 반경 0.8m). 그래서 놓치는 것보다 헛짚는
쪽이 비싸고, 게이트를 보수적으로 잡았다.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from agent import geometry as geo

KEY_HEIGHT = 0.35
MIN_PIXELS = 25
MAX_RANGE = 12.0
# 밑동 행이 이보다 더 어긋나면 같은 물체로 보지 않는다(px).
FOOT_TOL = 6
# 열쇠 세로 픽셀 상한 배율. 0.45m는 열쇠 높이 + 여유.
HEIGHT_SLACK = 1.8


def yellow_mask(frame: np.ndarray) -> np.ndarray:
    """순수 노랑(파랑 채널이 죽은 유채색) 픽셀 마스크. HUD 아래만."""
    b = frame[geo.HUD_H:, :, :].astype(np.int16)
    r, g, bl = b[:, :, 0], b[:, :, 1], b[:, :, 2]
    mx = np.maximum(r, g)
    return (
        (mx > 90)
        & (np.abs(r - g) < 0.30 * np.maximum(mx, 1))   # r≈g
        & (bl * 100 < mx * 45)                          # 파랑이 확실히 죽었다
    )


def _foot_groups(lowest: np.ndarray) -> List[Tuple[int, int]]:
    """밑동 행이 이어지는 열 묶음 구간 [i, j)들."""
    out: List[Tuple[int, int]] = []
    i = 0
    n = len(lowest)
    while i < n:
        if lowest[i] < 0:
            i += 1
            continue
        j = i + 1
        while j < n and lowest[j] >= 0 and abs(int(lowest[j]) - int(lowest[j - 1])) <= FOOT_TOL:
            j += 1
        out.append((i, j))
        i = j
    return out


def detect(frame: np.ndarray, n_cols: int = 64
           ) -> Optional[Tuple[float, float]]:
    """가장 그럴듯한 열쇠의 (상대 방위 deg, 거리 m). 없으면 None."""
    mask = yellow_mask(frame)
    horizon = max(0, int(geo.CY) - geo.HUD_H)
    mask[:horizon, :] = False
    if int(mask.sum()) < MIN_PIXELS:
        return None

    cw = geo.FRAME_W // n_cols
    binned = mask[:, :n_cols * cw].reshape(mask.shape[0], n_cols, cw).any(axis=2)
    bearings = geo.column_bearings(n_cols)

    lowest = np.full(n_cols, -1, dtype=int)
    for k in range(n_cols):
        rows = np.flatnonzero(binned[:, k])
        if len(rows):
            lowest[k] = int(rows.max())

    best: Optional[Tuple[float, float, int]] = None   # (bearing, dist, px)
    for i, j in _foot_groups(lowest):
        px = int(mask[:, i * cw:j * cw].sum())
        if px < MIN_PIXELS:
            continue
        rows = np.flatnonzero(binned[:, i:j].any(axis=1))
        y_bot = int(np.median(lowest[i:j])) + geo.HUD_H
        y_top = int(rows.min()) + geo.HUD_H
        col_mid = (i + j) * cw / 2.0

        if y_bot >= geo.FRAME_H - 1:
            dist = 1.2                    # 밑동이 화면 밖 = 아주 가깝다
        else:
            dist = geo.dist_at_row(y_bot, col_mid)
        if not (0.8 <= dist <= MAX_RANGE):
            continue
        if (y_bot - y_top + 1) > HEIGHT_SLACK * geo.FOCAL * 0.45 / dist:
            continue                      # 잎사귀처럼 세로로 길다
        if best is None or px > best[2]:
            best = (float((bearings[i] + bearings[j - 1]) / 2.0), float(dist), px)

    return (best[0], best[1]) if best else None
