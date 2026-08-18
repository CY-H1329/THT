"""정지 프레임 차분으로 적을 찾는다 — 외형을 전혀 보지 않는 탐지기.

핵심 사실 하나에서 출발한다. **이 env의 렌더러는 완전 결정적이다.**
domain_rand=False이고 애니메이션도 노이즈도 없다. 살아 있는 적을 전부
없앤 뒤 3.6 시뮬초를 흘려보내며 프레임을 비교하면 바뀐 픽셀이 0개,
최대 채널차가 0이다(실측). 그리고 이 월드에서 **움직이는 것은 적뿐이다**
— 벽·사진·3D 오브젝트·열쇠·문은 전부 정적이다.

따라서 에이전트가 제자리에 있는 동안 HUD 아래에서 바뀐 픽셀은 곧 적이다.
색·질감·형태·얼굴을 하나도 보지 않으므로, held-out 에셋(적 얼굴, 3D
오브젝트, 벽 사진, 방 이름)이 무엇으로 바뀌든 이 탐지기는 영향을 받지
않는다. enemies.py의 외형 휴리스틱과 달리 일반화 걱정이 원리적으로 없다.

'제자리'의 조건:

* ``NO_OP`` / ``ATTACK`` — ATTACK은 레이캐스트만 하고 에이전트를 전혀
  움직이지 않는다(실측: 50회 연속 공격 후 이동 0.000m, 회전 0.000rad,
  바뀐 픽셀 0개). 그래서 **때리는 동안에도 탐지가 공짜로 돌아간다**.
* 막힌 MOVE_FORWARD/BACK — miniworld는 부분 이동이 없어서 충돌하면
  위치가 그대로다.

회전은 충돌 판정이 없어 항상 성공하므로 정지 쌍이 될 수 없다.

민감도(실측, 4스텝=0.05 시뮬초 유지 후 차분):

    4m 279px | 6m 87px | 8m 182px | 12m 1205px | 16m 미검출(벽 뒤)

1스텝(0.012 시뮬초)만으로도 5m 앞의 적이 76px로 잡힌다. 즉 "몇 초 멈추기"가
아니라 "한두 스텝 안 움직이기"면 충분하다. 탐지 거리도 외형 탐지기의
2.4~3m보다 훨씬 길어서, 방을 가로질러 오는 적을 문간에서부터 본다.

**한계**: 적이 1.5m 안에 들어오면 attack 상태가 되어 완전히 멈춘다(실측
2.4초간 이동 0.000m). 가장 위험한 순간에 이 탐지기는 눈이 먼다. 그
구간은 (a) 적이 멈췄으므로 여전히 유효한 트랙의 마지막 위치와, (b) HP
감소 신호가 대신 메운다. mapper.EnemyTrack / explorer._threat 참고.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from agent import enemies as EN
from agent import geometry as geo

# 정적 대조군에서 최대 채널차가 0이므로 임계값은 사실상 자유롭다. 다른
# GPU/드라이버에서 미세한 디더링이 생길 가능성만 감안해 여유를 둔다.
DIFF_THRESH = 8
# 이만큼은 바뀌어야 블롭으로 인정. 정적 대조군이 0px라 낮게 잡아도 되지만,
# 12m 밖 적도 80px 이상 나오므로 굳이 극단까지 내리지 않는다.
MIN_PIXELS = 10
# 화면이 이 비율 이상 바뀌었으면 적이 아니라 자기 이동이다(추측항법이
# 틀려서 정지로 오판한 경우). 안전판.
MAX_CHANGED_FRAC = 0.25
# 열 묶음 사이 이만큼의 빈틈은 같은 블롭으로 잇는다. 적의 '이전 위치'와
# '현재 위치' 실루엣이 떨어져 찍히면 두 조각으로 갈리기 때문이다.
GAP_BINS = 2
# 실루엣이 '사라졌다'고 보려면 직전 프레임에 이만큼은 팔레트 색 픽셀이
# 있어야 한다. 2~3m 거리의 몹은 수백~수천 px이라 여유 있게 잡는다.
VANISH_MIN_PIXELS = 120
# 사망 후 남아도 되는 팔레트 픽셀 비율. 0이 아닌 이유는 죽은 적 뒤로 다른
# 적이 겹쳐 보이거나 실루엣 가장자리가 조금 남을 수 있어서다.
VANISH_RATIO = 0.25


@dataclass
class MotionBlob:
    bearing: float          # 현재 heading 기준 상대 방위(도, 좌 +)
    distance: float         # m (바닥 접점 행 역투영; 2.6m 안쪽은 포화)
    pixels: int             # 바뀐 픽셀 수
    col_range: Tuple[int, int]   # 원본 프레임 열 범위
    resolved: bool          # 바닥 접점으로 거리를 실측했는지
    # 이 열 구간에서 몹 팔레트 색 픽셀이 '이전 프레임엔 있었는데 지금은
    # 없다' = 실루엣이 통째로 사라졌다 = 방금 죽었다. 움직임과는 갈린다:
    # 몹이 걸어서 옮겨간 경우엔 새 위치의 픽셀이 여전히 팔레트 색이다.
    vanished: bool = False
    colors_before: List[str] = None   # 사라지기 직전의 팔레트 색(QA용)


class MotionDetector:
    """직전 프레임을 들고 있다가, 포즈가 같을 때만 차분한다.

    포즈 동일 여부는 추측항법 좌표(explorer.PoseTracker)로 판정한다.
    '움직였다고 착각한 이동'이 있으면 프레임 전체가 바뀌는데, 그건
    MAX_CHANGED_FRAC 안전판이 걸러낸다.
    """

    def __init__(self, n_cols: int = 64):
        self.n_cols = n_cols
        self._prev: Optional[np.ndarray] = None
        self._prev_key: Optional[tuple] = None
        self.last_changed = 0        # 진단용: 직전 유효 차분의 픽셀 수
        # 이번 observe에서 실제로 차분을 돌렸는지. 호출 측은 이 값으로
        # "관측했는데 조용했다"와 "관측 자체를 못 했다"를 구분해야 한다.
        # 포즈 키는 HUD heading 보정으로도 흔들릴 수 있어서, stationary
        # 플래그만으로는 쌍이 성립했는지 알 수 없다.
        self.paired = False

    def observe(self, frame: np.ndarray, pose_key: tuple, stationary: bool,
                banner: bool) -> List[MotionBlob]:
        """(블롭 목록). 정지 쌍이 아니거나 배너 중이면 빈 목록."""
        prev, prev_key = self._prev, self._prev_key
        self.paired = False
        # 배너 프레임은 화면 중앙 55%를 덮어 차분이 무의미하다. 다음 쌍이
        # 오염되지 않도록 직전 프레임 자체를 버린다.
        if banner:
            self._prev, self._prev_key = None, None
            return []
        self._prev, self._prev_key = frame, pose_key

        if prev is None or not stationary or prev_key != pose_key:
            return []
        self.paired = True
        return self._blobs(prev, frame)

    def _blobs(self, prev: np.ndarray, cur: np.ndarray) -> List[MotionBlob]:
        # HUD(0~28행)는 초 단위 타이머가 계속 바뀌므로 통째로 제외한다.
        a = prev[geo.HUD_H:].astype(np.int16)
        b = cur[geo.HUD_H:].astype(np.int16)
        mask = np.abs(a - b).max(axis=2) >= DIFF_THRESH
        self.last_changed = int(mask.sum())
        if self.last_changed < MIN_PIXELS:
            return []
        if mask.mean() > MAX_CHANGED_FRAC:
            return []                       # 자기 이동 — 적으로 볼 수 없다

        n, cw = self.n_cols, geo.FRAME_W // self.n_cols
        binned = mask[:, :n * cw].reshape(mask.shape[0], n, cw).any(axis=2)
        hit = binned.any(axis=0)
        bearings = geo.column_bearings(n)

        out: List[MotionBlob] = []
        for i, j in _runs(hit, GAP_BINS):
            seg = binned[:, i:j]
            px = int(mask[:, i * cw:j * cw].sum())
            if px < MIN_PIXELS:
                continue
            # 블롭의 바닥 접점 행 = 적의 발. 그 행을 역투영하면 거리가
            # 나온다. 다만 **열마다 따로 구해 중앙값**을 쓴다. 전체 최대
            # 행을 쓰면, GAP_BINS로 이어 붙인 두 조각(예: 깊이가 다른 두
            # 적)이 한 블롭이 됐을 때 가까운 쪽의 바닥 행이 블롭 전체의
            # 거리가 되어, 먼 적이 코앞 좌표로 찍힌다. 실측 시드 7에서
            # 진짜 적과 10m 어긋난 유령 트랙이 이렇게 만들어졌고, 거기에
            # 조준 공격 168회가 들어갔다.
            per_col = [int(np.flatnonzero(seg[:, k]).max())
                       for k in range(seg.shape[1]) if seg[:, k].any()]
            if not per_col:
                continue
            y_bot = int(np.median(per_col)) + geo.HUD_H
            resolved = y_bot < geo.FRAME_H - 1
            col_mid = (i + j) * cw / 2.0
            dist = (geo.dist_at_row(y_bot, col_mid) if resolved
                    else geo.NEAR_RANGE)
            c0, c1 = int(i * cw), int(j * cw)
            before = EN.palette_pixels(prev, c0, c1)
            after = EN.palette_pixels(cur, c0, c1)
            vanished = before >= VANISH_MIN_PIXELS and after <= before * VANISH_RATIO
            out.append(MotionBlob(
                bearing=float((bearings[i] + bearings[j - 1]) / 2.0),
                distance=float(dist), pixels=px,
                col_range=(c0, c1), resolved=bool(resolved),
                vanished=bool(vanished),
                colors_before=(EN.palette_sample(prev, c0, c1)
                               if vanished else []),
            ))
        out.sort(key=lambda m: m.distance)
        return out

    def invalidate(self) -> None:
        """다음 스텝의 차분을 포기한다(포즈 보정 등으로 기준이 깨졌을 때)."""
        self._prev, self._prev_key = None, None


def _runs(flags: np.ndarray, gap: int) -> List[Tuple[int, int]]:
    """True 구간 [i, j)들. `gap` 이하의 빈틈은 이어 붙인다."""
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    out: List[Tuple[int, int]] = []
    start = prev = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k - prev - 1 > gap:
            out.append((start, prev + 1))
            start = k
        prev = k
    out.append((start, prev + 1))
    return out
