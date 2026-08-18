"""벽 사진 기억 — 지나가면서 액자를 잡아 정면 크롭으로 펴서 방에 붙인다.

파이프라인은 세 단계다.

1. **키프레임 선택.** 사진은 여러 프레임에 걸쳐 보이고, 대부분은 비스듬하거나
   작거나 잘려 있다. 매 프레임 캡션을 뜨는 건 낭비이자 품질 저하다. 그래서
   사각형을 프레임 간에 추적하고(중심 근접), 트랙이 끝날 때 **가장 좋았던
   한 프레임만** 확정한다. 점수는 면적 × 정면성이다.
2. **직사각 크롭.** Canny → contour → 4각형 근사 → 원근 워프. 이 env는
   벽이 단색이고 액자는 텍스처라 에지가 아주 깨끗하게 나온다.
3. **저장.** 크롭·캡션·서명을 방 노드(mapper.RoomNode.photos)에 붙인다.

**진짜 어려운 부분은 사각형 찾기가 아니라 '이게 사진인가'다.** 순진하게
4각형을 다 받으면 단색 벽면, 바닥 체커보드, 몹의 몸통, 잠긴 문의 자물쇠
판이 전부 걸린다(실측: 14개 후보 중 사진은 4개뿐이었다). 그래서 세 가지로
거른다. 셋 다 held-out 에셋과 무관한 근거만 쓴다:

* **팔레트 비율** — 벽과 몹은 miniworld 팔레트 단색이라 팔레트 방향에
  붙는 픽셀이 92~100%다. 사진은 0%다(enemies.palette_pixels 재사용).
* **자물쇠 서명** — 따뜻한 노랑 바탕 + 진회색 아이콘. env 코드에 박힌
  고정 텍스처라 평가에서도 그대로다.
* **색 다양성 / 디테일** — 단색 벽(고유색 1개, 라플라시안 0.0)과 바닥
  무아레를 걷어낸다.

실측 14개 후보에서 이 게이트가 14/14로 사진만 골라냈다. 여유도 크다 —
사진은 팔레트 비율 0.00, 벽/몹은 0.92 이상이다.

캡션은 **선택 사항**이다. `PhotoCapture(captioner=...)`에 vlm.Captioner를
넘기면 사진이 확정되는 순간 캡션 요청이 큐에 들어간다(제출 자체는 0.3ms라
act를 블로킹하지 않는다). 키가 없으면 크롭과 서명만 남는다.
서명만으로도 "같은 카테고리 사진 두 장이 있는 방"(힌트 템플릿 3) 같은
유사도 질문은 답할 수 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from agent import enemies as EN
from agent import geometry as geo

try:                                   # OpenCV는 [play] extra라 없을 수 있다.
    import cv2
except ImportError:                    # pragma: no cover
    cv2 = None

CROP_PX = 128
MIN_QUAD_AREA = 900          # 이보다 작으면 캡션 품질이 안 나온다
MIN_COMMIT_AREA = 1600       # 확정(키프레임)으로 인정할 최소 면적
TRACK_MAX_GAP = 3            # 이 스텝 이상 안 보이면 트랙 종료
TRACK_MATCH_PX = 45          # heading 보정 후 중심이 이 거리 안이면 같은 사각형
# 같은 방 안에서 서명 코사인이 이 이상이면 같은 사진으로 보고 합친다.
DEDUP_COS = 0.90

# --- 사진다움 게이트 (위 주석의 실측 근거) ---
MAX_PALETTE_FRAC = 0.25      # 벽/몹은 0.92~1.00, 사진은 0.00
MIN_COLORS = 14              # 단색 벽 1, 바닥 무아레 10
MIN_DETAIL = 5.5             # 라플라시안 표준편차
LOCK_WARM_FRAC = 0.30        # 자물쇠 판: 따뜻한 노랑 비율
LOCK_DARK_FRAC = 0.05        # 그 안의 진회색 아이콘


@dataclass
class Photo:
    """확정된 사진 한 장."""
    room_key: str
    wall: Optional[str]              # 'N'|'E'|'S'|'W' (추정)
    crop: np.ndarray                 # CROP_PX x CROP_PX RGB, 정면으로 편 것
    sig: np.ndarray                  # 유사도 비교용 서명
    area: float                      # 확정 당시 화면 면적(px)
    step: int
    caption: str = ""                # VLM이 붙기 전까지는 빈 문자열
    bearing: float = 0.0
    distance: float = 0.0


@dataclass
class _Track:
    center: Tuple[float, float]
    best_score: float
    best_quad: np.ndarray
    best_frame: np.ndarray
    last_step: int


def signature(crop: np.ndarray, n: int = 16) -> np.ndarray:
    """조명 배율에 둔감한 작은 색 서명. 코사인 비교용으로 정규화."""
    if cv2 is not None:
        small = cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA)
    else:                                            # pragma: no cover
        small = np.array(crop[::max(1, crop.shape[0] // n),
                              ::max(1, crop.shape[1] // n)][:n, :n])
    v = small.astype(np.float32).reshape(-1)
    v -= v.mean()
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 1e-6 else v


def looks_like_photo(crop: np.ndarray) -> bool:
    """크롭이 실제 벽 사진인지. 벽/몹/자물쇠/바닥을 걸러낸다."""
    if cv2 is None:
        return False
    h, w = crop.shape[:2]
    # 팔레트 단색면(벽·몹). palette_pixels는 HUD 아래를 보므로 위를 덧대준다.
    padded = np.vstack([np.zeros((geo.HUD_H, w, 3), np.uint8), crop])
    if EN.palette_pixels(padded, 0, w) / float(h * w) > MAX_PALETTE_FRAC:
        return False
    # 잠긴 문의 자물쇠 판
    warm = float(geo._warm_yellow(crop).mean())
    gray = crop.astype(np.int32).sum(axis=-1) / 3.0
    dark = float(((gray < 70) & (crop.max(-1) - crop.min(-1) < 40)).mean())
    if warm > LOCK_WARM_FRAC and dark > LOCK_DARK_FRAC:
        return False
    # 단색 벽 / 바닥 체커보드
    q = (crop // 32).astype(np.int32).reshape(-1, 3)
    if len(np.unique(q, axis=0)) < MIN_COLORS:
        return False
    detail = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY),
                                 cv2.CV_32F).std())
    return detail >= MIN_DETAIL


def _order_quad(q: np.ndarray) -> np.ndarray:
    s, d = q.sum(axis=1), np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)],
                     q[np.argmax(s)], q[np.argmax(d)]], np.float32)


def _frontality(q: np.ndarray) -> float:
    """0~1. 정면일수록 1. 좌우 변의 길이가 비슷하면 정면으로 본다."""
    o = _order_quad(q)
    left = np.linalg.norm(o[3] - o[0])
    right = np.linalg.norm(o[2] - o[1])
    if max(left, right) < 1e-6:
        return 0.0
    return float(min(left, right) / max(left, right))


def find_quads(frame: np.ndarray,
               wall_rgb: Optional[Tuple[int, int, int]] = None
               ) -> List[Tuple[float, np.ndarray]]:
    """프레임에서 (면적, 4각형) 후보들. 큰 것부터.

    **에지가 아니라 영역으로 찾는다.** 처음에는 Canny → contour →
    approxPolyDP를 썼는데 회수율이 처참했다(방 8: 24개 방위를 다 훑어 후보가
    5개, 사진 3장 중 0장). 원인은 팽창시킨 에지 맵에서 findContours가 사각형
    하나당 바깥/안쪽 두 개의 **고리** 윤곽을 주고, 얇은 고리에 approxPolyDP를
    걸면 꼭짓점이 4개로 안 떨어지기 때문이다.

    이 월드에는 훨씬 강한 사전지식이 있다. **벽은 팔레트 단색이고 사진은
    텍스처다.** 그래서 "국소 벽색과 다른 픽셀"로 마스크를 만들면 사진이
    통째로 덩어리로 잡히고, 그 덩어리의 최소면적 사각형이 곧 액자다.
    회전에도 강하고(minAreaRect가 기울기를 그대로 준다) 고리 문제도 없다.
    """
    if cv2 is None:
        return []
    band = frame[geo.HUD_H:, :, :]
    h, w = band.shape[:2]
    # 벽색 기준. 프레임에서 그냥 최빈색을 뽑으면 바닥 체커보드의 검은 타일
    # (12,12,12)이나 몹의 단색 몸통(예: 84,36,156)이 뽑혀서, 마스크가 화면
    # 전체로 번지고 사진이 하나도 안 잡힌다(실측: 방 8에서 12개 방위 중
    # 10개가 벽이 아닌 색을 골랐고 회수 0장).
    #
    # 그래서 (a) 호출자가 지도에 기록해 둔 방 벽색을 넘겨주면 그걸 쓰고,
    # (b) 없으면 지평선 위쪽의 유채색만 보는 기존 추정기를 쓴다. 둘 다
    # 바닥·천장을 애초에 보지 않는다.
    wall = wall_rgb or geo.wall_color_estimate(frame)
    if wall is None:
        return []
    wall = np.array(wall, dtype=np.int16)
    diff = np.abs(band.astype(np.int16) - wall).max(axis=2)
    mask = (diff > 40).astype(np.uint8)
    # 천장(무채색)과 바닥(체커보드)은 사진일 수 없다 → 지평선 근처만 본다.
    mask[:max(0, int(geo.CY) - geo.HUD_H - 95), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_QUAD_AREA:
            continue
        if bw < 12 or bh < 12:
            continue
        if area < 0.55 * bw * bh:
            continue                     # 액자는 꽉 찬 사각형이다
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue                     # 화면 밖으로 잘렸다
        pts = cv2.findNonZero((labels == i).astype(np.uint8))
        quad = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)
        quad[:, 1] += geo.HUD_H          # 원본 프레임 좌표로 되돌린다
        out.append((float(area), quad))
    out.sort(key=lambda t: -t[0])
    return out


def rectify(frame: np.ndarray, quad: np.ndarray, size: int = CROP_PX) -> np.ndarray:
    """4각형을 정면 정사각 크롭으로 편다."""
    src = _order_quad(quad)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                   np.float32)
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst),
                               (size, size))


class PhotoCapture:
    """프레임 간 사각형 추적 + 키프레임 확정.

    ``observe``를 매 스텝 부를 필요는 없다. 방을 훑는 SCAN 중에만 불러도
    네 벽을 다 지나가므로 충분하다(그리고 Canny가 프레임당 ~2ms라 매 스텝
    돌리면 예산을 먹는다).
    """

    def __init__(self, captioner=None):
        self._tracks: List[_Track] = []
        self.pending: List[Photo] = []      # 확정된 사진들
        self._last_heading: Optional[float] = None
        # 캡션기(vlm.Captioner). 없거나 키가 없으면 크롭+서명만 남는다.
        # submit은 큐에 넣기만 하므로 act를 블로킹하지 않는다(실측 0.3ms).
        self.captioner = captioner

    def observe(self, frame: np.ndarray, step: int, room_key: str,
                heading: float,
                wall_rgb: Optional[Tuple[int, int, int]] = None) -> List[Photo]:
        """이번 프레임의 사각형들을 추적에 반영. 이번에 확정된 사진 목록."""
        if cv2 is None:
            return []
        # 회전한 만큼 기존 트랙의 중심을 미리 밀어 준다. 스캔은 15°씩 도는데
        # 그건 화면에서 FOCAL*tan(15°) ≈ 56px라, 보정 없이 고정 반경으로
        # 매칭하면 매 스텝 새 트랙이 생겨 같은 사진이 서너 번 확정된다
        # (실측: bridge 4번, dog 4번).
        if self._last_heading is not None:
            d = geo.wrap180(heading - self._last_heading)
            if abs(d) > 0.5:
                shift = geo.FOCAL * math.tan(math.radians(max(-60.0, min(60.0, d))))
                for tr in self._tracks:
                    tr.center = (tr.center[0] + shift, tr.center[1])
        self._last_heading = heading

        seen: List[_Track] = []
        for area, quad in find_quads(frame, wall_rgb)[:4]:
            cx, cy = float(quad[:, 0].mean()), float(quad[:, 1].mean())
            score = area * (0.4 + 0.6 * _frontality(quad))
            tr = self._match((cx, cy))
            if tr is None:
                tr = _Track((cx, cy), score, quad, frame.copy(), step)
                self._tracks.append(tr)
            else:
                tr.center = (cx, cy)
                tr.last_step = step
                if score > tr.best_score:
                    tr.best_score, tr.best_quad = score, quad
                    tr.best_frame = frame.copy()
            seen.append(tr)

        done: List[Photo] = []
        alive: List[_Track] = []
        for tr in self._tracks:
            if tr in seen or step - tr.last_step <= TRACK_MAX_GAP:
                alive.append(tr)
                continue
            photo = self._commit(tr, room_key, heading, step)
            if photo is not None:
                done.append(photo)
        self._tracks = alive
        return self._store(done)

    def flush(self, room_key: str, heading: float, step: int) -> List[Photo]:
        """방을 떠날 때 남은 트랙을 전부 확정한다."""
        done = [p for p in (self._commit(t, room_key, heading, step)
                            for t in self._tracks) if p is not None]
        self._tracks = []
        self._last_heading = None
        return self._store(done)

    def _store(self, done: List[Photo]) -> List[Photo]:
        """중복을 합치며 pending에 넣는다. 실제로 새로 들어간 것만 반환.

        추적이 끊기는 경우(가려짐, 방을 다시 방문)까지는 못 막으므로,
        같은 방 안에서 서명이 거의 같은 사진은 여기서 하나로 접는다. 더 크게
        잡힌 쪽을 남긴다 — 캡션 품질이 면적에 직결된다.
        """
        fresh: List[Photo] = []
        for p in done:
            dup = None
            for q in self.pending:
                if q.room_key == p.room_key and float(p.sig @ q.sig) >= DEDUP_COS:
                    dup = q
                    break
            if dup is None:
                self.pending.append(p)
                fresh.append(p)
                if self.captioner is not None:
                    self.captioner.submit(p)
            elif p.area > dup.area:
                self.pending[self.pending.index(dup)] = p
        return fresh

    # --- 내부 ----------------------------------------------------------
    def _match(self, center: Tuple[float, float]) -> Optional[_Track]:
        best, best_d = None, TRACK_MATCH_PX
        for tr in self._tracks:
            d = math.hypot(center[0] - tr.center[0], center[1] - tr.center[1])
            if d < best_d:
                best, best_d = tr, d
        return best

    def _commit(self, tr: _Track, room_key: str, heading: float,
                step: int) -> Optional[Photo]:
        if tr.best_score < MIN_COMMIT_AREA * 0.4:
            return None
        crop = rectify(tr.best_frame, tr.best_quad)
        if not looks_like_photo(crop):
            return None
        cx = float(tr.best_quad[:, 0].mean())
        bearing = -math.degrees(math.atan((cx - geo.CX) / geo.FOCAL))
        y_bot = float(tr.best_quad[:, 1].max())
        dist = geo.dist_at_row(y_bot, cx)
        return Photo(room_key=room_key, wall=None, crop=crop,
                     sig=signature(crop),
                     area=float(cv2.contourArea(tr.best_quad)),
                     step=step, bearing=bearing, distance=dist)
