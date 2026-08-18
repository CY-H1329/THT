"""잠긴 문 힌트 해석 — 문장에서 '열쇠가 있는 방'의 조건을 뽑아낸다.

힌트는 시드마다 하나이고, 템플릿은 네 가지 중 하나다(env world/doors.py의
HINT_TEMPLATES, 선호 순위대로):

    1. color             "the room with sage walls"
    2. image_count_wall  "the room with two images on its east wall"
    3. image_category    "the room with two tiger images"
    4. object            "the room with a barrel"
    5. (복합)            "the room with sage walls and a barrel"

README가 명시하듯 평가 시에는 **어휘가 held-out**이다. 그런데 벽 색만은
예외다. world/walls.py가 직접 밝혀 둔다:

    The palette is intentionally shared across public and heldout sessions:
    forcing a heldout palette would only test color-name matching, not vision.

즉 12색 팔레트의 **이름과 RGB가 둘 다 고정**이다. 그래서 1번(그리고 5번의
색 부분)은 완전히 풀 수 있다 — 이름을 RGB로 바꿔서, 스캔 때 기록해 둔
방별 벽 색과 맞추면 된다. 실측상 공개 시드 10개가 **전부 rank 1(color)**
이었다. env가 유일하게 식별되는 가장 선호되는 템플릿을 고르기 때문이다.

2번은 어휘가 아예 필요 없다(개수 + 벽 방향). 3·4번은 카테고리 이름이
held-out이라 이름으로는 못 짚고, 개수 제약만 쓸 수 있다. 못 풀면
`Target.kind == "unknown"`으로 두고, 탐색 쪽에서 방을 돌며 노란 열쇠를
눈으로 찾는 쪽으로 넘긴다(열쇠 색은 env 코드에 고정돼 있어 held-out이 아니다).

배너 OCR은 폰트 28에서 글자를 자주 헷갈린다(실측: s→a, t→I, i→I, d→r,
k→I:). 예: "the room with muaIard walla". 그래서 색 이름은 정확 비교가
아니라 12개 후보에 대한 **최근접 문자열 매칭**으로 고른다. 후보가 12개뿐이고
서로 충분히 다르기 때문에 이 정도 오독은 흡수된다.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# world/walls.py::WALL_PALETTE 사본. memory_fps_env.world.*는 import가
# 금지돼 있어(README 규칙) 값을 옮겨 적는다. 위 주석대로 이 팔레트는
# 공개/heldout 세션에서 동일하다.
WALL_PALETTE: List[Tuple[str, Tuple[int, int, int]]] = [
    ("terracotta", (200, 110, 90)),
    ("sage", (155, 180, 145)),
    ("slate", (95, 115, 135)),
    ("mustard", (210, 175, 55)),
    ("plum", (130, 80, 130)),
    ("sand", (220, 200, 160)),
    ("teal", (60, 150, 150)),
    ("brick", (175, 80, 60)),
    ("olive", (135, 140, 70)),
    ("denim", (75, 110, 170)),
    ("lavender", (190, 175, 220)),
    ("rust", (180, 90, 50)),
]

_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4}
_WALLS = ("south", "east", "north", "west")
# env의 벽 이름 → 우리 지도의 방향 라벨(mapper.DIRS)
WALL_TO_DIR = {"south": "S", "east": "E", "north": "N", "west": "W"}

# OCR이 실제로 저지르는 치환들(실측). 색 이름 매칭 전에 양쪽을 같은
# 표준형으로 접어서 오독을 흡수한다.
_FOLD = str.maketrans({
    "i": "l", "I": "l", "t": "l", "1": "l", "|": "l",
    "a": "s", "e": "s", "c": "o", "0": "o", "r": "d",
})

# 색 이름은 후보가 12개뿐이고 서로 충분히 달라서 위처럼 과감하게 접어도
# 되지만, 수사(one/two/...)와 벽 이름에까지 같은 폴딩을 쓰면 엉뚱한 단어가
# 걸린다("barrel"이 "three"로 잡혔다). 그쪽은 세로획 혼동(i/I/t/1/|)만 접는다.
_FOLD_L = str.maketrans({"i": "l", "I": "l", "t": "l", "1": "l", "|": "l"})

_MIN_COLOR_SCORE = 0.55


@dataclass
class Target:
    """열쇠가 있는 방을 가리키는 조건."""
    kind: str                  # 'color' | 'image_count_wall' | 'count' | 'unknown'
    color: Optional[str] = None
    rgb: Optional[Tuple[int, int, int]] = None
    count: Optional[int] = None
    wall: Optional[str] = None      # 'N'|'E'|'S'|'W'
    raw: str = ""
    confidence: float = 0.0


def _fold(word: str) -> str:
    return word.lower().translate(_FOLD)


def _fold_l(word: str) -> str:
    return word.lower().translate(_FOLD_L)


def _close(word: str, target: str, cutoff: float = 0.7) -> bool:
    return difflib.SequenceMatcher(None, _fold_l(word),
                                   _fold_l(target)).ratio() >= cutoff


# 수사 매칭에서 반드시 건너뛸 단어들. "the"가 "three"와 0.75로 붙어서
# "the room with a barrel"이 count=3으로 잡혔다.
_STOP = {"the", "room", "with", "a", "an", "its", "it", "on", "and",
         "wall", "walls", "walla", "image", "images"}


def _count_word(words: List[str]) -> Optional[int]:
    """수사(one/two/three/four)를 오독 허용으로 찾는다."""
    best, best_score = None, 0.0
    for w in words:
        if w in _STOP:
            continue
        for name, n in _COUNT_WORDS.items():
            score = difflib.SequenceMatcher(None, _fold_l(w), _fold_l(name)).ratio()
            if score > best_score and score >= 0.8:
                best, best_score = n, score
    return best


def match_color(token: str) -> Tuple[Optional[str], float]:
    """OCR로 읽은 단어 하나를 팔레트 색 이름에 맞춘다. (이름, 점수)."""
    t = _fold(token)
    best, best_score = None, 0.0
    for name, _rgb in WALL_PALETTE:
        score = difflib.SequenceMatcher(None, t, _fold(name)).ratio()
        if score > best_score:
            best, best_score = name, score
    return (best, best_score) if best_score >= _MIN_COLOR_SCORE else (None, best_score)


def rgb_for(name: str) -> Optional[Tuple[int, int, int]]:
    for n, rgb in WALL_PALETTE:
        if n == name:
            return rgb
    return None


def parse(text: str) -> Target:
    """힌트 문장 → Target. 못 읽으면 kind='unknown'."""
    raw = (text or "").strip()
    low = raw.lower()
    words = re.findall(r"[a-z0-9:]+", low)

    # 2) "... two images on its east wall" — 어휘가 필요 없는 유일한 템플릿.
    #    색 판정보다 **먼저** 본다. OCR이 "two"를 "lwo"로 뭉개면 이 갈래가
    #    조용히 실패하고, 뒤의 색 갈래가 "Images"를 색 이름으로 오인한다.
    if any(_close(w, "on") for w in words) and any(_close(w, "wall") for w in words):
        cnt = _count_word(words)
        wall = None
        for w in words:
            folded = [_fold_l(x) for x in _WALLS]
            m = difflib.get_close_matches(_fold_l(w), folded, n=1, cutoff=0.65)
            if m:
                wall = WALL_TO_DIR[_WALLS[folded.index(m[0])]]
                break
        if cnt and wall:
            return Target(kind="image_count_wall", count=cnt, wall=wall,
                          raw=raw, confidence=0.9)

    # 1) / 5) 색. "with <color> walls" 자리의 단어를 후보로 본다. OCR이
    # "walls"를 "walla"로 읽으므로 마지막 단어도 느슨하게 본다.
    cand: List[str] = []
    for i, w in enumerate(words):
        if w in ("with", "wilh", "wilth") and i + 1 < len(words):
            cand.append(words[i + 1])
    # 'wall(s)' 바로 앞 단어도 후보.
    for i, w in enumerate(words):
        if difflib.SequenceMatcher(None, _fold(w), _fold("walls")).ratio() > 0.7 and i:
            cand.append(words[i - 1])
    best_name, best_score = None, 0.0
    for c in cand:
        if c in ("a", "an", "the", "room"):
            continue
        name, score = match_color(c)
        if name and score > best_score:
            best_name, best_score = name, score
    if best_name:
        return Target(kind="color", color=best_name, rgb=rgb_for(best_name),
                      raw=raw, confidence=best_score)

    # 3) "two tiger images" — 카테고리는 held-out이라 개수만 남는다.
    cnt = _count_word(words)
    if cnt:
        return Target(kind="count", count=cnt, raw=raw, confidence=0.4)

    return Target(kind="unknown", raw=raw, confidence=0.0)
