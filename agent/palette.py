"""방 벽 색 팔레트.

memory_fps_env.world.walls.WALL_PALETTE와 값이 같다 — 하지만 그 모듈을
import하지 않는다(world.* import 금지 규칙). README에 "팔레트는 public/
heldout 세션에 공유된다"고 명시돼 있어서, 이 12개 (이름, RGB) 값 자체는
공개된 상수이고 여기 복사해서 쓰는 건 규칙 위반이 아니다 — 우리는 이걸
'세계 상태를 몰래 읽는 것'이 아니라 '사람이 색을 보고 이름 붙이듯, 픽셀
색을 알려진 색 이름에 매칭하는 것'으로 쓴다.
"""

from typing import List, Optional, Tuple

WALL_PALETTE: List[Tuple[str, Tuple[int, int, int]]] = [
    ("terracotta", (200, 110,  90)),
    ("sage",       (155, 180, 145)),
    ("slate",      ( 95, 115, 135)),
    ("mustard",    (210, 175,  55)),
    ("plum",       (130,  80, 130)),
    ("sand",       (220, 200, 160)),
    ("teal",       ( 60, 150, 150)),
    ("brick",      (175,  80,  60)),
    ("olive",      (135, 140,  70)),
    ("denim",      ( 75, 110, 170)),
    ("lavender",   (190, 175, 220)),
    ("rust",       (180,  90,  50)),
]


def nearest_color_name(rgb: Tuple[int, int, int], max_dist: float = 55.0) -> Optional[str]:
    """rgb에 가장 가까운 팔레트 색 이름. max_dist보다 멀면 None(불확실)."""
    best_name, best_dist = None, float("inf")
    r, g, b = rgb
    for name, (pr, pg, pb) in WALL_PALETTE:
        d = ((r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_dist > max_dist:
        return None
    return best_name
