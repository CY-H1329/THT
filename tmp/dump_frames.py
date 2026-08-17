"""PHASE 3-1 디버그 스크립트.

env를 몇 스텝 돌려서 obs 프레임을 tmp/debug_frames/에 PNG로 저장한다.
목적: HUD 바의 정확한 픽셀 y범위, 글자 시작 x좌표, 힌트 배너의 실제
테두리 좌표를 육안+픽셀 분석으로 확인해서 ocr.py 크롭 좌표를 확정하기
위함. agent/ 코드가 아니라 순수 개발용 스크립트이며, 실제 obs를 만드는
데는 memory_fps_env.env(허용된 최상위 API)만 사용한다.
"""

import sys
from pathlib import Path

# 이 venv에서는 .py 파일을 직접 실행할 때 editable-install(.pth)의 파인더
# 등록 코드가 실행되지 않는 현상이 있어(-c 인라인 실행에서는 정상 동작),
# 개발용 스크립트는 프로젝트 루트를 sys.path에 직접 넣어 우회한다.
# agent/ 패키지 자체(평가 시 `from agent import Agent`로 로드)는 이 문제와 무관.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from memory_fps_env.env import Action, MemoryFPSEnv

OUT_DIR = Path(__file__).parent / "debug_frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dump(seed: int, n_steps: int = 40) -> None:
    env = MemoryFPSEnv(seed=seed)
    obs, info = env.reset()
    Image.fromarray(obs).save(OUT_DIR / f"seed{seed}_step000.png")
    print(f"seed={seed} step=0 phase={info['phase']} shape={obs.shape} dtype={obs.dtype}")

    # 방을 몇 개 지나가며 다양한 HUD 상태(방이름 변화, 헤딩 변화)를 확보.
    actions = (
        [Action.MOVE_FORWARD] * 8
        + [Action.TURN_RIGHT] * 6
        + [Action.MOVE_FORWARD] * 8
        + [Action.TURN_LEFT] * 4
        + [Action.MOVE_FORWARD] * 8
    )
    for i, a in enumerate(actions[:n_steps], start=1):
        obs, _r, term, trunc, info = env.step(int(a))
        if i in (5, 10, 15, 20, 25, 30):
            Image.fromarray(obs).save(OUT_DIR / f"seed{seed}_step{i:03d}.png")
        if term or trunc:
            print(f"seed={seed} terminated at step {i}, phase={info['phase']}")
            break
    else:
        print(f"seed={seed} ran {n_steps} steps without terminating")

    # HUD 바 픽셀 분석: y=0..31 범위에서 각 row의 0이 아닌(텍스트) 픽셀 수.
    bar = obs[:32, :, :]
    for y in range(32):
        nonzero = int(np.count_nonzero(bar[y].sum(axis=-1) > 30))
        if nonzero > 0:
            print(f"  row y={y}: {nonzero} non-black px (first/last x with text)")

    env.close()


if __name__ == "__main__":
    for seed in (0, 42):
        dump(seed)
    print("done ->", OUT_DIR)
