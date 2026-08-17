"""에이전트가 스스로 돌아다니는 걸 눈으로 보는 스크립트 (개발용).

obs 프레임을 확대해서 띄우고, 아래 패널에 에이전트 내부 상태(상태 기계,
추정 위치/방위, 목표, 방금 고른 액션, 지도 크기)를 같이 그린다.

    python tmp/watch_agent.py                 # 시드 0, 기본 15 steps/s
    python tmp/watch_agent.py 42 --fps 5      # 더 느리게
    python tmp/watch_agent.py 7 --fps 0       # 제한 없이 (원래 속도)
    python tmp/watch_agent.py 1 --headless    # 창 없이 로그만
    python tmp/watch_agent.py 0 --no-save     # 녹화 없이

한 판을 돌리면 tmp/runs/<날짜시각>_seed<N>/ 아래에 다음이 남는다:

    frames/       프레임 PNG. 기본은 10스텝마다 + 이벤트가 난 순간마다.
                  파일명 뒤에 이벤트 태그가 붙는다(f_000482_hpdrop.png).
    log.jsonl     매 스텝의 상태/좌표/HP/목표 (한 줄 = 한 스텝).
    summary.json  최종 지도, 통계, 상태별 스텝 수, 힌트/잠긴 문 기록.

조작:
    SPACE  일시정지 / 재개
    n      (일시정지 중) 한 스텝만 진행
    [ / ]  속도 낮추기 / 높이기
    q, ESC 종료

주의: env의 시뮬레이션 시계는 step() 사이의 실제 경과 시간으로 흐른다.
느리게 보면 한 스텝당 적이 더 많이 움직이고 T- 카운트다운도 빨리 줄어든다.
즉 '느린 관찰 모드'는 에이전트에게 더 불리한 조건이다 — 성능 측정은
tmp/run_explore.py로 하고, 이 스크립트는 동작 확인용으로만 쓴다.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                           # noqa: E402
from PIL import Image                         # noqa: E402

from agent import Agent                       # noqa: E402
from memory_fps_env.env import MemoryFPSEnv    # noqa: E402

ACTION_NAMES = ["TURN_LEFT", "TURN_RIGHT", "FORWARD", "BACK",
                "ATTACK", "NO_OP", "END"]


class Recorder:
    """한 판을 통째로 남긴다: 프레임 PNG + 스텝 로그 + 최종 요약.

    프레임을 전부 저장하면 600초 에피소드가 4만 장(≈1GB)이 되므로,
    기본은 N스텝마다 한 장이다. 대신 **이벤트 프레임**(피격, 충돌, 방
    이동, 힌트 배너, 잠긴 문 발견, 종료)은 주기와 무관하게 항상 남긴다 —
    나중에 결과를 볼 때 실제로 필요한 건 거의 그 순간들이다.
    """

    def __init__(self, root: Path, seed: int, every: int, max_frames: int):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / f"{stamp}_seed{seed}"
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.max_frames = max_frames
        self.saved = 0
        self.events = []
        self.log = open(self.dir / "log.jsonl", "w", encoding="utf-8")
        self._prev = {}

    def step(self, steps: int, frame: np.ndarray, ex, action: int) -> None:
        rec = {
            "step": steps, "state": ex.state, "action": ACTION_NAMES[int(action)],
            "room": ex.cur_name, "cell": list(ex.cur_cell),
            "x": round(ex.pose.x, 2), "z": round(ex.pose.z, 2),
            "heading": round(ex.pose.heading),
            "hp": ex._last_hp, "hp_max": ex._last_hp_max,
            "goal": f"{ex._goal_kind}/{ex._goal_dir}",
            "rooms": len(ex.map.rooms),
            "blocked": ex.stats["blocked"], "attacks": ex.stats["attacks"],
        }
        self.log.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tags = []
        prev = self._prev
        if prev.get("hp") is not None and rec["hp"] is not None \
                and rec["hp"] < prev["hp"]:
            tags.append("hpdrop")
        if rec["blocked"] > prev.get("blocked", 0):
            tags.append("bump")
        if rec["room"] != prev.get("room"):
            tags.append("room")
        if rec["state"] != prev.get("state"):
            tags.append("state")
        self._prev = rec

        periodic = (self.every > 0 and steps % self.every == 0
                    and self.saved < self.max_frames)
        if periodic or tags:
            self.save(steps, frame, "_".join(tags) if tags else "")
            if tags:
                self.events.append({"step": steps, "tags": tags,
                                    "state": rec["state"], "hp": rec["hp"]})

    def save(self, steps: int, frame: np.ndarray, tag: str = "") -> None:
        name = f"f_{steps:06d}" + (f"_{tag}" if tag else "") + ".png"
        Image.fromarray(frame).save(self.frames_dir / name)
        self.saved += 1

    def finish(self, ex, env, steps: int, info: dict, frame: np.ndarray) -> None:
        self.save(steps, frame, "final")
        summary = {
            "seed": getattr(env, "seed_value", None),
            "steps": steps,
            "phase": info.get("phase"),
            "hp": f"{ex._last_hp}/{ex._last_hp_max}",
            "stats": ex.stats,
            "state_steps": ex.state_steps,
            "map": ex.map.summary(),
            "memory": {
                "hints": ex.memory["hints"],
                "locked_door": ex.memory["locked_door"],
                "events": ex.memory["events"][-50:],
            },
            "recorded_events": self.events,
        }
        (self.dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log.close()


def _panel_lines(ex, action, steps, fps, paused):
    room = ex.map.get(ex.cur_cell)
    return [
        f"step {steps:6d}   {ACTION_NAMES[int(action)]:10s} "
        f"{'[PAUSED]' if paused else ''}",
        f"state {ex.state:10s} goal {ex._goal_kind}/{ex._goal_dir}",
        f"room  {str(ex.cur_name)[:22]:22s} cell {ex.cur_cell}",
        f"pose  ({ex.pose.x:6.2f}, {ex.pose.z:6.2f})  hdg {ex.pose.heading:3.0f}deg",
        f"map   {len(ex.map.rooms)} rooms, cell {ex.map.cell_size:.0f}m   "
        f"walls {room.walls if room else '-'}",
        f"hp    {ex._last_hp}/{ex._last_hp_max}   "
        f"blocked {ex.stats['blocked']}  attacks {ex.stats['attacks']}   "
        f"fps {'max' if fps <= 0 else fps}",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", nargs="?", type=int, default=0)
    ap.add_argument("--fps", type=float, default=15.0,
                    help="초당 스텝 수 (0이면 제한 없음)")
    ap.add_argument("--scale", type=int, default=3, help="창 확대 배율")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--headless", action="store_true", help="창 없이 콘솔 로그만")
    ap.add_argument("--out", default="tmp/runs",
                    help="녹화 폴더의 상위 경로")
    ap.add_argument("--save-every", type=int, default=10,
                    help="N스텝마다 프레임 저장 (0이면 정기 저장 안 함)")
    ap.add_argument("--max-frames", type=int, default=3000,
                    help="정기 저장 프레임 수 상한 (이벤트 프레임은 별도)")
    ap.add_argument("--no-save", action="store_true", help="녹화하지 않음")
    args = ap.parse_args()

    rec = None if args.no_save else Recorder(
        Path(args.out), args.seed, args.save_every, args.max_frames)

    cv2 = None
    if not args.headless:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            print("opencv가 없어 headless로 진행합니다 (pip install -e .[play])")

    env = MemoryFPSEnv(seed=args.seed)
    obs, _ = env.reset()
    agent = Agent()
    ex = agent.explorer

    win = f"agent seed={args.seed}"
    if cv2 is not None:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 320 * args.scale, 240 * args.scale + 130)

    fps = args.fps
    paused = False
    t0 = time.time()
    done = False
    steps = 0
    action = 5
    info = {}
    last_log = ""

    try:
      while not done and steps < args.steps:
          step_start = time.perf_counter()
          advance = not paused

          if advance:
              action = agent.act(obs)
              obs, _r, term, trunc, info = env.step(int(action))
              done = term or trunc
              steps += 1
              if rec is not None:
                  rec.step(steps, obs, ex, action)

          if cv2 is not None:
              big = cv2.resize(obs[..., ::-1], (320 * args.scale, 240 * args.scale),
                               interpolation=cv2.INTER_NEAREST)
              panel = np.zeros((130, big.shape[1], 3), dtype=np.uint8)
              for i, line in enumerate(_panel_lines(ex, action, steps, fps, paused)):
                  cv2.putText(panel, line, (10, 20 + i * 19),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1,
                              cv2.LINE_AA)
              cv2.imshow(win, np.vstack([big, panel]))

              # 남은 시간만큼 창 이벤트를 돌리며 대기 (일시정지 중엔 계속 대기)
              budget = 0.0 if fps <= 0 else max(0.0, 1.0 / fps -
                                                (time.perf_counter() - step_start))
              wait_ms = max(1, int(budget * 1000)) if (budget > 0 or paused) else 1
              key = cv2.waitKey(wait_ms) & 0xFF
              if key in (27, ord("q")):
                  break
              if key == ord(" "):
                  paused = not paused
              elif key == ord("n") and paused:
                  paused = False           # 한 스텝만 진행하고 다시 멈춘다
                  action = agent.act(obs)
                  obs, _r, term, trunc, info = env.step(int(action))
                  done = term or trunc
                  steps += 1
                  if rec is not None:
                      rec.step(steps, obs, ex, action)
                  paused = True
              elif key == ord("["):
                  fps = max(1.0, (fps if fps > 0 else 60.0) / 1.5)
              elif key == ord("]"):
                  fps = 0.0 if fps >= 60 else fps * 1.5
          else:
              line = (f"{ex.state:9s} room={str(ex.cur_name)[:18]:18s} "
                      f"pos=({ex.pose.x:5.1f},{ex.pose.z:5.1f}) "
                      f"hdg={ex.pose.heading:3.0f} "
                      f"goal={ex._goal_kind}/{ex._goal_dir} "
                      f"rooms={len(ex.map.rooms)}")
              if line[:40] != last_log[:40]:
                  print(f"[{steps:6d}] {line}")
                  last_log = line
              if fps > 0:
                  time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - step_start)))

    except KeyboardInterrupt:
        print('\n중단됨 (Ctrl-C)')

    dt = time.time() - t0
    print(f"\n{steps} steps in {dt:.1f}s ({steps/max(dt,1e-9):.0f}/s) "
          f"phase={info.get('phase')}")
    print(f"rooms mapped: {len(ex.map.rooms)}  cell={ex.map.cell_size}  "
          f"stats={ex.stats}")
    for cell, room in sorted(ex.map.rooms.items()):
        print(f"  {cell} {room.key:22s} walls={room.walls} color={room.wall_color}")
    if ex.memory["hints"]:
        print("hints:", ex.memory["hints"])
    if ex.memory["locked_door"]:
        print("locked door:", ex.memory["locked_door"])
    if rec is not None:
        rec.finish(ex, env, steps, info, obs)
        print(f"\n녹화 저장: {rec.dir}  "
              f"(프레임 {rec.saved}장, 이벤트 {len(rec.events)}건)")
        print("  frames/  log.jsonl  summary.json")
    if cv2 is not None:
        cv2.destroyAllWindows()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
