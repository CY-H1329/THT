"""Play 중 프레임을 방 단위로 모았다가, 방을 나갈 때 백그라운드로 VLM에
넘겨 agent/content_db.py를 채우는 적재기(ingest).

왜 이렇게 하나:
  - act()는 5초 예산이라 프레임마다 VLM을 동기 호출할 수 없다. 그래서
    "방을 나가는 순간, 그 방에서 모은 대표 프레임 몇 장을 한 번에" 단일
    워커 스레드로 던지고 act()는 즉시 반환한다(tmp/qa_ingest.py에서
    검증한 패턴을, agent/vlm.py의 값싼 단일 호출 설계에 맞춰 옮긴 것).
  - VLM이 필요 없는 사실(HP 하락 위치, [KEY] 태그 전이, 방문 순서)은
    여기서 HUD OCR 결과만으로 직접 기록한다 — 네트워크가 통째로 죽어도
    이 정보들은 그대로 남고 QA에서 쓸 수 있다.
  - HUD OCR은 프레임당 ~140ms로 비싸서, explorer가 이미 캐싱해둔 판독
    결과를 그대로 재사용한다(직접 read_hud를 또 부르면 비용이 두 배).
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from agent.config import VLM_MAX_CALLS_PER_EPISODE
from agent.content_db import ContentDB
from agent.hint_resolve import resolve_hint_room
from agent.vision import sample_wall_color
from agent.vlm import update_room_db

# 방 하나를 VLM에 넘길 때의 프레임 수. 너무 적으면 벽 4면을 다 못 보고,
# 너무 많으면 토큰/지연이 커진다. SURVEY가 4방향을 정면으로 보므로
# 15도 버킷으로 최대 6장이면 방 전체가 대체로 덮인다.
_MAX_FRAMES_PER_FLUSH = 6
_HEADING_BUCKET_DEG = 15
# SURVEY 프레임이 하나도 없는 방(스폰 직후 문을 그대로 통과해버려서 서베이를
# 못 받은 경우 — explorer.py의 알려진 사례)을 위한 폴백 버퍼 크기.
_MAX_FALLBACK_FRAMES = 4
# 적이 보이는 프레임: 서베이(벽 4면 훑기)만 보내면 적을 아예 못 보는
# 경우가 실측으로 확인됐다(seed=0: 3번 교전했는데 enemies 기록이 전부
# 빈 리스트). 적 외형/색은 README가 명시한 QA 카테고리라, 방 묶음에
# 몇 장 끼워 보낸다.
_MAX_COMBAT_FRAMES = 2
# 벽 색 CV 투표 주기(프레임마다 돌릴 필요는 없다).
_COLOR_VOTE_EVERY = 5
# 한 방을 몇 번이나 다시 보낼지 상한(재방문 때마다 새 프레임이 생기지만,
# 같은 방에 예산을 다 쓰지 않도록).
_MAX_FLUSHES_PER_ROOM = 2
# 동결된 기억을 사람이 확인할 수 있게 남기는 위치. README 규칙상 agent가
# 디스크에 쓸 수 있는 건 "우리가 관리하는 tmp/ 디렉터리"뿐이라 그 안에만
# 쓰고, 실패해도(읽기 전용 환경 등) 조용히 넘어간다 — QA 답변 자체는 전부
# 메모리에서 나오므로 이 파일이 없어도 동작에 아무 영향이 없다.
_DB_DUMP_PATH = Path(__file__).resolve().parent.parent / "tmp" / "qa_out" / "db.json"

# finalize()는 answer()의 첫 호출 안에서 일어난다 — 질문당 10초 예산을
# 넘기면 그 답은 0점이므로(README), 여기서 기다리는 시간을 넉넉히 잘라둔다.
_FINALIZE_TIMEOUT_S = 6.0


@dataclass
class _Kept:
    frame: np.ndarray
    heading: int


class _RoomBuffer:
    def __init__(self) -> None:
        self.by_heading: dict = {}   # 15도 버킷 -> _Kept (SURVEY 프레임)
        self.fallback: list = []     # 서베이를 못 받은 방용 예비 프레임
        self.combat: list = []       # 적이 보이는/맞은 프레임
        self.color_votes: dict = {}  # CV로 잰 벽 색 투표
        self.seen: int = 0
        self.flushes: int = 0
        self.pending: bool = False   # 아직 워커가 처리 중인 잡이 있는지

    def majority_color(self):
        if not self.color_votes:
            return None
        return max(self.color_votes, key=self.color_votes.get)


class ContentIngest:
    def __init__(self, max_calls: int = VLM_MAX_CALLS_PER_EPISODE) -> None:
        self.content = ContentDB()
        self.max_calls = max_calls
        self.calls_used = 0
        self.cost_usd = 0.0

        self._lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._futures: list = []
        self._buffers: dict = {}
        self._room: Optional[str] = None
        self._prev_hp: Optional[int] = None
        self._prev_key: bool = False
        self._was_fleeing: bool = False
        self._flee_room: Optional[str] = None
        self._explorer = None
        self._finalized = False

    # --- Play phase ---------------------------------------------------
    def on_frame(self, obs: np.ndarray, explorer) -> None:
        self._explorer = explorer
        hud = explorer._read_hud_cached(obs)  # 이미 계산된 결과 재사용(위 주석)
        if not hud.ok:
            return

        self.content.note_hp(hud.hp, hud.hp_max)
        if hud.seconds_remaining is not None:
            # "에피소드가 어떻게 끝났나" 판정용 — 마지막으로 본 남은 시간이
            # 아직 넉넉한데 QA가 시작됐다면 시간 초과가 아니라 사망이다
            # (평가 하네스는 종료 후 act()를 더 부르지 않으므로, HP 0인
            # 마지막 프레임 자체는 우리 눈에 안 들어온다).
            self.content.seconds_left_last = hud.seconds_remaining
        room = None
        if hud.room_name and not hud.is_corridor:
            room = explorer._canonicalize(hud.room_name)

        # 1) HP 하락 = 그 방에 적이 실제로 있었다는 픽셀 근거.
        if hud.hp is not None:
            if self._prev_hp is not None and hud.hp < self._prev_hp:
                self.content.record_damage(self._room or room, self._prev_hp - hud.hp)
                if room:
                    self._keep_combat_frame(room, obs)
            self._prev_hp = hud.hp

        # 2) [KEY] 태그 전이 — 켜지면 "여기가 열쇠가 있던 방", 꺼지면
        #    "열쇠를 써서 문을 열었다"(README: 열쇠는 해제에만 소모됨).
        if hud.has_key and not self._prev_key:
            self.content.note_key_found(self._room or room)
        elif self._prev_key and not hud.has_key:
            self.content.note_key_consumed()
        self._prev_key = hud.has_key

        # 3) 전투 교전 수(처치 수 추정의 근거). FLEE는 HP가 깎여야만
        #    시작되므로(explorer.step), 교전 하나 = 실제로 적과 붙은 사건
        #    하나다. 실제로 ATTACK을 낸 교전만 센다.
        fleeing = explorer.state == "FLEE"
        if fleeing and not self._was_fleeing:
            self._flee_room = self._room or room
        elif self._was_fleeing and not fleeing:
            if self._flee_room and getattr(explorer, "flee_ever_attacked", False):
                self.content.get_or_create(self._flee_room).combat_bouts += 1
            self._flee_room = None
        self._was_fleeing = fleeing

        # 4) 방 전환 감지 — 나가는 순간 그 방의 프레임 묶음을 백그라운드로.
        if room is None:
            return  # 복도: 방 프레임으로 섞이면 안 되므로 버퍼링하지 않는다.
        if self._room is not None and room != self._room:
            self._schedule_flush(self._room)
        self._room = room
        self.content.get_or_create(room)

        self._buffer_frame(room, obs, hud, explorer)

    def _buffer_frame(self, room: str, obs, hud, explorer) -> None:
        buf = self._buffers.setdefault(room, _RoomBuffer())
        buf.seen += 1

        # 벽 색은 VLM 없이 픽셀만으로도 꽤 정확하게 나온다(agent/vision.py).
        # 방마다 다수결로 모아두면, VLM 응답이 없거나 실패해도 "그 방 벽이
        # 무슨 색이었나"와 색 힌트 매칭은 그대로 답할 수 있다.
        if buf.seen % _COLOR_VOTE_EVERY == 1:
            color = sample_wall_color(obs)
            if color:
                buf.color_votes[color] = buf.color_votes.get(color, 0) + 1

        # 적이 보이는 프레임은 따로 챙긴다(explorer가 이번 틱에 이미 돌린
        # 탐지 결과를 재사용 — 여기서 CV를 다시 돌리지 않는다).
        if any(getattr(m, "score", 0) >= 0.6 and getattr(m, "distance", 99) <= 4.0
               for m in getattr(explorer, "last_mobs", []) or []):
            self._keep_combat_frame(room, obs)

        if hud.heading is None:
            return
        if explorer.state == "SURVEY":
            # 문턱에 서 있을 때 찍힌 프레임은 옆방 내용이 통째로 들어와
            # 오염된다(report.md의 doorway bleed-through). 방 안쪽으로
            # 들어와 4방향을 둘러보는 SURVEY 프레임이 1순위다 — 각 방향을
            # 정면으로 보는 시점이라 벽 그림/오브젝트가 가장 잘 보인다.
            bucket = (int(hud.heading) % 360) // _HEADING_BUCKET_DEG
            if bucket not in buf.by_heading:
                buf.by_heading[bucket] = _Kept(
                    frame=np.ascontiguousarray(obs.copy()), heading=int(hud.heading))
        elif len(buf.fallback) < _MAX_FALLBACK_FRAMES and buf.seen % _COLOR_VOTE_EVERY == 1:
            # 서베이를 한 번도 못 받는 방이 실제로 있다(스폰 방에서 입구
            # 방향이 우연히 문이라 그대로 통과해버리는 경우 — explorer.py
            # 주석 참고). 그런 방이 통째로 빈 기록으로 남지 않게 예비
            # 프레임을 조금 모아둔다.
            buf.fallback.append(_Kept(
                frame=np.ascontiguousarray(obs.copy()), heading=int(hud.heading)))

    def _keep_combat_frame(self, room: str, obs) -> None:
        buf = self._buffers.setdefault(room, _RoomBuffer())
        if len(buf.combat) >= _MAX_COMBAT_FRAMES:
            return
        buf.combat.append(_Kept(frame=np.ascontiguousarray(obs.copy()), heading=0))

    # --- VLM 백그라운드 잡 ---------------------------------------------
    def _pick_frames(self, buf: _RoomBuffer) -> list:
        """버킷을 헤딩 순으로 고르게 골라 최대 _MAX_FRAMES_PER_FLUSH장 +
        적이 보인 프레임 몇 장."""
        keys = sorted(buf.by_heading)
        if keys:
            if len(keys) <= _MAX_FRAMES_PER_FLUSH:
                picked = keys
            else:
                stride = len(keys) / float(_MAX_FRAMES_PER_FLUSH)
                picked = [keys[int(i * stride)] for i in range(_MAX_FRAMES_PER_FLUSH)]
            frames = [buf.by_heading[k].frame for k in picked]
        else:
            frames = [k.frame for k in buf.fallback]
        frames += [k.frame for k in buf.combat]
        return frames

    def _room_snapshot(self, room: str) -> dict:
        """VLM에게 "지금까지 이렇게 기록돼 있다"고 보여줄 이 방의 현재 상태."""
        with self._lock:
            rec = self.content.rooms.get(room)
            if rec is None:
                return {}
            return {
                "wall_color": rec.wall_color.value,
                "images": list(rec.images),
                "objects": list(rec.objects),
                "enemies": list(rec.enemies),
            }

    def _schedule_flush(self, room: str) -> None:
        buf = self._buffers.get(room)
        if buf is None or buf.pending or buf.flushes >= _MAX_FLUSHES_PER_ROOM:
            return
        if self.calls_used >= self.max_calls:
            return
        frames = self._pick_frames(buf)
        if not frames:
            return
        buf.pending = True
        buf.flushes += 1
        # 다음 방문에서 다시 모은다(같은 프레임을 두 번 보내지 않도록).
        buf.by_heading = {}
        buf.fallback = []
        buf.combat = []
        self.calls_used += 1
        if self._pool is None:
            # 워커 1개 = 호출이 겹치지 않고 순서대로 처리됨(ContentDB 병합이
            # 단순해지고, OpenRouter 쪽 동시 요청도 안 늘어난다).
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")
        self._futures.append(self._pool.submit(self._flush_job, room, frames, buf))

    def _flush_job(self, room: str, frames: list, buf: _RoomBuffer) -> None:
        try:
            result = update_room_db(frames, self._room_snapshot(room), room_hint=room)
            if result.ok and result.data:
                with self._lock:
                    self.content.apply_update(room, result.data)
                    self.cost_usd += result.cost_usd
        except Exception:
            # 어떤 이유로든 VLM 쪽이 터져도 Play는 계속돼야 한다.
            pass
        finally:
            buf.pending = False

    # --- QA 시작 시점의 동결 --------------------------------------------
    def finalize(self, explorer=None, timeout_s: float = _FINALIZE_TIMEOUT_S) -> ContentDB:
        """마지막 방을 밀어 넣고, 진행 중인 잡을 (시간 제한 안에서) 기다린 뒤
        ContentDB를 동결한다. answer()의 첫 호출에서 딱 한 번만 실행된다."""
        if self._finalized:
            return self.content
        self._finalized = True
        explorer = explorer or self._explorer

        if self._room:
            self._schedule_flush(self._room)
            self._room = None
        if self._futures:
            wait(self._futures, timeout=timeout_s)
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

        scene = getattr(explorer, "scene", None)
        self._merge_scene_facts(scene)
        with self._lock:
            self.content.freeze(scene)
        self._dump_db(scene)
        return self.content

    def _dump_db(self, scene=None, path: Optional[Path] = None) -> None:
        """동결된 기억을 tmp/qa_out/db.json으로 저장(확인/디버깅용).

        탐험 그래프(방 연결/출구 상태/힌트)는 ContentDB에 없는 정보라 함께
        넣어, 이 파일 하나만 봐도 에피소드에서 무엇을 기억했는지 다 보이게
        한다. 어떤 이유로든 저장에 실패하면 그냥 넘어간다.
        """
        target = Path(path) if path else _DB_DUMP_PATH
        try:
            with self._lock:
                dump = self.content.to_dict()
            dump["scene"] = self._scene_dict(scene)
            dump["vlm_calls_used"] = self.calls_used
            dump["vlm_cost_usd"] = round(self.cost_usd, 6)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(dump, indent=1, ensure_ascii=False),
                              encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _scene_dict(scene) -> dict:
        if scene is None:
            return {}
        return {
            "rooms": {
                name: {
                    "wall_color": node.wall_color,
                    "entry_heading": node.entry_heading,
                    "exits": node.exits,
                    "exit_leads_to": node.exit_leads_to,
                    "done": node.done,
                }
                for name, node in scene.nodes.items()
            },
            "visited_order": scene.visited_order,
            "key_hint_text": scene.key_hint_text,
            "key_hint_room": scene.key_hint_room,
            "key_hint_heading": scene.key_hint_heading,
        }

    def _merge_scene_facts(self, scene) -> None:
        """탐험 FSM이 픽셀로 직접 확인한 사실(힌트 배너 원문, 잠긴 문 위치)을
        ContentDB로 옮긴다. VLM 판단보다 신뢰도가 높아서 덮어쓴다."""
        if scene is None:
            return
        with self._lock:
            if scene.key_hint_text:
                self.content.hint_text.value = scene.key_hint_text
                self.content.hint_text.confidence = 1.0
            if scene.key_hint_room:
                self.content.locked_door_room = scene.key_hint_room
                self.content.locked_door_heading = scene.key_hint_heading
            else:
                for name, node in scene.nodes.items():
                    if any(st == "locked" for st in node.exits.values()):
                        self.content.locked_door_room = name
                        break
            for name, node in scene.nodes.items():
                rec = self.content.get_or_create(name)
                if rec.wall_color.value is None and node.wall_color:
                    # CV(픽셀 최빈색)로 잰 벽 색 — VLM이 못 채운 방의 폴백.
                    rec.wall_color.value = node.wall_color
                    rec.wall_color.confidence = 0.9
            for name, buf in self._buffers.items():
                rec = self.content.get_or_create(name)
                if rec.wall_color.value is None and buf.majority_color():
                    # 서베이조차 못 받은 방의 최후 폴백 — 방에 있는 동안
                    # 모아둔 픽셀 색 투표의 다수결.
                    rec.wall_color.value = buf.majority_color()
                    rec.wall_color.confidence = 0.85
            # 열쇠 방: [KEY]를 실제로 주운 방이 있으면 그게 정답이고,
            # 없으면(열쇠를 못 찾았으면) 힌트 문구로 추정한다.
            if self.content.key_found_in:
                self.content.key_hint_room = self.content.key_found_in
            else:
                guess = resolve_hint_room(self.content.hint_text.value, self.content)
                if guess:
                    self.content.key_hint_room = guess
