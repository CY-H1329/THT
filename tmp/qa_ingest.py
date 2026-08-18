"""실시간처럼 프레임을 한 장씩 받아 QA DB를 채운다.

방마다 GPT-4o surveyor + GPT-4o verifier 를 병렬 호출 (images only, 이전 DB 없음).
act() 는 on_frame 만 (OCR). VLM 은 방을 나갈 때 백그라운드.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from agent.ocr import hint_banner_active, read_hint_text, read_hud
from agent.vision import sample_wall_color, something_in_front, wall_color_fraction
from qa_db import EpisodeDB, _canon_phrase, _same_image, _similar
from qa_portal import best_wall_color, detect_lock_panel
from qa_vlm import (
    LIVE_SURVEY_MODEL,
    SURVEY_MODEL,
    CallResult,
    survey_and_verify_parallel,
    witness_enemies,
)

def _norm_color(c) -> Optional[str]:
    t = str(c or "").strip().lower()
    if t in ("gray", "grey"):
        return "grey"
    return t or None


def _enemy_key(e: dict) -> tuple:
    return (
        _norm_color(e.get("shirt_color") or e.get("torso_color")),
        _norm_color(e.get("pants_color")),
        _norm_color(e.get("skin_color") or e.get("head_color")),
        str(e.get("body_shape") or "").lower() or None,
    )


def _intersect_images(a: list, b: list) -> list:
    """Les deux GPT doivent voir la même photo — sinon c'est souvent le voisin."""
    if a and not b:
        return a
    if b and not a:
        return b
    keep, seen = [], set()
    for im in a:
        if float(im.get("confidence") or 0) < 0.8:
            continue
        cat = _canon_phrase(im.get("category") or im.get("desc") or "")
        if cat in seen:
            continue
        if any(_same_image(im, other) for other in b):
            seen.add(cat)
            keep.append(im)
    return keep


def _merge_images(a: list, b: list, known_cats: set) -> list:
    """Les deux GPT d'accord = on garde. Un seul GPT = on garde seulement
    si ce sujet n'est pas déjà dans une autre salle (fuite par la porte)."""
    if a and not b:
        pool, agreed = a, set()
    elif b and not a:
        pool, agreed = b, set()
    else:
        agreed_list = _intersect_images(a, b)
        agreed = {
            _canon_phrase(im.get("category") or im.get("desc") or "")
            for im in agreed_list
        }
        pool = list(agreed_list)
        for im in list(a) + list(b):
            cat = _canon_phrase(im.get("category") or im.get("desc") or "")
            if not cat or cat in agreed or float(im.get("confidence") or 0) < 0.8:
                continue
            if cat in known_cats or any(_same_image(im, x) for x in pool):
                continue
            agreed.add(cat)
            pool.append(im)
        return pool
    keep, seen = [], set()
    for im in pool:
        cat = _canon_phrase(im.get("category") or im.get("desc") or "")
        if not cat or cat in seen or cat in known_cats:
            continue
        if float(im.get("confidence") or 0) < 0.8:
            continue
        seen.add(cat)
        keep.append(im)
    return keep


def _intersect_objects(a: list, b: list) -> list:
    if a and not b:
        return a
    if b and not a:
        return b
    keep, seen = [], set()
    for obj in a:
        if float(obj.get("confidence") or 0) < 0.8:
            continue
        name = _canon_phrase(obj.get("name") or "")
        if not name or name in seen:
            continue
        if any(
            _canon_phrase(o.get("name") or "") == name
            or _similar(name, o.get("name") or "")
            for o in b
        ):
            seen.add(name)
            keep.append(obj)
    return keep


def _union_enemies(a: list, b: list) -> list:
    keep, seen = [], set()
    for e in list(a) + list(b):
        if float(e.get("confidence") or 0) < 0.8:
            continue
        k = _enemy_key(e)
        if k in seen:
            continue
        seen.add(k)
        keep.append(e)
    return keep


COVERAGE_MIN = 0.38
HEADING_STEP = 15
EVERY_N = 5
MAX_EVERY5 = 40
MAX_COMBAT = 10
BATCH_SIZE = 6
VERIFY_N = 12
SPIN_MIN_STEPS = 18


def canon_name(name: str, known: list[str]) -> str:
    if name in known:
        return name
    stripped = name.rstrip("…").rstrip(".").rstrip()
    if len(stripped) < 6:
        return name
    for k in known:
        kk = k.rstrip("…").rstrip(".").rstrip()
        if stripped == kk or stripped.startswith(kk) or kk.startswith(stripped):
            return k
    return name


def _view_score(coverage: float) -> float:
    if coverage < COVERAGE_MIN:
        return -1.0
    return -abs(coverage - 0.68)


@dataclass
class Kept:
    frame: np.ndarray
    heading: int
    coverage: float
    step: int
    kind: str


@dataclass
class VisitBuf:
    color_votes: dict = field(default_factory=dict)
    by_heading: dict = field(default_factory=dict)  # 15° bucket -> Kept
    every5: list = field(default_factory=list)
    combat: list = field(default_factory=list)
    sent_steps: set = field(default_factory=set)
    sent_combat: set = field(default_factory=set)
    n_seen: int = 0
    verified: bool = False
    last_heading: Optional[int] = None
    spin_run: int = 0
    spin_done: bool = False
    live_submitted: bool = False
    survey_enemies: list = field(default_factory=list)

    def majority_color(self) -> Optional[str]:
        if not self.color_votes:
            return None
        return max(self.color_votes, key=self.color_votes.get)


class QaIngest:
    def __init__(
        self,
        *,
        dry_run: bool = False,
        cost_limit_usd: float = 10.0,
        strategy: str = "angles_then_verify",
        api_key: Optional[str] = None,
        on_event: Optional[Callable] = None,
        live_mode: bool = False,
        survey_model: Optional[str] = None,
    ) -> None:
        self.strategy = strategy
        self.dry_run = dry_run
        self.cost_limit_usd = cost_limit_usd
        self.api_key = api_key
        self.on_event = on_event
        # live_mode: act() 5초 예산. 프레임마다 VLM 금지.
        # 방을 떠날 때 대표 6장 1호출만 백그라운드 스레드로.
        self.live_mode = live_mode
        self.survey_model = survey_model or (LIVE_SURVEY_MODEL if live_mode else SURVEY_MODEL)
        self._lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="qa-vlm")
            if live_mode else None
        )
        self._futures: list = []

        self.db = EpisodeDB()
        self.total_cost = 0.0
        self.n_frames = 0
        self.n_surveys = 0
        self.n_witness = 0
        self.n_verify = 0
        self.calls: list[CallResult] = []
        self.stopped = False

        self._room: Optional[str] = None
        self._known: list[str] = []
        self._buf: dict[str, VisitBuf] = {}
        self._prev_hp: Optional[int] = None
        self._prev_key = False
        self._exit_heading: Optional[int] = None
        self._last_room_before_corridor: Optional[str] = None

    def on_frame(self, obs: np.ndarray, step: int = -1) -> None:
        self.n_frames += 1
        hud = read_hud(obs)
        if not hud.ok:
            return
        self.db.note_hp(hud.hp, hud.hp_max)

        if hud.hp is not None and self._prev_hp is not None and hud.hp < self._prev_hp:
            dmg = self._prev_hp - hud.hp
            room = self._room
            self.db.apply_damage(room, dmg)
            if room:
                self._keep_combat(room, obs, hud.heading or 0, step)
        if hud.hp is not None:
            self._prev_hp = hud.hp

        if hud.has_key and not self._prev_key:
            self.db.apply_key_pickup(self._room)
        if self._prev_key and not hud.has_key:
            self.db.apply_key_consumed()
        self._prev_key = hud.has_key

        if hint_banner_active(obs):
            text = (read_hint_text(obs) or "").strip()
            if text:
                self.db.apply_hint(text, self._room)
            if self._room is not None:
                self.db.apply_cv_lock(self._room, hud.heading)

        if hud.is_corridor or hud.room_name is None:
            if self._room is not None:
                self._exit_heading = hud.heading if hud.heading is not None else self._exit_heading
                self._last_room_before_corridor = self._room
                self._schedule_room_flush(self._room)
                self._room = None
            return
        if hud.heading is None:
            return

        room = canon_name(hud.room_name, self._known)
        if room not in self._known:
            self._known.append(room)

        if self._room is not None and room != self._room:
            self.db.apply_transition(
                self._room, room,
                self._exit_heading if self._exit_heading is not None else hud.heading,
            )
            self._schedule_room_flush(self._room)
        elif self._last_room_before_corridor and room != self._last_room_before_corridor:
            self.db.apply_transition(self._last_room_before_corridor, room, self._exit_heading)
            self._last_room_before_corridor = None

        if self._room != room:
            self.db.get_or_create(room).visits += 1
        self._room = room
        self._exit_heading = hud.heading
        rec = self.db.get_or_create(room)
        rec.n_frames += 1
        self._ingest_in_room(room, obs, hud.heading, step)

    def finalize(self, *, wait_s: float = 4.5) -> EpisodeDB:
        if self._room:
            self._schedule_room_flush(self._room)
            self._room = None
        if self.live_mode:
            self._wait_worker(timeout_s=wait_s)
        else:
            for name in list(self._buf.keys()):
                self._flush_survey(name)
        self._witness_damage_rooms()
        self.db.freeze()
        return self.db

    def _two_batches(self, buf: VisitBuf, n: int = BATCH_SIZE) -> tuple[list, list]:
        """Deux jeux d'angles décalés pour surveyor / verifier."""
        buckets = sorted(buf.by_heading.keys())
        if not buckets:
            allk = self._all_kept(buf)
            a, b = allk[:n], allk[n:2 * n]
            return a, (b or list(a))
        step = max(1, len(buckets) // n) if len(buckets) > n else 1
        a = []
        for bkt in buckets[::step]:
            a.append(buf.by_heading[bkt])
            if len(a) >= n:
                break
        offset = max(1, step // 2)
        b = []
        seen = {k.step for k in a}
        for bkt in buckets[offset::step]:
            item = buf.by_heading[bkt]
            if item.step in seen and len(buckets) >= n * 2:
                continue
            b.append(item)
            if len(b) >= n:
                break
        if not b:
            b = list(a)
        return a, b

    def _schedule_room_flush(self, room: str) -> None:
        if not self.live_mode or self._pool is None:
            self._flush_survey(room)
            return
        buf = self._buf.get(room)
        if buf is None or buf.sent_steps:
            return
        majority = buf.majority_color()
        if majority:
            self.db.apply_cv_wall_color(room, majority)
        batch_a, batch_b = self._two_batches(buf, BATCH_SIZE)
        if not batch_a:
            return
        for k in batch_a + batch_b:
            buf.sent_steps.add(k.step)
        buf.live_submitted = True
        fut = self._pool.submit(
            self._live_dual_job, room, batch_a, batch_b, majority,
        )
        self._futures.append(fut)

    def _live_dual_job(self, room, batch_a, batch_b, majority) -> None:
        if self.dry_run or not self._budget_ok():
            return
        ra_rb = self._call_dual(room, batch_a, batch_b, majority)
        with self._lock:
            self._record_dual(room, *ra_rb, batch_a, batch_b)

    def _call_dual(self, room, batch_a, batch_b, majority):
        frames_a = [k.frame for k in batch_a]
        frames_b = [k.frame for k in batch_b]
        color = best_wall_color(frames_a + frames_b) or majority
        ra, rb = survey_and_verify_parallel(
            frames_a, [k.heading for k in batch_a],
            frames_b, [k.heading for k in batch_b],
            room, color, self.api_key, self.survey_model, True,
        )
        return color, ra, rb

    def _record_dual(self, room, color, ra: CallResult, rb: CallResult, batch_a, batch_b) -> None:
        if color:
            self.db.apply_cv_wall_color(room, color)
        self.total_cost += ra.cost_usd + rb.cost_usd
        self.n_surveys += 1
        self.n_verify += 1
        self.calls.extend([ra, rb])
        da = ra.data or {} if ra.ok else {}
        db_ = rb.data or {} if rb.ok else {}
        if da:
            self.db.apply_survey(room, {
                "doors": da.get("doors"),
                "wall_color": da.get("wall_color"),
            })
        if db_:
            self.db.apply_survey(room, {
                "doors": db_.get("doors"),
                "wall_color": db_.get("wall_color"),
            })
        known = set()
        for name, rec in self.db.rooms.items():
            if name == room:
                continue
            for im in rec.images:
                cat = _canon_phrase(im.get("category") or im.get("desc") or "")
                if cat:
                    known.add(cat)
        self.db.apply_survey(room, {
            "images": _merge_images(da.get("images") or [], db_.get("images") or [], known),
            "objects": _intersect_objects(da.get("objects") or [], db_.get("objects") or []),
        })
        buf = self._buf.get(room)
        if buf is not None:
            buf.survey_enemies = _union_enemies(
                da.get("enemies_visible") or da.get("enemies") or [],
                db_.get("enemies_visible") or db_.get("enemies") or [],
            )
        if self.on_event:
            self.on_event("survey", room, ra, [k.heading for k in batch_a])
            self.on_event("verify", room, rb, [k.heading for k in batch_b])

    def _witness_damage_rooms(self) -> None:
        """Ennemis = frames de combat des salles où on a pris des dégâts, pas le spin mural."""
        for name, rec in self.db.rooms.items():
            if rec.damage_taken <= 0:
                rec.enemies = []
        if self.dry_run or not self._budget_ok():
            return
        for name, rec in self.db.rooms.items():
            if rec.damage_taken <= 0:
                continue
            buf = self._buf.get(name)
            frames = [k.frame for k in (buf.combat if buf else [])][:6]
            ens: list = []
            if frames:
                wr = witness_enemies(
                    frames,
                    api_key=self.api_key,
                    model=self.survey_model,
                    cv_wall_color=(buf.majority_color() if buf else rec.wall_color.value),
                )
                self.total_cost += wr.cost_usd
                self.n_witness += 1
                self.calls.append(wr)
                if self.on_event:
                    self.on_event(
                        "witness", name, wr,
                        [k.heading for k in (buf.combat if buf else [])][:6],
                    )
                if wr.ok and wr.data:
                    ens = wr.data.get("enemies") or wr.data.get("enemies_visible") or []
            if not ens and buf:
                ens = buf.survey_enemies
            if not ens:
                continue
            rec.enemies = []
            self.db.apply_survey(name, {"enemies_visible": ens})

    def _wait_worker(self, timeout_s: float = 4.5) -> None:
        if not self._futures:
            return
        wait(self._futures, timeout=timeout_s)

    def _buf_for(self, room: str) -> VisitBuf:
        if room not in self._buf:
            self._buf[room] = VisitBuf()
        return self._buf[room]

    def _ingest_in_room(self, room: str, obs: np.ndarray, heading: int, step: int) -> None:
        buf = self._buf_for(room)
        if detect_lock_panel(obs):
            self.db.apply_cv_lock(room, heading)
        buf.n_seen += 1
        color = sample_wall_color(obs)
        if color:
            buf.color_votes[color] = buf.color_votes.get(color, 0) + 1
        majority = buf.majority_color()
        cov = wall_color_fraction(obs, majority) if majority else 0.0

        if buf.last_heading is not None:
            d = (heading - buf.last_heading) % 360
            if d in (15, 345):
                buf.spin_run += 1
                if buf.spin_run >= SPIN_MIN_STEPS:
                    buf.spin_done = True
            else:
                buf.spin_run = 0
        buf.last_heading = heading

        copy = np.ascontiguousarray(obs.copy())
        kept = Kept(frame=copy, heading=heading, coverage=cov, step=step, kind="angle")

        bucket = (heading // HEADING_STEP) * HEADING_STEP
        old = buf.by_heading.get(bucket)
        if old is None or _view_score(cov) > _view_score(old.coverage) or (
            cov >= COVERAGE_MIN and old.coverage < COVERAGE_MIN
        ):
            buf.by_heading[bucket] = kept

        if buf.n_seen % EVERY_N == 0 and cov >= COVERAGE_MIN:
            if all(k.step != step for k in buf.every5):
                buf.every5.append(Kept(frame=copy, heading=heading, coverage=cov,
                                       step=step, kind="every5"))
                if len(buf.every5) > MAX_EVERY5:
                    # 시간적으로 균등하게: 앞쪽을 버림
                    buf.every5 = buf.every5[-MAX_EVERY5:]

        if majority and something_in_front(obs, majority):
            self._keep_combat(room, obs, heading, step)

    def _keep_combat(self, room: str, obs: np.ndarray, heading: int, step: int) -> None:
        buf = self._buf_for(room)
        if any(k.step == step for k in buf.combat):
            return
        buf.combat.append(Kept(
            frame=np.ascontiguousarray(obs.copy()),
            heading=heading or 0, coverage=1.0, step=step, kind="combat",
        ))
        del buf.combat[MAX_COMBAT:]

    def _all_kept(self, buf: VisitBuf) -> list[Kept]:
        by_step: dict[int, Kept] = {}
        for k in buf.by_heading.values():
            by_step[k.step] = k
        for k in buf.every5:
            by_step.setdefault(k.step, k)
        for k in buf.combat:
            by_step.setdefault(k.step, k)
        return sorted(by_step.values(), key=lambda k: k.step)

    def _unsent(self, buf: VisitBuf) -> list[Kept]:
        return [k for k in self._all_kept(buf) if k.step not in buf.sent_steps]

    def _verify_picks(self, buf: VisitBuf) -> list[Kept]:
        """서로 다른 heading이 최대한 퍼지도록 VERIFY_N장."""
        allk = self._all_kept(buf)
        if len(allk) <= VERIFY_N:
            return allk
        # 15° 버킷에서 고르게
        buckets = sorted(buf.by_heading.keys())
        picked = []
        if buckets:
            step = max(1, len(buckets) // VERIFY_N)
            for b in buckets[::step]:
                picked.append(buf.by_heading[b])
                if len(picked) >= VERIFY_N:
                    break
        # every5로 구멍 메움
        for k in buf.every5:
            if len(picked) >= VERIFY_N:
                break
            if k.step not in {p.step for p in picked}:
                picked.append(k)
        return picked[:VERIFY_N]

    def _flush_survey(self, room: str) -> None:
        buf = self._buf.get(room)
        if buf is None or buf.sent_steps:
            return
        majority = buf.majority_color()
        if majority:
            self.db.apply_cv_wall_color(room, majority)
        batch_a, batch_b = self._two_batches(buf, BATCH_SIZE)
        if not batch_a:
            return
        for k in batch_a + batch_b:
            buf.sent_steps.add(k.step)
        if self.dry_run or not self._budget_ok():
            dummy = CallResult(ok=True, data=None, cost_usd=0.0, error="dry_run", n_frames=len(batch_a))
            self.n_surveys += 1
            self.n_verify += 1
            self.calls.extend([dummy, dummy])
            return
        ra_rb = self._call_dual(room, batch_a, batch_b, majority)
        self._record_dual(room, *ra_rb, batch_a, batch_b)

    def _budget_ok(self) -> bool:
        if self.stopped or self.total_cost >= self.cost_limit_usd:
            self.stopped = True
            return False
        return True
