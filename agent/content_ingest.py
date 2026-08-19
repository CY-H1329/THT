"""Ingest: buffers frames per room during play, then hands them to the
VLM in the background when leaving a room, to fill agent/content_db.py.

Why it's built this way:
  - act() has a 5 s budget, so we can't make a synchronous VLM call
    every frame. Instead, the moment we leave a room, we hand off the
    few representative frames we collected there to a single worker
    thread all at once, and act() returns immediately (this carries
    over the pattern validated in the tmp/qa_ingest.py prototype,
    fitted to agent/vlm.py's cheap-single-call design).
  - Facts that don't need the VLM at all (where HP dropped, [KEY] tag
    transitions, visit order) are recorded here directly from the HUD
    OCR result -- so even if the network is completely down, this
    information survives and QA can still use it.
  - HUD OCR costs ~140 ms per frame, so we reuse the reading the
    explorer already cached instead of calling read_hud() again
    ourselves (which would double the cost).
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

# Number of frames sent to the VLM per room. Too few and we miss some
# of the 4 walls; too many and tokens/latency grow. Since SURVEY looks
# at all 4 directions face-on, bucketing by 15-degree headings and
# capping at 6 frames covers the room reasonably completely.
_MAX_FRAMES_PER_FLUSH = 6
_HEADING_BUCKET_DEG = 15
# Fallback buffer size for a room that never got a single SURVEY frame
# (e.g. right after spawn, walking straight through a door before
# survey even started -- a known case, see explorer.py).
_MAX_FALLBACK_FRAMES = 4
# Frames where an enemy is visible: measured directly that sending only
# SURVEY frames (a sweep of the 4 walls) can miss enemies entirely
# (seed 0: fought 3 times, but the enemies field ended up empty every
# time). Enemy appearance/color is an explicit README QA category, so
# we mix a few combat frames into the room's batch.
_MAX_COMBAT_FRAMES = 2
# How often to run the CV wall-color vote (no need to do it every frame).
_COLOR_VOTE_EVERY = 5
# Cap on how many times a single room gets flushed to the VLM again (a
# revisit produces new frames, but we don't want to spend the whole
# budget re-covering one room).
_MAX_FLUSHES_PER_ROOM = 2
# Where the frozen memory gets dumped for a human to inspect. Per the
# README, the only place the agent is allowed to write to disk is its
# own tmp/ directory, so we write only there, and silently do nothing
# if that fails (e.g. a read-only environment) -- answer() itself is
# entirely memory-based, so this file being absent has no effect on
# behavior.
_DB_DUMP_PATH = Path(__file__).resolve().parent.parent / "tmp" / "qa_out" / "db.json"

# finalize() runs inside the first answer() call -- per the README, if
# a question takes more than 10 s it scores zero, so we bound how long
# we're willing to wait here with a generous but finite timeout.
_FINALIZE_TIMEOUT_S = 6.0


@dataclass
class _Kept:
    frame: np.ndarray
    heading: int


class _RoomBuffer:
    def __init__(self) -> None:
        self.by_heading: dict = {}   # 15-degree bucket -> _Kept (SURVEY frame)
        self.fallback: list = []     # backup frames for a room that never got surveyed
        self.combat: list = []       # frames with an enemy visible / that landed a hit
        self.color_votes: dict = {}  # CV-measured wall-color votes
        self.seen: int = 0
        self.flushes: int = 0
        self.pending: bool = False   # whether a worker job is still in flight

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

    # Play phase
    def on_frame(self, obs: np.ndarray, explorer) -> None:
        self._explorer = explorer
        hud = explorer._read_hud_cached(obs)  # reuse the already-computed reading (see module docstring)
        if not hud.ok:
            return

        self.content.note_hp(hud.hp, hud.hp_max)
        if hud.seconds_remaining is not None:
            # Used to judge how the episode ended: if there was still
            # plenty of time left the last time we saw the HUD when QA
            # started, the episode ended in death, not a timeout (the
            # evaluation harness stops calling act() after termination,
            # so we never actually see the HP-0 frame itself).
            self.content.seconds_left_last = hud.seconds_remaining
        room = None
        if hud.room_name and not hud.is_corridor:
            room = explorer._canonicalize(hud.room_name)

        # 1) An HP drop is pixel-level proof that an enemy was really in this room.
        if hud.hp is not None:
            if self._prev_hp is not None and hud.hp < self._prev_hp:
                self.content.record_damage(self._room or room, self._prev_hp - hud.hp)
                if room:
                    self._keep_combat_frame(room, obs)
            self._prev_hp = hud.hp

        # 2) [KEY] tag transition -- turning on means "this is the room
        #    the key was in"; turning back off means "the key was used
        #    to open the door" (per the README, the key is only
        #    consumed by unlocking).
        if hud.has_key and not self._prev_key:
            self.content.note_key_found(self._room or room)
        elif self._prev_key and not hud.has_key:
            self.content.note_key_consumed()
        self._prev_key = hud.has_key

        # 3) Combat engagement count (basis for the kill estimate). FLEE
        #    only starts once HP actually drops (explorer.step), so one
        #    engagement = one real encounter with an enemy. We only
        #    count engagements where we actually landed an ATTACK.
        fleeing = explorer.state == "FLEE"
        if fleeing and not self._was_fleeing:
            self._flee_room = self._room or room
        elif self._was_fleeing and not fleeing:
            if self._flee_room and getattr(explorer, "flee_ever_attacked", False):
                self.content.get_or_create(self._flee_room).combat_bouts += 1
            self._flee_room = None
        self._was_fleeing = fleeing

        # 4) Detect a room change -- the moment we leave, flush that room's frame batch in the background.
        if room is None:
            return  # In the corridor: don't buffer, so we never mix these frames into a room's set.
        if self._room is not None and room != self._room:
            self._schedule_flush(self._room)
        self._room = room
        self.content.get_or_create(room)

        self._buffer_frame(room, obs, hud, explorer)

    def _buffer_frame(self, room: str, obs, hud, explorer) -> None:
        buf = self._buffers.setdefault(room, _RoomBuffer())
        buf.seen += 1

        # Wall color can be estimated fairly accurately from pixels
        # alone, no VLM needed (agent/vision.py). Collecting a
        # per-room majority vote means we can still answer "what color
        # were the walls" and match color hints even if the VLM never
        # responds or fails.
        if buf.seen % _COLOR_VOTE_EVERY == 1:
            color = sample_wall_color(obs)
            if color:
                buf.color_votes[color] = buf.color_votes.get(color, 0) + 1

        # Keep frames where an enemy is visible separately (reusing the
        # detection the explorer already ran this tick -- we don't run
        # CV again here).
        if any(getattr(m, "score", 0) >= 0.6 and getattr(m, "distance", 99) <= 4.0
               for m in getattr(explorer, "last_mobs", []) or []):
            self._keep_combat_frame(room, obs)

        if hud.heading is None:
            return
        if explorer.state == "SURVEY":
            # A frame captured while standing right at a doorway gets
            # contaminated with the neighboring room's content whole
            # (the doorway bleed-through issue noted in report.md).
            # SURVEY frames -- taken after moving into the room and
            # looking at all 4 directions -- are the priority, since
            # each direction is viewed face-on, giving the clearest
            # look at wall images/objects.
            bucket = (int(hud.heading) % 360) // _HEADING_BUCKET_DEG
            if bucket not in buf.by_heading:
                buf.by_heading[bucket] = _Kept(
                    frame=np.ascontiguousarray(obs.copy()), heading=int(hud.heading))
        elif len(buf.fallback) < _MAX_FALLBACK_FRAMES and buf.seen % _COLOR_VOTE_EVERY == 1:
            # There really are rooms that never get a SURVEY (e.g. the
            # spawn room, if the entry heading happens to be a door and
            # we just walk straight through -- see explorer.py). This
            # keeps a small backup buffer so such a room doesn't end up
            # with an entirely empty record.
            buf.fallback.append(_Kept(
                frame=np.ascontiguousarray(obs.copy()), heading=int(hud.heading)))

    def _keep_combat_frame(self, room: str, obs) -> None:
        buf = self._buffers.setdefault(room, _RoomBuffer())
        if len(buf.combat) >= _MAX_COMBAT_FRAMES:
            return
        buf.combat.append(_Kept(frame=np.ascontiguousarray(obs.copy()), heading=0))

    # Background VLM job
    def _pick_frames(self, buf: _RoomBuffer) -> list:
        """Pick buckets evenly spaced by heading, up to
        _MAX_FRAMES_PER_FLUSH, plus a few frames with an enemy visible."""
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
        """The current state of this room, shown to the VLM as "here's what's recorded so far."""
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
        # Start collecting fresh on the next visit (so we never send the same frame twice).
        buf.by_heading = {}
        buf.fallback = []
        buf.combat = []
        self.calls_used += 1
        if self._pool is None:
            # A single worker keeps calls from overlapping and
            # processes them in order (simplifies the ContentDB merge,
            # and doesn't increase concurrent load on OpenRouter either).
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
            # Whatever goes wrong on the VLM side, play has to keep going.
            pass
        finally:
            buf.pending = False

    # Freezing at the start of QA
    def finalize(self, explorer=None, timeout_s: float = _FINALIZE_TIMEOUT_S) -> ContentDB:
        """Flush the last room, wait (within the time limit) for
        in-flight jobs, then freeze the ContentDB. Runs exactly once,
        on the first answer() call."""
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
        """Save the frozen memory to tmp/qa_out/db.json for a human to inspect (debugging aid).

        The exploration graph (room connections/exit states/hints) isn't
        part of ContentDB, so it's included here too, so this one file
        alone shows everything the episode remembered. Any failure to
        save is just ignored.
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
        """Move facts the explorer FSM confirmed directly from pixels
        (the hint banner's exact text, the locked door's location) into
        ContentDB. These are more reliable than a VLM judgment, so they overwrite it."""
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
                    # CV-measured (pixel mode) wall color -- fallback for a room the VLM never filled in.
                    rec.wall_color.value = node.wall_color
                    rec.wall_color.confidence = 0.9
            for name, buf in self._buffers.items():
                rec = self.content.get_or_create(name)
                if rec.wall_color.value is None and buf.majority_color():
                    # Last-resort fallback for a room that never even got
                    # surveyed -- the majority vote of pixel colors
                    # collected while we were in it.
                    rec.wall_color.value = buf.majority_color()
                    rec.wall_color.confidence = 0.85
            # The key's room: if we have a room where [KEY] was actually
            # picked up, that's the answer; otherwise (never found the
            # key) fall back to guessing from the hint text.
            if self.content.key_found_in:
                self.content.key_hint_room = self.content.key_found_in
            else:
                guess = resolve_hint_room(self.content.hint_text.value, self.content)
                if guess:
                    self.content.key_hint_room = guess
