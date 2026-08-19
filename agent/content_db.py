"""Confidence-gated content DB -- everything the VLM has "seen," rolled up per room.

Deliberately kept separate from agent/memory.py (the exploration
SceneGraph):
  - SceneGraph is live exploration state, updated every step during play.
  - This ContentDB is "observed content," filled in by the VLM after
    play ends (or on the first answer() call) -- a completely different
    update cadence and nature.
  - The two are linked only by room name (string) as a shared key. Even
    if exploration is flaky, this DB (the heart of the 60% QA score)
    stays independently robust.

Core rule: an observation with confidence < CONFIDENCE_THRESHOLD is
never committed, so a shaky recognition can't pollute the DB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

CONFIDENCE_THRESHOLD = 0.8

# Actual game-structure bounds confirmed in world/layout.py (true for
# every seed):
#   - objects: exactly 1-2 per room (`objects_rng.randint(1, 2)`).
#   - images: 0-2 per wall x 4 walls = 8 max.
# An observation past this cap is likely contamination (something from
# an adjacent room leaking in through a doorway), so once we hit it we
# swap in the new item only if it beats the lowest-confidence existing
# entry -- dropping the new one outright would let an early bad reading
# stick around forever, so replacing is safer.
MAX_OBJECTS_PER_ROOM = 2
MAX_IMAGES_PER_ROOM = 8


@dataclass
class ConfidentValue:
    """A single value (e.g. a room's wall color) plus its confidence."""
    value: Optional[str] = None
    confidence: float = 0.0

    def maybe_update(self, new_value: str, new_confidence: float) -> bool:
        """Ignore if new_confidence is below the threshold. Otherwise
        update only if it beats the current confidence (or there's no
        value yet). Returns True if it updated."""
        if new_confidence < CONFIDENCE_THRESHOLD:
            return False
        if self.value is None or new_confidence >= self.confidence:
            self.value = new_value
            self.confidence = new_confidence
            return True
        return False


@dataclass
class RoomContent:
    name: str
    wall_color: ConfidentValue = field(default_factory=ConfidentValue)
    has_locked_door: ConfidentValue = field(default_factory=ConfidentValue)
    images: list = field(default_factory=list)   # [{desc, wall_dir, confidence}]
    objects: list = field(default_factory=list)  # [{name, color, confidence}]
    enemies: list = field(default_factory=list)  # [{desc, color, confidence}]
    frame_count: int = 0
    # Total HP actually lost while in this room (filled in every frame
    # by agent/content_ingest.py from the HUD's HP drop). This is the
    # core evidence for the kill-count estimate: getting hit means an
    # enemy was really there, which distinguishes it from the VLM
    # mistakenly seeing an enemy in the next room through a doorway
    # (a heuristic already validated in the tmp/qa_db.py prototype).
    damage_taken: int = 0
    # Number of engagements where we actually landed an ATTACK in this
    # room (FLEE strike bouts) -- the upper bound for the kill estimate
    # (at most 2 enemies per room).
    combat_bouts: int = 0

    def image_count_on_wall(self, wall_dir) -> int:
        return sum(1 for im in self.images if im.get("wall_dir") == wall_dir)


class ContentDB:
    def __init__(self) -> None:
        self.rooms: dict[str, RoomContent] = {}
        self.hint_text = ConfidentValue()
        self.locked_door_room: Optional[str] = None
        self.locked_door_heading: Optional[int] = None
        self.key_found: bool = False
        self.key_found_in: Optional[str] = None   # room we were in when [KEY] first turned on
        self.door_unlocked: bool = False          # [KEY] turned back off = key consumed = door opened
        self.key_hint_room: Optional[str] = None
        self.killed_enemies: int = 0
        self.damage_taken_total: int = 0
        self.hp_start: Optional[int] = None
        self.hp_end: Optional[int] = None
        self.hp_max: Optional[int] = None
        # Last seconds-remaining value read off the HUD. If there was
        # still plenty of time left when QA started, the episode ended
        # in death, not a timeout.
        self.seconds_left_last: Optional[int] = None
        self.visited_order: list = []
        self.frozen: bool = False

    def get_or_create(self, room_name: str) -> RoomContent:
        if room_name not in self.rooms:
            self.rooms[room_name] = RoomContent(name=room_name)
            self.visited_order.append(room_name)
        return self.rooms[room_name]

    # Facts filled in directly from CV/HUD during play (always trustworthy, no VLM needed)
    def record_damage(self, room_name: Optional[str], amount: int) -> None:
        if self.frozen or amount <= 0:
            return
        self.damage_taken_total += amount
        if room_name:
            self.get_or_create(room_name).damage_taken += amount

    def note_hp(self, hp: Optional[int], hp_max: Optional[int]) -> None:
        if self.frozen:
            return
        if hp_max is not None:
            self.hp_max = hp_max
        if hp is None:
            return
        if self.hp_start is None:
            self.hp_start = hp
        self.hp_end = hp

    def note_key_found(self, room_name: Optional[str]) -> None:
        """HUD's [KEY] tag went off -> on. Whatever room we were in the
        moment we picked it up is, by definition, the room the key was
        in -- a fact we're 100% sure of from pixels alone."""
        if self.frozen:
            return
        self.key_found = True
        if room_name and self.key_found_in is None:
            self.key_found_in = room_name

    def note_key_consumed(self) -> None:
        """[KEY] tag went on -> off. Per the README, the key is only consumed by unlocking the door."""
        if self.frozen:
            return
        self.door_unlocked = True

    def apply_update(self, room_name: str, update: dict) -> None:
        """Merge one JSON result from vlm.update_room_db() into this room.

        Update shape (matches the schema in agent/vlm.py):
          {
            "wall_color": {"value": str, "confidence": float} | None,
            "has_locked_door": bool | None (True/False itself may have
                no confidence field; if present we still gate it at 0.8),
            "images": [{"desc": str, "wall_dir": int|None, "confidence": float}],
            "objects": [{"name": str, "confidence": float}],
            "enemies": [{"desc": str, "color": str, "confidence": float}],
            "hint_text": {"value": str, "confidence": float} | None,
          }
        """
        room = self.get_or_create(room_name)
        room.frame_count += 1

        wc = update.get("wall_color")
        if wc and wc.get("value"):
            room.wall_color.maybe_update(wc["value"], wc.get("confidence", 0.0))

        hld = update.get("has_locked_door")
        hld_conf = update.get("has_locked_door_confidence", 1.0 if hld is not None else 0.0)
        if hld is not None:
            room.has_locked_door.maybe_update(str(bool(hld)), hld_conf)
            if hld and hld_conf >= CONFIDENCE_THRESHOLD:
                self.locked_door_room = room_name

        for img in update.get("images", []) or []:
            if img.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
                continue
            if not _has_similar(room.images, img, text_key="desc"):
                _add_with_cap(room.images, img, MAX_IMAGES_PER_ROOM)

        for obj in update.get("objects", []) or []:
            if obj.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
                continue
            if not _has_similar(room.objects, obj, text_key="name"):
                _add_with_cap(room.objects, obj, MAX_OBJECTS_PER_ROOM)

        for en in update.get("enemies", []) or []:
            if en.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
                continue
            if not _has_similar(room.enemies, en, text_key="desc"):
                room.enemies.append(en)

        hint = update.get("hint_text")
        if hint and hint.get("value"):
            if self.hint_text.maybe_update(hint["value"], hint.get("confidence", 0.0)):
                self.key_hint_room = room_name

    # Freezing memory at the start of QA
    # README: "Memory is frozen at the start of the QA phase." So this is
    # the one place we do cleanup that could only be judged correctly
    # after seeing the whole episode -- not something we could know
    # mid-play. All three rules were already validated in the tmp/qa_db.py
    # prototype:
    #   1) Drop any recorded image on the locked-door wall -- nothing
    #      actually hangs on a padlock panel. This catches the VLM
    #      mistaking the yellow panel/padlock icon for a picture.
    #   2) If two adjacent rooms recorded the same-looking image, and one
    #      of them was recorded on the wall that holds the door between
    #      them, treat it as bleed-through from the doorway and drop it
    #      (the direct fix for the doorway bleed-through issue noted in
    #      report.md).
    #   3) If the same-looking enemy is recorded in two adjacent rooms,
    #      keep it only in the one where damage_taken is actually higher
    #      -- there's really just one mob, visible through the doorway
    #      and double-recorded in both rooms.
    def freeze(self, scene=None) -> None:
        if self.frozen:
            return
        adjacency = _adjacency_from_scene(scene)
        self._drop_lock_wall_images(scene)
        self._prune_adjacent_image_leaks(adjacency)
        self._prune_adjacent_enemy_leaks(adjacency)
        self._estimate_kills()
        self.frozen = True

    def _drop_lock_wall_images(self, scene) -> None:
        if scene is None:
            return
        for name, room in self.rooms.items():
            node = scene.nodes.get(name)
            if node is None:
                continue
            lock_dirs = {h for h, st in node.exits.items() if st == "locked"}
            if not lock_dirs:
                continue
            room.images = [im for im in room.images
                           if _snap_heading(im.get("wall_dir")) not in lock_dirs]

    def _prune_adjacent_image_leaks(self, adjacency: dict) -> None:
        for a, links in adjacency.items():
            room_a = self.rooms.get(a)
            if room_a is None:
                continue
            for dir_ab, b in links.items():
                room_b = self.rooms.get(b)
                if room_b is None:
                    continue
                # Among the images recorded on a's "wall facing b," drop
                # only the ones where b also has a similar-looking image
                # (both rooms could genuinely hang similar pictures, so
                # we only act when it's also the shared wall with the
                # door).
                room_a.images = [
                    im for im in room_a.images
                    if not (_snap_heading(im.get("wall_dir")) == dir_ab
                            and _has_similar(room_b.images, im, text_key="desc"))
                ]

    def _prune_adjacent_enemy_leaks(self, adjacency: dict) -> None:
        for a, links in adjacency.items():
            room_a = self.rooms.get(a)
            if room_a is None:
                continue
            for b in links.values():
                room_b = self.rooms.get(b)
                if room_b is None or room_a is room_b:
                    continue
                # Whichever room we actually took more damage in is the
                # one the enemy was really in. If it's a tie, leave both
                # alone -- not enough evidence either way.
                if room_a.damage_taken >= room_b.damage_taken:
                    continue
                room_a.enemies = [
                    en for en in room_a.enemies
                    if not _same_enemy_in(room_b.enemies, en)
                ]

    def _estimate_kills(self) -> None:
        """Kill-count estimate: only count rooms where we were actually
        hit (so an enemy was definitely there) AND we fought back in
        that room. Each room has at most 2 enemies, so we cap the bout
        count at 2. Our policy never retreats -- it fights to the end
        wherever it's hit (see the measured evidence in the comment at
        the top of explorer.py) -- so one engagement roughly corresponds
        to one kill. Not perfect, but far better for QA than answering
        "0" or "I don't know."
        """
        kills = 0
        for room in self.rooms.values():
            if room.damage_taken <= 0 or room.combat_bouts <= 0:
                continue
            kills += min(2, room.combat_bouts)
        self.killed_enemies = kills

    def to_dict(self) -> dict:
        return {
            "rooms": {
                name: {
                    "wall_color": r.wall_color.value,
                    "has_locked_door": r.has_locked_door.value,
                    "images": r.images,
                    "objects": r.objects,
                    "enemies": r.enemies,
                }
                for name, r in self.rooms.items()
            },
            "hint_text": self.hint_text.value,
            "locked_door_room": self.locked_door_room,
            "key_found": self.key_found,
            "key_found_in": self.key_found_in,
            "door_unlocked": self.door_unlocked,
            "key_hint_room": self.key_hint_room,
            "killed_enemies": self.killed_enemies,
            "damage_taken_total": self.damage_taken_total,
            "hp_start": self.hp_start,
            "hp_end": self.hp_end,
            "hp_max": self.hp_max,
            "seconds_left_last": self.seconds_left_last,
            "visited_order": self.visited_order,
            "frozen": self.frozen,
        }


def _add_with_cap(existing: list, new_item: dict, cap: int) -> None:
    """Append new_item to existing, but once past cap, swap in the new
    item only if it beats the lowest-confidence existing entry (past
    the game's own structural limit, an observation is almost certainly
    contamination either way, so we'd rather replace than let the list
    grow unbounded)."""
    if len(existing) < cap:
        existing.append(new_item)
        return
    worst_idx = min(range(len(existing)), key=lambda i: existing[i].get("confidence", 0.0))
    if new_item.get("confidence", 0.0) > existing[worst_idx].get("confidence", 0.0):
        existing[worst_idx] = new_item
    # else: the new item is even weaker than the current worst -- drop it, keep the cap.


def _has_similar(existing: list, new_item: dict, text_key: str, threshold: float = 0.35) -> bool:
    """True if new_item looks close enough to an existing entry that
    it's plausibly the VLM describing the same thing in different words.
    The VLM avoiding duplicates on its own (by seeing the current DB) is
    the first line of defense; this is the second-line safety net for
    whatever still slips through."""
    new_text = str(new_item.get(text_key, "")).lower()
    new_tokens = set(new_text.split())
    if not new_tokens:
        return False
    for item in existing:
        old_tokens = set(str(item.get(text_key, "")).lower().split())
        if not old_tokens:
            continue
        inter = len(new_tokens & old_tokens)
        union = len(new_tokens | old_tokens)
        if union and inter / union >= threshold:
            return True
    return False


CARDINAL_HEADINGS = (0, 90, 180, 270)


def _snap_heading(heading):
    """Snap a VLM-reported wall_dir to the nearest cardinal heading.
    Doors and walls only ever exist at 0/90/180/270 (the grid layout in
    world/layout.py)."""
    if heading is None:
        return None
    try:
        h = int(heading) % 360
    except (TypeError, ValueError):
        return None
    return min(CARDINAL_HEADINGS, key=lambda c: min(abs(h - c), 360 - abs(h - c)))


def _adjacency_from_scene(scene) -> dict:
    """SceneGraph -> {room: {exit heading: neighboring room}}. Only
    includes doors we've actually walked through (explorer only fills
    in exit_leads_to then)."""
    if scene is None:
        return {}
    return {name: dict(node.exit_leads_to) for name, node in scene.nodes.items()}


def _same_enemy_in(existing: list, new_item: dict) -> bool:
    """Treat two enemies as the same mob if their appearance
    (shirt/pants/skin/body shape) matches. If none of the color fields
    were ever filled in, fall back to description-text similarity."""
    key = tuple(new_item.get(k) for k in
                ("shirt_color", "pants_color", "skin_color", "body_shape"))
    if any(v for v in key):
        for old in existing:
            if tuple(old.get(k) for k in
                     ("shirt_color", "pants_color", "skin_color", "body_shape")) == key:
                return True
        return False
    return _has_similar(existing, new_item, text_key="desc")
