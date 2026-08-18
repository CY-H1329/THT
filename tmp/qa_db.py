"""QA용 에피소드 콘텐츠 DB. 탐험(explorer) / 전투 VLM과 분리.

규칙(difficulty.yaml + world/layout.py + README):
  - 방 8~12개, 문은 0/90/180/270만.
  - 오브젝트: 방당 1~2개.
  - 벽 그림: 벽당 0~2장, 방당 최대 8장.
  - 적: 스폰방 제외, 방당 0~2마리. 6큐브 몹. shirt/pants/skin + short/tall.
  - 잠긴 문 1개, 힌트 1개, 열쇠 1개.
  - QA 시작 이후 메모리는 동결(README).

이 모듈은 agent/* 를 수정하지 않는다. 나중에 agent/qa_db.py 로 옮기면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

CONFIDENCE_THRESHOLD = 0.8
MAX_OBJECTS_PER_ROOM = 2
MAX_IMAGES_PER_WALL = 2
MAX_IMAGES_PER_ROOM = 8
MAX_ENEMIES_PER_ROOM = 2
CARDINAL = (0, 90, 180, 270)

WALL_COLOR_NAMES = (
    "terracotta", "sage", "slate", "mustard", "plum", "sand",
    "teal", "brick", "olive", "denim", "lavender", "rust",
)

# 픽셀로 항상 같은 것(미니월드 기본 바닥/천장). QA가 "바닥/천장"을 물을 수 있어 기록.
GAME_LOOK = {
    "floor": "two-tone checkerboard tiles",
    "ceiling": "plain grey concrete",
    "turn_step_deg": 15,
    "doors_only_cardinal": True,
}


def _tokens(text: str) -> set[str]:
    return {w for w in str(text).lower().replace("-", " ").split() if w}


def _canon_phrase(text: str) -> str:
    t = " ".join(str(text or "").lower().replace("-", " ").split())
    if t.endswith("ies") and len(t) > 5:
        t = t[:-3] + "y"
    elif t.endswith("s") and not t.endswith("ss") and len(t) > 4:
        t = t[:-1]
    return OBJECT_ALIASES.get(t, t)


OBJECT_ALIASES = {
    "duck": "rubber duck",
    "ducky": "rubber duck",
    "duckie": "rubber duck",
    "rubber ducky": "rubber duck",
    "cone": "traffic cone",
    "trafficcone": "traffic cone",
    "orange cone": "traffic cone",
    "office chair": "chair",
    "taxi": "car",
    "yellow car": "car",
    "racecar": "car",
    "race car": "car",
    "foliage": "plant",
    "leaves": "plant",
}

# trop générique pour fusionner deux photos différentes
_GENERIC_CATS = frozenset({
    "abstract", "art", "painting", "photo", "picture", "image",
    "poster", "pattern", "wall", "scene",
})


def _similar(a: str, b: str, threshold: float = 0.30) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:
        return True
    return len(ta & tb) / len(ta | tb) >= threshold


def _same_image(old: dict, new: dict) -> bool:
    oc = _canon_phrase(old.get("category") or "")
    nc = _canon_phrase(new.get("category") or "")
    if oc and nc and oc == nc and oc not in _GENERIC_CATS:
        return True
    return _similar(old.get("desc") or "", new.get("desc") or "")


def _snap_heading(h: Optional[int]) -> Optional[int]:
    if h is None:
        return None
    return min(CARDINAL, key=lambda c: min(abs(h - c), 360 - abs(h - c)))


@dataclass
class ConfidentValue:
    value: Optional[str] = None
    confidence: float = 0.0

    def maybe_update(self, new_value: str, new_confidence: float) -> bool:
        if new_confidence < CONFIDENCE_THRESHOLD or not new_value:
            return False
        if self.value is None or new_confidence >= self.confidence:
            self.value = new_value
            self.confidence = new_confidence
            return True
        return False


@dataclass
class DoorSlot:
    status: str = "unknown"  # open | locked | wall | unknown
    leads_to: Optional[str] = None
    neighbor_color: Optional[str] = None
    confidence: float = 0.0
    from_transition: bool = False
    from_cv_lock: bool = False


@dataclass
class RoomRecord:
    name: str
    wall_color: ConfidentValue = field(default_factory=ConfidentValue)
    images: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    doors: dict = field(default_factory=lambda: {d: DoorSlot() for d in CARDINAL})
    enemies: list = field(default_factory=list)
    has_locked_door: bool = False
    damage_taken: int = 0
    visits: int = 0
    surveyed: bool = False
    n_frames: int = 0

    def image_count_on_wall(self, wall_dir: Optional[int]) -> int:
        return sum(1 for im in self.images if im.get("wall_dir") == wall_dir)

    def door_count(self) -> int:
        return sum(1 for d in self.doors.values() if d.status in ("open", "locked"))


@dataclass
class EpisodeDB:
    rooms: dict[str, RoomRecord] = field(default_factory=dict)
    hint_text: ConfidentValue = field(default_factory=ConfidentValue)
    key_hint_room: Optional[str] = None
    key_found: bool = False
    key_found_in: Optional[str] = None
    key_consumed: bool = False
    locked_door_room: Optional[str] = None
    locked_from_cv: bool = False
    unlocked_door: bool = False
    damage_taken_total: int = 0
    hp_start: Optional[int] = None
    hp_end: Optional[int] = None
    hp_max: Optional[int] = None
    killed_enemies: int = 0
    visited_order: list = field(default_factory=list)
    frozen: bool = False
    game_look: dict = field(default_factory=lambda: dict(GAME_LOOK))

    def get_or_create(self, name: str) -> RoomRecord:
        if name not in self.rooms:
            self.rooms[name] = RoomRecord(name=name)
            self.visited_order.append(name)
        return self.rooms[name]

    def apply_survey(self, room_name: str, update: dict) -> None:
        if self.frozen:
            return
        room = self.get_or_create(room_name)
        room.surveyed = True

        wc = update.get("wall_color") or {}
        if wc.get("value"):
            room.wall_color.maybe_update(str(wc["value"]), float(wc.get("confidence") or 0))

        # 문은 이미지/오브젝트보다 먼저: 이후 필터가 열린 문 방향을 알 수 있게.
        doors = update.get("doors") or {}
        for d in CARDINAL:
            raw = doors.get(f"dir_{d}") or doors.get(str(d))
            if not raw:
                continue
            self._merge_door(room.doors[d], raw, from_transition=False)

        for img in update.get("images") or []:
            self._add_image(room, img)
        for obj in update.get("objects") or []:
            self._add_object(room, obj)
        for en in update.get("enemies_visible") or update.get("enemies") or []:
            self._add_enemy(room, en)

    def apply_cv_lock(self, room_name: str, heading: Optional[int]) -> None:
        """Plaque cadenas vue en pixels. Une seule porte verrouillée par épisode."""
        if self.frozen or not room_name:
            return
        room = self.get_or_create(room_name)
        d = _snap_heading(heading)
        if d is None:
            return
        slot = room.doors[d]
        if slot.from_transition:
            return
        slot.status = "locked"
        slot.confidence = 1.0
        slot.from_cv_lock = True
        room.has_locked_door = True
        self.locked_from_cv = True
        self.locked_door_room = room_name

    def apply_cv_wall_color(self, room_name: str, color: str, confidence: float = 0.95) -> None:
        if self.frozen or not color:
            return
        self.get_or_create(room_name).wall_color.maybe_update(color, confidence)

    def apply_transition(self, src: str, dst: str, src_heading: Optional[int]) -> None:
        """A에서 B로 실제로 걸어 넘어감. VLM보다 신뢰도 높음."""
        if self.frozen or not src or not dst or src == dst:
            return
        a = self.get_or_create(src)
        b = self.get_or_create(dst)
        exit_dir = _snap_heading(src_heading)
        if exit_dir is not None:
            slot = a.doors[exit_dir]
            slot.status = "open"
            slot.leads_to = dst
            slot.confidence = 1.0
            slot.from_transition = True
            # B의 입구는 등 뒤(대략 반대 방향)
            enter_dir = (exit_dir + 180) % 360
            bslot = b.doors[enter_dir]
            bslot.status = "open"
            bslot.leads_to = src
            bslot.confidence = max(bslot.confidence, 0.85)
            bslot.from_transition = True

    def apply_hint(self, text: str, seen_in_room: Optional[str], confidence: float = 0.95) -> None:
        if self.frozen or not text:
            return
        if self.hint_text.maybe_update(text, confidence) and seen_in_room:
            # 힌트는 잠긴 문 앞에서 뜸. 열쇠 방은 힌트 내용으로 나중에 매칭.
            if self.locked_door_room is None:
                self.locked_door_room = seen_in_room
            self.get_or_create(seen_in_room).has_locked_door = True

    def apply_key_pickup(self, room_name: Optional[str]) -> None:
        if self.frozen:
            return
        self.key_found = True
        if room_name:
            self.key_found_in = room_name

    def apply_key_consumed(self) -> None:
        if self.frozen:
            return
        self.key_consumed = True
        self.unlocked_door = True

    def apply_damage(self, room_name: Optional[str], amount: int) -> None:
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

    def resolve_hint_room(self) -> None:
        """힌트 문구를 관측된 방 특징에 매칭해 key가 있던/있어야 할 방을 추정."""
        if self.frozen:
            return
        text = (self.hint_text.value or "").lower()
        if not text:
            return
        best, best_score = None, 0
        for name, room in self.rooms.items():
            score = 0
            color = (room.wall_color.value or "").lower()
            if color and color in text:
                score += 3
            for obj in room.objects:
                n = str(obj.get("name") or "").lower()
                if n and n in text:
                    score += 3
            for im in room.images:
                cat = str(im.get("category") or im.get("desc") or "").lower()
                if cat:
                    for w in cat.split():
                        if len(w) > 3 and w in text:
                            score += 2
                            break
            # "two images on its east wall" 류
            if "east" in text and room.image_count_on_wall(0) >= 2:
                score += 2
            if "south" in text and room.image_count_on_wall(90) >= 2:
                score += 2
            if "west" in text and room.image_count_on_wall(180) >= 2:
                score += 2
            if "north" in text and room.image_count_on_wall(270) >= 2:
                score += 2
            if score > best_score:
                best, best_score = name, score
        if best and best_score >= 2:
            self.key_hint_room = best

    def estimate_kills(self) -> None:
        """적 메시가 사라진 방 ≈ 처치. 과대추정 방지를 위해 데미지를 받은 방만."""
        if self.frozen:
            return
        kills = 0
        for room in self.rooms.values():
            if room.damage_taken <= 0:
                continue
            # 데미지를 받았는데 최종 서베이에 적이 없으면 죽였거나 다른 방으로 追격
            visible = [e for e in room.enemies if e.get("seen_alive")]
            if not visible:
                kills += 1
            else:
                # 보인 적 중 killed 플래그
                kills += sum(1 for e in room.enemies if e.get("killed"))
        self.killed_enemies = kills

    def _rooms_adjacent(self, a: str, b: str) -> bool:
        ra, rb = self.rooms.get(a), self.rooms.get(b)
        if not ra or not rb:
            return False
        for slot in ra.doors.values():
            if slot.leads_to == b:
                return True
        for slot in rb.doors.values():
            if slot.leads_to == a:
                return True
        return False

    def prune_adjacent_enemy_leaks(self) -> None:
        """Même mob vu dans deux salles voisines → garder celle où on a pris des dégâts."""
        from collections import defaultdict
        loc: dict[tuple, list[str]] = defaultdict(list)
        for name, room in self.rooms.items():
            for e in room.enemies:
                key = (e.get("shirt_color"), e.get("pants_color"), e.get("skin_color"), e.get("body_shape"))
                loc[key].append(name)
        for key, names in loc.items():
            uniq = list(dict.fromkeys(names))
            if len(uniq) < 2:
                continue
            pairs = [(x, y) for i, x in enumerate(uniq) for y in uniq[i + 1:] if self._rooms_adjacent(x, y)]
            if not pairs:
                continue
            involved = {n for p in pairs for n in p}
            best = max(involved, key=lambda n: self.rooms[n].damage_taken)
            for n in involved:
                if n == best:
                    continue
                self.rooms[n].enemies = [
                    e for e in self.rooms[n].enemies
                    if (e.get("shirt_color"), e.get("pants_color"), e.get("skin_color"), e.get("body_shape")) != key
                ]

    def prune_adjacent_image_leaks(self) -> None:
        """Même sujet vu sur le mur-porte qui mène à la salle voisine → fuite, on jette."""
        from collections import defaultdict
        by_cat: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for name, room in self.rooms.items():
            for im in room.images:
                cat = _canon_phrase(im.get("category") or im.get("desc") or "")
                if not cat or cat in _GENERIC_CATS:
                    continue
                by_cat[cat].append((name, im))

        drop_ids: dict[str, set[int]] = defaultdict(set)

        for _cat, locs in by_cat.items():
            rooms = list(dict.fromkeys(n for n, _ in locs))
            if len(rooms) < 2:
                continue
            for i, a in enumerate(rooms):
                for b in rooms[i + 1:]:
                    if not self._rooms_adjacent(a, b):
                        continue
                    a_ims = [im for n, im in locs if n == a]
                    b_ims = [im for n, im in locs if n == b]
                    if not a_ims or not b_ims:
                        continue
                    dir_ab = next(
                        (d for d, s in self.rooms[a].doors.items() if s.leads_to == b),
                        None,
                    )
                    dir_ba = next(
                        (d for d, s in self.rooms[b].doors.items() if s.leads_to == a),
                        None,
                    )
                    for im in a_ims:
                        if dir_ab is not None and im.get("wall_dir") == dir_ab:
                            drop_ids[a].add(id(im))
                    for im in b_ims:
                        if dir_ba is not None and im.get("wall_dir") == dir_ba:
                            drop_ids[b].add(id(im))

        for name, ids in drop_ids.items():
            self.rooms[name].images = [
                im for im in self.rooms[name].images if id(im) not in ids
            ]

    def drop_lock_wall_images(self) -> None:
        """Rien n'est accroché sur le panneau cadenas."""
        for room in self.rooms.values():
            lock_dirs = {
                d for d, s in room.doors.items()
                if s.status == "locked" or s.from_cv_lock
            }
            if not lock_dirs:
                continue
            room.images = [im for im in room.images if im.get("wall_dir") not in lock_dirs]

    def freeze(self) -> None:
        self.drop_lock_wall_images()
        self.prune_adjacent_image_leaks()
        self.prune_adjacent_enemy_leaks()
        for room in self.rooms.values():
            if room.damage_taken <= 0:
                room.enemies = []
        if self.locked_from_cv and self.locked_door_room:
            for name, room in self.rooms.items():
                if name == self.locked_door_room:
                    continue
                room.has_locked_door = False
                for slot in room.doors.values():
                    if slot.from_cv_lock:
                        continue
                    if slot.status == "locked" and not slot.from_transition:
                        slot.status = "unknown"
        self.resolve_hint_room()
        self.estimate_kills()
        self.frozen = True

    def _add_image(self, room: RoomRecord, img: dict) -> None:
        conf = float(img.get("confidence") or 0)
        if conf < CONFIDENCE_THRESHOLD:
            return
        desc = (img.get("desc") or "").strip()
        if not desc:
            return
        wall = img.get("wall_dir")
        if wall is not None:
            try:
                wall = _snap_heading(int(wall))
            except (TypeError, ValueError):
                wall = None
        cat = _canon_phrase((img.get("category") or "").strip()) or None
        blob = f"{desc} {cat or ''}".lower()
        lockish = (
            "lock", "padlock", "keyhole", "plaque", "door panel",
            "golden panel", "gold panel", "yellow panel", "cadenas",
        )
        if any(w in blob for w in lockish):
            return
        item = {
            "desc": desc,
            "category": cat,
            "wall_dir": wall,
            "confidence": conf,
        }
        for old in room.images:
            if _same_image(old, item):
                if conf > float(old.get("confidence") or 0):
                    old.update(item)
                return
        if wall is not None and room.image_count_on_wall(wall) >= MAX_IMAGES_PER_WALL:
            self._replace_worst(room.images, item, lambda im: im.get("wall_dir") == wall)
            return
        if len(room.images) >= MAX_IMAGES_PER_ROOM:
            self._replace_worst(room.images, item, lambda _im: True)
            return
        room.images.append(item)

    def _add_object(self, room: RoomRecord, obj: dict) -> None:
        conf = float(obj.get("confidence") or 0)
        if conf < CONFIDENCE_THRESHOLD:
            return
        name = _canon_phrase((obj.get("name") or "").strip())
        if not name:
            return
        blocked = {"block", "panel", "lock", "door", "wall", "cube", "shape", "rectangle", "pillar"}
        if any(b in name.lower() for b in blocked):
            return
        item = {
            "name": name,
            "color": obj.get("color"),
            "confidence": conf,
        }
        for old in room.objects:
            if _canon_phrase(old.get("name") or "") == name or _similar(old.get("name") or "", name):
                if conf > float(old.get("confidence") or 0):
                    old.update(item)
                return
        if len(room.objects) >= MAX_OBJECTS_PER_ROOM:
            self._replace_worst(room.objects, item, lambda _o: True)
            return
        room.objects.append(item)

    def _add_enemy(self, room: RoomRecord, en: dict) -> None:
        conf = float(en.get("confidence") or 0)
        if conf < CONFIDENCE_THRESHOLD:
            return
        item = {
            "shirt_color": en.get("shirt_color") or en.get("torso_color"),
            "pants_color": en.get("pants_color"),
            "skin_color": en.get("skin_color") or en.get("head_color"),
            "body_shape": en.get("body_shape"),
            "desc": en.get("desc") or "",
            "seen_alive": bool(en.get("alive_looking", True)),
            "killed": bool(en.get("killed", False)),
            "confidence": conf,
        }
        key = (item["shirt_color"], item["pants_color"], item["skin_color"], item["body_shape"])
        for old in room.enemies:
            old_key = (old.get("shirt_color"), old.get("pants_color"), old.get("skin_color"), old.get("body_shape"))
            if old_key == key or _similar(old.get("desc") or "", item["desc"] or "x"):
                if conf >= float(old.get("confidence") or 0):
                    old.update(item)
                return
        if len(room.enemies) >= MAX_ENEMIES_PER_ROOM:
            return
        room.enemies.append(item)

    @staticmethod
    def _merge_door(slot: DoorSlot, raw: dict, from_transition: bool) -> None:
        conf = float(raw.get("confidence") or 0)
        if conf < CONFIDENCE_THRESHOLD:
            return
        if slot.from_cv_lock and not from_transition:
            nc = raw.get("neighbor_color")
            if nc:
                slot.neighbor_color = nc
            return
        if slot.from_transition and not from_transition:
            # 실제 통과가 VLM 추측보다 우선. neighbor_color만 보강.
            nc = raw.get("neighbor_color")
            if nc:
                slot.neighbor_color = nc
            return
        if conf < slot.confidence:
            return
        status = raw.get("status") or slot.status
        if status == "locked":
            status = "open" if raw.get("neighbor_color") else "unknown"
        if status in ("open", "wall", "unknown"):
            slot.status = status
        slot.confidence = conf
        if raw.get("neighbor_color"):
            slot.neighbor_color = raw["neighbor_color"]
        if raw.get("leads_to"):
            slot.leads_to = raw["leads_to"]

    @staticmethod
    def _replace_worst(existing: list, new_item: dict, pred) -> None:
        idxs = [i for i, it in enumerate(existing) if pred(it)]
        if not idxs:
            return
        worst = min(idxs, key=lambda i: existing[i].get("confidence", 0.0))
        if new_item.get("confidence", 0.0) > existing[worst].get("confidence", 0.0):
            existing[worst] = new_item

    def to_dict(self) -> dict:
        rooms = {}
        for name, r in self.rooms.items():
            rooms[name] = {
                "wall_color": r.wall_color.value,
                "images": r.images,
                "image_count": len(r.images),
                "images_per_wall": {str(d): r.image_count_on_wall(d) for d in CARDINAL},
                "objects": r.objects,
                "object_count": len(r.objects),
                "doors": {
                    str(d): {
                        "status": slot.status,
                        "leads_to": slot.leads_to,
                        "neighbor_color": slot.neighbor_color,
                    }
                    for d, slot in r.doors.items()
                },
                "door_count": r.door_count(),
                "enemies": r.enemies,
                "has_locked_door": r.has_locked_door,
                "damage_taken": r.damage_taken,
                "surveyed": r.surveyed,
                "visits": r.visits,
                "n_frames": r.n_frames,
            }
        ending = "unknown"
        if self.hp_end == 0:
            ending = "died"
        elif self.unlocked_door:
            ending = "unlocked_then_continued"
        return {
            "game_look": self.game_look,
            "rooms": rooms,
            "visited_order": self.visited_order,
            "hint_text": self.hint_text.value,
            "key_hint_room": self.key_hint_room,
            "key_found": self.key_found,
            "key_found_in": self.key_found_in,
            "key_consumed": self.key_consumed,
            "locked_door_room": self.locked_door_room,
            "unlocked_door": self.unlocked_door,
            "killed_enemies": self.killed_enemies,
            "damage_taken_total": self.damage_taken_total,
            "hp_start": self.hp_start,
            "hp_end": self.hp_end,
            "hp_max": self.hp_max,
            "ending": ending,
            "frozen": self.frozen,
        }
