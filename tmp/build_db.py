"""tmp/vlm_results.jsonl(OCR+VLM 원본 결과)을 방 단위 씬그래프 DB로 집계.

- 방 이름: 말줄임표 트렁케이션 병합(dev_log.md).
- wall_color: 각 프레임을 다시 열어 agent.vision.sample_wall_color로 직접
  샘플링(공짜, VLM 결과에 없어서 여기서 별도로 계산) 후 방별 최빈값.
- 문/연결: 파일명의 tick 순서대로 방 전환을 추적해서 "이 방향에서 저 방으로
  넘어갔다"를 기록. 수동 플레이 데이터라 4방향을 다 확인했다는 보장은 없어서
  "관찰된 것만" 기록하고 나머지는 unknown으로 남긴다(정직한 하한선).
- images/objects: category_guess 기준으로 중복 묶어서 리스트로.
- enemies_seen: enemy_visible=true인 프레임들의 설명을 모음.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from agent.vision import sample_wall_color

RESULTS = ROOT / "tmp" / "vlm_results.jsonl"
DATASET = ROOT / "tmp" / "vlm_dataset"
OUT_JSON = ROOT / "tmp" / "db.json"

records = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]


def _tick(fname: str) -> int:
    m = re.search(r"(\d+)\.png$", fname)
    return int(m.group(1)) if m else 0


records.sort(key=lambda r: _tick(r["file"]))


_STOPWORDS = {"a", "an", "the", "on", "of", "in", "and", "with", "is", "its", "to"}


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _caption_similar(a: str, b: str, threshold: float = 0.35) -> bool:
    """캡션 문장의 단어 겹침(Jaccard)으로 "같은 걸 다르게 부른 것"인지 판단.
    category_guess 완전일치만 보면 duck/rubber duck, chimp/ape 같은 동의어를
    다른 걸로 세는 문제가 있어서(실측 확인), 캡션 유사도로 보강한다."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb)
    return union > 0 and inter / union >= threshold


def _merge_dedup(existing: list, new_item: dict) -> bool:
    """new_item이 existing 리스트의 기존 항목과 캡션이 비슷하면 스킵(True 반환),
    아니면 리스트에 추가(False 반환 = 새로 추가됨)."""
    for item in existing:
        if _caption_similar(item.get("caption", ""), new_item.get("caption", "")):
            return True
    existing.append(new_item)
    return False


def canon(name, known):
    if name is None:
        return None
    if name in known:
        return name
    stripped = name.rstrip("…")
    for k in known:
        kk = k.rstrip("…")
        if stripped == kk or stripped.startswith(kk) or kk.startswith(stripped):
            return k
    return name


known_rooms = []
for r in records:
    if r["room"]:
        c = canon(r["room"], known_rooms)
        if c not in known_rooms:
            known_rooms.append(c)

rooms = {
    name: {
        "wall_color": None,
        "_wall_color_votes": {},
        "doors": {"exits": {0: "unknown", 90: "unknown", 180: "unknown", 270: "unknown"}},
        "images": [],   # 캡션 유사도로 중복 제거된 리스트
        "objects": [],  # 캡션 유사도로 중복 제거된 리스트
        "enemies_seen": [],
    }
    for name in known_rooms
}

key_info = {"found": False, "room_found_in": None}
door_info = {"touched": False, "unlocked": False}

prev_room = None
prev_has_key = False
for r in records:
    room = canon(r["room"], known_rooms)
    if room is None:
        continue
    node = rooms[room]

    # 벽 색 투표(공짜 CV, VLM 결과에 없어서 여기서 재계산).
    frame = np.array(Image.open(DATASET / r["file"]).convert("RGB"))
    c = sample_wall_color(frame)
    if c:
        node["_wall_color_votes"][c] = node["_wall_color_votes"].get(c, 0) + 1

    # 방 전환 -> 문/연결 기록(관찰된 것만, 하한선).
    if prev_room is not None and prev_room != room and r["heading"] is not None:
        bucket = (r["heading"] // 90) * 90
        # 관측 시점 heading은 "새 방에서" 찍힌 것이라, 대략적인 근사치임을 감안.
        rooms[prev_room]["doors"]["exits"].setdefault(bucket, "open")
        rooms[prev_room]["doors"]["exits"][bucket] = f"open->{room}"

    # 열쇠 상태 전이.
    if r["has_key"] and not prev_has_key:
        key_info["found"] = True
        key_info["room_found_in"] = room
    if prev_has_key and not r["has_key"]:
        door_info["unlocked"] = True
    prev_has_key = r["has_key"]

    # VLM 결과 병합.
    if r["vlm_ok"] and r["vlm_data"]:
        for img in r["vlm_data"].get("wall_images", []):
            if img.get("caption"):
                _merge_dedup(node["images"], img)
        for obj in r["vlm_data"].get("objects", []):
            key = obj["category_guess"].strip().lower()
            # 프롬프트로 걸러달라고 했지만 VLM이 가끔 안 따름 — 애매한
            # 표현/잠긴 문 판넬은 여기서 한 번 더 걸러낸다(안전망).
            blocked = {"block", "shape", "geometric", "cube", "panel",
                       "lock", "door", "pillar", "rectangle"}
            if any(b in key for b in blocked):
                continue
            if obj.get("caption"):
                _merge_dedup(node["objects"], obj)
        if r["vlm_data"].get("enemy_visible"):
            appearance = {
                "shirt_color": r["vlm_data"].get("enemy_shirt_color"),
                "pants_color": r["vlm_data"].get("enemy_pants_color"),
                "skin_color": r["vlm_data"].get("enemy_skin_color"),
                "body_shape": r["vlm_data"].get("enemy_body_shape"),
            }
            # 색 조합(enum이라 정확히 같은 값끼리만 매칭됨)이 이미 있으면 중복 제외.
            if appearance not in [e["appearance"] for e in node["enemies_seen"]]:
                node["enemies_seen"].append({"appearance": appearance, "outcome": "seen"})

    prev_room = room

# 벽 색 최빈값 확정 + 내부 카운터 제거, images/objects를 리스트로.
final_rooms = {}
for name, node in rooms.items():
    votes = node.pop("_wall_color_votes")
    wall_color = max(votes, key=votes.get) if votes else None
    final_rooms[name] = {
        "wall_color": wall_color,
        "doors": {
            "count": sum(1 for v in node["doors"]["exits"].values() if str(v).startswith("open")),
            "exits": {str(k): v for k, v in node["doors"]["exits"].items()},
        },
        "images": {"count": len(node["images"]), "items": node["images"]},
        "objects": {"count": len(node["objects"]), "items": node["objects"]},
        "enemies_seen": node["enemies_seen"],
    }

db = {
    "rooms": final_rooms,
    "key": key_info,
    "door": door_info,
    "enemies_killed_total": 0,  # 아직 킬 감지 로직 없음(한계로 명시된 부분)
}

OUT_JSON.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"방 {len(final_rooms)}개 집계 완료 -> {OUT_JSON}")
for name, node in final_rooms.items():
    print(f"  {name!r}: color={node['wall_color']} doors={node['doors']['count']} "
          f"images={node['images']['count']} objects={node['objects']['count']} "
          f"enemies={len(node['enemies_seen'])}")
