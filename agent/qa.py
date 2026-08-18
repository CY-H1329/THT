"""answer(question) 라우팅.

방이름 퍼지매칭 -> 키워드 기반 의도 분류 -> 결정론적 문자열 답변 조립
(<10ms, 헛소리 위험 없음). 확정 카테고리를 못 찾을 때만 1회에 한해
저가 텍스트 LLM(agent.vlm.qa_fallback)에 "이 JSON 사실만 근거로" 답하게
위임한다. 예외 경로 포함 항상 문자열을 반환한다(0점 방지).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from agent.content_db import ContentDB
from agent.memory import SceneGraph

_STOPWORDS = {
    "the", "a", "an", "room", "did", "you", "was", "were", "in", "what",
    "which", "how", "many", "is", "are", "of", "on", "to", "that", "this",
    "your", "and", "there", "with", "for", "any", "does", "do", "it",
}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


def _find_room(question: str, scene: SceneGraph) -> Optional[str]:
    """질문 텍스트에서 언급된 방을 scene.nodes 키와 대조해 찾는다.
    말줄임표(...)로 잘린 방 이름도 앞부분 부분 문자열로 매치되도록 처리."""
    q_lower = question.lower()
    best, best_len = None, 0
    for name in scene.nodes:
        n = name.rstrip("…").strip()
        if n and n.lower() in q_lower and len(n) > best_len:
            best, best_len = name, len(n)
    if best:
        return best

    q_tokens = _tokens(question)
    if not q_tokens:
        return None
    best, best_score = None, 0.0
    for name in scene.nodes:
        n_tokens = _tokens(name)
        if not n_tokens:
            continue
        overlap = len(q_tokens & n_tokens) / len(n_tokens)
        if overlap > best_score:
            best, best_score = name, overlap
    return best if best_score >= 0.5 else None


def answer(question: str, scene: SceneGraph, db: ContentDB) -> str:
    try:
        return _answer(question, scene, db)
    except Exception:
        return "I don't remember that."


def _answer(question: str, scene: SceneGraph, db: ContentDB) -> str:
    q = question.lower()
    room = _find_room(question, scene)
    room_content = db.rooms.get(room) if room else None

    # --- 킬 수 ---
    if any(w in q for w in ("kill", "killed", "defeat", "defeated")):
        n = db.killed_enemies
        if n == 0:
            return "I didn't kill any enemies."
        return f"I killed {n} enem{'y' if n == 1 else 'ies'}."

    # --- 잠긴 문 뒤에 뭐가 있었는지 ---
    if "behind" in q and ("locked" in q or "door" in q):
        if scene.door_unlocked and scene.key_hint_room:
            node = scene.nodes.get(scene.key_hint_room)
            dest = node.exit_leads_to.get(scene.key_hint_heading) if node else None
            if dest:
                return f"Behind the locked door was {dest}."
        return "I don't know what was behind the locked door."

    # --- 문 해제 여부 ---
    if "unlock" in q:
        if scene.door_unlocked:
            return "Yes, I found the key and unlocked the door."
        if scene.key_found:
            return "I found the key, but I did not unlock the door."
        return "No, I did not unlock the locked door."

    # --- 힌트 내용 ---
    if "hint" in q:
        if scene.key_hint_text:
            return f'The hint said: "{scene.key_hint_text}".'
        return "I never saw a hint about the key."

    # --- 열쇠가 있던 방 ---
    if "key" in q and ("which room" in q or "what room" in q or "room" in q and "contain" in q):
        if scene.key_target_room:
            return f"The key was in {scene.key_target_room}."
        return "I don't know which room had the key."

    # --- 열쇠를 찾았는지 ---
    if "key" in q:
        if scene.key_found:
            return "Yes, I found the key."
        return "No, I never found the key."

    # --- 몇 개 방을 돌았는지 (개별 방 "visit" 질문보다 먼저 검사) ---
    if "how many room" in q:
        return f"I visited {len(scene.visited_order)} room(s)."

    # --- 방문 여부 ---
    if any(w in q for w in ("visit", "been to", "went to", "enter", "go to")):
        if room:
            return f"Yes, I visited {room}."
        return "I'm not sure which room you mean, so I can't say."

    # --- 벽 색 ---
    if "color" in q and "wall" in q:
        node = scene.nodes.get(room) if room else None
        color = node.wall_color if node else None
        if not color and room_content:
            color = room_content.wall_color.value
        if color and room:
            return f"The walls in {room} were {color}."
        return "I don't remember that wall color."

    # --- 벽 이미지 ---
    if any(w in q for w in ("image", "photo", "picture")):
        if room_content and room_content.images:
            descs = "; ".join(img.get("desc", "") for img in room_content.images if img.get("desc"))
            return f"In {room}, I saw: {descs}."
        if room:
            return f"I don't remember any images in {room}."
        return "I don't remember that."

    # --- 3D 오브젝트 ---
    if "object" in q or "3d" in q:
        if room_content and room_content.objects:
            names = ", ".join(o.get("name", "") for o in room_content.objects if o.get("name"))
            return f"In {room}, I saw these objects: {names}."
        if room:
            return f"I don't remember any objects in {room}."
        return "I don't remember that."

    # --- 적(외형/색) ---
    if any(w in q for w in ("enemy", "enemies", "monster")):
        if room_content and room_content.enemies:
            descs = "; ".join(
                f"{e.get('desc', '')} ({e.get('color', '')})".strip()
                for e in room_content.enemies
            )
            return f"In {room}, I encountered: {descs}."
        return "I don't remember encountering an enemy there." if room else "I don't remember that."

    # --- 색(오브젝트/일반) 폴백 ---
    if "color" in q and room_content:
        colored = [o for o in room_content.objects if o.get("color")]
        if colored:
            o = colored[0]
            return f"The {o.get('name', 'object')} in {room} was {o.get('color')}."

    # --- 최종 폴백: 저가 텍스트 LLM에 1회 위임 ---
    return _llm_fallback(question, scene, db)


# answer()의 전체 예산은 10초(README) — urllib의 timeout=은 소켓 유휴
# 시간 기준이라 응답이 계속 조금씩 들어오면 전체 호출이 그보다 훨씬
# 오래 걸릴 수 있다(실측: QA_TIMEOUT_S=4.0인데 실제 9.46초 걸린 사례
# 확인됨). 그래서 여기서 스레드로 감싸 진짜 벽시계 상한을 강제한다 —
# 이 상한을 넘기면 스레드는 백그라운드에서 계속 돌아도 무시하고 즉시
# 안전한 폴백 문자열을 반환한다(호출자 프로세스가 죽거나 블록되지 않음).
_LLM_FALLBACK_WALL_CLOCK_BUDGET_S = 6.0


def _llm_fallback(question: str, scene: SceneGraph, db: ContentDB) -> str:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    from agent.vlm import qa_fallback

    facts = {
        "rooms_visited": scene.visited_order,
        "content_by_room": db.to_dict()["rooms"],
        "enemies_killed": db.killed_enemies,
        "key_found": scene.key_found,
        "door_unlocked": scene.door_unlocked,
        "key_hint_text": scene.key_hint_text,
        "key_hint_room_guess": scene.key_target_room,
    }
    facts_json = json.dumps(facts, ensure_ascii=False, default=str)

    # 일부러 `with`(=shutdown(wait=True))를 안 쓴다 — 컨텍스트 매니저
    # 종료 시 그 백그라운드 스레드가 끝날 때까지 또 블록해버려서, 아래
    # future.result() 타임아웃으로 일찍 빠져나온 의미가 없어진다.
    # shutdown(wait=False)는 우리 쪽만 즉시 반환하고, 스레드는 저 혼자
    # 끝나게 내버려 둔다(결과는 버려짐, 프로세스를 막지 않음).
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(qa_fallback, question, facts_json)
    try:
        result = future.result(timeout=_LLM_FALLBACK_WALL_CLOCK_BUDGET_S)
    except FutureTimeoutError:
        pool.shutdown(wait=False)
        return "I don't remember that."
    pool.shutdown(wait=False)

    if result.ok and result.data and result.data.get("answer"):
        return str(result.data["answer"])
    return "I don't remember that."
