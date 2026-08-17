"""탐험 정책: 픽셀만 보고 방을 최대한 많이 도는 규칙기반 상태머신.

상태:
  INIT     — 첫 프레임에서 스폰된 방을 그래프의 루트로 등록.
  SURVEY   — 방금 들어온 방에서 4방향(0/90/180/270)을 돌아보며 벽 색과
             비교해 "문일 가능성이 있는 방향"을 추린다. 문은 오직 이
             4방향에만 존재한다(world/layout.py의 격자 구조로 보장됨,
             README의 heading<->wall 대응표와도 일치) — 그래서 임의의
             각도를 찔러볼 필요가 없다.
  SEEK     — 후보 방향 하나를 향해 전진. 막히면(벽이든 물건이든 구분 없이)
             조금씩 회전하며 재시도, 그래도 안 되면 "이 방향엔 문 없음"
             으로 기록하고 다음 후보로. 방 이름이 바뀌면 문을 찾은 것.
  RETURN   — 이미 가본 방으로 연결됐거나, 이 방의 4방향을 다 확인해서
             이전 방으로 되돌아가야 할 때(DFS backtrack).
  COMBAT   — HP가 실제로 깎이는 걸 감지하면(=근접한 적이 서서 공격 중이라
             모션으론 못 잡음) 최우선으로 진입. 제자리에서 한 바퀴 돌며
             매 방향마다 공격을 시도해 적을 찾아 처치한다.
  DONE     — 갈 수 있는 새 방을 다 찾음(도달 가능한 범위 완료).
"""

from __future__ import annotations

from agent.memory import SceneGraph, CARDINAL_HEADINGS
from agent.ocr import read_hud
from agent.vision import is_blocked, sample_wall_color, something_in_front

_TURN_TOLERANCE = 7        # 목표 방향과 이 각도 이내면 "정렬됨"
_RECOVER_MAX_ATTEMPTS = 8  # 이만큼 회전해도 계속 막히면 "이 방향엔 문 없음"
_COMBAT_MAX_TICKS = 48     # 회전+공격을 번갈아 최대 이만큼(=한 바퀴 왕복분)
_HP_CRITICAL_RATIO = 0.4  # 최대 HP의 이 비율 이하면 싸우지 않고 도망
_FLEE_TICKS = 20           # 한 번 도망칠 때 최소 이만큼은 반대 방향으로 이동


def _heading_diff(a: int, b: int) -> int:
    """b - a를 [-180, 180]로 정규화. TURN_RIGHT는 헤딩을 줄이고 TURN_LEFT는
    늘리므로(실측 확인됨), 양수(목표가 더 큼)면 TURN_LEFT, 음수면
    TURN_RIGHT가 목표에 가까워지는 방향이다."""
    d = (b - a) % 360
    if d > 180:
        d -= 360
    return d


class ExplorerPolicy:
    def __init__(self) -> None:
        from memory_fps_env.env import Action
        self._Action = Action

        self.scene = SceneGraph()
        self.state = "INIT"
        self.last_hp = None
        self.prev_frame = None

        self.survey_queue: list = []
        self.survey_samples: dict = {}

        self.target_heading = None
        self.seek_origin_room = None
        self.return_target_room = None
        self.recover_attempts = 0
        self._need_align = False
        self._last_action_was_forward = False

        self.pre_combat_state = None
        self.pre_combat_target = None
        self.combat_ticks = 0
        self._pending_attack = False
        self._extra_attacks = 0

        self.flee_ticks = 0
        self.flee_heading = None

    # --- 메인 진입점 ------------------------------------------------
    def step(self, obs):
        hud = read_hud(obs)

        # 매 스텝 공통: HP 하락 감지 (화면에 뭐가 보이든 최우선으로 반응).
        # HP가 위험 수준(40% 이하)이면 싸우지 않고 무조건 도망 — 도망 중에도
        # 계속 맞으면 매번 다시 반대 방향을 잡아 도망을 갱신한다.
        if hud.ok and hud.hp is not None:
            if self.last_hp is not None and hud.hp < self.last_hp:
                critical = bool(hud.hp_max) and (hud.hp / hud.hp_max) <= _HP_CRITICAL_RATIO
                if critical:
                    if self.state != "FLEE":
                        self.pre_combat_state = self.state
                        self.pre_combat_target = self.target_heading
                    self.state = "FLEE"
                    self.flee_ticks = 0
                    self.flee_heading = (hud.heading + 180) % 360
                    self._last_action_was_forward = False
                elif self.state not in ("COMBAT", "FLEE"):
                    self.pre_combat_state = self.state
                    self.pre_combat_target = self.target_heading
                    self.state = "COMBAT"
                    self.combat_ticks = 0
                    self._pending_attack = True  # 첫 틱은 바로 공격 시도
                    self._extra_attacks = 0
            self.last_hp = hud.hp

        if self.state == "FLEE":
            action = self._step_flee(hud, obs)
            self.prev_frame = obs
            return int(action)

        if self.state == "COMBAT":
            action = self._step_combat(hud, obs)
            self.prev_frame = obs
            return int(action)

        if not hud.ok or hud.room_name is None:
            # HUD 못 읽음/복도(방 이름 없음) — 그냥 전진해서 방에 도착하길 기다림.
            self.prev_frame = obs
            return int(self._Action.MOVE_FORWARD)

        action = self._dispatch(hud, obs)
        self.prev_frame = obs
        return int(action)

    def _dispatch(self, hud, obs):
        if self.state == "INIT":
            return self._enter_room(hud, entry_heading=None, parent=None)
        if self.state == "SURVEY":
            return self._step_survey(hud, obs)
        if self.state == "SEEK":
            return self._step_seek(hud, obs)
        if self.state == "RETURN":
            return self._step_return(hud, obs)
        if self.state == "DONE":
            return self._Action.TURN_RIGHT  # 더 갈 새 방 없음 — 제자리 대기
        return self._enter_room(hud, entry_heading=None, parent=None)

    # --- 방 이름 정규화 (말줄임표로 잘린 것 병합, dev_log.md 참고) -----
    def _canonicalize(self, name):
        if name is None:
            return None
        if name in self.scene.nodes:
            return name
        stripped = name.rstrip("…")
        for known in self.scene.nodes:
            k = known.rstrip("…")
            if stripped == k or stripped.startswith(k) or k.startswith(stripped):
                return known
        return name

    # --- 방 진입 -------------------------------------------------------
    def _enter_room(self, hud, entry_heading, parent):
        name = self._canonicalize(hud.room_name)
        node = self.scene.get_or_create(name)
        if node.entry_heading is None:
            node.entry_heading = entry_heading
        if parent is not None and entry_heading is not None:
            backward = (entry_heading + 180) % 360
            if node.exits.get(backward) == "unknown":
                node.exits[backward] = "open"
                node.exit_leads_to[backward] = parent
            self.scene.add_edge(name, parent)
        self.scene.stack.append((name, entry_heading))

        self.survey_queue = list(node.unknown_headings()) or list(CARDINAL_HEADINGS)
        self.survey_samples = {}
        self.state = "SURVEY"
        return self._Action.NO_OP

    # --- SURVEY: 4방향 돌아보며 벽 색 비교 -------------------------------
    def _step_survey(self, hud, obs):
        canon = self._canonicalize(hud.room_name)
        node = self.scene.nodes[canon]
        if not self.survey_queue:
            return self._finish_survey(canon, node)

        target = self.survey_queue[0]
        diff = _heading_diff(hud.heading, target)
        if abs(diff) > _TURN_TOLERANCE:
            return self._Action.TURN_LEFT if diff > 0 else self._Action.TURN_RIGHT

        self.survey_samples[target] = sample_wall_color(obs)
        self.survey_queue.pop(0)
        if self.survey_queue:
            return self._Action.NO_OP
        return self._finish_survey(canon, node)

    def _finish_survey(self, canon, node):
        # 화면 전체 최빈색으로는 문 방향도 옆 벽 색에 묻혀 "벽"으로 오판되기
        # 쉬웠다(실측으로 확인, dev_log.md). 그래서 여기선 방 벽 색을
        # 기록(나중에 QA/기억용)만 하고, 문인지 아닌지는 SEEK에서 실제로
        # 걸어가서(is_blocked) 확인한다 — 더 느리지만 확실하다.
        colors = [c for c in self.survey_samples.values() if c]
        if colors and node.wall_color is None:
            node.wall_color = max(set(colors), key=colors.count)
        return self._start_seek(canon, node)

    # --- SEEK: 후보 방향으로 전진, 막히면 회피, 문 찾으면 진입 ------------
    def _start_seek(self, canon, node):
        candidates = node.unknown_headings()
        if not candidates:
            node.done = True
            return self._start_backtrack(canon)
        self.target_heading = candidates[0]
        self.seek_origin_room = canon
        self.recover_attempts = 0
        self._need_align = True
        self._last_action_was_forward = False
        self.state = "SEEK"
        return self._Action.NO_OP

    def _step_seek(self, hud, obs):
        canon = self._canonicalize(hud.room_name) if hud.room_name else None
        if canon is not None and canon != self.seek_origin_room:
            return self._on_seek_arrival(hud, canon)

        if self._need_align:
            diff = _heading_diff(hud.heading, self.target_heading)
            if abs(diff) > _TURN_TOLERANCE:
                return self._Action.TURN_LEFT if diff > 0 else self._Action.TURN_RIGHT
            self._need_align = False

        if self._last_action_was_forward and self.prev_frame is not None:
            if is_blocked(self.prev_frame, obs):
                self.recover_attempts += 1
                if self.recover_attempts > _RECOVER_MAX_ATTEMPTS:
                    node = self.scene.nodes[self.seek_origin_room]
                    node.exits[self.target_heading] = "wall"
                    return self._start_seek(self.seek_origin_room, node)
                self._last_action_was_forward = False
                return self._Action.TURN_RIGHT  # 조금씩 회피 회전 후 재시도

        self._last_action_was_forward = True
        return self._Action.MOVE_FORWARD

    def _on_seek_arrival(self, hud, new_room_canon):
        node = self.scene.nodes[self.seek_origin_room]
        node.exits[self.target_heading] = "open"
        node.exit_leads_to[self.target_heading] = new_room_canon
        self.scene.add_edge(self.seek_origin_room, new_room_canon)

        if new_room_canon in self.scene.nodes:
            # 이미 가본 방으로 연결됨 — 원래 방으로 되돌아간다.
            self.return_target_room = self.seek_origin_room
            self.target_heading = (self.target_heading + 180) % 360
            self.recover_attempts = 0
            self._need_align = True
            self._last_action_was_forward = False
            self.state = "RETURN"
            return self._Action.NO_OP

        # 새 방 발견!
        return self._enter_room(hud, entry_heading=self.target_heading,
                                 parent=self.seek_origin_room)

    # --- RETURN: 특정 방으로 되돌아가기(백트랙 겸용) ----------------------
    def _start_backtrack(self, canon):
        self.scene.stack.pop()
        if not self.scene.stack:
            self.state = "DONE"
            return self._Action.NO_OP
        parent_name, _parent_entry = self.scene.stack[-1]
        node = self.scene.nodes[canon]
        self.return_target_room = parent_name
        self.target_heading = (node.entry_heading + 180) % 360
        self.recover_attempts = 0
        self._need_align = True
        self._last_action_was_forward = False
        self.state = "RETURN"
        return self._Action.NO_OP

    def _step_return(self, hud, obs):
        canon = self._canonicalize(hud.room_name) if hud.room_name else None
        if canon == self.return_target_room:
            node = self.scene.nodes[canon]
            if node.done:
                return self._start_backtrack(canon)
            return self._start_seek(canon, node)

        if self._need_align:
            diff = _heading_diff(hud.heading, self.target_heading)
            if abs(diff) > _TURN_TOLERANCE:
                return self._Action.TURN_LEFT if diff > 0 else self._Action.TURN_RIGHT
            self._need_align = False

        if self._last_action_was_forward and self.prev_frame is not None:
            if is_blocked(self.prev_frame, obs):
                # 원래 뚫려있던 길이라 정상적으론 안 막혀야 하지만, 안전망으로
                # 살짝 회전 후 재시도(무한 루프 방지를 위해 카운트만 하고 계속 진행).
                self.recover_attempts += 1
                if self.recover_attempts > _RECOVER_MAX_ATTEMPTS:
                    self.recover_attempts = 0
                self._last_action_was_forward = False
                return self._Action.TURN_RIGHT

        self._last_action_was_forward = True
        return self._Action.MOVE_FORWARD

    # --- FLEE: HP 위험 시 반대 방향으로 이동 -------------------------------
    def _step_flee(self, hud, obs):
        self.flee_ticks += 1
        if self.flee_ticks > _FLEE_TICKS:
            return self._resume_after_combat()

        if hud.ok and hud.heading is not None:
            diff = _heading_diff(hud.heading, self.flee_heading)
            if abs(diff) > _TURN_TOLERANCE:
                return self._Action.TURN_LEFT if diff > 0 else self._Action.TURN_RIGHT

        if self._last_action_was_forward and self.prev_frame is not None:
            if is_blocked(self.prev_frame, obs):
                self._last_action_was_forward = False
                return self._Action.TURN_RIGHT  # 막히면 살짝 틀어서 다른 쪽으로

        self._last_action_was_forward = True
        return self._Action.MOVE_FORWARD

    # --- COMBAT: 제자리에서 돌며 공격 -------------------------------------
    def _step_combat(self, hud, obs):
        self.combat_ticks += 1
        if self.combat_ticks > _COMBAT_MAX_TICKS:
            return self._resume_after_combat()

        # 지금 보고 있는 방향에 뭔가(적일 가능성) 있으면, 그냥 다음 방향으로
        # 넘어가지 않고 여기에 몇 번 더 공격을 퍼붓는다 — 한 바퀴 다 도는
        # 동안에도 적은 자기 쿨다운대로 계속 우리를 때리므로, 보이면 최대한
        # 빨리 끝내는 게 낫다.
        if self._extra_attacks > 0:
            self._extra_attacks -= 1
            return self._Action.ATTACK

        if self._pending_attack:
            self._pending_attack = False
            node = self.scene.nodes.get(self._canonicalize(hud.room_name)) if hud.room_name else None
            wall_color = node.wall_color if node else None
            if something_in_front(obs, wall_color):
                self._extra_attacks = 2
            return self._Action.ATTACK

        self._pending_attack = True
        return self._Action.TURN_RIGHT

    def _resume_after_combat(self):
        self.state = self.pre_combat_state or "SURVEY"
        self.target_heading = self.pre_combat_target
        self._need_align = True
        self._last_action_was_forward = False
        return self._Action.NO_OP
