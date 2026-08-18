"""자율 주행/탐색 상태 기계.

한 방에 들어가면 제자리 360° 스캔으로 (a) 방 크기와 내 위치, (b) 네 벽의
문 유무, (c) 방 안 장애물 위치를 한꺼번에 얻는다. 그 다음 아직 안 가본
방 쪽 문으로 웨이포인트 주행 → 통과 → 다시 스캔. 갈 곳이 없으면 알려진
문만 따라 가장 가까운 미탐색 방으로 되돌아간다(백트래킹).

상태:
    SCAN    제자리 24회 좌회전(=360°)하며 표본 수집 → 지도 갱신 → 목표 선정
    GOTO    웨이포인트 리스트를 반응형 회피 주행으로 소화
    ENTER   문을 통과한 직후, HUD 방 이름으로 도착 확인
    PROBE   상태 미상인 벽으로 밀고 들어가 문/벽/잠금 판별
    COMBAT  HP가 깎였을 때 근처 적을 때림
    BANNER  힌트 배너가 떠 있는 동안 정지 (화면 중앙이 가려져 주행 불가)
    DONE    모든 방 스캔 완료 → 가장 오래 안 간 방을 다시 돈다

주행 자체는 "목표 방위로 돌고, 앞이 뚫려 있으면 전진, 막히면 가장 목표에
가까운 뚫린 방위로 우회"하는 반응형 제어기다. 전진이 실패하면(=충돌)
그 지점을 장애물로 기록하고 후진→우회한다.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from agent import enemies as EN
from agent import geometry as geo
from agent import localize as L
from agent import mapper as M
from agent import motion as MOT
from agent.perception import Percept, Perception
from agent.pose import (ATTACK, MOVE_BACK, MOVE_FORWARD, NO_OP, TURN_LEFT,
                        TURN_RIGHT, PoseTracker)

# --- 주행 파라미터 ------------------------------------------------------
HEADING_TOL = 8.0          # 이 각도 안이면 "목표를 보고 있다"
ARRIVE_R = 0.45            # 웨이포인트 도착 반경 (m)
SAFE_CLEARANCE = 0.85      # 이 이상 뚫려 있어야 전진 (m)
DETOUR_TURN = 60.0         # 막혔을 때 옆으로 트는 각도
DETOUR_STEPS = 16          # 우회 기동에서 직진하는 스텝 수 (≈2.4m)
SCAN_TURNS = 24            # 15° × 24 = 360°
STUCK_LIMIT = 150          # 웨이포인트 하나에 이만큼 쓰면 포기하고 재계획
BLACKLIST_STEPS = 4000     # 통과 실패한 문을 다시 시도하지 않는 기간
DOOR_STANDOFF = 2.3        # 문 앞 확인 지점 (배너 유발 반경 1.5m 밖)
ATTACK_GIVEUP = 10         # 이만큼 때려도 안 사라지면 적이 아니라 오브젝트
ALERT_STEPS = 600          # 피격 후 경계 모드 지속 스텝
SWEEP_LIMIT = 52           # 피격 중 '때리며 훑기' 최대 스텝(≈한 바퀴)
UNDER_ATTACK_STEPS = 150   # 마지막 피격 이후 이 스텝까지는 교전 우선
ATTACK_CONE_DEG = 20.0     # env의 공격 콘 반각 — 이 안이면 바로 때린다
ATTACK_RANGE = 3.0         # env의 공격 사거리(m)
MOB_MIN_SCORE = 0.6        # 몹 탐지기 신뢰도 하한
MOB_SURE_SCORE = 0.78      # 피격 중 '확신' 기준 (확인 가능한 단서를 모두 통과)
FORCE_SWEEP_STEPS = 52     # 목표를 잘못 짚었다고 판단됐을 때 강제 스윕 기간
COMBAT_BUDGET = 70         # 한 교전에서 제자리에 쓸 수 있는 총 스텝
COMBAT_COOLDOWN = 150      # 이만큼 전투가 없으면 예산을 리셋
DISENGAGE_STEPS = 200      # 예산 초과 시 '서지 않고 이동' 기간
COMBAT_MAX_TURNS = 8       # 한 교전에서 조준에 쓸 수 있는 회전 스텝
SENTRY_TURN_EVERY = 8      # 보초 자세에서 15° 회전 주기(스텝)
SENTRY_PROBE_HP = 0.6      # 이 체력 비율 이상일 때만 추가 확인에 나선다
CELL_CANDIDATES = [6.0, 7.0, 8.0, 9.0, 10.0]   # difficulty.yaml의 방 한 변 범위
MIN_MATCH_SCORE = 0.45     # 스캔 정합 인라이어 비율이 이 밑이면 위치 갱신 보류

# --- 움직임 탐지 (motion.py) -------------------------------------------
# 정지한 두 프레임의 차분은 적만 잡아낸다(렌더러가 결정적이므로 정적
# 장면의 차분은 0px). 그래서 '가끔 멈춰서 보는' 것만으로 외형과 무관한
# 적 탐지가 공짜로 돌아간다. 멈춤 비용은 스텝 두 개(≈0.024 시뮬초)뿐이다.
LOOK_EVERY = 24            # 주행 중 정지 관측 주기(스텝)
LOOK_HOLD = 2              # 정지 관측을 유지하는 스텝 수
TRACK_FRESH = 400          # 기억(사망 판정·QA)용 트랙 보존 기간
# 조준용 유효 기간은 훨씬 짧아야 한다. 적은 1.5 m/s로 걷기 때문에 400스텝
# (≈4.8 시뮬초) 전 위치는 이미 7m쯤 어긋나 있다. 실측 시드 7에서 458스텝
# 묵은 트랙을 계속 겨냥하며 허공에 168대를 휘둘렀다. 60스텝이면 적이
# 움직이는 거리는 3m 이내라, 사거리(3m) 안에서는 조준이 대체로 유효하다.
# 60까지 조였더니 교전 자체가 안 걸려 10시드 사망이 1→4로 늘었다.
TRACK_AIM_FRESH = 150
# 목격 1회짜리 트랙은 조준하지 않는다. 지평선 근처는 1픽셀 행이 1m를
# 넘어서, 먼 적이 엉뚱하게 가까운 좌표로 투영되는 일이 잦다. 진짜로
# 다가오는 적은 목격이 금방 쌓인다(실측 12회).
TRACK_MIN_SIGHTINGS = 2
TRACK_MAX_TURNS = 26       # 트랙 조준에 허용하는 회전 스텝(≈한 바퀴 반)
# 역투영 위치 오차는 거리 제곱에 비례해 커진다(지평선 근처 1행 ≈ 1m).
# 이보다 먼 움직임은 '저기서 뭔가 움직인다'로만 쓰고 트랙은 만들지 않는다.
MOTION_TRACK_MAX = 7.0
# 사거리·콘 안에서 이만큼 때렸는데도 안 죽고 안 움직이면 트랙 위치가 틀린
# 것이다(적 HP 상한 4 → 4대면 죽는다). 실측에서 유령 트랙 하나에 400회
# 넘게 공격을 쏟아붓는 판이 있었다. 그런 트랙은 조준에서 빼고 킬로도
# 세지 않는다.
TRACK_GIVEUP = 24
# 누적 조준 공격이 이만큼이면 영구 배제한다. 적 HP 상한이 4라, 정말 거기
# 있었다면 진작 열 번은 죽었다.
TRACK_RETIRE = 60
PHANTOM_STEPS = 240        # 유령으로 강등된 트랙을 다시 볼 때까지의 유예
# 사망 판정에 쓰는 콘/사거리는 env 값(±20°, 3m)보다 조금 넉넉하게 잡는다.
# 블롭 중심 방위와 바닥 접점 거리가 추정값이라, env 값을 그대로 쓰면
# 실제로 죽인 적을 놓친다(실측: 실제 3킬 중 1킬만 검출).
KILL_CONE_DEG = 26.0
KILL_RANGE = 3.8


class Explorer:
    def __init__(self, trace: bool = False):
        self.trace = [] if trace else None
        self.perc = Perception()
        self.motion = MOT.MotionDetector()
        self.map = M.WorldMap()
        self.pose = PoseTracker()
        self.step_i = 0
        self.last_action = NO_OP
        self.state = "BOOT"

        self.cur_cell: Tuple[int, int] = (0, 0)
        self.cur_name: Optional[str] = None

        # SCAN 상태
        self._scan_samples: List[Tuple[float, "list"]] = []
        self._scan_seen: set = set()
        self._scan_colors: List[tuple] = []
        self._scan_started = 0
        self._first_scan_done = False
        self._enter_wait = 0

        # GOTO 상태
        self._waypoints: List[Tuple[float, float]] = []
        self._goal_kind: Optional[str] = None      # 'through' | 'probe' | 'roam'
        self._goal_dir: Optional[str] = None
        self._goal_cell: Optional[Tuple[int, int]] = None
        self._wp_steps = 0

        # 회피/복구
        self._detour_heading = 0.0
        self._detour_steps = 0
        self._last_detour_side = 0.0
        self._last_detour_step = -999
        self._recover: List[int] = []              # 미리 계획된 강제 액션 큐

        # 전투
        self._last_hp: Optional[int] = None
        self._last_hp_max: Optional[int] = None
        self._combat_until = -1
        self._combat_swing = 0
        self._threat_seen = -99
        self._last_hit_step = -999
        self._alert_until = -1
        self._combat_turns = 0
        self._last_mob = None
        self._hits_in_combat = 0
        self._force_sweep_until = -1
        self._combat_budget = 0
        self._last_combat_step = -999
        self._disengage_until = -1
        self._sentry_tick = 0
        self._revisit_target = None
        self._locked_at = None
        # 움직임 탐지
        self._scan_phase = 0            # SCAN의 회전/정지 사이클 위치
        self._aim_track = None          # 이번 스텝 조준 대상 트랙 (없으면 None)
        self._motion_steps = 0          # 유효 정지 쌍을 얻은 스텝 수(진단용)

        self._fails: Dict[Tuple[Tuple[int, int], Optional[str]], int] = {}
        self._blacklist: Dict[Tuple[Tuple[int, int], Optional[str]], int] = {}

        # 장애물 기억: 0.5m 격자 → 마지막으로 부딪힌 스텝
        self.bumps: Dict[Tuple[int, int], int] = {}

        # QA용 원시 기록 (이후 memory 모듈에서 소비)
        self.memory: Dict = {
            "rooms": {},          # room key → dict
            "hints": [],
            "events": [],
            "locked_door": None,
            "enemies": [],        # 확정 사망한 적 (map.tracks가 원본)
        }
        self.stats = {"blocked": 0, "attacks": 0, "scans": 0, "transitions": 0,
                      "looks": 0, "motion_hits": 0, "kills": 0}
        self.state_steps: Dict[str, int] = {}   # 상태별 소요 스텝 (진단용)
        self._step_label = "BOOT"

    # ------------------------------------------------------------------
    def act(self, obs) -> int:
        self.step_i += 1
        # HUD OCR은 프레임당 ~20ms로 가장 비싼 연산이라 주기를 상태별로 둔다.
        # heading은 액션으로 정확히 적분되므로 자주 읽을 이유가 없고, 방
        # 이름(전이 확인)과 HP(피격 감지)만 제때 보면 된다.
        interval = 4 if self.state in ("BOOT", "ENTER", "PROBE") else 12
        want_hud = self.step_i <= 2 or self.step_i % interval == 0
        p = self.perc.observe(obs, force_hud=want_hud,
                              want_wall_color=(self.state == "SCAN"
                                               and self.step_i % 4 == 0))
        self.pose.update(self.last_action, p.moved,
                         p.heading if p.hud_fresh else None)

        # 움직임 탐지는 포즈 갱신 **뒤에** 돌려야 한다. 정지 쌍 판정이
        # 갱신된 포즈 키의 동일성으로 이뤄지기 때문이다.
        self._observe_motion(obs, p)

        self._track_room(p)
        self._track_hp(p)
        if self.last_action == MOVE_FORWARD and not p.moved:
            self._on_bump(p)

        self._step_label = self.state
        action = self._decide(p)
        self.state_steps[self._step_label] = (
            self.state_steps.get(self._step_label, 0) + 1)
        self.last_action = action
        if self.trace is not None:
            self.trace.append((self.step_i, self.state, self.cur_name,
                               round(self.pose.x, 2), round(self.pose.z, 2),
                               round(self.pose.heading), self._goal_kind,
                               self._goal_dir, int(action), p.hp))
        return int(action)

    # --- 공통 갱신 -----------------------------------------------------
    def _track_room(self, p: Percept) -> None:
        """HUD 방 이름으로 현재 방을 확정한다. 이름은 에피소드 내 유일하므로
        위치 추정이 흔들려도 방 정체성은 이름이 최종 권한을 갖는다."""
        if not p.hud_fresh or p.room_name is None or p.is_corridor:
            return
        name = M.normalize_name(p.room_name)
        if self.cur_name and M.names_match(self.cur_name, name):
            room = self.map.get(self.cur_cell)
            if room and len(name) > len(room.key):
                room.key = name
                self.cur_name = name
            return

        known = self.map.find_by_name(name)
        if known is not None:
            cell = known.cell
        elif (self.state in ("ENTER", "PROBE") and self._goal_cell is not None
              and self._goal_cell not in self.map.rooms):
            # 어느 문으로 나갔는지 알고 있으면 그 셀이 가장 확실한 근거다.
            cell = self._goal_cell
        else:
            cell = self.map.cell_of(*self.pose.xz)
        if known is None and cell in self.map.rooms:
            # 위치 추정이 가리키는 셀에 다른 이름의 방이 이미 있다 →
            # 추정이 밀린 것이므로 아직 안 쓰인 이웃 셀로 보정한다.
            cell = self._nearest_free_cell(cell)
        room = self.map.ensure_room(cell, name, self.step_i)
        room.visits += 1
        if self.cur_name is not None:
            self.stats["transitions"] += 1
        self.cur_cell, self.cur_name = cell, room.key
        self.memory["rooms"].setdefault(room.key, {
            "cell": list(cell), "first_step": self.step_i,
        })

    def _nearest_free_cell(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        gx, gz = cell
        for d in M.DIRS:
            dgx, dgz = M.DIR_DELTA[d]
            cand = (gx + dgx, gz + dgz)
            if cand not in self.map.rooms:
                return cand
        return cell

    def _track_hp(self, p: Percept) -> None:
        if p.hp is None:
            return
        if self._last_hp is not None and p.hp < self._last_hp:
            # 맞았다 = 적이 1.5m 안에 있다. 다만 제자리에 서서 눈먼 스핀
            # 어택을 하면 그 사이에 또 맞는다(실측: 한 바퀴 도는 1.3초에
            # 1~2대). 우리는 적보다 8배 빠르므로 계속 움직이는 편이 낫다.
            # 대신 한동안 '경계 모드'로 두어, 보이는 적은 더 먼 거리에서도
            # 선제 공격하고 스캔 회전 중에는 공격을 끼워 넣는다.
            self._alert_until = self.step_i + ALERT_STEPS
            self._last_hit_step = self.step_i
            # 때린 적은 1.5m 안에 확실히 있다. 예전엔 여기서 '탐지만 하며
            # 도는' 둘러보기를 돌렸는데, 탐지가 그 적을 걸러내면(정적
            # 오브젝트로 오인, 벽색과 비슷한 셔츠 등) 공격은 한 번도 안 하고
            # 제자리에서 돌기만 하다 죽었다. 이제는 곧바로 교전 모드로
            # 들어가고, 못 찾으면 '때리면서' 훑는다(_combat 참고).
            self._hits_in_combat += 1
            if self.step_i > self._combat_until:      # 새 교전의 시작
                self._hits_in_combat = 1
                self._combat_swing = 0
                self._combat_turns = 0
            self._combat_until = self.step_i + SWEEP_LIMIT
            self.memory["events"].append(
                {"type": "hp_drop", "step": self.step_i, "hp": p.hp}
            )
        self._last_hp = p.hp
        self._last_hp_max = p.hp_max

    # --- 움직임 탐지 ---------------------------------------------------
    def _pose_key(self) -> tuple:
        return (round(self.pose.x, 3), round(self.pose.z, 3),
                round(self.pose.heading, 1))

    def _observe_motion(self, obs, p: Percept) -> None:
        """정지 쌍이면 차분을 돌려 적 트랙을 갱신하고, 사망을 판정한다.

        '정지'는 직전 액션이 에이전트를 움직이지 않았다는 뜻이다. ATTACK은
        레이캐스트만 하므로 정지에 포함된다 — 때리는 동안에도 탐지가
        계속 돌아간다는 게 이 설계의 핵심이다(motion.py 참고).
        """
        stationary = (
            self.last_action in (NO_OP, ATTACK)
            or (self.last_action in (MOVE_FORWARD, MOVE_BACK) and not p.moved)
        )
        blobs = self.motion.observe(obs, self._pose_key(), stationary, p.banner)
        if not self.motion.paired:
            return
        self._motion_steps += 1
        # 사망으로 판정한 블롭은 트랙 입력에서 빼야 한다. 안 그러면 방금
        # 죽은 것으로 표시한 트랙이 같은 블롭에 의해 곧바로 되살아난다.
        killed_blob = self._detect_kill_event(blobs, p)
        self._ingest_motion([b for b in blobs if b is not killed_blob], p)

    def _ingest_motion(self, blobs, p: Percept) -> set:
        """차분 블롭을 월드 좌표로 역투영해 트랙에 반영. 갱신된 트랙 집합."""
        touched = set()
        for b in blobs:
            # 2.6m 안쪽은 바닥 접점이 화면 밖이라 거리가 포화값으로만 온다.
            # 그 경우 '코앞'이라는 사실만 쓰고 대표값으로 찍는다.
            d = b.distance if b.resolved else 1.6
            if d > MOTION_TRACK_MAX:
                continue        # 위치를 믿을 수 없는 거리 — 트랙을 만들지 않는다
            hx, hz = geo.heading_to_vec(self.pose.heading + b.bearing)
            x, z = self.pose.x + hx * d, self.pose.z + hz * d
            t, revived = self.map.note_motion(x, z, self.cur_name or "?",
                                              self.step_i)
            if not touched:
                self.stats["motion_hits"] += 1
            touched.add(id(t))
            for name in EN.palette_sample(p.frame, *b.col_range):
                if name not in t.colors:
                    t.colors.append(name)
        return touched

    def _detect_kill_event(self, blobs, p: Percept):
        """방금 때린 프레임에서 실루엣이 통째로 사라졌으면 킬로 센다.

        트랙의 생사를 추론해서 세는 방식은 실패했다. 적이 움직이는 데다
        역투영 위치에 오차가 있어서 같은 적이 여러 트랙으로 갈리고, 갈린
        조각마다 사망 판정이 붙어 실제 3킬에 19킬을 주장했다. 트랙 연결을
        아무리 조여도 '몇 마리였나'를 트랙 수로 세는 한 이 오차는 남는다.

        대신 **사건**을 센다. 적이 죽는 순간 메시가 통째로 제거되므로
        (env._resolve_agent_attack), 그 프레임의 차분에는 몹 실루엣이
        있던 자리가 통째로 잡히고 **새 프레임에는 팔레트 색이 남지 않는다**.
        걸어서 이동한 경우와는 여기서 갈린다 — 이동이면 옮겨간 자리에
        여전히 팔레트 색 픽셀이 있다.

        세는 규칙이 정확한 이유는 env의 공격 판정 때문이다. ATTACK 한 번은
        콘 안에서 **가장 가까운 적 하나**만 때리므로, 한 스텝에 죽을 수
        있는 적은 최대 한 마리다. 그래서 스텝당 최대 1킬로 못 박는다.
        """
        if self.last_action != ATTACK:
            return None                  # 차분 쌍이 공격을 사이에 두지 않았다
        best = None
        for b in blobs:
            if not b.vanished:
                continue
            if abs(b.bearing) > KILL_CONE_DEG:
                continue                 # env의 공격 콘 밖 — 우리가 죽인 게 아니다
            d = b.distance if b.resolved else 1.6
            if d > KILL_RANGE:
                continue
            if best is None or b.pixels > best.pixels:
                best = b
        if best is None:
            return None
        self.stats["kills"] += 1
        d = best.distance if best.resolved else 1.6
        hx, hz = geo.heading_to_vec(self.pose.heading + best.bearing)
        rec = {"room": self.cur_name, "step": self.step_i,
               "colors": list(best.colors_before or []),
               "pos": [round(self.pose.x + hx * d, 1),
                       round(self.pose.z + hz * d, 1)]}
        self.memory["enemies"].append(rec)
        self.memory["events"].append(dict(rec, type="enemy_killed"))
        # 그 자리의 트랙은 죽은 것으로 표시해 조준 대상에서 뺀다.
        t = self.map.track_near(rec["pos"][0], rec["pos"][1], 2.0,
                                self.step_i, TRACK_FRESH)
        if t is not None:
            t.alive = False
            t.killed_step = self.step_i
        return best

    def enemy_report(self) -> dict:
        """QA용 적 요약. 메모리는 QA 시작 시점에 동결되므로 조회만 한다.

        색은 팔레트 이름(red/green/blue/yellow/purple/grey)으로 저장돼 있다.
        이 6색은 env 코드에 박힌 miniworld 렌더 팔레트라 평가 때 바뀌는
        에셋이 아니므로, held-out 시드에서도 그대로 쓸 수 있다.
        """
        killed = self.memory["enemies"]
        seen = list(self.map.tracks)
        by_room: Dict[str, int] = {}
        for rec in killed:
            key = rec["room"] or "?"
            by_room[key] = by_room.get(key, 0) + 1
        return {
            "killed": len(killed),
            "seen": len(seen),
            "killed_by_room": by_room,
            "killed_detail": list(killed),
        }

    def _on_bump(self, p: Percept) -> None:
        """전진 실패 지점을 기록하고, 그게 벽이면 지도에 반영한다."""
        self.stats["blocked"] += 1
        bx, bz = self.pose.ahead(geo.AGENT_RADIUS + 0.1)
        self.bumps[(int(bx * 2), int(bz * 2))] = self.step_i

        wall_dir = self._wall_at(bx, bz)
        if wall_dir is not None:
            room = self.map.get(self.cur_cell)
            door_x, door_z = self.map.door_point(self.cur_cell, wall_dir)
            at_doorway = math.hypot(bx - door_x, bz - door_z) < 1.0
            locked = at_doorway or geo.lock_visible(p.frame)
            if locked:
                # 문인 줄 알았는데 막혔고 자물쇠가 보인다 = 잠긴 문.
                self.map.link(self.cur_cell, wall_dir, M.LOCKED)
                self.memory["locked_door"] = {
                    "room": self.cur_name, "dir": wall_dir, "step": self.step_i,
                }
            elif room is not None:
                if room.walls[wall_dir] == M.UNKNOWN:
                    self.map.link(self.cur_cell, wall_dir, M.WALL)
                room.verified.add(wall_dir)   # 몸으로 확인한 벽
        elif self._under_attack():
            # 맞고 있는 중에 벽 아닌 것에 막혔다 = 몸으로 막고 있는 적이
            # 코앞(0.5m)에 있다는 뜻. 옆으로 비켜가지 말고 바로 때린다.
            self._recover = [ATTACK] * 6
            self._combat_until = self.step_i + 12
        elif self._combat_until < self.step_i:
            # 벽이 아닌 것에 막혔다 → 오브젝트이거나 적. 일단 몇 대 쳐 본다.
            self._combat_until = self.step_i + 12
        self._detour_steps = 0        # 새 상황이므로 진행 중이던 우회는 무효

    def _wall_at(self, x: float, z: float, tol: float = 0.55) -> Optional[str]:
        """(x, z)가 현재 방의 어느 벽면 위인지 (아니면 None)."""
        for d in M.DIRS:
            if abs(self.map.wall_distance(x, z, self.cur_cell, d)) <= tol:
                return d
        return None

    # --- 상태 분기 -----------------------------------------------------
    def _decide(self, p: Percept) -> int:
        if p.banner:
            self._step_label = "BANNER"
            if p.hint_text:
                self._record_hint(p.hint_text)
            # 배너는 "잠긴 문에 닿았을 때"만 뜬다 → 지금 붙어 있는 문이
            # 그 잠긴 문이라는 확실한 증거다(색 판정보다 훨씬 신뢰도가 높다).
            self._mark_nearby_door_locked()
            # 배너는 시야 중앙 55%를 3초간 가려서 주행이 불가능하다. 다만
            # 그동안 적에게 맞고만 있을 수는 없으니, 피격 중이면 시야 없이도
            # 되는 스핀 어택은 계속한다.
            if self.step_i <= self._combat_until:
                self._combat_swing += 1
                return self._sweep_action()
            return NO_OP

        if self._recover:
            return self._recover.pop(0)

        # 생존: 적은 1.5m까지 붙어야 때리지만 우리 공격은 3m/±20°에 쿨다운도
        # 없다. 다만 가만히 서서 오브젝트를 때리다 죽는 게 최악이라, 선제
        # 공격은 "이미 스캔해서 정적 장애물 목록이 있는 방"에서만 한다.
        # 제자리 전투 총량 제한. 문간에 선 채로 130스텝을 훑다가 HP 13→1이
        # 된 판이 있었다(24스텝마다 한 대씩). 우리는 적보다 8배 빠르니
        # 오래 붙잡혀 있는 것 자체가 지는 길이다 — 예산을 넘기면 교전을
        # 접고 임무로 돌아가 이동으로 거리를 벌린다.
        if self.step_i - self._last_combat_step > COMBAT_COOLDOWN:
            self._combat_budget = 0
        if self._combat_budget > COMBAT_BUDGET:
            self._disengage_until = self.step_i + DISENGAGE_STEPS
            self._combat_budget = 0
            self._end_combat()

        threat = self._threat(p)
        if self.step_i <= self._disengage_until:
            # 이탈 중에도 '돌 필요 없이 바로 닿는' 적은 때린다(1스텝, 이동
            # 손실 없음). 조준하려고 서는 것만 금지한다.
            if threat is not None and abs(threat[0]) <= ATTACK_CONE_DEG:
                self._step_label = "COMBAT"
                self._last_combat_step = self.step_i
                return self._aimed_attack()
            threat = None
        if (threat is not None and self._aim_track is None
                and self._threat_seen != self.step_i - 1):
            # 한 프레임짜리 오검출로 교전에 들어가지 않도록 한 스텝 지연.
            # 움직임 트랙에는 걸지 않는다 — 정적 장면의 차분이 0px이라
            # 한 프레임짜리 오검출이라는 게 존재하지 않는다.
            self._threat_seen = self.step_i
            threat = None
        elif threat is not None:
            self._threat_seen = self.step_i
        if self.step_i <= self._combat_until or threat is not None:
            self._step_label = "COMBAT"      # 상태별 집계는 전투로 잡는다
            self._combat_budget += 1
            self._last_combat_step = self.step_i
            return self._combat(p)

        if self.state == "BOOT":
            self._begin_scan()
            return self._scan(p)
        if self.state == "SCAN":
            return self._scan(p)
        if self.state == "GOTO":
            return self._goto(p)
        if self.state == "ENTER":
            return self._enter(p)
        if self.state == "DOORCHECK":
            return self._doorcheck(p)
        if self.state == "PROBE":
            return self._probe(p)
        if self.state == "DONE":
            return self._sentry(p)
        return NO_OP

    def _mark_nearby_door_locked(self) -> None:
        room = self.map.get(self.cur_cell)
        if room is None:
            return
        best, best_d = None, 2.8
        for d in M.DIRS:
            if room.walls[d] not in (M.DOOR, M.UNKNOWN):
                continue
            dist = self.pose.distance_to(*self.map.door_point(self.cur_cell, d))
            if dist < best_d:
                best, best_d = d, dist
        if best is None:
            return
        # 잠긴 문은 에피소드당 정확히 하나다. 새로 확정되면 이전에 잠김으로
        # 표시했던 곳은 판정 보류로 되돌려 다시 시도할 수 있게 한다.
        prev = self._locked_at
        if prev is not None and prev != (self.cur_cell, best):
            prev_room = self.map.get(prev[0])
            if prev_room is not None and prev_room.walls[prev[1]] == M.LOCKED:
                prev_room.walls[prev[1]] = M.UNKNOWN
        self.map.link(self.cur_cell, best, M.LOCKED)
        self._locked_at = (self.cur_cell, best)
        self.memory["locked_door"] = {
            "room": self.cur_name, "dir": best, "step": self.step_i,
        }
        if self._goal_dir == best:
            self._waypoints.clear()

    def _record_hint(self, text: str) -> None:
        if text and (not self.memory["hints"] or self.memory["hints"][-1]["text"] != text):
            self.memory["hints"].append({"text": text, "step": self.step_i})

    # --- SCAN ----------------------------------------------------------
    def _begin_scan(self) -> None:
        self.state = "SCAN"
        self._scan_samples = []
        self._scan_seen = set()
        self._scan_colors = []
        self._scan_started = self.step_i
        self._scan_phase = 0

    def _scan(self, p: Percept) -> int:
        """15° 버킷 24개가 다 찰 때까지, 회전 1 + 공격 2를 반복한다.

        회전 횟수가 아니라 '어느 방위를 봤는지'로 종료를 판정하므로,
        중간에 전투로 끊겼다 돌아와도 스캔이 정상적으로 완성된다.

        회전 사이에 ATTACK을 두 번 끼우는 이유는 두 가지다.

        1. ATTACK은 에이전트를 움직이지 않으므로 그 두 프레임이 **정지 쌍**이
           되고, 차분이 곧 적 탐지다(motion.py). 방을 처음 훑는 이 스캔이
           방 안 적을 전부 트랙으로 잡는 자리가 된다.
        2. 공격은 쿨다운이 없고 사거리 3m·콘 ±20°라, 도는 김에 사거리 안의
           적에게 그대로 데미지가 들어간다. 비용은 없다 — 시뮬 시간은
           step() 사이 실제 경과 시간으로만 흐르므로 스캔이 24스텝에서
           72스텝이 되어도 0.3초가 0.9초로 늘 뿐이다.

        깊이 표본은 새 버킷에서만 모은다. 같은 방위 표본을 세 배로 쌓으면
        정합(localize.match_all) 비용만 늘고 정보는 늘지 않는다.
        """
        bucket = int(round(self.pose.heading / 15.0)) % 24
        if p.profile is not None and bucket not in self._scan_seen:
            self._scan_samples.append((self.pose.heading, p.profile))
            self._scan_seen.add(bucket)
        if p.wall_color:
            self._scan_colors.append(p.wall_color)
        if len(self._scan_seen) < SCAN_TURNS and self.step_i - self._scan_started < 900:
            self._scan_phase = (self._scan_phase + 1) % 3
            if self._scan_phase == 0:
                return TURN_LEFT
            self.stats["looks"] += 1
            return ATTACK
        self._finish_scan()
        return self._choose_goal()

    def _finish_scan(self) -> None:
        """스캔 표본 → 내 위치(정합), 셀 크기, 벽별 문 유무, 장애물 위치."""
        self.stats["scans"] += 1
        room = self.map.ensure_room(self.cur_cell, self.cur_name, self.step_i)
        rays = self._rays()

        if not self.map.cell_locked:
            # 셀 크기가 아직 안 굳었으면 후보별로 정합해 점수를 누적한다.
            results = L.match_all(rays, CELL_CANDIDATES)
            cell = self.map.vote_cell_size({c: r[2] for c, r in results.items()})
            x_in, z_in, score = results[cell]
            self._first_scan_done = True
        else:
            cell = self.map.cell_size
            x_in, z_in, score = L.match_scan(rays, cell)

        ox, oz = self.map.room_origin(self.cur_cell)
        if score >= MIN_MATCH_SCORE:
            self.pose.x, self.pose.z = ox + x_in, oz + z_in
        else:
            # 정합 실패(적/오브젝트가 시야를 다 막은 경우 등) → 추측항법 유지.
            x_in, z_in = self.pose.x - ox, self.pose.z - oz
        self._last_match_score = score

        openings = L.wall_openings(rays, x_in, z_in, cell)
        for d, state in openings.items():
            if room.walls[d] in (M.UNKNOWN, M.WALL) or state == M.DOOR:
                self.map.link(self.cur_cell, d, state)
        room.static_blobs = [(ox + px, oz + pz)
                             for px, pz in L.obstacle_points(rays, x_in, z_in, cell)]
        room.scanned = True
        if self._scan_colors:
            room.wall_color = max(set(self._scan_colors), key=self._scan_colors.count)

        self.memory["rooms"].setdefault(room.key, {}).update({
            "cell": list(self.cur_cell),
            "wall_color_rgb": room.wall_color,
            "walls": dict(room.walls),
            "scanned_step": self.step_i,
            "match_score": round(score, 2),
        })

    def _rays(self) -> List[Tuple[float, float]]:
        """스캔 표본 → (절대 heading, 거리) 리스트."""
        out = []
        for heading, profile in self._scan_samples:
            bearings = geo.column_bearings(len(profile))
            for b, d in zip(bearings, profile):
                out.append(((heading + b) % 360.0, float(d)))
        return out

    # --- 목표 선정 -----------------------------------------------------
    def _choose_goal(self) -> int:
        room = self.map.get(self.cur_cell)
        if room is None:
            self._begin_scan()
            return TURN_LEFT

        # 1) 이 방에서 바로 갈 수 있는, 아직 안 가본 방
        for d in self._sorted_dirs(self._usable_dirs(room)):
            dgx, dgz = M.DIR_DELTA[d]
            nxt = (self.cur_cell[0] + dgx, self.cur_cell[1] + dgz)
            nb = self.map.get(nxt)
            if nb is None or not nb.scanned:
                return self._start_traverse(d)

        # 2) 상태 미상인 벽 확인
        unknown = self._sorted_dirs(room.unknown_dirs())
        if unknown:
            return self._start_probe(unknown[0])

        # 3) 남은 미탐색 방으로 백트래킹 (알려진 문만 따라 최단 경로)
        hop = self._route_to_frontier(include_unverified=False)
        if hop:
            return self._start_traverse(hop)

        # 4) 갈 곳이 없다 → 보초 자세로. 남는 시간과 체력이 있으면 거기서
        #    "스캔으로만 벽이라 본" 방향을 되짚는다(_sentry 참고). 스캔은
        #    가구에 가려 문 구간을 못 보면 벽으로 오판할 수 있는데, 그
        #    재확인은 비싸고 위험해서 탐색 본류에서는 하지 않는다.
        return self._park_pick()

    def _route_to_frontier(self, include_unverified: bool) -> Optional[str]:
        """가장 가까운 '할 일 남은 방'으로 가는 첫 이동 방향."""
        best = None
        for cell in self.map.frontier_rooms(include_unverified=include_unverified):
            if cell == self.cur_cell:
                continue
            path = self.map.route(self.cur_cell, cell)
            if path and (best is None or len(path) < len(best)):
                best = path
        return best[0] if best else None

    def _usable_dirs(self, room) -> List[str]:
        """열려 있다고 믿는 문 중, 최근에 통과 실패하지 않은 것."""
        return [d for d in room.open_dirs()
                if self._blacklist.get((room.cell, d), -1) < self.step_i]

    def _sorted_dirs(self, dirs: List[str]) -> List[str]:
        """지금 보고 있는 방향에 가까운 문부터 — 회전 낭비를 줄인다."""
        return sorted(dirs, key=lambda d: abs(geo.wrap180(
            M.DIR_HEADING[d] - self.pose.heading)))

    def _start_traverse(self, direction: str) -> int:
        dgx, dgz = M.DIR_DELTA[direction]
        self._goal_kind, self._goal_dir = "through", direction
        self._goal_cell = (self.cur_cell[0] + dgx, self.cur_cell[1] + dgz)
        # 문 앞 1.4m까지만 웨이포인트로 잡는다. 거기서 문을 정면으로 보고
        # 잠김 여부를 확인한 뒤에야 밀고 나간다(DOORCHECK).
        self._waypoints = [self.map.approach_point(self.cur_cell, direction,
                                           back=DOOR_STANDOFF)]
        self._wp_steps = 0
        self.state = "GOTO"
        return NO_OP

    def _start_probe(self, direction: str) -> int:
        self._goal_kind, self._goal_dir = "probe", direction
        dgx, dgz = M.DIR_DELTA[direction]
        self._goal_cell = (self.cur_cell[0] + dgx, self.cur_cell[1] + dgz)
        self._waypoints = [
            self.map.approach_point(self.cur_cell, direction, back=1.2),
            self.map.exit_point(self.cur_cell, direction, ahead=1.2),
        ]
        self._wp_steps = 0
        self.state = "PROBE"
        return NO_OP

    # --- 주행 ----------------------------------------------------------
    def _goto(self, p: Percept) -> int:
        # 주행 중 정기 정지 관측. 이동은 매 스텝 포즈를 바꾸므로 정지 쌍이
        # 아예 안 생긴다 — 방을 가로지르는 동안 다가오는 적을 못 본다는
        # 뜻이다. LOOK_HOLD 스텝만 멈춰 주면 그 구간이 메워진다. 비용은
        # LOOK_EVERY당 두 스텝(≈0.024 시뮬초)뿐이다.
        if self.step_i % LOOK_EVERY < LOOK_HOLD and not p.banner:
            # NO_OP이 아니라 ATTACK으로 멈춘다. 정지 효과는 같은데(ATTACK은
            # 에이전트를 안 움직인다) 사거리 안에 적이 있으면 그대로 데미지가
            # 들어간다. env의 레이캐스트는 우리가 무엇을 겨냥한다고 '믿는지'와
            # 무관하게 콘 안의 가장 가까운 적을 때리므로, 목표가 없어도
            # 휘두르는 것 자체가 방어가 된다.
            self.stats["looks"] += 1
            self._register_blind_hit()
            return ATTACK
        if self._check_locked_door(p):
            return self._choose_goal()
        status, action = self._drive(p)
        if status == "driving":
            return action
        if status == "giveup":
            return self._traverse_failed()
        if self._goal_kind == "through":
            self.state = "DOORCHECK"
            return NO_OP
        if self._goal_kind == "push":
            self.state = "ENTER"
            self._enter_wait = 0
            return NO_OP
        self._begin_scan()
        return TURN_LEFT

    def _doorcheck(self, p: Percept) -> int:
        """문 앞에서 문을 정면으로 보고 잠김 여부를 확인한 뒤 통과를 시작한다.

        에피소드마다 하나 있는 잠긴 문(노란 자물쇠 판)은 열쇠 없이 절대
        못 지난다. 모르고 밀면 수백 스텝을 버리므로, 통과 전에 반드시
        정면에서 한 번 본다. 이 확인은 회전 몇 스텝이면 끝난다.
        """
        d = self._goal_dir
        if d is None:
            return self._choose_goal()
        err = geo.wrap180(M.DIR_HEADING[d] - self.pose.heading)
        if abs(err) > HEADING_TOL:
            return TURN_LEFT if err > 0 else TURN_RIGHT
        if not p.banner and geo.lock_visible(p.frame):
            self.map.link(self.cur_cell, d, M.LOCKED)
            self.memory["locked_door"] = {
                "room": self.cur_name, "dir": d, "step": self.step_i,
            }
            self.memory["events"].append(
                {"type": "locked_door_seen", "room": self.cur_name,
                 "dir": d, "step": self.step_i})
            return self._choose_goal()
        self._goal_kind = "push"
        self._waypoints = [self.map.exit_point(self.cur_cell, d, ahead=1.8)]
        self._wp_steps = 0
        self.state = "GOTO"
        return MOVE_FORWARD

    def _check_locked_door(self, p: Percept) -> bool:
        """향하는 문이 잠긴 문(노란 자물쇠 판)인지 가까이서 확인한다.

        에피소드마다 딱 하나 있는 잠긴 문은 열쇠 없이는 절대 못 지나가므로,
        모르고 계속 밀면 수천 스텝을 버린다(개발 중 실제로 그랬다). 문
        2.4m 앞에서 판이 보이면 그 벽을 LOCKED로 확정하고 목표를 바꾼다.
        """
        if p.banner or self._goal_kind != "through" or self._goal_dir is None:
            return False
        dx, dz = self.map.door_point(self.cur_cell, self._goal_dir)
        if self.pose.distance_to(dx, dz) > 2.4:
            return False
        if abs(self.pose.bearing_to(dx, dz)) > 30.0:
            return False
        if not geo.lock_visible(p.frame):
            return False
        self.map.link(self.cur_cell, self._goal_dir, M.LOCKED)
        self.memory["locked_door"] = {
            "room": self.cur_name, "dir": self._goal_dir, "step": self.step_i,
        }
        self.memory["events"].append(
            {"type": "locked_door_seen", "room": self.cur_name,
             "dir": self._goal_dir, "step": self.step_i})
        self._waypoints.clear()
        return True

    def _probe(self, p: Percept) -> int:
        """벽으로 밀어붙여 문/벽/잠금을 몸으로 확정한다.

        종료 조건은 "문/잠금으로 밝혀짐" 또는 "부딪혀서 벽 확인(verified)"
        이다. 예전엔 '벽이면 종료'로 두는 바람에, 스캔이 벽이라 판정한 곳을
        재확인하러 와서는 매 스텝 목표만 다시 고르는 무한루프에 빠졌다.
        """
        d = self._goal_dir
        room = self.map.get(self.cur_cell)
        if room is None or d is None:
            return self._choose_goal()
        if room.walls[d] in (M.DOOR, M.LOCKED) or d in room.verified:
            return self._choose_goal()          # 이미 결론이 났다
        status, action = self._drive(p)
        if status == "giveup":
            self.map.link(self.cur_cell, d, M.WALL)   # 접근조차 못 하면 벽으로 간주
            room = self.map.get(self.cur_cell)
            if room is not None:
                room.verified.add(d)
            return self._choose_goal()
        if status == "driving":
            return action
        # 벽 평면을 넘어 이웃 셀까지 들어갔다 = 문이 있었다.
        self.map.link(self.cur_cell, d, M.DOOR)
        room = self.map.get(self.cur_cell)
        if room is not None:
            room.verified.add(d)
        self.state = "ENTER"
        self._goal_kind = "through"
        self._enter_wait = 0
        return NO_OP

    def _enter(self, p: Percept) -> int:
        """문 통과 직후: HUD 이름이 바뀔 때까지 기다렸다가 새 방을 스캔한다."""
        self._enter_wait = getattr(self, "_enter_wait", 0) + 1
        target = self._goal_cell
        if self.cur_cell == target:
            # 새 방 진입 확정. 위치는 여기서 추정하지 않는다 — 곧바로 도는
            # 스캔 정합이 훨씬 정확하다(과거엔 여기서 문 안쪽 좌표로
            # 스냅했는데, 이름은 벽 평면을 넘자마자 바뀌므로 1~2m씩 틀렸다).
            if self.state == "DONE":
                self._goal_kind = "revisit"
            self._begin_scan()
            return TURN_LEFT
        if self._enter_wait > 60:
            return self._traverse_failed()
        # 아직 문 안쪽 — 조금 더 밀고 들어간다.
        hx, hz = M._dir_vec(self._goal_dir)
        tx = self.pose.x + hx * 1.0
        tz = self.pose.z + hz * 1.0
        _, action = self._drive_towards(p, tx, tz)
        return action

    def _traverse_failed(self) -> int:
        """문으로 갔는데 옆방에 못 들어갔을 때.

        위치 추정이 밀렸거나(→ 다시 스캔해서 정합), 애초에 문이 아니었을
        수 있다(→ 두 번 실패하면 벽으로 확정하고 다른 벽을 판다).
        """
        key = (self.cur_cell, self._goal_dir)
        self._fails[key] = self._fails.get(key, 0) + 1
        if self._goal_dir is not None:
            # 한 번 실패하면 한동안 후보에서 빼고, 세 번이면 벽으로 확정한다.
            self._blacklist[key] = self.step_i + BLACKLIST_STEPS
            if self._fails[key] >= 3:
                self.map.link(self.cur_cell, self._goal_dir, M.WALL)
        self._waypoints.clear()
        self._goal_kind = self._goal_dir = None
        self._begin_scan()
        return TURN_LEFT

    def _park_pick(self) -> int:
        """탐색이 끝난 뒤의 자세: 방 구석에 자리 잡고 보초를 선다.

        방을 계속 순회하면 QA에 새로 얻을 게 없는데 적에게 맞을 기회만
        늘어난다(실측: 지도 완성 후 순회하다 전멸). 구석에 붙으면 접근
        가능한 방향이 전방 반구로 줄어들고, 천천히 돌면서 보면 들어오는
        적을 사거리(3m) 안에서 먼저 칠 수 있다.
        """
        self.state = "DONE"
        ox, oz = self.map.room_origin(self.cur_cell)
        c = self.map.cell_size
        # 현재 위치에서 가장 가까운 구석에서 1.2m 안쪽
        cx = ox + (1.2 if self.pose.x - ox < c / 2 else c - 1.2)
        cz = oz + (1.2 if self.pose.z - oz < c / 2 else c - 1.2)
        self._waypoints = [(cx, cz)]
        self._goal_kind, self._goal_dir = "park", None
        self._wp_steps = 0
        return NO_OP

    def _hp_ratio(self) -> float:
        hp, hp_max = self._last_hp, self._last_hp_max
        if not hp or not hp_max:
            return 1.0
        return hp / float(hp_max)

    def _sentry_probe(self) -> Optional[int]:
        """미확인 벽이 남은 가장 가까운 방으로 향한다. 시작했으면 그 액션.

        시작 함수들이 상태(PROBE/GOTO)를 직접 세팅하므로 여기서 나온 액션을
        그대로 돌려줘야 한다. 예전엔 이 뒤에 _choose_goal()을 불러서
        방금 세운 목표를 곧바로 park로 덮어쓰는 루프가 생겼다.
        """
        room = self.map.get(self.cur_cell)
        if room is not None and room.unverified_walls():
            return self._start_probe(self._sorted_dirs(room.unverified_walls())[0])
        hop = self._route_to_frontier(include_unverified=True)
        if hop:
            return self._start_traverse(hop)
        return None

    def _revisit_pick(self) -> Optional[int]:
        """가장 방문 횟수가 적은 방으로 이동을 시작한다."""
        target, best = None, None
        for cell, room in self.map.rooms.items():
            if cell == self.cur_cell:
                continue
            key = (room.visits, room.first_seen_step)
            if (best is None or key < best) and self.map.route(self.cur_cell, cell):
                target, best = cell, key
        if target is None:
            return None
        path = self.map.route(self.cur_cell, target)
        if not path:
            return None
        action = self._start_traverse(path[0])
        self._revisit_target = target
        return action

    def _sentry(self, p: Percept) -> int:
        """보초 자세: 구석까지 간 뒤 천천히 회전하며 감시한다.

        전투 판정은 상태 분기보다 앞에서 돌기 때문에, 여기서는 시야만
        돌려주면 접근하는 적은 자동으로 요격된다.
        """
        if self._waypoints:
            status, action = self._drive(p)
            if status == "driving":
                return action
            self._waypoints.clear()
            if self._goal_kind == "revisit":
                self._begin_scan()           # 다시 돌아본 방은 새로 스캔
                return TURN_LEFT
        # 새로 할 일이 생겼으면 탐색으로 복귀
        if self.map.frontier_rooms():
            return self._choose_goal()
        # 체력에 여유가 있으면 스캔으로만 벽이라 본 방향을 몸으로 확인한다.
        # 여기서 문이 나오면 지도가 통째로 늘어난다.
        if self._hp_ratio() >= SENTRY_PROBE_HP:
            action = self._sentry_probe()
            if action is not None:
                return action
            # 확인할 벽도 없다 → 가장 안 가본 방을 다시 돈다. 스캔의 문
            # 판정은 가구에 가려 틀릴 수 있어서, 다른 위치에서 다시 보면
            # 놓친 문이 드러나는 경우가 실제로 많다(커버리지가 크게 갈린다).
            action = self._revisit_pick()
            if action is not None:
                return action
        self._sentry_tick += 1
        return TURN_LEFT if self._sentry_tick % SENTRY_TURN_EVERY == 0 else NO_OP

    # --- 반응형 제어기 --------------------------------------------------
    def _drive(self, p: Percept) -> Tuple[str, int]:
        """웨이포인트 큐를 소화한다. ('arrived'|'driving'|'giveup', 액션) 반환."""
        if not self._waypoints:
            return "arrived", NO_OP
        self._wp_steps += 1
        if self._wp_steps > STUCK_LIMIT:
            self._waypoints.clear()
            self._wp_steps = 0
            return "giveup", NO_OP
        tx, tz = self._waypoints[0]
        if self.pose.distance_to(tx, tz) <= ARRIVE_R:
            self._waypoints.pop(0)
            self._wp_steps = 0
            if not self._waypoints:
                return "arrived", NO_OP
            tx, tz = self._waypoints[0]
        _, action = self._drive_towards(p, tx, tz)
        return "driving", action

    def _drive_towards(self, p: Percept, tx: float, tz: float) -> Tuple[bool, int]:
        """목표점을 향해 한 스텝. (도착여부, 액션).

        회피는 '방위만 틀기'가 아니라 **옆으로 실제로 이동하는 우회 기동**
        이어야 한다. 깊이 센서는 2.6m 안쪽 거리를 분해하지 못해서, 물체
        앞에 서면 어느 방향으로 틀어도 "가깝다"만 나온다. 초기 버전은 그
        상태에서 좌우로 15°씩 왔다갔다만 하며 수천 스텝을 태웠다.

        그래서: 목표 방위로 돌고 전진하되, 막히면(범프 또는 예상 밖 근접
        장애물) 여유가 더 많은 쪽으로 60° 틀어 일정 거리를 **직진**한 뒤
        다시 목표를 겨눈다(bug 알고리즘). 방이 6~10m이고 오브젝트는
        1.5m 이하라 몇 번의 우회로 반드시 빠져나온다.
        """
        # 1) 진행 중인 우회 기동
        if self._detour_steps > 0:
            self._detour_steps -= 1
            err = geo.wrap180(self._detour_heading - self.pose.heading)
            if abs(err) > HEADING_TOL:
                return False, (TURN_LEFT if err > 0 else TURN_RIGHT)
            return False, MOVE_FORWARD

        # 2) 목표 조준
        err = self.pose.bearing_to(tx, tz)
        if abs(err) > HEADING_TOL:
            return False, (TURN_LEFT if err > 0 else TURN_RIGHT)

        # 3) 전진 가부. 물리적으로 막힌 지점(범프 기억)은 확실한 근거이고,
        #    센서는 "지도가 멀다는데 가깝게 찍힌" 경우에만 신뢰한다.
        goal_dist = self.pose.distance_to(tx, tz)
        if self._bump_ahead():
            self._start_detour(p)
            return False, self._detour_action()
        if p.profile is not None:
            clearance = geo.cone_clearance(p.profile, 0.0, 12.0)
            predicted = self.map.predicted_range(
                self.pose.x, self.pose.z, self.pose.heading)
            surprise = (clearance <= geo.FLOOR_MIN_RANGE and predicted > 3.2
                        and goal_dist > 1.2)
            if surprise:
                self._start_detour(p)
                return False, self._detour_action()
        return False, MOVE_FORWARD

    def _start_detour(self, p: Percept) -> None:
        """여유가 더 많은 쪽으로 60° 틀어 DETOUR_STEPS만큼 직진하도록 예약."""
        side = 1.0
        if p.profile is not None:
            left = geo.cone_clearance(p.profile, 45.0, 25.0)
            right = geo.cone_clearance(p.profile, -45.0, 25.0)
            side = 1.0 if left >= right else -1.0
        if self._last_detour_side == side and self.step_i - self._last_detour_step < 60:
            side = -side          # 같은 쪽으로 계속 실패하면 반대로 틀어본다
        self._last_detour_side = side
        self._last_detour_step = self.step_i
        self._detour_heading = (self.pose.heading + side * DETOUR_TURN) % 360.0
        self._detour_steps = DETOUR_STEPS

    def _detour_action(self) -> int:
        err = geo.wrap180(self._detour_heading - self.pose.heading)
        if abs(err) > HEADING_TOL:
            return TURN_LEFT if err > 0 else TURN_RIGHT
        return MOVE_FORWARD

    def _bump_ahead(self) -> bool:
        """바로 앞 0.5m 칸이 최근에 부딪힌 지점인지."""
        bx, bz = self.pose.ahead(geo.AGENT_RADIUS + 0.1)
        step = self.bumps.get((int(bx * 2), int(bz * 2)))
        return step is not None and self.step_i - step < 150

    # --- 전투 ----------------------------------------------------------
    def _track_target(self) -> Optional[Tuple[float, float]]:
        """사거리 안의 움직임 트랙 → (상대 방위, 거리). 없으면 None."""
        t = self.map.track_near(self.pose.x, self.pose.z, ATTACK_RANGE,
                                self.step_i, TRACK_AIM_FRESH,
                                min_sightings=TRACK_MIN_SIGHTINGS)
        if t is None:
            return None
        self._aim_track = t
        return (self.pose.bearing_to(t.x, t.z),
                self.pose.distance_to(t.x, t.z))

    def _threat(self, p: Percept) -> Optional[Tuple[float, float]]:
        """선제 공격 대상의 (상대 방위, 거리).

        1순위는 **움직임 트랙**이다(motion.py). 정적 장면의 프레임 차분이
        0px라 오검출이 원리적으로 없고, 색·질감·형태를 하나도 안 보므로
        held-out 에셋(적 얼굴/3D 오브젝트)에 영향을 받지 않는다. 게다가
        적이 등 뒤로 돌아가도 트랙의 마지막 위치가 남아 조준이 가능하다.

        2순위가 외형 몹 탐지기(agent/enemies.py)다. 트랙이 아직 없는
        첫 조우(=한 번도 정지 관측을 못 한 상태)를 메운다. 둘 다 놓치면
        맞고 있을 때에 한해 기하 휴리스틱으로 보완한다.
        """
        self._aim_track = None
        room = self.map.get(self.cur_cell)
        alert = self.step_i <= self._alert_until
        under = self._under_attack()
        if self.state_steps.get("COMBAT", 0) > 1800:
            return None                      # 전투에 너무 많이 쓰면 그만둔다
        track = self._track_target()
        if track is not None:
            return track
        if room is None and not under:
            return None
        max_dist = 3.0 if (alert or under) else 2.4
        mob = self._detect_mob(p, max_dist, relaxed=under)
        if mob is not None:
            return mob
        if not under:
            return None
        # 맞고 있는데 몹이 안 보인다 → 기하 단서라도 쓴다.
        if room is None or not room.scanned:
            return None
        return self._surprise(p, max_dist=max_dist, relaxed=True)

    def _detect_mob(self, p: Percept, max_dist: float, relaxed: bool,
                    min_score: float = MOB_MIN_SCORE) -> Optional[Tuple[float, float]]:
        """몹 탐지기 결과 중 사거리 안의 가장 가까운 후보."""
        # 값싼 사전 필터: 사거리 안에 아무것도 없으면 탐지기를 돌리지 않는다.
        # (탐지기는 프레임당 ~1.5ms라 매 스텝 무조건 돌리면 예산을 먹는다)
        if not relaxed and p.profile is not None:
            if geo.cone_clearance(p.profile, 0.0, 34.0) > max_dist + 1.0:
                return None
        room = self.map.get(self.cur_cell)
        wall_rgb = room.wall_color if room is not None else None
        statics = room.static_blobs if (room is not None and not relaxed) else []
        for mob in EN.detect(p.frame, wall_rgb=wall_rgb, sat=p.sat):
            if mob.distance > max_dist or abs(mob.bearing) > 34.0:
                continue
            if mob.score < min_score:
                continue
            hx, hz = geo.heading_to_vec(self.pose.heading + mob.bearing)
            px, pz = (self.pose.x + hx * mob.distance,
                      self.pose.z + hz * mob.distance)
            if any(math.hypot(px - sx, pz - sz) < 1.0 for sx, sz in statics):
                continue                     # 스캔 때 본 정적 오브젝트
            self._last_mob = mob
            return mob.bearing, mob.distance
        return None

    def _under_attack(self) -> bool:
        """최근에 실제로 HP가 깎였는가 = 적이 확실히 붙어 있는 상태."""
        return self.step_i - self._last_hit_step < UNDER_ATTACK_STEPS

    def _surprise(self, p: Percept, max_dist: float = 3.0,
                  relaxed: bool = False) -> Optional[Tuple[float, float]]:
        """지도로 설명되지 않는 **좁은** 근접 장애물의 (상대 방위, 거리).

        두 가지 필터가 핵심이다.

        1. 지도 예측과의 비교 — 깊이 센서는 2.6m보다 가까우면 거리를
           분해하지 못하고 포화값만 준다. 그래서 벽·문틀·이미 아는
           오브젝트 앞에서도 "뭔가 가깝다"가 뜬다. 지도가 그 방향으로
           비어 있다고 말할 때만 예상 밖 물체다.
        2. 폭 — 적(사람형 몹)은 폭 0.5~0.9m라 2m 거리에서 20° 남짓,
           40개 열 중 3~6개만 차지한다. 벽면이나 문틀은 훨씬 넓게
           걸린다. 이 폭 필터를 넣기 전에는 공격 358회 중 345회가
           헛스윙이었다.
        """
        if p.profile is None:
            return None
        room = self.map.get(self.cur_cell)
        statics = room.static_blobs if room else []
        bearings = geo.column_bearings(len(p.profile))
        n = len(p.profile)

        # 열별로 "예상보다 가까운가"를 먼저 표시한다.
        flags = []
        for b, d in zip(bearings, p.profile):
            near = (abs(b) <= 35.0 and d < max_dist
                    and d < self.map.predicted_range(
                        self.pose.x, self.pose.z, self.pose.heading + b) - 0.8)
            flags.append(near)

        best = None
        i = 0
        while i < n:
            if not flags[i]:
                i += 1
                continue
            j = i
            while j < n and flags[j]:
                j += 1
            width = abs(bearings[j - 1] - bearings[i])
            d_min = min(p.profile[t] for t in range(i, j))
            # 폭 1.2m짜리 몹이 그 거리에서 차지하는 각도 + 여유. 다만 2.6m
            # 안쪽은 거리 분해가 안 돼 포화값(0.6)이 오므로, 그때는 폭 대신
            # 색으로 판단한다(벽은 팔레트 단색, 몹은 셔츠/바지/피부 3색).
            saturated = d_min <= geo.NEAR_RANGE + 0.01
            cap = 60.0 if saturated else min(
                110.0, max(34.0, math.degrees(2 * math.atan(1.2 / max(d_min, 0.5)))))
            cw = geo.FRAME_W / n
            # 피격 중(relaxed)에는 색/정적목록 필터를 끈다. 적이 이미 때리고
            # 있는 상황에서 "벽색과 비슷하다", "여기 예전에 통이 있었다"는
            # 이유로 후보를 버리면 공격 한 번 못 하고 맞아 죽는다.
            looks_like_wall = (
                not relaxed
                and room is not None and room.wall_color is not None
                and geo.color_close(
                    geo.region_color(p.frame, int(i * cw), int(j * cw)),
                    room.wall_color)
            )
            if width <= cap and not looks_like_wall:
                k = min(range(i, j), key=lambda t: p.profile[t])
                b, d = bearings[k], float(p.profile[k])
                hx, hz = geo.heading_to_vec(self.pose.heading + b)
                reach = min(d, geo.FLOOR_MIN_RANGE)
                px, pz = self.pose.x + hx * reach, self.pose.z + hz * reach
                known = (not relaxed) and any(
                    math.hypot(px - sx, pz - sz) < 1.2 for sx, sz in statics)
                if not known and (best is None or d < best[1]):
                    best = (b, d)
            i = j
        return best

    def _combat(self, p: Percept) -> int:
        """붙은 적을 향해 돌고 때린다.

        규칙은 두 가지다.

        * **회전만 하는 상태를 만들지 않는다.** 맞고 있는데 대상이 안 보이면
          (등 뒤에서 접근, 벽색과 비슷한 셔츠, 예전에 오브젝트가 있던 자리
          등) 제자리 회전 대신 '때리면서' 훑는다. 우리 공격은 쿨다운이 없고
          콘이 ±20°라, 도는 동안 적 방향을 스치기만 해도 여러 대가 들어간다.
        * **목표가 보이면 조준은 최소로.** 공격 콘이 ±20°이므로 그 안에
          들어오는 순간 바로 때린다. 조준만 반복하는 건 그 자체가 피해다.
        """
        self._combat_swing += 1
        relaxed = self._under_attack()
        # 안전판: 때리는 동안에도 계속 맞고 있다면 목표를 잘못 짚은 것이다.
        # (실측: 벽 사진/소품을 확신 있게 오인해 440회를 때리는 동안 진짜
        # 적은 등 뒤 101°에 있었다.) 잠시 탐지를 무시하고 스윕으로 찾는다.
        if self._hits_in_combat >= 2 and self.step_i > self._force_sweep_until:
            self._force_sweep_until = self.step_i + FORCE_SWEEP_STEPS
            self._hits_in_combat = 0
            self._combat_swing = 0
        forced = self.step_i <= self._force_sweep_until

        self._aim_track = None
        if forced:
            target = None
        else:
            # 트랙이 최우선. 움직임으로 확인된 적이라 '오브젝트를 때리고
            # 있는 것 아닌가' 하는 의심 자체가 필요 없다.
            target = self._track_target()
        if target is None and not forced:
            if relaxed:
                # 맞고 있을 때 정면에서 보이는 어중간한 후보를 때리다가,
                # 정작 옆·뒤에서 때리는 적을 놓치는 게 최악이다(실측: 공격
                # 413회 전부 콘 밖, 킬 0). 확신 있는 몹 탐지만 목표로 삼고,
                # 아니면 때리면서 도는 스윕으로 찾는다.
                target = self._detect_mob(p, ATTACK_RANGE, relaxed=True,
                                          min_score=MOB_SURE_SCORE)
            else:
                target = self._detect_mob(p, ATTACK_RANGE, relaxed=False)

        if target is None:
            if (relaxed or forced) and self._combat_swing <= SWEEP_LIMIT:
                return self._sweep_action()
            self._end_combat()
            return NO_OP

        on_track = self._aim_track is not None
        limit = SWEEP_LIMIT if (relaxed or on_track) else ATTACK_GIVEUP
        if self._combat_swing > limit:
            if not relaxed and not on_track:
                # 계속 때려도 안 사라진다 → 적이 아니라 정적 오브젝트로 학습.
                # 트랙 목표에는 적용하지 않는다 — 움직인 것이 확인된 대상을
                # 정적 장애물로 학습하면 그 뒤로 영영 무시하게 된다.
                room = self.map.get(self.cur_cell)
                if room is not None:
                    hx, hz = geo.heading_to_vec(self.pose.heading + target[0])
                    reach = min(target[1], geo.FLOOR_MIN_RANGE)
                    room.static_blobs.append(
                        (self.pose.x + hx * reach, self.pose.z + hz * reach))
            self._end_combat()
            return NO_OP

        self._combat_until = max(self._combat_until, self.step_i + 6)
        bearing = target[0]
        if abs(bearing) > ATTACK_CONE_DEG:
            self._combat_turns += 1
            # 트랙은 위치를 알고 겨누는 것이므로 한 바퀴를 돌아서라도
            # 조준할 값어치가 있다. 외형 후보는 오검출일 수 있어 짧게 끊는다.
            max_turns = TRACK_MAX_TURNS if on_track else COMBAT_MAX_TURNS
            if self._combat_turns > max_turns:
                if relaxed:
                    return self._aimed_attack()
                self._end_combat()
                return NO_OP
            return TURN_LEFT if bearing > 0 else TURN_RIGHT
        return self._aimed_attack()

    def _aimed_attack(self) -> int:
        """조준이 끝난 공격 한 대. 트랙 목표면 사망 판정용으로 기록한다.

        env의 명중 조건(사거리 3m 안 + 콘 ±20° 안)을 그대로 다시 계산해서,
        그 조건을 만족한 공격만 `hits`로 센다. 트랙 위치가 맞다면 이건 곧
        실제 명중 수이고, 4대면 적 HP 상한(4)을 넘으므로 사망이 확정된다.
        """
        self.stats["attacks"] += 1
        t = self._aim_track
        if t is not None:
            t.attacked += 1
            t.attacked_total += 1
            t.last_attack_step = self.step_i
            self._check_phantom(t)
        return ATTACK

    def _check_phantom(self, t) -> None:
        """때릴 만큼 때렸는데 아무 반응이 없는 트랙을 유령으로 강등한다."""
        if t.is_phantom(self.step_i):
            return
        if t.attacked_total >= TRACK_RETIRE:
            t.phantom_until = M.RETIRE_MARK      # 영구 배제
            self._end_combat()
            return
        if t.attacked < TRACK_GIVEUP:
            return
        t.phantom_until = self.step_i + PHANTOM_STEPS
        t.attacked = 0
        self._end_combat()

    def _sweep_action(self) -> int:
        """목표를 못 짚었을 때의 '때리며 훑기' — 공격 2 + 회전 1의 반복.

        공격을 연속 두 번 넣는 건 데미지 때문만이 아니다. ATTACK은 에이전트를
        움직이지 않으므로 연속 두 프레임이 **정지 쌍**이 되고, 그 차분이
        곧 적 탐지다(motion.py). 즉 훑는 동안 매 15°마다 무료로 한 번씩
        주변을 살피게 되어, 스윕이 끝날 때쯤이면 대개 트랙이 잡혀 있다.

        비용은 거의 없다. 시뮬 시간은 step() 사이의 실제 경과 시간으로만
        흐르므로(env.py), 스텝 ~12ms 기준 한 바퀴(72스텝)가 0.86 시뮬초다.
        적의 공격 쿨다운이 0.8~2.0초라 한 바퀴 도는 동안 많아야 한 대 맞는다.

        호출 측에서 이미 _combat_swing을 올린 뒤에 부른다.
        """
        if self._combat_swing % 3 == 0:
            return TURN_LEFT
        self.stats["attacks"] += 1
        self._register_blind_hit()
        return ATTACK

    def _register_blind_hit(self) -> None:
        """조준 없이 때린 공격도 명중 조건을 만족하면 그 트랙에 기록한다.

        훑기로 죽인 적이 사망 판정에서 빠지는 걸 막는다. env의 레이캐스트는
        콘 안에서 **가장 가까운** 적을 때리므로 여기서도 같은 규칙을 쓴다.
        """
        best, best_d = None, None
        for t in self.map.live_tracks(self.step_i, TRACK_AIM_FRESH,
                                      min_sightings=TRACK_MIN_SIGHTINGS):
            d = self.pose.distance_to(t.x, t.z)
            if d > ATTACK_RANGE:
                continue
            if abs(self.pose.bearing_to(t.x, t.z)) > ATTACK_CONE_DEG:
                continue
            if best_d is None or d < best_d:
                best, best_d = t, d
        if best is not None:
            best.attacked += 1
            best.attacked_total += 1
            best.last_attack_step = self.step_i
            self._check_phantom(best)

    def _end_combat(self) -> None:
        self._hits_in_combat = 0
        self._combat_until = -1
        self._combat_swing = 0
        self._combat_turns = 0
        self._threat_seen = -99
