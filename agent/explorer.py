"""탐험 정책: 픽셀만 보고 방을 최대한 많이 도는 규칙기반 상태머신.

상태:
  INIT     — 첫 프레임에서 스폰된 방을 그래프의 루트로 등록.
  RECENTER — 방에 막 들어왔을 때(문 바로 앞), agent.geometry로 입구에서
             측정한 정면 클리어런스의 절반쯤(=대충 방 중앙 방향)까지
             전진 — "문 바로 앞"에 어정쩡하게 서 있지 않고 방 안쪽으로
             들어가기 위함(사용자 피드백: 입구에서 멈춰서 도는 게 어색해
             보임). VLM으로 방 내용을 캡처할 때 문틈으로 옆방이 섞여
             보이는 오염을 줄이는 목적도 겸한다(dev_log.md 참고).
  SURVEY   — 방 안쪽(RECENTER 끝난 위치)에서 4방향(0/90/180/270)을 돌아보며
             벽 색을 기록한다. 문은 오직 이 4방향에만 존재한다
             (world/layout.py의 격자 구조로 보장됨, README의 heading<->wall
             대응표와도 일치) — 그래서 임의의 각도를 찔러볼 필요가 없다.
             agent.py는 이 상태일 때 프레임을 버퍼링해서 나중에 VLM에 보낸다.
             각 방향을 정면으로 본 시점에 예산이 남아있으면
             agent.vlm.classify_heading()도 한 번 불러 "문처럼 보이는지"
             미리 물어본다(사용자 지시: 움직이기 전에 먼저 VLM에게 확인).
             VLM이 "문 아님"이라고 확신하면 그 자리에서 바로 exits를
             "wall"로 기록해 SEEK의 물리 접근(부딪혀보기+180도 등 CV
             안전망) 자체를 건너뛴다 — 명백한 벽에 헛되이 부딪히는 시도를
             줄이는 게 목적. "문일 수 있음"이면 vlm_door_hint에만 남기고
             exits는 여전히 unknown으로 둔다(진짜 문인지, 어디로
             이어지는지는 결국 실제로 걸어가야 알 수 있으므로) — SEEK가
             후보를 고를 때 이 힌트가 있는 방향을 먼저 시도한다. VLM이
             실패/타임아웃/예산소진이면 기존과 완전히 동일하게 동작
             (VLM은 어디까지나 주력, CV 물리 확인이 최종 보험 — 사용자
             확인: "VLM이 주력, CV는 보험").
  SEEK     — 후보 방향 하나를 향해 전진(vlm_door_hint가 있는 방향 우선).
             막히면(벽이든 물건이든 구분 없이) 조금씩 회전하며 재시도,
             그래도 안 되면 "이 방향엔 문 없음"으로 기록하고 다음 후보로.
             방 이름이 바뀌면 문을 찾은 것.
  RETURN   — 이미 가본 방으로 연결됐거나, 이 방의 4방향을 다 확인해서
             이전 방으로 되돌아가야 할 때(DFS backtrack).
  FLEE     — HP가 실제로 깎이는 걸 감지하면(=적이 근처에 있다는 뜻) 최우선
             진입, 곧바로 strike(공격) 상태로. 원래는 "일단 후진해서 거리를
             만든 뒤 반격"을 시도했지만, 실측으로 직접 확인해보니(dev_log.md)
             적의 추격 속도가 내 후진 속도와 거의 같아서 아무리 후진해도
             적과의 거리(항상 1.3~1.5m, 자기 공격 사거리 안)가 전혀 안
             줄어들고 그냥 계속 맞기만 했다 — 즉 "후진하면 거리가 벌어진다"
             는 전제 자체가 이 게임에서는 성립하지 않는다(사용자 확인).
             그래서 후진 없이 맞은 그 자리에서 바로 대응한다: 적이 이미
             항상 내 공격 사거리(3.0m)보다 가까운 1.3~1.5m 안에 있으므로
             거리 문제는 없고, 방향만 맞으면 된다. 2단 안전망으로 매 틱
             조준한다:
               1) agent.enemies.detect(): 바닥 체커보드 채도 경계로 각
                  방향의 실측 거리를 얻고(agent.geometry, 카메라가
                  domain_rand=False로 고정이라 픽셀→미터가 정확), 원색
                  채도·바닥 접점·키(1.15~2.45m)로 몹을 식별해 정확한 도
                  단위 방위/거리를 낸다. 프레임당 ~1~2ms라 사실상 즉시 —
                  대부분의 틱은 여기서 끝난다. 콘·사거리 안이면 바로 공격,
                  아니면 정확히 그 방향으로 회전.
               2) 못 찾으면(실패/예산 소진 포함) 결정론적 sweep으로 전환
                  — 공격 콘이 정면 ±20도(전체 40도), 회전 1회가 15도
                  (실측/소스 확인)이므로 "공격 2번 → 15도 회전"을 24번
                  (360도) 반복하면 적이 어느 각도에 있든 반드시 콘 안에
                  걸리는 순간이 생긴다는 게 기하학적으로 보장된다.
             VLM(agent.vlm.locate_enemy)은 조준 결정에서 뺐다 — "왼쪽/
             가운데/오른쪽" 3단계 판정이 실제 공격 콘보다 훨씬 좁게
             "가운데"를 매겨서, 콘 안에 이미 들어와 있는데도 계속 좌우로
             흔들리며 한 번도 공격 못 하고 맞기만 하는 게 실측으로
             확인됐다(dev_log.md). sweep 도중 이미 한 번 맞춘 뒤에는
             "아직 살아있는지" 확인용으로만 VLM을 보조로 쓴다(방위 판단이
             아니라 단순 존재 여부만 물으므로 이 문제가 없음).
             한 번 sweep으로 전환하면(진행 중 다시 맞아도 처음부터 다시
             시작하지 않고) 끝까지 진행한다. 다 돌아도(또는 안전 상한 틱을
             넘어도) 끝나면 하던 일로 복귀 — 그 뒤 다시 HP가 깎이면(=아직
             살아있거나 두 번째 적) step() 최상단의 HP-드랍 감지가 새로
             FLEE를 트리거한다(자동으로 반복).
  DONE     — 갈 수 있는 새 방을 다 찾음(도달 가능한 범위 완료).
"""

from __future__ import annotations

import re

from agent import enemies as EN
from agent import geometry as geo
from agent.memory import SceneGraph, CARDINAL_HEADINGS
from agent.ocr import hint_banner_active, read_hint_text, read_hud
from agent.palette import WALL_PALETTE
from agent.vision import is_blocked, sample_wall_color
from agent.vlm import classify_heading, locate_door, locate_enemy, recover_action
from agent.config import VLM_MAX_CALLS_PER_EPISODE

_WALL_RGB_BY_NAME = dict(WALL_PALETTE)
_MOB_MIN_SCORE = 0.6              # agent.enemies 탐지 신뢰도 하한
_ATTACK_CONE_HALF_DEG = 20.0      # env의 공격 콘 반각(소스 확인)
_ATTACK_RANGE_M = 3.0             # env의 공격 사거리(소스 확인)
# 전투 중 CV 탐지는 이 거리보다 먼 후보를 아예 무시한다. 실측(dev_log.md):
# 우리 WALL_PALETTE/벽그림은 채도가 높은 색도 많아서(mustard 155 등),
# 먼 벽·그림이 "20m 밖의 사람 키 몹"으로 오탐되는 경우가 실제로 있었다
# (에피소드 첫 프레임부터 재현됨). 전투 중 진짜 적은 항상 1.3~1.5m
# 이내(실측 확인)이므로, 이 범위 밖 후보는 원거리 오탐일 뿐 필요도 없다.
_MOB_MAX_COMBAT_DIST_M = 4.0
# CV가 나무 같은 3D 소품(키가 사람과 비슷해서 안 걸러짐, dev_log.md)을
# 적으로 오인해 "콘·사거리 안"이라고 계속 우기는 걸 실측으로 확인했다 —
# 진짜 적이면 HP 1~4라 몇 방이면 죽어야 정상인데 콘/사거리 조건을 계속
# 만족하며 공격이 이만큼 연속되면 오탐으로 보고 sweep으로 넘어간다.
_FASTPATH_ATTACK_STREAK_MAX = 4
# close_range_bearing()은 FLEE strike 진입 시점(=이미 맞아서 적이 코앞)
# 에만 쓰여서 오탐 위험이 훨씬 낮다 — 실측(dev_log.md, seed=7): 적 HP가
# 1~4인데 위 4번 cap에 걸려 매번 sweep으로 밀려나서, 근접 탐지가 성공해도
# 총 전투 시간이 거의 안 줄었다(그대로 매번 sweep을 다시 타서 39틱대
# 유지). 진짜 적이 확실한 상황이라 cap을 넉넉히 늘려 fastpath만으로
# 끝까지 죽일 기회를 준다(빗맞음 몇 번 있어도 충분하도록 최대 HP의 2배).
_CLOSE_RANGE_ATTACK_STREAK_MAX = 8

# 전진 전에 미리 확인하는 정면 클리어런스(agent.geometry 깊이 프로파일).
# AGENT_RADIUS(0.4m) + 한 스텝(0.15m) + 여유. 이걸로 "가보고 막히면 반응"
# 대신 "가기 전에 이미 안다"로 바꿔서 벽에 부딪히는 움직임을 없앤다
# (사용자 지시: 이동이 어색하다는 실측 피드백).
_SEEK_SAFE_CLEARANCE = 0.75
_SEEK_CONE_HALF_DEG = 15.0
# 나무·소품처럼 국지적인 장애물을 만났을 때, 목표 방향을 포기하고 벽
# 판정으로 넘어가기 전에 옆으로 살짝 비켜 지나갈 수 있는지 먼저 본다
# (사용자 지시: "각 obstacle에 경계가 있다고 여기고 contourner"). FOV가
# ±37.5도(agent.geometry.FOV_X_DEG)라 이 범위 밖은 이번 프레임에 안
# 보이므로, 그 안에서만 옆 방향을 찔러본다.
_SIDESTEP_OFFSETS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)
_SIDESTEP_CONE_HALF_DEG = 8.0
_SIDESTEP_MAX_PROBE_DEG = 35.0  # FOV_X_DEG(~75)의 절반 안쪽만 신뢰

_TURN_TOLERANCE = 7        # 목표 방향과 이 각도 이내면 "정렬됨"
_RECOVER_MAX_ATTEMPTS = 3  # 이만큼 회전해도 계속 막히면 "이 방향엔 문 없음"
# 문/벽은 항상 0/90/180/270 카디널에만 존재하므로, 15도 단위 미세조정을
# 오래 붙잡고 있을 이유가 없다 — 몇 번만 시도해보고 안 되면 바로 다음
# 카디널 후보로 넘어간다(실측 피드백: seed 0에서 Pearl Vault/Ebon Annex
# 에서 한 방향에 너무 오래 매달려 제자리서 도는 것처럼 보임, dev_log.md).
# 중간에 성공적 전진이 껴서 recover_attempts가 리셋되더라도, "이 목표
# 헤딩에서 180도-포기(구석 몰림)를 통째로 몇 번 겪었는지"는 별도로 세서
# 무한히 방을 맴도는 걸 막는다(실측으로 발견, dev_log.md).
_MAX_CORNERED_PER_TARGET = 1
# RECENTER: 이 이상 재면 벽이 아니라 문/개방부로 보고(그 방향 거리는
# 못 믿음) 반대쪽 벽만으로 중앙을 추정한다. 실측(ground truth로 확인,
# dev_log.md): 방이 10x10이라 입구 바로 앞에서 정면 벽까지가 이미 9.9m
# 가까이 나온다 — 처음엔 6.0m로 잡았다가 이 진짜 벽 거리까지 "문"으로
# 오판해서 전혀 안 움직이는 버그가 났다. geo.MAX_RANGE(20.0, 완전히
# 뚫린 문이 나오는 값)보다는 확실히 낮고, 이 env의 실제 방 크기보다는
# 넉넉히 큰 값으로 올림.
_RECENTER_MEASURE_CAP_M = 15.0

# HINT_CAPTURE: door_touch_radius=1.5m(difficulty.yaml 실측 확인). 자물쇠가
# 보일 만큼 가까이 있는 "구석에 몰린" 상태는 대개 이미 그 안쪽이거나
# 한두 스텝(forward_step=0.15m) 안이라 20스텝(=최대 3.0m 전진 시도)이면
# 충분하고도 남는다 — 못 미쳐도 collision이 먼저 막아줌.
_HINT_CAPTURE_MAX_APPROACH_TICKS = 20

# 공격 콘 정면 ±20도(전체 40도), 회전 1회 15도(_ATTACK_CONE_HALF_RAD/turn_step,
# 소스로 확인) — 15도 간격으로 돌면서 매 위치마다 공격하면 콘끼리 겹쳐서
# 어느 각도에 있든 반드시 한 번은 걸린다. sweep 안전망의 기반 수치.
_FLEE_SWEEP_HEADINGS = 24              # 360 / 15도
_FLEE_SWEEP_ATTACKS_PER_HEADING = 2    # 적 HP 1~4 감안, 걸렸을 때 한 방에 안 죽어도 잡도록
_FLEE_SWEEP_RECHECK_EVERY = 4          # 이만큼 방향을 돌 때마다 VLM으로 "아직 있는지" 재확인
# 실측(dev_log.md, seed=7): close_range_bearing()으로 이미 명중까지
# 시켰는데(flee_ever_attacked=True) 그 다음 틱에 갑자기 못 찾으면, 적
# HP가 1~4라 대부분 죽어서 사라진 경우다 — 이럴 때도 못 찾은 이유를
# "안 보였을 뿐일 수도 있다"고 24방향(최대 72틱) 풀스윕을 다 도는 건
# 낭비다. 짧게만 확인하고 없으면 바로 복귀한다.
_FLEE_SWEEP_QUICK_CHECK_HEADINGS = 4
_FLEE_STRIKE_MAX_TICKS = (             # 안전 상한: VLM 조준 실패 후 sweep 풀 사이클까지 감안
    _FLEE_SWEEP_HEADINGS * (_FLEE_SWEEP_ATTACKS_PER_HEADING + 1) + 4
)
_FLEE_VLM_MAX_CALLS = VLM_MAX_CALLS_PER_EPISODE  # 전투 조준 + 막힌 방향 VLM 확인이 이 예산을 공유

# VLM_RECOVER: 같은 액션을 이만큼 연속으로 냈는데(그리고 그동안 화면도
# 거의 안 바뀌었으면) "제자리에서 맴돈다"고 보고 VLM에게 잠깐 조종을
# 맡긴다(사용자 지시). 너무 낮으면(예: 3~4) 정상적인 반복 행동(SURVEY의
# NO_OP 연속 등)까지 오탐하고, 너무 높으면 정체를 늦게 알아챈다 —
# 회전만으로 한 바퀴(24스텝) 도는 SEEK 복구 루프보다는 확실히 짧게.
_STUCK_TICKS_THRESHOLD = 10
_VLM_RECOVER_MAX_TICKS = 20  # VLM이 계속 "아직 안 열림"이라 해도 여기서 강제 종료
# 실측으로 발견한 별도 정체 패턴(dev_log.md): 액션은 계속 바뀌는데(좌우
# 오락가락 등) 방은 전혀 못 넘어가는 경우 — RETURN이 목표 방을 못 찾고
# 500틱 넘게 같은 방에 갇힌 사례를 실측으로 확인함. SEEK가 한 방향의
# 4단계 복구 사다리를 정상적으로 다 밟는 데도 수십~백 틱 정도는 걸릴 수
# 있어(단계마다 6~15액션 반복 x 최대 4단계 x 후보 방향 여러 개) 너무
# 낮추면 정상 동작까지 방해하므로, 확실히 비정상인 수준으로 넉넉하게 잡음.
_STUCK_ROOM_TICKS_THRESHOLD = 150

# 열쇠 찾기: 방 하나를 훑을 때 정면/좌/뒤/우 순으로 몇 걸음씩 걸어보는
# 십자(+) 패턴. key_pickup_radius(0.8m)가 좁고 열쇠 위치가 방 안 랜덤이라
# (실측 확인, dev_log.md) 완벽한 커버리지는 아니지만, 중앙 한 점보다는
# 훨씬 넓게 훑는다.
_KEY_SWEEP_HEADINGS_OFFSET = (0, 90, 180, 270)  # 진입 시 헤딩 기준 상대 오프셋
_KEY_SWEEP_STEPS_PER_LEG = 15  # 15 * 0.15m ≈ 2.25m


def _heading_diff(a: int, b: int) -> int:
    """b - a를 [-180, 180]로 정규화. TURN_RIGHT는 헤딩을 줄이고 TURN_LEFT는
    늘리므로(실측 확인됨), 양수(목표가 더 큼)면 TURN_LEFT, 음수면
    TURN_RIGHT가 목표에 가까워지는 방향이다."""
    d = (b - a) % 360
    if d > 180:
        d -= 360
    return d


# 힌트 텍스트(doors.py::HINT_TEMPLATES)를 지금까지 알아낸 방 정보와
# 대조해서 후보 방을 추정한다. 플레이 도중(=answer() 전, content_db가
# 아직 안 채워진 시점)에는 wall_color와 방 이름만 확실히 갖고 있으므로,
# 그 두 템플릿("the room with {color} walls", "the room called {name}")
# 만 직접 매칭한다 — 나머지(이미지/오브젝트 개수) 템플릿은 지금 갖고
# 있는 정보로는 판단 불가능하니 매칭 안 되면 그냥 None(포기)을 반환한다.
# color 템플릿은 전체 방(8~12개) 기준으로는 고유하다고 doors.py가
# 보장하지만, 우리가 "지금까지 찾은" 부분집합에서는 여러 방이 같은
# 색일 수 있어 모호할 수 있다 — 모호하면 그냥 첫 매칭을 최선의 추정으로
# 쓴다(확신 낮음, report.md에 한계로 기록 예정).
def _match_hint_to_room(hint_text: str, scene: SceneGraph) -> Optional[str]:
    if not hint_text:
        return None
    text = hint_text.lower()
    m = re.search(r"room called ([a-z .…]+)", text)
    if m:
        candidate = m.group(1).strip().rstrip(".")
        for name in scene.nodes:
            norm = name.lower().rstrip("…").strip()
            if norm and (norm in candidate or candidate in norm):
                return name
    for name, node in scene.nodes.items():
        if node.wall_color and f"{node.wall_color} walls" in text:
            return name
    return None


class ExplorerPolicy:
    def __init__(self) -> None:
        from memory_fps_env.env import Action
        self._Action = Action

        self.scene = SceneGraph()
        self.state = "INIT"
        self.last_hp = None
        self.prev_frame = None

        self.pre_flee_state = None
        self.pre_flee_target = None
        self.flee_phase = None       # None | "strike"
        self.flee_ticks = 0
        self.flee_ever_attacked = False  # 이번 전투에서 한 번이라도 ATTACK을 냈는지
        self.flee_sweep_active = False   # CV가 못 찾아서(또는 안 죽어서) 결정론적 sweep으로 전환했는지
        self.flee_sweep_headings_done = 0
        self.flee_sweep_attacks_done = 0
        self.flee_sweep_heading_limit = _FLEE_SWEEP_HEADINGS
        self.flee_fastpath_attack_streak = 0  # CV 조준으로 같은 자리를 연속 공격한 횟수
        self.vlm_calls_used = 0      # strike 단계 locate_enemy() 호출 수(예산 상한용)
        self._proactive_attack_streak = 0  # 선제공격(비FLEE)이 연속 몇 틱째인지 — 무한루프 방지용

        # VLM_RECOVER: 정체 감지 + VLM 조종 인계
        self._stuck_last_action = None
        self._stuck_streak = 0
        self._stuck_ref_frame = None  # 지금 스트릭이 시작된 시점의 프레임(그 이후 실제 진전이 있었는지 판정용)
        self._stuck_last_room = None
        self._stuck_room_ticks = 0
        self.pre_recover_state = None
        self.pre_recover_target = None
        self._vlm_recover_ticks = 0

        # DFS 스택: 부모 방 도착 확인 전까지는 pop을 미룬다(위 _start_backtrack 주석 참고)
        self._stack_pop_pending = None

        # 열쇠 찾기 + 잠긴 문 열기(DONE 도달 후, 힌트가 있으면 시도)
        self.key_hunt_attempted = False  # 에피소드당 한 번만 시도
        self.goto_target_room = None
        self.goto_purpose = None  # "key_room" | "locked_door"
        self.key_sweep_idx = 0
        self.key_sweep_steps_done = 0

        self.recenter_room = None
        self.recenter_entry_heading = None
        self.recenter_phase = None       # "measure" | "move"
        self.recenter_measure_idx = 0
        self.recenter_clearances: dict = {}
        self.recenter_move_plan: list = []

        self.survey_queue: list = []
        self.survey_samples: dict = {}

        self.target_heading = None
        self.seek_origin_room = None
        self.return_target_room = None
        self.recover_attempts = 0
        self._need_align = False
        self._last_action_was_forward = False
        self._nav_bias_deg = 0.0  # SEEK/RETURN 중 국지 장애물을 비켜가는 임시 헤딩 오프셋
        self._nav_fail_count = 0     # 지금 목표를 몇 번째 "완전히 못 뚫음"으로 포기했는지(단계적 복구용)
        self._nav_action_queue: list = []  # 단계적 복구 중 미리 정해둔 액션 시퀀스(후진/옆걸음 등)
        self._nav_probed_center = False  # 이 방향으로 정렬한 뒤 전진을 한 번이라도 실제로 시도해봤는지
        self._nav_cornered_count = 0  # 이 목표 헤딩에서 180도-포기를 통째로 몇 번 겪었는지
        self._hud_cache_key = None
        self._hud_cache_value = None

        # HINT_CAPTURE: 자물쇠 판을 발견하면 잠깐 이 상태로 빠져 문 쪽으로
        # 더 다가가(터치 반경 안으로) 힌트 배너를 띄우고 읽은 뒤, 원래
        # 있던 자리 근처로 물러난다. seek_origin_room/target_heading은
        # SEEK에서 그대로 물려받아 안 바꾸므로 별도로 저장할 필요가 없다.
        self._hint_capture_phase = None  # None | "approach" | "retreat"
        self._hint_capture_forward_steps = 0
        self._hint_capture_retreat_remaining = 0
        self._hint_capture_return_state = None
        self._hint_capture_confirmed_locked = False  # 배너가 실제로 떴는지
        self._hint_captured_this_door = False  # 같은 문에서 재시도 방지

    # HUD OCR은 프레임당 ~140ms(글리프 템플릿 매칭)로 압도적인 병목이었다
    # (dev_log.md 실측: EN.detect 1.6ms/depth_profile 1.4ms와 비교해 100배).
    # 근데 HUD 바 픽셀은 방 이름/HP/방향이 안 바뀌는 대부분의 틱에서 완전히
    # 그대로다(방향만 회전할 때 바뀌고, 시간은 1초에 한 번). 참고 레포의
    # perception.py도 같은 이유로 HUD 바를 해시해서 캐싱한다 — 같은 방식을
    # 적용해 실제로 바뀐 프레임에서만 OCR을 다시 돌린다.
    def _read_hud_cached(self, obs):
        bar_key = hash(obs[:28, :, :].tobytes())
        if bar_key != self._hud_cache_key:
            self._hud_cache_value = read_hud(obs)
            self._hud_cache_key = bar_key
        return self._hud_cache_value

    @property
    def current_room(self):
        """지금까지 확인된 가장 최근 방 이름(canonicalized). agent.py가
        SURVEY 중 프레임을 어느 방에 버퍼링할지 정할 때 쓴다."""
        return self._stuck_last_room

    # --- 메인 진입점 ------------------------------------------------
    def step(self, obs):
        hud = self._read_hud_cached(obs)

        # QA용 영구 플래그: hud.has_key는 문을 여는 순간 다시 꺼지므로
        # "열쇠를 찾은 적이 있는지"는 여기서 한 번 True가 되면 계속 True로
        # 남긴다(memory.py::SceneGraph.key_found 참고).
        if hud.ok and hud.has_key:
            self.scene.key_found = True

        # 정체 감지용 보조 신호: 같은 방에 계속 머물러 있는지(=방을 못
        # 넘어가는지) 액션 종류와 무관하게 추적한다. 실측으로 발견한 문제
        # (dev_log.md): "같은 액션 반복" 기준만으로는 좌우로 오락가락하며
        # 액션 자체는 계속 바뀌는데 방은 전혀 안 바뀌는 경우(RETURN이
        # 목표 방을 못 찾고 500틱 넘게 한 방에 갇힘)를 못 잡는다. 방
        # 이름은 HUD OCR로 매 틱 거의 공짜로 읽으니 이것도 같이 본다.
        if hud.ok and hud.room_name is not None:
            canon_now = self._canonicalize(hud.room_name)
            if canon_now == self._stuck_last_room:
                self._stuck_room_ticks += 1
            else:
                self._stuck_room_ticks = 0
                self._stuck_last_room = canon_now
        else:
            self._stuck_room_ticks = 0

        # 매 스텝 공통: HP 하락 감지 (화면에 뭐가 보이든 최우선으로 반응).
        # 후진해도 적과의 거리가 안 벌어진다는 게 실측 확인됐으므로(위 상단
        # 주석 참고), 맞은 그 자리에서 바로 VLM 조준으로 대응한다. 이미
        # FLEE 중에 또 맞은 거면(전투가 계속 이어지는 중) 진행 중인
        # sweep/조준 상태를 리셋하지 않는다 — 예전엔 매번 리셋해서 sweep이
        # 절대 끝까지 못 돌았던 버그가 있었음(dev_log.md).
        if hud.ok and hud.hp is not None:
            if self.last_hp is not None and hud.hp < self.last_hp and self.state != "FLEE":
                self.pre_flee_state = self.state
                self.pre_flee_target = self.target_heading
                self.state = "FLEE"
                self.flee_phase = "strike"
                self.flee_ticks = 0
                self.flee_ever_attacked = False
                self.flee_sweep_active = False
                self.flee_sweep_headings_done = 0
                self.flee_sweep_attacks_done = 0
                self.flee_sweep_heading_limit = _FLEE_SWEEP_HEADINGS
                self.flee_fastpath_attack_streak = 0
            self.last_hp = hud.hp

        if self.state == "FLEE":
            action = self._step_flee(hud, obs)
            self.prev_frame = obs
            return int(action)

        # 선제 공격: 지금까진 HP가 깎여야만(=한 대 맞아야만) 전투를 시작해서
        # "적을 보고도 안 죽이고 맞고 나서야 죽인다"는 실측 피드백이 있었다
        # (사용자 지시). agent.enemies는 이미 프레임당 ~1~2ms라 매 틱 상시
        # 돌려도 예산에 거의 안 걸리므로, FLEE 중이 아닐 때도 매 틱 확인해서
        # 콘·사거리 안에 적이 있으면 하던 일(탐험 상태)은 그대로 둔 채
        # 그 한 틱만 공격으로 가로챈다.
        # 실측으로 확인된 버그(dev_log.md): 안전장치 없이 매 틱 이 조건이면
        # 무조건 가로채니까, 죽지 않는 대상(오탐이거나, 실제 명중이 안 되는
        # 경우)을 만나면 완전히 같은 자리에서 ATTACK만 영원히 반복하며
        # 탐험이 통째로 멈췄다(seed=1, 60틱 넘게 위치/각도 고정). 적 HP가
        # 최대 4라 몇 방이면 죽어야 정상이므로, 이 이상 연속되면 오탐/명중
        # 실패로 보고 이번 틱은 하던 일(탐험)을 계속하게 양보한다 — 진짜
        # 적이면 언젠가 다시 맞고 정식 FLEE(sweep 포함) 경로로 넘어간다.
        mobs = EN.detect(obs, wall_rgb=self._current_wall_rgb(hud) if hud.ok else None)
        best = next((m for m in mobs if m.score >= _MOB_MIN_SCORE
                     and m.distance <= _MOB_MAX_COMBAT_DIST_M), None)
        if (best is not None and abs(best.bearing) <= _ATTACK_CONE_HALF_DEG
                and best.distance <= _ATTACK_RANGE_M
                and self._proactive_attack_streak < _FASTPATH_ATTACK_STREAK_MAX):
            self._proactive_attack_streak += 1
            self._last_action_was_forward = False  # 하던 상태의 is_blocked 계산이 꼬이지 않게
            self.prev_frame = obs
            return int(self._Action.ATTACK)
        self._proactive_attack_streak = 0

        # "사거리 밖 적을 멈춰서 지켜보다 다가오면 공격" 로직을 시도했다가
        # 실측으로 되돌렸다(dev_log.md) — 이 게임은 후진해도 적과의 거리가
        # 안 벌어진다는 게 이미 확인된 사실이라, 멈춰서 기다리든 탐험을
        # 계속하든 "언젠가 다가와서 맞는" 타이밍 자체는 거의 안 바뀐다.
        # 대신 멈춰서 지켜보는 동안 탐험 진행(다음 문 찾기 등)이 완전히
        # 멎어서, 적이 많은 방(Coral Vault 등)에 에이전트가 계속 묶여
        # 있다가 반복 교전으로 죽는 결과가 났다(같은 seed=7: 이 로직
        # 켰을 때 640스텝만에 사망 vs 껐을 때 4000스텝 끝까지 생존,
        # 직접 A/B로 확인). 그래서 사거리 밖 적은 그냥 무시하고 하던
        # 탐험을 계속하며, 실제로 맞으면(HP 하락) 그때 FLEE로 반응한다.

        if not hud.ok or hud.room_name is None:
            # HUD 못 읽음(글리치)이거나 문틈을 지나는 중("복도" = 방 이름
            # 없음). 방금 전진 중이었다면(문을 막 통과하는 중) 그대로
            # 전진해서 마저 건너가고, 그게 아니면(회전 등 다른 동작을
            # 하던 중이었다면) 근거 없이 위치를 바꾸지 않고 직전 액션을
            # 그대로 반복한다 — "정지하고 생각하고 움직이라"는 원칙(사용자
            # 지시). 예전엔 이 분기에서 무조건 MOVE_FORWARD를 냈는데,
            # RECENTER 측정(제자리 회전) 중에 room_name이 잠깐 흔들리면
            # 그 틱에 억지로 전진해 문턱 옆벽에 부딪히거나 방금 나온
            # 방으로 도로 넘어가 두 방 사이를 오가는 원인이 됐다
            # (실측 피드백: seed 0, dev_log.md).
            self.prev_frame = obs
            if self._last_action_was_forward:
                return int(self._Action.MOVE_FORWARD)
            if self._stuck_last_action is not None:
                return int(self._stuck_last_action)
            return int(self._Action.NO_OP)

        action = self._dispatch(hud, obs)

        # 정체 감지: 같은 액션이 반복되는데 화면도 안 바뀌면(=제자리에서
        # 맴돔) 룰베이스를 잠깐 멈추고 VLM에게 조종을 맡긴다(사용자
        # 지시). VLM_RECOVER 자신은 이 감지 대상에서 뺀다(무한 재진입
        # 방지) — DONE도 뺀다(더 갈 곳이 없어 의도적으로 대기 중이므로).
        # HINT_CAPTURE도 뺀다 — 문 앞까지 접근/후퇴하는 고정 스텝 수
        # 시퀀스라 원래도 MOVE_FORWARD가 여러 틱 연속되고(막힌 문에
        # 다가가는 중이라 화면도 잘 안 바뀜), 그 자체가 정상 동작이라
        # 정체로 오탐하면 자체 20틱 상한보다 먼저 끊겨버린다.
        if self.state not in ("VLM_RECOVER", "DONE", "HINT_CAPTURE"):
            if int(action) == self._stuck_last_action:
                self._stuck_streak += 1
            else:
                self._stuck_streak = 0
                self._stuck_last_action = int(action)
                self._stuck_ref_frame = obs
            tight_loop = (self._stuck_streak >= _STUCK_TICKS_THRESHOLD
                          and self._stuck_ref_frame is not None
                          and geo.frame_motion(self._stuck_ref_frame, obs) < geo.BLOCKED_MOTION)
            # 넓은 의미의 정체(액션은 바뀌어도 방을 못 넘어감)는 SEEK/
            # RETURN/RECENTER에서만 본다 — SURVEY는 원래 한 방에 오래
            # 머물며 4방향을 다 보는 게 정상 동작이라 이 기준에서 뺀다.
            wide_loop = (self.state in ("SEEK", "RETURN", "RECENTER")
                         and self._stuck_room_ticks >= _STUCK_ROOM_TICKS_THRESHOLD)
            if tight_loop or wide_loop:
                action = self._start_vlm_recover(obs)

        self.prev_frame = obs
        return int(action)

    def _dispatch(self, hud, obs):
        if self.state == "INIT":
            return self._enter_room(hud, entry_heading=None, parent=None)
        if self.state == "RECENTER":
            return self._step_recenter(hud, obs)
        if self.state == "SURVEY":
            return self._step_survey(hud, obs)
        if self.state == "SEEK":
            return self._step_seek(hud, obs)
        if self.state == "HINT_CAPTURE":
            return self._step_hint_capture(hud, obs)
        if self.state == "RETURN":
            return self._step_return(hud, obs)
        if self.state == "VLM_RECOVER":
            return self._step_vlm_recover(obs)
        if self.state == "GOTO_ROOM":
            return self._step_goto_room(hud, obs)
        if self.state == "KEY_SWEEP":
            return self._step_key_sweep(hud, obs)
        if self.state == "KEY_UNLOCK":
            return self._step_key_unlock(hud, obs)
        if self.state == "DONE":
            return self._step_done(hud, obs)
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
        return self._start_recenter(name, entry_heading)

    # --- RECENTER: 방 4벽까지의 거리를 실측해 기하학적으로 정확한 중앙까지 이동 ---
    # 사용자 지시: "공간 자체가 geometry니까 그걸 계산해서 중간에 도착해야
    # 한다". 예전엔 "입구 클리어런스의 절반쯤"이라는 어림값으로 앞으로
    # 걷고, 그 다음에 따로 좌우를 재서 보정하는 2단계 방식이었는데, 이건
    # 진짜 중앙을 보장하지 않는다(추정치 위에 추정치를 쌓는 식). 대신
    # 여기서는: 절대 방위 0/90/180/270 네 방향 모두 벽까지의 거리를 먼저
    # 다 재고(방은 축에 정렬된 상자형이라 이 네 방향이 곧 방의 두 축과
    # 일치함, world/layout.py 격자 구조로 보장), 두 축(0-180축, 90-270축)
    # 각각에서 "두 벽 사이의 정확한 중점"까지 걸을 거리를 계산해서 실행한다.
    def _start_recenter(self, canon, entry_heading=None):
        self.recenter_room = canon
        self.recenter_entry_heading = entry_heading
        self.recenter_phase = "measure"
        self.recenter_measure_idx = 0
        self.recenter_clearances = {}
        self.recenter_move_plan = []
        self._last_action_was_forward = False
        self.state = "RECENTER"
        return self._Action.NO_OP

    def _plan_recenter_moves(self):
        c = self.recenter_clearances
        entry = self.recenter_entry_heading
        plan = []
        for pos, neg in ((0, 180), (90, 270)):
            d_pos = c.get(pos, _RECENTER_MEASURE_CAP_M)
            d_neg = c.get(neg, _RECENTER_MEASURE_CAP_M)
            if entry is not None and entry in (pos, neg):
                # 방금 들어온 문이 있는 축 — 문 쪽(뒤)은 재도 못 믿는다
                # (아직 열려 있는 문틈으로 "옆방까지" 거리가 잡혀서 방금
                # 지나온 문을 진짜 벽으로 오인하는 버그를 ground truth로
                # 실측 확인함, dev_log.md). 대신 "몇 걸음 안 걸어서 아직
                # 그 문 벽 바로 앞이다"라는 사실 자체를 근거로 쓴다 —
                # 정면(entry 방향) 클리어런스의 절반만큼 더 걸으면 된다.
                d_fwd = c.get(entry, _RECENTER_MEASURE_CAP_M)
                dist = min(d_fwd, _RECENTER_MEASURE_CAP_M) / 2.0
                heading = entry
            else:
                pos_open = d_pos >= _RECENTER_MEASURE_CAP_M
                neg_open = d_neg >= _RECENTER_MEASURE_CAP_M
                if pos_open or neg_open:
                    # 이 축의 한쪽(또는 양쪽)이 문/개방부라 반대쪽 벽
                    # 하나만으로는 이 방의 진짜 폭을 알 수 없다 — 방이
                    # 대칭이라고 가정할 근거가 약하므로, 억지로 추정하지
                    # 않고 이 축은 그냥 안 움직인다(안전한 쪽).
                    continue
                # 양쪽 다 실측 벽 — 정확한 중점까지 남은 거리를 바로 계산.
                diff = (d_pos - d_neg) / 2.0
                if abs(diff) < 0.1:
                    continue
                dist, heading = (diff, pos) if diff > 0 else (-diff, neg)
            n_steps = max(1, int(round(dist / geo.FORWARD_STEP)))
            n_steps = min(n_steps, int(_RECENTER_MEASURE_CAP_M / geo.FORWARD_STEP))
            plan.append([heading, n_steps])
        self.recenter_move_plan = plan

    def _step_recenter(self, hud, obs):
        A = self._Action
        # 측정/이동 중 실수로 문을 넘어가면(문 방향을 재는 중일 수도,
        # 이동 중일 수도) 그 자체가 "이 방향에 문이 있다"는 확실한
        # 증거다 — 이전 방 노드에도 기록해둔다(실측 확인된 문제,
        # dev_log.md — 안 그러면 그 방이 서베이를 한 번도 못 받고
        # 나머지 방향이 영원히 unknown으로 남았다).
        canon = self._canonicalize(hud.room_name) if hud.room_name else None
        if canon is not None and canon != self.recenter_room:
            old_node = self.scene.nodes.get(self.recenter_room)
            if old_node is not None and hud.ok and hud.heading is not None:
                cardinal = min(CARDINAL_HEADINGS,
                                key=lambda h: abs(_heading_diff(h, hud.heading)))
                if old_node.exits.get(cardinal) == "unknown":
                    old_node.exits[cardinal] = "open"
                    old_node.exit_leads_to[cardinal] = canon
            return self._enter_room(hud, entry_heading=hud.heading, parent=self.recenter_room)

        if hint_banner_active(obs):
            # 배너 중엔 깊이 측정이 무의미 — 드문 경우라 그냥 잠깐 대기.
            return A.NO_OP

        if self.recenter_phase == "measure":
            headings = (0, 90, 180, 270)
            target = headings[self.recenter_measure_idx]
            diff = _heading_diff(hud.heading, target)
            if abs(diff) > _TURN_TOLERANCE:
                return A.TURN_LEFT if diff > 0 else A.TURN_RIGHT
            clearance = geo.cone_clearance(geo.depth_profile(obs), 0.0, _SEEK_CONE_HALF_DEG)
            self.recenter_clearances[target] = min(clearance, _RECENTER_MEASURE_CAP_M)
            self.recenter_measure_idx += 1
            if self.recenter_measure_idx >= 4:
                self._plan_recenter_moves()
                self.recenter_phase = "move"
            return A.NO_OP

        # phase == "move": 계산해둔 두 축(최대 2개) 이동을 순서대로 실행.
        if not self.recenter_move_plan:
            return self._start_survey_scan()
        target_heading, steps_remaining = self.recenter_move_plan[0]
        diff = _heading_diff(hud.heading, target_heading)
        if abs(diff) > _TURN_TOLERANCE:
            return A.TURN_LEFT if diff > 0 else A.TURN_RIGHT
        if steps_remaining <= 0:
            self.recenter_move_plan.pop(0)
            self._last_action_was_forward = False
            return A.NO_OP
        if self._last_action_was_forward and self.prev_frame is not None:
            if is_blocked(self.prev_frame, obs):
                # 계산한 거리보다 먼저 막힘(가구 등) — 이 축은 여기서
                # 포기하고 다음 축(또는 서베이)으로 넘어간다.
                self.recenter_move_plan.pop(0)
                self._last_action_was_forward = False
                return A.NO_OP
        clearance = geo.cone_clearance(geo.depth_profile(obs), 0.0, _SEEK_CONE_HALF_DEG)
        if clearance < _SEEK_SAFE_CLEARANCE:
            self.recenter_move_plan.pop(0)
            self._last_action_was_forward = False
            return A.NO_OP
        self.recenter_move_plan[0][1] = steps_remaining - 1
        self._last_action_was_forward = True
        return A.MOVE_FORWARD

    def _start_survey_scan(self):
        node = self.scene.nodes[self.recenter_room]
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
        if self.vlm_calls_used < _FLEE_VLM_MAX_CALLS:
            self.vlm_calls_used += 1
            result = classify_heading(obs)
            if result.ok:
                if result.data.get("is_door_or_opening"):
                    node.vlm_door_hint.add(target)
                elif node.exits.get(target) == "unknown":
                    node.exits[target] = "wall"
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
        # SURVEY에서 VLM이 "문처럼 보인다"고 한 방향을 먼저 시도한다 —
        # 문이 아닐 후보를 먼저 물리적으로 부딪혀보며 시간을 낭비하지
        # 않기 위함.
        candidates = sorted(candidates, key=lambda h: h not in node.vlm_door_hint)
        self.target_heading = candidates[0]
        self.seek_origin_room = canon
        self.recover_attempts = 0
        self._need_align = True
        self._last_action_was_forward = False
        self._nav_bias_deg = 0.0
        self._nav_fail_count = 0
        self._nav_action_queue = []
        self._nav_probed_center = False
        self._nav_cornered_count = 0
        self._hint_captured_this_door = False
        self.state = "SEEK"
        return self._Action.NO_OP

    def _step_seek(self, hud, obs):
        canon = self._canonicalize(hud.room_name) if hud.room_name else None
        if canon is not None and canon != self.seek_origin_room:
            return self._on_seek_arrival(hud, canon)

        def on_final_give_up():
            node = self.scene.nodes[self.seek_origin_room]
            node.exits[self.target_heading] = "wall"
            return self._start_seek(self.seek_origin_room, node)

        def on_exhausted():
            return self._escalate_nav_failure(obs, on_final_give_up)

        return self._navigate_step(hud, obs, self.target_heading, on_exhausted)

    # 물리 충돌 확인(is_blocked)만으로 8번 넘게 시도해도 못 뚫으면, 단계적으로
    # 강한 복구를 시도한다(사용자 지시: 벽치기를 반복하다 죽는 문제를 실측
    # 재현해서 확인함, dev_log.md — RETURN 상태에서 같은 자리 각도만 오락
    # 가락하다 70틱 넘게 한 발짝도 못 움직이고 적을 만나 죽었다):
    #   1번째 실패: 짧게 후진(2스텝) 후 처음부터 재정렬 — 접근 각도를 살짝
    #               바꿔서 재시도.
    #   2번째 실패: 옆으로 걸음(90도 틀어 3걸음 이동 후 원래 방향으로 복귀)
    #               — 후진은 "같은 선 위에서 거리만" 벌리지만, 문 앞에서
    #               좌우로 살짝 비켜서 있는 경우(사용자 지시: "문이 내
    #               정중앙 앞에 있어야 똑바로 나갈 수 있다")는 후진만으론
    #               못 고친다. 실측으로 확인된 문제: 진짜 문인데 접근
    #               각도가 안 맞아서 물리적으로 계속 막혀 SEEK이 "벽"으로
    #               잘못 마킹하고 방을 통째로 놓치는 사례가 있었다
    #               (dev_log.md).
    #   3번째 실패: 더 멀리 후진(5스텝)으로 완전히 새 위치에서 재접근.
    #   4번째 실패: VLM에게 "진짜 문/틈이 보이냐"를 한 번 확인 — 보이면 그
    #               방향으로 bias를 주고 후진 후 재시도.
    #   그래도 안 되면: on_final_give_up() 호출(SEEK는 벽으로 마킹하고 다음
    #               후보로, RETURN은 이 목표를 포기하고 한 단계 더 백트랙).
    def _escalate_nav_failure(self, obs, on_final_give_up):
        self._nav_fail_count += 1
        self._nav_bias_deg = 0.0
        self._last_action_was_forward = False
        self.recover_attempts = 0
        A = self._Action
        if self._nav_fail_count == 1:
            self._nav_action_queue = [A.MOVE_BACK, A.MOVE_BACK]
        elif self._nav_fail_count == 2:
            side = A.TURN_LEFT if self._nav_fail_count % 2 == 0 else A.TURN_RIGHT
            back = A.TURN_RIGHT if side == A.TURN_LEFT else A.TURN_LEFT
            self._nav_action_queue = (
                [side] * 6 + [A.MOVE_FORWARD] * 3 + [back] * 6
            )
        elif self._nav_fail_count == 3:
            self._nav_action_queue = [A.MOVE_BACK] * 5
        elif self._nav_fail_count == 4 and self.vlm_calls_used < _FLEE_VLM_MAX_CALLS:
            self.vlm_calls_used += 1
            result = locate_door(obs)
            if result.ok and result.data.get("door_or_opening_visible"):
                bearing = result.data.get("bearing")
                self._nav_bias_deg = {"left": 25.0, "right": -25.0}.get(bearing, 0.0)
                self._nav_action_queue = [A.MOVE_BACK, A.MOVE_BACK]
            else:
                return on_final_give_up()
        else:
            return on_final_give_up()
        return self._nav_action_queue.pop(0)

    # 목표 헤딩으로 정렬 -> 정면이 막히면 옆(가까운 오프셋 우선)으로 비켜
    # 지나가기 -> 지나갔으면 원래 목표로 복귀, 를 매 틱 반복하는 공용
    # 이동 로직. SEEK/RETURN 둘 다 씀(사용자 지시: 나무 같은 국지 장애물은
    # "경계가 있다고 보고 우회"하되, 지나가면 원래 가려던 방향으로 되돌아
    # 오길 원함 — contourner).
    def _navigate_step(self, hud, obs, target_heading, on_exhausted):
        # 단계적 복구(_escalate_nav_failure) 중 미리 정해둔 액션 시퀀스가
        # 남아있으면 그것부터 소진한다 — 정렬/전진 판단보다 우선.
        if self._nav_action_queue:
            self._last_action_was_forward = False
            return self._nav_action_queue.pop(0)

        effective_target = (target_heading + self._nav_bias_deg) % 360
        diff = _heading_diff(hud.heading, effective_target)
        if abs(diff) > _TURN_TOLERANCE:
            self._last_action_was_forward = False
            self._nav_probed_center = False
            return self._Action.TURN_LEFT if diff > 0 else self._Action.TURN_RIGHT

        # depth profile로 미리 피하는 건 어디까지나 "부딪히는 모양새를
        # 줄이는" 보조 수단이고, 실제로 움직였는지의 최종 판정은 항상
        # is_blocked(프레임 diff)로 한다 — 실측으로 확인된 버그: 벽에 걸린
        # 그림 앞에서 depth profile이 "3.43m 뚫려있다"고 계속 우겨서(벽 색
        # 채도가 임계값 근처라 경계 판정이 살짝 어긋남), 실제로는 완전히
        # 제자리인데(위치 좌표까지 확인함, dev_log.md) recover_attempts가
        # 한 번도 안 올라가고 400스텝 넘게 MOVE_FORWARD만 반복했다. depth가
        # "괜찮다"고 해도 직전 전진이 실제로 안 먹혔으면 무조건 막힌 걸로
        # 취급한다.
        if self._last_action_was_forward and self.prev_frame is not None:
            if is_blocked(self.prev_frame, obs):
                self.recover_attempts += 1
                if self.recover_attempts > _RECOVER_MAX_ATTEMPTS:
                    self._nav_bias_deg = 0.0
                    return on_exhausted()
                # 회복 회전도 sidestep과 같은 bias로 취급한다 — 그냥
                # TURN_RIGHT만 반환하면 다음 틱에 맨 위 정렬 체크가 "목표랑
                # 안 맞네" 하고 바로 되돌려버려서 recover_attempts가 절대
                # 못 쌓이는 버그가 있었다(실측 확인, dev_log.md). bias를
                # 같이 옮겨서 이 새 방향을 "당분간의 목표"로 인정해야
                # 실제로 그 방향을 테스트해볼 기회가 생긴다.
                self._nav_bias_deg = geo.wrap180(self._nav_bias_deg - 15.0)
                self._last_action_was_forward = False
                return self._Action.TURN_RIGHT
            # 직전 전진이 실제로 먹혔음 — 회복 카운트 리셋, bias도 원래
            # 목표 쪽으로 15도만 되돌린다(사용자 지시: 장애물 지나가면
            # 원래 방향으로 복귀). depth profile 예측만 보고 한 틱만에
            # bias를 통째로 0으로 되돌렸다가, 실제로는 그 자리에서 못
            # 움직였는데 목표 헤딩으로 즉시 재정렬 → 다시 막힘 → 또 되돌림
            # ...으로 무한 진동한 버그가 있었다(실측 확인, dev_log.md).
            # "실제로 한 걸음 전진에 성공했을 때만" 15도씩 서서히 되돌리면
            # 이 문제가 없다.
            self.recover_attempts = 0
            if self._nav_bias_deg > 0:
                self._nav_bias_deg = max(0.0, self._nav_bias_deg - 15.0)
            elif self._nav_bias_deg < 0:
                self._nav_bias_deg = min(0.0, self._nav_bias_deg + 15.0)

        if hint_banner_active(obs):
            # 배너 중엔 깊이 측정이 무의미 — 위 is_blocked 판정만으로 진행.
            self._last_action_was_forward = True
            return self._Action.MOVE_FORWARD

        profile = geo.depth_profile(obs)
        # 정면 안전 체크는 원래 검증된 15도 콘(_SEEK_CONE_HALF_DEG) 그대로
        # 유지 — sidestep용 8도 콘을 여기 재사용했다가 실측으로 버그를
        # 발견했다: 좁은 콘은 실제 에이전트 폭(AGENT_RADIUS 0.4m)보다 좁은
        # 틈도 "뚫림"으로 오판했다(dev_log.md).
        center_clear = geo.cone_clearance(
            profile, 0.0, _SEEK_CONE_HALF_DEG) >= _SEEK_SAFE_CLEARANCE
        # 실측으로 새로 발견한 버그(dev_log.md): 문/통로를 몇 m 떨어져서
        # 정면으로 바라볼 때, 문 폭이 15도 콘보다 좁으면 콘 양옆이 문틀
        # 옆 벽을 걸치고, floor_boundary()가 "바닥이 화면 맨 아래까지 안
        # 보이면 아주 가까운 장애물"로 간주하는 휴리스틱(NEAR_RANGE 폴백)
        # 때문에 그 벽까지의 실제 거리와 무관하게 0.6m로 뭉개진다 — 그
        # 결과 VLM이 방금 "문 맞다"고 확인해준 방향인데도 전진을 단 한
        # 번도 시도 안 하고 곧장 막힌 걸로 포기해버렸다. depth profile은
        # 어디까지나 "미리 피하는" 보조 신호이지 전진 자체를 막는
        # 근거여선 안 된다 — 이 방향으로 정렬한 뒤 아직 한 번도 실제로
        # 전진해본 적이 없다면, depth가 뭐라 하든 일단 한 번은 실제로
        # 시도해서 is_blocked()로 물리 확인한다(그래도 진짜 막혀 있으면
        # 바로 다음 틱에 정상적으로 recover_attempts가 올라가 기존 복구
        # 로직으로 이어진다 — 안전망은 그대로 유지됨).
        if center_clear or not self._nav_probed_center:
            # bias를 원래 목표로 되돌리는 건 "실제로 전진에 성공했을 때만"
            # 위(is_blocked 확인 지점)에서 서서히 한다 — 여기서 depth
            # 예측만 보고 되돌렸다가 무한 진동한 버그가 있었다(위 주석 참고).
            self._last_action_was_forward = True
            self._nav_probed_center = True
            return self._Action.MOVE_FORWARD

        # 정면이 막힘 — 옆으로 비켜갈 수 있는지 확인. 매 틱 오프셋을
        # 처음부터 다시 스캔하면, 이미 한쪽으로 틀어놓은 상태에서 다음
        # 틱에 반대쪽이 근소하게 더 넓게 측정되면 곧바로 반대로 확 틀어
        # 버려서 좌우로 계속 뒤집는 게 빙글빙글 도는 것처럼 보였다(사용자
        # 피드백: "정지하고 생각하고 움직이라는거야"). 이미 기운 bias가
        # 있으면 같은 쪽을 먼저 본다 — 반대쪽은 같은 쪽이 전부 막혔을
        # 때만 시도한다.
        offsets = _SIDESTEP_OFFSETS_DEG[1:]
        if self._nav_bias_deg > 0:
            offsets = sorted(offsets, key=lambda o: (o <= 0, abs(o)))
        elif self._nav_bias_deg < 0:
            offsets = sorted(offsets, key=lambda o: (o >= 0, abs(o)))
        for off in offsets:
            clearance = geo.cone_clearance(profile, off, _SIDESTEP_CONE_HALF_DEG)
            if clearance >= _SEEK_SAFE_CLEARANCE:
                self._nav_bias_deg = geo.wrap180(self._nav_bias_deg + off)
                self._last_action_was_forward = False
                return self._Action.TURN_LEFT if off > 0 else self._Action.TURN_RIGHT

        # 정면 + 좌우(±15,±30) 전부 막힘 = 구석에 몰린 상태(실측 확인:
        # 이 경우 15도씩 찔끔찔끔 돌면서 뚫린 틈을 찾을 때까지 방 안을
        # 몇 백 틱씩 헤매는 것처럼 보였다 — 사용자 피드백: "이상하게
        # 오른쪽으로 돌고... 벽에 계속 붙히쳐서". 실측으로 확인해보니 이
        # 막힘의 정체가 실제로 "잠긴 문"(자물쇠 판)인 사례가 있었다 —
        # 열쇠 없인 몇 번을 재시도해도 못 지나가므로, 자물쇠가 보이면
        # 회전/재시도로 시간 낭비 말고 곧장 포기한다. 다만 색상 휴리스틱
        # (lock_visible)만으로 exits에 "locked"를 확정하진 않는다 — 오탐
        # 위험이 있어서다. 대신 SEEK을 잠깐 멈추고(HINT_CAPTURE) 문
        # 터치 반경 안까지 다가가 힌트 배너(hint_banner_active, 100%
        # 신뢰 가능)가 실제로 뜨는지로 최종 확인한다 — 뜨면 "locked"로
        # 확정하고 힌트 텍스트도 같이 챙긴다(사용자 지시: 문 앞까지 가서
        # 힌트 받고 원래 자리로 복귀). 같은 문에서 이미 한 번 시도했으면
        # (_hint_captured_this_door) 재시도 없이 바로 포기한다.
        if geo.lock_visible(obs):
            if self.state == "SEEK" and not self._hint_captured_this_door:
                return self._start_hint_capture()
            self._nav_bias_deg = 0.0
            return on_exhausted()

        # 자물쇠가 아니면 사용자 지시대로 조금씩 돌지 말고 곧장 180도
        # (뒤돌기)로 반응한다 — 최소한 지금 막힌 구석에서는 확실히
        # 벗어나는 방향이라 다음 판단이 훨씬 빨리(적은 틱으로) 정리된다.
        self.recover_attempts += 1
        # 실측으로 새로 발견한 무한루프(dev_log.md): 180도로 등 돌려
        # 방 안을 크게 돌다 보면 그 도중에 "실제로 전진 성공"이 여러 번
        # 나오는데, 그때마다 recover_attempts가 0으로 리셋된다(바로 위
        # is_blocked 분기의 설계 의도 — 나무 같은 국지 장애물을 지나가면
        # 카운트를 지워야 하니까). 문제는 door_visible=True인데
        # lock_visible=False인 좁은 문(자물쇠는 아니지만 15도 안전 콘보다
        # 좁아 depth가 계속 "막힘"으로 오판하는 경우) 앞에서는, 방을 크게
        # 돌아 결국 bias가 다시 0으로 수렴해 같은 문에 재접근 →
        # 재충돌 → 다시 180도 회전, 을 recover_attempts가 한 번도 8을
        # 못 넘긴 채 영원히 반복한다(실측: 600틱 넘게 같은 문 주변을
        # 맴돎, 방 전환 전혀 없음). 그래서 "이번 목표 헤딩에서 통째로
        # 몇 번이나 이 180도-포기를 겪었는지"는 중간의 성공적 전진으로
        # 리셋되지 않는 별도 카운터로 센다.
        self._nav_cornered_count += 1
        if (self.recover_attempts > _RECOVER_MAX_ATTEMPTS
                or self._nav_cornered_count > _MAX_CORNERED_PER_TARGET):
            self._nav_bias_deg = 0.0
            return on_exhausted()
        self._nav_bias_deg = geo.wrap180(self._nav_bias_deg + 180.0)
        self._last_action_was_forward = False
        return self._Action.TURN_RIGHT

    # --- HINT_CAPTURE: 잠긴 문 쪽으로 다가가 힌트 배너를 읽고 물러남 -----
    def _start_hint_capture(self):
        self._hint_capture_phase = "approach"
        self._hint_capture_forward_steps = 0
        self._hint_capture_confirmed_locked = False
        self._hint_captured_this_door = True  # 성공 여부와 무관하게 한 번만 시도
        self._hint_capture_return_state = self.state  # 보통 "SEEK"
        self.state = "HINT_CAPTURE"
        # HINT_CAPTURE는 정체 감지 대상에서 빠지므로, 복귀 후 엉뚱한
        # 과거 스트릭이 이어지지 않게 여기서 리셋해둔다.
        self._stuck_streak = 0
        self._stuck_last_action = None
        return self._Action.NO_OP

    def _step_hint_capture(self, hud, obs):
        A = self._Action
        if self._hint_capture_phase == "approach":
            if hint_banner_active(obs):
                self._hint_capture_confirmed_locked = True
                # 배너 텍스트는 처음 확보한 것만 채택 — 문이 여러 개라도
                # 첫 힌트가 유효하다(재방문 시 다시 덮어쓰지 않음).
                if self.scene.key_hint_text is None:
                    text = read_hint_text(obs)
                    if text:
                        self.scene.key_hint_text = text
                        self.scene.key_hint_room = self.seek_origin_room
                        self.scene.key_hint_heading = self.target_heading
                self._hint_capture_retreat_remaining = self._hint_capture_forward_steps
                self._hint_capture_phase = "retreat"
                return A.MOVE_BACK if self._hint_capture_forward_steps > 0 else self._finish_hint_capture()
            if self._hint_capture_forward_steps >= _HINT_CAPTURE_MAX_APPROACH_TICKS:
                # 터치 반경에 못 들어갔다(색 오탐이었거나 이미 열쇠를 들고
                # 있어 배너 대신 바로 열렸을 수도 있음) — 포기하고 후퇴.
                self._hint_capture_retreat_remaining = self._hint_capture_forward_steps
                self._hint_capture_phase = "retreat"
                return (A.MOVE_BACK if self._hint_capture_forward_steps > 0
                        else self._finish_hint_capture())
            self._hint_capture_forward_steps += 1
            return A.MOVE_FORWARD

        # retreat
        if hint_banner_active(obs):
            self._hint_capture_confirmed_locked = True
            if self.scene.key_hint_text is None:
                text = read_hint_text(obs)
                if text:
                    self.scene.key_hint_text = text
                    self.scene.key_hint_room = self.seek_origin_room
                    self.scene.key_hint_heading = self.target_heading
        self._hint_capture_retreat_remaining -= 1
        if self._hint_capture_retreat_remaining > 0:
            return A.MOVE_BACK
        return self._finish_hint_capture()

    def _finish_hint_capture(self):
        node = self.scene.nodes[self.seek_origin_room]
        node.exits[self.target_heading] = (
            "locked" if self._hint_capture_confirmed_locked else "wall"
        )
        self._hint_capture_phase = None
        self.state = self._hint_capture_return_state or "SEEK"
        return self._start_seek(self.seek_origin_room, node)

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
            self._nav_bias_deg = 0.0
            self._nav_fail_count = 0
            self._nav_action_queue = []
            self._nav_probed_center = False
            self._nav_cornered_count = 0
            self.state = "RETURN"
            return self._Action.NO_OP

        # 새 방 발견!
        return self._enter_room(hud, entry_heading=self.target_heading,
                                 parent=self.seek_origin_room)

    # --- RETURN: 특정 방으로 되돌아가기(백트랙 겸용) ----------------------
    def _start_backtrack(self, canon):
        # 실측으로 발견한 버그(dev_log.md): 예전엔 여기서 곧바로
        # stack.pop()을 했는데, 그러면 "부모 방으로 실제로 돌아가는 데
        # 성공했다"는 확인 전에 스택이 이미 줄어든다. 되돌아가는 길이
        # 막혀서 도중에 포기하면(_step_return의 on_final_give_up) 지금
        # 물리적으로 있는 방 기준으로 _start_seek을 다시 부르는데, 그
        # 방이 이미 done이면 _start_seek이 _start_backtrack을 다시 불러
        # 스택을 한 번 더 pop한다 — 즉 실제로는 한 번만 떠났는데 스택은
        # 두 번 줄어드는 "이중 pop"이 나서, 아직 안 가본 조상 방(예:
        # 스폰 방의 나머지 방향들)이 통째로 건너뛰어지고 스택이 예정보다
        # 일찍 텅 비어 DONE으로 빠졌다(실측: 8~12개 방 중 5개만 방문하고
        # 스폰 방에 unknown 방향이 3개나 남은 채 종료). 그래서 pop은
        # "부모 방에 실제로 도착"을 _step_return이 확인한 순간에만
        # 하도록 미룬다 — 도중에 실패해도 스택이 그대로라 재시도가 항상
        # 안전(멱등)하다.
        if not self.scene.stack:
            self.state = "DONE"
            return self._Action.NO_OP
        if len(self.scene.stack) == 1:
            self.scene.stack.pop()
            self.state = "DONE"
            return self._Action.NO_OP
        parent_name, _parent_entry = self.scene.stack[-2]
        node = self.scene.nodes[canon]
        self._stack_pop_pending = canon
        self.return_target_room = parent_name
        self.target_heading = (node.entry_heading + 180) % 360
        self.recover_attempts = 0
        self._need_align = True
        self._last_action_was_forward = False
        self._nav_bias_deg = 0.0
        self._nav_fail_count = 0
        self._nav_action_queue = []
        self._nav_probed_center = False
        self._nav_cornered_count = 0
        self.state = "RETURN"
        return self._Action.NO_OP

    def _step_return(self, hud, obs):
        canon = self._canonicalize(hud.room_name) if hud.room_name else None
        if canon == self.return_target_room:
            if (self._stack_pop_pending is not None
                    and self.scene.stack
                    and self.scene.stack[-1][0] == self._stack_pop_pending):
                self.scene.stack.pop()
            self._stack_pop_pending = None
            node = self.scene.nodes[canon]
            if node.done:
                return self._start_backtrack(canon)
            if node.wall_color is None:
                # RECENTER 중 실수로 문을 넘어가서 이 방이 서베이(벽색
                # 기록)를 한 번도 못 받은 경우 — 되돌아온 김에 지금 마저
                # 받는다(실측 확인된 문제, dev_log.md 참고). 지금 서있는
                # 위치가 방 중앙일 리 없으므로(RETURN으로 막 도착한
                # 지점) 곧장 서베이로 가지 않고 RECENTER부터 다시 거친다
                # (360도 서베이는 반드시 방 중앙에서만 — 사용자 지시).
                return self._start_recenter(canon)
            return self._start_seek(canon, node)

        # 실측으로 발견한 버그(dev_log.md): RETURN이 장애물을 피하다
        # 엉뚱한 문(예: 방금 나온 방으로 되돌아가는 문)으로 잘못 들어가면,
        # 시작할 때 한 번 계산해둔 target_heading(그때 있던 방 기준)이
        # 지금 물리적으로 있는 방과 안 맞아 방향을 잃고 두 방 사이를
        # 몇백 틱씩 맴돌았다(실측: seed=7에서 Coral Vault<->Ivory Library
        # 사이 500틱+ 정체). 매 틱 SceneGraph.find_path로 "지금 있는
        # 방 -> 최종 목표 방"의 다음 홉을 다시 계산해서, 엉뚱한 방으로
        # 새도 스스로 교정한다 — 이미 다 파악된 방들 사이의 작은 그래프
        # BFS라 비용이 거의 없다.
        if canon is not None and canon in self.scene.nodes:
            path = self.scene.find_path(canon, self.return_target_room)
            if path:
                self.target_heading = path[0][1]

        def on_final_give_up():
            # 여러 번 단계적으로 시도해도 원래 뚫려있어야 할 길을 못
            # 찾으면(사용자 지시로 재현/확인된 버그: 같은 자리에서 각도만
            # 오락가락하며 70틱 넘게 한 발짝도 못 움직이다 적을 만나 죽음,
            # dev_log.md) 이 목표는 포기한다. _start_backtrack을 직접 다시
            # 부르면 이미 팝된 스택을 또 팝할 위험이 있어서(백트랙 도중
            # 막힌 경우), 대신 지금 있는 방 기준으로 _start_seek을 새로
            # 호출한다 — 안 가본 다른 방향이 있으면 그쪽을, 없으면
            # _start_seek이 알아서 정상적으로 백트랙한다(같은, 검증된
            # 경로라 스택이 꼬이지 않음).
            if canon is not None and canon in self.scene.nodes:
                return self._start_seek(canon, self.scene.nodes[canon])
            self.recover_attempts = 0
            self._need_align = True
            self._last_action_was_forward = False
            self._nav_probed_center = False
            self._nav_cornered_count = 0
            return self._Action.MOVE_BACK

        def on_exhausted():
            return self._escalate_nav_failure(obs, on_final_give_up)

        return self._navigate_step(hud, obs, self.target_heading, on_exhausted)

    # --- FLEE: 맞은 자리에서 바로 공격(strike) — CV 조준, 실패 시 sweep ---
    def _step_flee(self, hud, obs):
        if self.flee_phase == "strike":
            return self._step_flee_strike(hud, obs)
        return self._resume_after_flee()

    def _current_wall_rgb(self, hud):
        node = self.scene.nodes.get(self._canonicalize(hud.room_name)) if hud.room_name else None
        if node is None or node.wall_color is None:
            return None
        return _WALL_RGB_BY_NAME.get(node.wall_color)

    def _step_flee_strike(self, hud, obs):
        # 후진은 거리를 못 벌린다는 게 실측 확인됐으므로(위 상단 주석 참고)
        # 맞은 그 자리에서 바로 대응한다. 1순위는 agent.enemies의 순수 CV
        # 탐지(~1~2ms, 왕복 지연 없음, 정확한 도 단위 방위) — 콘·사거리 안이면
        # 바로 공격, 아니면 정확한 방위로 회전. 못 찾으면 그 순간부터 결정론적
        # sweep(공격 2번 → 15도 회전, 24번 = 360도)으로 전환 — 공격 콘(±20도)
        # 이 회전 간격(15도)보다 넓어서 기하학적으로 반드시 걸린다.
        #
        # VLM(locate_enemy)은 조준 결정에서 뺐다 — 실측으로 확인된 버그:
        # VLM의 "왼쪽/가운데/오른쪽" 3단계 판정이 실제 공격 콘(±20도)보다
        # 훨씬 좁은 기준으로 "가운데"를 매겨서, 콘 안에 이미 들어와 있는데도
        # "가운데 아님"으로 계속 오판 → 두 헤딩 사이를 76틱 넘게 왕복하며
        # 한 번도 공격 못 하고 계속 맞기만 하는 걸 확인함(dev_log.md). CV가
        # 정확한 도 단위 방위를 주므로 이 모호함이 없고, 못 찾을 때는 VLM
        # 판정을 더 기다리는 대신 곧장 100% 보장되는 sweep으로 넘어간다.
        # 한 번 sweep으로 전환하면 중간에 또 맞아도(step()에서 이미 FLEE
        # 중이면 리셋 안 함) 처음부터 다시 시작하지 않고 이어서 진행한다.
        self.flee_ticks += 1
        if self.flee_ticks > _FLEE_STRIKE_MAX_TICKS:
            return self._resume_after_flee()

        if not self.flee_sweep_active:
            # 1순위: 근접 전용 방위 추정(enemies.close_range_bearing). 실측
            # 확인(dev_log.md, seed=7): strike 진입 시점엔 적이 이미
            # 1.3~1.5m 코앞이라 몸통이 화면을 거의 다 채워서, detect()의
            # 사람형 비율/바닥접점 판정이 구조적으로 0개만 반환하고 매
            # 전투가 곧장 24방향 sweep(최대 72틱)으로 빠졌다 — "예전처럼
            # 즉각적으로 안 죽인다"는 원인. close_range_bearing은 사람형
            # 판정 없이 "벽도 바닥도 아닌 큰 덩어리"만 보므로 이 거리에서도
            # 먹힌다(이미 HP가 깎여 진입한 상태라 오탐 위험도 낮음).
            # 아직 한 번도 못 맞춘 시점(flee_ever_attacked=False)엔 "죽여서
            # 사라진 자리를 계속 공격" 오탐이 원천적으로 불가능하므로
            # 발밑 조건(require_bottom_band)을 꺼서 문틈 사이로 비스듬히
            # 보이는 적도 더 적극적으로 잡는다(실측으로 확인된 두 번째
            # 교전 놓침 사례, dev_log.md).
            bearing = EN.close_range_bearing(
                obs, wall_rgb=self._current_wall_rgb(hud),
                require_bottom_band=self.flee_ever_attacked)
            if bearing is not None and self.flee_fastpath_attack_streak < _CLOSE_RANGE_ATTACK_STREAK_MAX:
                if abs(bearing) <= _ATTACK_CONE_HALF_DEG:
                    self.flee_ever_attacked = True
                    self.flee_fastpath_attack_streak += 1
                    return self._Action.ATTACK
                return self._Action.TURN_LEFT if bearing > 0 else self._Action.TURN_RIGHT

            # 2순위: 중간 거리용 사람형 탐지(위 1순위가 실패한 경우 —
            # 예: 적이 아직 근접하기 전, 또는 방금 물러난 경우).
            mobs = EN.detect(obs, wall_rgb=self._current_wall_rgb(hud))
            best = next((m for m in mobs if m.score >= _MOB_MIN_SCORE
                         and m.distance <= _MOB_MAX_COMBAT_DIST_M), None)
            if best is not None and self.flee_fastpath_attack_streak < _FASTPATH_ATTACK_STREAK_MAX:
                if abs(best.bearing) <= _ATTACK_CONE_HALF_DEG:
                    if best.distance <= _ATTACK_RANGE_M:
                        self.flee_ever_attacked = True
                        self.flee_fastpath_attack_streak += 1
                        return self._Action.ATTACK
                    # 방향(콘)은 이미 맞는데 사거리(3.0m) 밖 — 실측으로
                    # 확인된 버그: 이 경우도 "정렬 안 됨"으로 보고 회전만
                    # 시켰더니, 회전은 거리를 못 좁히니 bearing 부호가
                    # 살짝씩 뒤집히며 좌우로 영원히 진동만 하고 한 번도
                    # 공격을 못 했다(dev_log.md). 전진해서 거리부터 좁힌다.
                    self._last_action_was_forward = True
                    return self._Action.MOVE_FORWARD
                return self._Action.TURN_LEFT if best.bearing > 0 else self._Action.TURN_RIGHT

            # 둘 다 못 찾으면 곧장 sweep으로(VLM 조준은 안 씀 — 위 주석 참고).
            self.flee_sweep_active = True
            self.flee_sweep_headings_done = 0
            self.flee_sweep_attacks_done = 0
            # 이미 근접/중간거리 탐지로 명중까지 시켰던 상태에서 갑자기
            # 못 찾은 거라면(위 _FLEE_SWEEP_QUICK_CHECK_HEADINGS 주석
            # 참고) 죽어서 사라졌을 확률이 높으니 짧게만 확인한다.
            self.flee_sweep_heading_limit = (
                _FLEE_SWEEP_QUICK_CHECK_HEADINGS if self.flee_ever_attacked
                else _FLEE_SWEEP_HEADINGS
            )

        # --- 결정론적 sweep: 공격 N번 → 15도 회전, 24번(또는 명중 후
        # 놓친 경우는 짧게) 반복 ---
        if self.flee_sweep_headings_done >= self.flee_sweep_heading_limit:
            return self._resume_after_flee()
        if self.flee_sweep_attacks_done < _FLEE_SWEEP_ATTACKS_PER_HEADING:
            self.flee_sweep_attacks_done += 1
            self.flee_ever_attacked = True
            return self._Action.ATTACK
        self.flee_sweep_attacks_done = 0
        self.flee_sweep_headings_done += 1

        # 이미 한 번이라도 맞춘 뒤라면, 몇 방향마다 VLM으로 "아직도 있는지"
        # 재확인해서 죽은 걸 계속 헛공격하며 sweep을 다 도는 낭비를 줄인다.
        # CV(공짜)로 먼저 보고, CV도 못 찾으면 VLM으로 한 번 더 확인.
        # 단, 최소 절반(180도)은 훑어본 뒤에만 이 조기 포기를 허용한다 —
        # 실측으로 확인된 버그(dev_log.md): flee_ever_attacked는 "ATTACK을
        # 내긴 했다"는 뜻일 뿐 "진짜 맞혔다"는 보장이 없다(오탐 대상을
        # 공격했을 수도 있음). 그 상태에서 적이 애초에 시야 밖(예: 뒤쪽)에
        # 있었다면, 몇 방향 안 돌아본 시점에 "안 보이니 죽었나보다"라고
        # 성급하게 포기해서 진짜 적을 아예 못 찾고 sweep을 접는 일이
        # 있었다. 최소 절반은 돌아본 뒤에야 이 판단을 신뢰한다.
        if (self.flee_ever_attacked
                and self.flee_sweep_headings_done >= _FLEE_SWEEP_HEADINGS // 2
                and self.flee_sweep_headings_done % _FLEE_SWEEP_RECHECK_EVERY == 0):
            mobs = EN.detect(obs, wall_rgb=self._current_wall_rgb(hud))
            if not any(m.score >= _MOB_MIN_SCORE and m.distance <= _MOB_MAX_COMBAT_DIST_M
                       for m in mobs):
                if self.vlm_calls_used < _FLEE_VLM_MAX_CALLS:
                    self.vlm_calls_used += 1
                    result = locate_enemy(obs)
                    if result.ok and not result.data.get("enemy_visible"):
                        return self._resume_after_flee()

        return self._Action.TURN_RIGHT

    # --- VLM_RECOVER: 정체(같은 액션 반복 + 화면 안 바뀜) 시 VLM 조종 인계 ---
    def _start_vlm_recover(self, obs):
        self.pre_recover_state = self.state
        self.pre_recover_target = self.target_heading
        self.state = "VLM_RECOVER"
        self._vlm_recover_ticks = 0
        self._stuck_streak = 0
        self._stuck_last_action = None
        return self._step_vlm_recover(obs)

    def _step_vlm_recover(self, obs):
        self._vlm_recover_ticks += 1
        if self._vlm_recover_ticks > _VLM_RECOVER_MAX_TICKS:
            return self._end_vlm_recover()
        if self.vlm_calls_used >= _FLEE_VLM_MAX_CALLS:
            return self._end_vlm_recover()
        self.vlm_calls_used += 1
        result = recover_action(obs)
        if not result.ok:
            return self._end_vlm_recover()
        if result.data.get("reached_open_area"):
            return self._end_vlm_recover()
        name = result.data.get("action")
        action_map = {
            "turn_left": self._Action.TURN_LEFT,
            "turn_right": self._Action.TURN_RIGHT,
            "move_forward": self._Action.MOVE_FORWARD,
            "move_back": self._Action.MOVE_BACK,
        }
        action = action_map.get(name)
        if action is None:
            return self._end_vlm_recover()
        self._last_action_was_forward = (action == self._Action.MOVE_FORWARD)
        return action

    def _end_vlm_recover(self):
        self.state = self.pre_recover_state or "SURVEY"
        self.target_heading = self.pre_recover_target
        self._need_align = True
        self._last_action_was_forward = False
        self._nav_bias_deg = 0.0
        self._nav_fail_count = 0
        self._nav_action_queue = []
        self._nav_probed_center = False
        self._nav_cornered_count = 0
        self.recover_attempts = 0
        self._stuck_room_ticks = 0  # 룰베이스에 다시 온전한 기회를 준다
        return self._Action.NO_OP

    # --- DONE: 더 갈 새 방 없음. 힌트가 있으면 열쇠 찾기를 한 번 시도 ---
    def _step_done(self, hud, obs):
        if (not self.key_hunt_attempted and self.scene.key_hint_text
                and hud.ok and not hud.has_key):
            self.key_hunt_attempted = True
            target = _match_hint_to_room(self.scene.key_hint_text, self.scene)
            self.scene.key_target_room = target
            canon = self._canonicalize(hud.room_name) if hud.room_name else None
            if target is not None and canon is not None and target != canon:
                return self._start_goto_room(target, "key_room", canon)
            if target is not None and canon == target:
                return self._start_key_sweep()
            # 매칭 실패 — 지금 가진 정보(방 색/이름)로는 못 찾음. 포기.
        return self._Action.TURN_RIGHT  # 제자리 대기

    # --- GOTO_ROOM: SceneGraph.find_path로 매 틱 다음 홉을 다시 계산하며 이동 ---
    def _start_goto_room(self, target_room, purpose, from_room):
        self.goto_target_room = target_room
        self.goto_purpose = purpose
        self.recover_attempts = 0
        self._need_align = True
        self._last_action_was_forward = False
        self._nav_bias_deg = 0.0
        self._nav_fail_count = 0
        self._nav_action_queue = []
        self._nav_probed_center = False
        self._nav_cornered_count = 0
        path = self.scene.find_path(from_room, target_room)
        if not path:
            return self._end_key_hunt()
        self.target_heading = path[0][1]
        self.state = "GOTO_ROOM"
        return self._Action.NO_OP

    def _step_goto_room(self, hud, obs):
        canon = self._canonicalize(hud.room_name) if hud.room_name else None
        if canon == self.goto_target_room:
            if self.goto_purpose == "key_room":
                return self._start_key_sweep()
            if self.goto_purpose == "locked_door":
                return self._start_key_unlock_approach()
            return self._end_key_hunt()

        if canon is not None and canon in self.scene.nodes:
            path = self.scene.find_path(canon, self.goto_target_room)
            if path:
                self.target_heading = path[0][1]
            else:
                return self._end_key_hunt()

        def on_final_give_up():
            return self._end_key_hunt()

        def on_exhausted():
            return self._escalate_nav_failure(obs, on_final_give_up)

        return self._navigate_step(hud, obs, self.target_heading, on_exhausted)

    # --- KEY_SWEEP: 열쇠 방에 도착 후, 십자 패턴으로 걸어보며 0.8m 픽업 유도 ---
    def _start_key_sweep(self):
        self.key_sweep_idx = 0
        self.key_sweep_steps_done = 0
        self._need_align = True
        self._last_action_was_forward = False
        self.state = "KEY_SWEEP"
        return self._Action.NO_OP

    def _step_key_sweep(self, hud, obs):
        if hud.ok and hud.has_key:
            locked_room = self.scene.key_hint_room
            if locked_room is not None and locked_room in self.scene.nodes:
                canon = self._canonicalize(hud.room_name) if hud.room_name else None
                return self._start_goto_room(locked_room, "locked_door", canon)
            return self._end_key_hunt()
        if self.key_sweep_idx >= len(_KEY_SWEEP_HEADINGS_OFFSET):
            return self._end_key_hunt()  # 이 방엔 없었다 — 포기

        target = _KEY_SWEEP_HEADINGS_OFFSET[self.key_sweep_idx]
        diff = _heading_diff(hud.heading, target) if hud.ok else 0
        if abs(diff) > _TURN_TOLERANCE:
            return self._Action.TURN_LEFT if diff > 0 else self._Action.TURN_RIGHT
        if self.key_sweep_steps_done >= _KEY_SWEEP_STEPS_PER_LEG:
            self.key_sweep_idx += 1
            self.key_sweep_steps_done = 0
            return self._Action.NO_OP
        profile = geo.depth_profile(obs)
        if geo.cone_clearance(profile, 0.0, _SEEK_CONE_HALF_DEG) >= _SEEK_SAFE_CLEARANCE:
            self.key_sweep_steps_done += 1
            self._last_action_was_forward = True
            return self._Action.MOVE_FORWARD
        # 이 방향은 막힘 — 그만 걷고 다음 방향으로.
        self.key_sweep_idx += 1
        self.key_sweep_steps_done = 0
        return self._Action.NO_OP

    # --- KEY_UNLOCK: 잠긴 문 방향으로 접근 — 터치 반경(1.5m) 안에서 자동 해제 ---
    def _start_key_unlock_approach(self):
        self.target_heading = self.scene.key_hint_heading
        self.recover_attempts = 0
        self._need_align = True
        self._last_action_was_forward = False
        self._nav_bias_deg = 0.0
        self._nav_fail_count = 0
        self._nav_action_queue = []
        self._nav_probed_center = False
        self._nav_cornered_count = 0
        self.state = "KEY_UNLOCK"
        return self._Action.NO_OP

    def _step_key_unlock(self, hud, obs):
        canon = self._canonicalize(hud.room_name) if hud.room_name else None
        if canon is not None and canon != self.scene.key_hint_room:
            # 문 통과 성공 — 새 방 발견 처리(잠긴 문 쪽 exits는 이미
            # "locked"로 기록돼 있으니 그대로 두고, 새 방을 정상 등록).
            self.scene.door_unlocked = True
            return self._enter_room(hud, entry_heading=self.target_heading,
                                     parent=self.scene.key_hint_room)

        def on_final_give_up():
            return self._end_key_hunt()

        def on_exhausted():
            return self._escalate_nav_failure(obs, on_final_give_up)

        return self._navigate_step(hud, obs, self.target_heading, on_exhausted)

    def _end_key_hunt(self):
        self.state = "DONE"
        return self._Action.TURN_RIGHT

    def _resume_after_flee(self):
        self.flee_phase = None
        self.flee_sweep_active = False
        self.flee_sweep_headings_done = 0
        self.flee_sweep_attacks_done = 0
        self.flee_sweep_heading_limit = _FLEE_SWEEP_HEADINGS
        self.flee_fastpath_attack_streak = 0
        self.state = self.pre_flee_state or "SURVEY"
        self.target_heading = self.pre_flee_target
        self._need_align = True
        self._last_action_was_forward = False
        # 전투 전에 국지 장애물을 피하던 중이었다면 그 bias가 남아있는데,
        # 리셋 안 하면 원래 목표 헤딩(pre_flee_target)이 아니라 엉뚱한
        # (bias 낀) 방향으로 복귀하려 든다 — "적 죽이고 나서 제자리로
        # 안 돌아온다"는 실측 피드백의 원인(dev_log.md).
        self._nav_bias_deg = 0.0
        self._nav_probed_center = False
        return self._Action.NO_OP
