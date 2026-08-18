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
  GOTO_HINT— 힌트 문구가 가리키는 "열쇠가 있는 방"이 이미 가본 방들 중
             하나로 특정되면, DFS 탐험을 잠시 멈추고 그 방까지
             scene.shortest_path()로 최단 경로를 따라 이동한다(leg="key").
             열쇠를 실제로 집으면([KEY] 태그) 같은 방식으로 잠긴 문이
             있는 방으로 되돌아가 문에 걸어 들어가 연다(leg="door") —
             그래야 "잠긴 문 뒤에 뭐가 있었나"를 답할 수 있다(README의
             QA 카테고리). 새 지각 로직은 하나도 안 쓰고 기존
             _navigate_step 이동 프리미티브만 재사용한다. 예산(틱 상한)을
             넘거나 경로가 없으면 조용히 원래 DFS 탐험으로 복귀한다.
  DONE     — 갈 수 있는 새 방을 다 찾음(도달 가능한 범위 완료). 이때는
             제자리에 서 있지 않고 _default_wander로 계속 돌아다닌다
             (남은 시간에 적을 만나 죽을 위험보다, 놓친 문/열쇠를 다시
             만날 가능성이 QA 점수에 더 도움이 된다).

행동 우선순위(_arbitrate): 도망/반격(FLEE) > 선제 공격 > 열쇠·문으로
이동(GOTO_HINT) > 탐험(INIT/RECENTER/SURVEY/SEEK/HINT_CAPTURE/RETURN) >
기본 배회(_default_wander).
"""

from __future__ import annotations

from agent import enemies as EN
from agent import geometry as geo
from agent.memory import SceneGraph, CARDINAL_HEADINGS
from agent.hint_resolve import resolve_hint_room
from agent.ocr import hint_banner_active, read_hint_text, read_hud
from agent.palette import WALL_PALETTE
from agent.vision import is_blocked, sample_wall_color
from agent.vlm import classify_heading, locate_door, locate_enemy
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
# close_range_bearing() 쪽 상한. 이건 이미 맞아서 적이 코앞인 게 확정된
# 상황에서만 쓰여 오탐 위험이 낮으므로 위 4번보다 여유를 준다. 적 HP는
# 최대 4(difficulty.yaml)라 조준이 맞았다면 4방이면 죽는다 — 연속 5방을
# 때렸는데 아직 살아있으면 지금 겨누고 있는 건 적이 아니다(문틈/벽 오탐).
# 더 때리는 대신 다시 훑는 게 빠르다. 예전 값은 8이었는데, 그건 방위
# 추정이 심하게 편향돼 있던 시절(enemies.close_range_bearing 주석 참고:
# 중앙값 오차 42도)에 헛스윙을 많이 허용해야 했기 때문이다.
_CLOSE_RANGE_ATTACK_STREAK_MAX = 5

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
_RECOVER_MAX_ATTEMPTS = 8  # 이만큼 회전해도 계속 막히면 "이 방향엔 문 없음"
# 중간에 성공적 전진이 껴서 recover_attempts가 리셋되더라도, "이 목표
# 헤딩에서 180도-포기(구석 몰림)를 통째로 몇 번 겪었는지"는 별도로 세서
# 무한히 방을 맴도는 걸 막는다(실측으로 발견, dev_log.md).
_MAX_CORNERED_PER_TARGET = 3
_RECENTER_STEPS = 8        # 힌트 배너 중(깊이 측정 불가) 폴백용 소량 고정 전진
_RECENTER_MAX_STEPS = 40   # clearance 기반 중앙 이동의 안전 상한(=최대 6m)
_RECENTER_BACKOFF = 3      # 막힌 뒤 벽에서 떨어지려고 후진하는 스텝 수

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
# 눈감고 휘두르는 sweep에 들어가기 전에 먼저 도는 "훑기(scan)" 패스.
# 실측(base 벤치, seed 0~3): 전투 한 번이 20~59틱인데 그 대부분이 sweep의
# 헛스윙이었다(공격 39번에 명중 1번 같은 식). sweep은 한 방향에 3틱씩
# (공격 2 + 회전 1) 쓰므로 적이 뒤에 있으면 정면에 들어오기까지만 36틱이
# 걸린다. 반면 공격 없이 회전만 하면 같은 각도를 12틱에 훑고, 그 사이
# 매 틱 CV(close_range_bearing ~1~2ms)로 보므로 시야(±37.5도)에 들어오는
# 순간 바로 조준 공격으로 넘어간다 — "돌아보고 조준해서 때린다"가
# "눈감고 한 바퀴 휘두른다"보다 훨씬 빠르다. CV가 끝내 아무것도 못 찾으면
# 그때 기존 sweep(기하학적으로 100% 걸리는 안전망)으로 넘어간다.
_FLEE_SCAN_HEADINGS = 24               # 360도 한 바퀴, 틱당 15도
# 이미 "조준한 상태로" 때린 적이 있는데 그 뒤 CV가 갑자기 아무것도 못 찾으면,
# 적 HP가 1~4라 대개 죽어서 사라진 것이다(sweep 쪽 _FLEE_SWEEP_QUICK_CHECK_
# HEADINGS와 같은 논리). 그럴 땐 한 바퀴를 다 훑는 대신 짧게만 확인하고
# 하던 탐험으로 돌아간다 — 실측(v3 벤치)에서 전투 하나가 38~76틱이었는데
# 그 상당 부분이 "이미 죽인 적을 마저 찾는" 시간이었다.
_FLEE_SCAN_QUICK_HEADINGS = 8
# CV가 준 방위로 계속 돌기만 하고 한 번도 공격 콘에 못 넣는 경우(대개
# 벽/문틈 같은 오탐 덩어리를 쫓는 중) 몇 틱 만에 끊는다. 진짜 적이면
# 시야 반각이 37.5도라 3틱이면 정면에 온다.
_FLEE_CV_TURN_STREAK_MAX = 4
_FLEE_SWEEP_RECHECK_EVERY = 4          # 이만큼 방향을 돌 때마다 VLM으로 "아직 있는지" 재확인
# 실측(dev_log.md, seed=7): close_range_bearing()으로 이미 명중까지
# 시켰는데(flee_ever_attacked=True) 그 다음 틱에 갑자기 못 찾으면, 적
# HP가 1~4라 대부분 죽어서 사라진 경우다 — 이럴 때도 못 찾은 이유를
# "안 보였을 뿐일 수도 있다"고 24방향(최대 72틱) 풀스윕을 다 도는 건
# 낭비다. 짧게만 확인하고 없으면 바로 복귀한다.
_FLEE_SWEEP_QUICK_CHECK_HEADINGS = 4
_FLEE_STRIKE_MAX_TICKS = (             # 안전 상한: scan 한 바퀴 + sweep 풀 사이클
    _FLEE_SCAN_HEADINGS
    + _FLEE_SWEEP_HEADINGS * (_FLEE_SWEEP_ATTACKS_PER_HEADING + 1) + 4
)
_FLEE_VLM_MAX_CALLS = VLM_MAX_CALLS_PER_EPISODE  # 전투 조준 + 막힌 방향 VLM 확인이 이 예산을 공유


# _arbitrate에서 "탐험 계열"로 묶어 _dispatch에 넘기는 상태들.
_EXPLORE_STATES = ("INIT", "RECENTER", "SURVEY", "SEEK", "HINT_CAPTURE", "RETURN")

# GOTO_HINT 안전 상한: 열쇠/문으로 가는 여정 하나가 이 틱을 넘으면 뭔가
# 잘못된 것이므로(길이 막혔거나 방 이름 오독) 조용히 탐험으로 복귀한다.
_GOTO_MAX_TICKS = 400
# 열쇠 방에 도착한 뒤, 열쇠를 실제로 밟기 위해 방 안을 훑는 예산.
# 열쇠 획득은 자동(README)이라 그 위를 지나가기만 하면 된다.
_GOTO_SWEEP_MAX_TICKS = 120


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

        self.pre_flee_state = None
        self.pre_flee_target = None
        self.flee_phase = None       # None | "strike"
        self.flee_ticks = 0
        self.flee_ever_attacked = False  # 이번 전투에서 한 번이라도 ATTACK을 냈는지
        self.flee_sweep_active = False   # CV가 못 찾아서(또는 안 죽어서) 결정론적 sweep으로 전환했는지
        self.flee_scan_headings_done = 0  # 공격 없이 돌며 CV로 찾는 훑기 패스 진행도
        self.flee_cv_turn_streak = 0      # CV를 따라 연속으로 돌기만 한 틱 수
        self.flee_aimed_attacks = 0       # CV로 조준해서(눈감고 말고) 때린 횟수
        self.flee_sweep_headings_done = 0
        self.flee_sweep_attacks_done = 0
        self.flee_sweep_heading_limit = _FLEE_SWEEP_HEADINGS
        self.flee_fastpath_attack_streak = 0  # CV 조준으로 같은 자리를 연속 공격한 횟수
        self.vlm_calls_used = 0      # strike 단계 locate_enemy() 호출 수(예산 상한용)
        self._proactive_attack_streak = 0  # 선제공격(비FLEE)이 연속 몇 틱째인지 — 무한루프 방지용

        self.recenter_room = None
        self.recenter_steps = 0
        self.recenter_backoff = 0
        self.recenter_target_clearance = None

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

        # GOTO_HINT(열쇠 방 -> 잠긴 문): agent/__init__.py가 ContentIngest의
        # ContentDB를 여기 꽂아준다. None이면(단독 실행 등) 이 기능은 그냥
        # 꺼진 채로 동작한다 — 탐험 자체는 전혀 영향받지 않는다.
        self.content_db = None
        self.goto_target_room = None
        self.goto_leg = None          # None | "key" | "door"
        self.goto_ticks = 0
        self.goto_sweep_ticks = 0
        self.goto_prev_room = None
        self._goto_key_attempted = False
        self._goto_door_attempted = False
        self.key_was_held = False
        self.door_unlocked = False
        # 이번 틱의 적 탐지 결과(agent.content_ingest가 "적이 보이는
        # 프레임"을 VLM에 같이 보낼지 고를 때 재사용 — 같은 프레임에
        # EN.detect를 두 번 돌리지 않기 위함).
        self.last_mobs = []
        self._wander_heading = None

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

    # --- 메인 진입점 ------------------------------------------------
    def step(self, obs):
        hud = self._read_hud_cached(obs)
        self._note_hud_transitions(hud)
        action = self._arbitrate(hud, obs)
        self.prev_frame = obs
        return int(action)

    # 행동 우선순위를 한눈에 보이게 모아둔 함수. 각 분기는 이미 검증된
    # 기존 로직을 그대로 부르는 껍데기다 — FLEE/공격/탐험의 실제 동작은
    # 리팩터링 전과 동일하고, 바뀐 건 (1) GOTO_HINT가 3순위로 새로 끼었고
    # (2) DONE에서 제자리 회전 대신 배회로 떨어진다는 두 가지뿐이다.
    def _arbitrate(self, hud, obs):
        # 1) 맞았으면(HP 하락) 그 자리에서 즉시 반격 — 무조건 최우선.
        if self.state == "FLEE":
            return self._step_flee(hud, obs)

        # 2) 선제 공격: 콘·사거리 안에 적이 보이면 하던 일은 그대로 둔 채
        #    그 한 틱만 가로챈다.
        attack = self._attack_should_engage(hud, obs)
        if attack is not None:
            return attack

        # 3) HUD를 못 읽었거나 복도(방 이름 없음) — 방에 도착할 때까지 전진.
        if not hud.ok or hud.room_name is None:
            return self._Action.MOVE_FORWARD

        # 4) 열쇠/잠긴 문으로 이동 중이면 탐험보다 우선.
        if self.state == "GOTO_HINT":
            return self._step_goto_hint(hud, obs)

        # 5) 탐험(DFS) 계열.
        if self.state in _EXPLORE_STATES:
            return self._dispatch(hud, obs)

        # 6) 그 외(DONE 등) — 벽을 피하며 계속 돌아다닌다.
        return self._default_wander(hud, obs)

    def _note_hud_transitions(self, hud):
        """매 스텝 공통: HP 하락 감지(FLEE 트리거)와 [KEY] 태그 전이 기록.

        후진해도 적과의 거리가 안 벌어진다는 게 실측 확인됐으므로(파일
        상단 주석 참고), 맞은 그 자리에서 바로 조준으로 대응한다. 이미
        FLEE 중에 또 맞은 거면(전투가 계속 이어지는 중) 진행 중인
        sweep/조준 상태를 리셋하지 않는다 — 예전엔 매번 리셋해서 sweep이
        절대 끝까지 못 돌았던 버그가 있었음(dev_log.md).
        """
        if not (hud.ok and hud.hp is not None):
            return
        if self.last_hp is not None and hud.hp < self.last_hp and self.state != "FLEE":
            self.pre_flee_state = self.state
            self.pre_flee_target = self.target_heading
            self.state = "FLEE"
            self.flee_phase = "strike"
            self.flee_ticks = 0
            self.flee_ever_attacked = False
            self.flee_sweep_active = False
            self.flee_scan_headings_done = 0
            self.flee_cv_turn_streak = 0
            self.flee_aimed_attacks = 0
            self.flee_sweep_headings_done = 0
            self.flee_sweep_attacks_done = 0
            self.flee_sweep_heading_limit = _FLEE_SWEEP_HEADINGS
            self.flee_fastpath_attack_streak = 0
        self.last_hp = hud.hp

        # [KEY] 태그가 켜졌다 꺼지면 = 열쇠를 써서 문이 열린 것(README).
        if self.key_was_held and not hud.has_key:
            self.door_unlocked = True
        self.key_was_held = hud.has_key

    def _attack_should_engage(self, hud, obs):
        """콘·사거리 안에 적이 있으면 Action.ATTACK, 아니면 None.

        선제 공격: 지금까진 HP가 깎여야만(=한 대 맞아야만) 전투를 시작해서
        "적을 보고도 안 죽이고 맞고 나서야 죽인다"는 실측 피드백이 있었다
        (사용자 지시). agent.enemies는 이미 프레임당 ~1~2ms라 매 틱 상시
        돌려도 예산에 거의 안 걸리므로, FLEE 중이 아닐 때도 매 틱 확인해서
        콘·사거리 안에 적이 있으면 하던 일(탐험 상태)은 그대로 둔 채
        그 한 틱만 공격으로 가로챈다.
        실측으로 확인된 버그(dev_log.md): 안전장치 없이 매 틱 이 조건이면
        무조건 가로채니까, 죽지 않는 대상(오탐이거나, 실제 명중이 안 되는
        경우)을 만나면 완전히 같은 자리에서 ATTACK만 영원히 반복하며
        탐험이 통째로 멈췄다(seed=1, 60틱 넘게 위치/각도 고정). 적 HP가
        최대 4라 몇 방이면 죽어야 정상이므로, 이 이상 연속되면 오탐/명중
        실패로 보고 이번 틱은 하던 일(탐험)을 계속하게 양보한다 — 진짜
        적이면 언젠가 다시 맞고 정식 FLEE(sweep 포함) 경로로 넘어간다.

        "사거리 밖 적을 멈춰서 지켜보다 다가오면 공격" 로직을 시도했다가
        실측으로 되돌렸다(dev_log.md) — 이 게임은 후진해도 적과의 거리가
        안 벌어진다는 게 이미 확인된 사실이라, 멈춰서 기다리든 탐험을
        계속하든 "언젠가 다가와서 맞는" 타이밍 자체는 거의 안 바뀐다.
        대신 멈춰서 지켜보는 동안 탐험 진행이 완전히 멎어서, 적이 많은
        방에 계속 묶여 있다가 반복 교전으로 죽는 결과가 났다(seed=7:
        640스텝만에 사망 vs 껐을 때 4000스텝 끝까지 생존, 직접 A/B).
        """
        mobs = EN.detect(obs, wall_rgb=self._current_wall_rgb(hud) if hud.ok else None)
        self.last_mobs = mobs
        best = next((m for m in mobs if m.score >= _MOB_MIN_SCORE
                     and m.distance <= _MOB_MAX_COMBAT_DIST_M), None)
        if (best is not None and abs(best.bearing) <= _ATTACK_CONE_HALF_DEG
                and best.distance <= _ATTACK_RANGE_M
                and self._proactive_attack_streak < _FASTPATH_ATTACK_STREAK_MAX):
            self._proactive_attack_streak += 1
            self._last_action_was_forward = False  # 하던 상태의 is_blocked 계산이 꼬이지 않게
            return self._Action.ATTACK
        self._proactive_attack_streak = 0
        return None

    # --- 기본 배회(우선순위 최하위) --------------------------------------
    # 탐험이 끝난(DONE) 뒤의 폴백. 예전엔 제자리에서 계속 회전만 했는데,
    # 그건 남은 시간을 통째로 버리는 것과 같다. 목표 헤딩을 "지금 보고 있는
    # 방향"으로 두고 검증된 _navigate_step(전진 -> 막히면 옆으로 우회 ->
    # 구석이면 180도)을 그대로 돌리면, 새 지각 로직 없이 "벽을 피하며 계속
    # 걷는다"가 된다. 완전히 막히면 목표를 90도 틀어 다시 시도한다.
    def _default_wander(self, hud, obs):
        if not hud.ok or hud.heading is None:
            return self._Action.MOVE_FORWARD
        if self._wander_heading is None:
            self._wander_heading = int(hud.heading) % 360

        def on_exhausted():
            self._wander_heading = (self._wander_heading + 90) % 360
            self._reset_nav_state()
            return self._Action.TURN_RIGHT

        return self._navigate_step(hud, obs, self._wander_heading, on_exhausted)

    def _reset_nav_state(self):
        """_navigate_step이 쓰는 임시 상태를 초기화(목표를 새로 잡을 때)."""
        self.recover_attempts = 0
        self._need_align = True
        self._last_action_was_forward = False
        self._nav_bias_deg = 0.0
        self._nav_fail_count = 0
        self._nav_action_queue = []
        self._nav_probed_center = False
        self._nav_cornered_count = 0

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
        # DONE/GOTO_HINT는 _arbitrate가 직접 처리한다(여기 오지 않음).
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
        return self._start_recenter(name)

    # --- RECENTER: 문 앞을 벗어나 방 중앙 쪽으로 -------------------------------
    def _start_recenter(self, canon):
        self.recenter_room = canon
        self.recenter_steps = 0
        self.recenter_backoff = 0
        self.recenter_target_clearance = None  # 첫 틱에 측정해서 채움
        self._last_action_was_forward = False
        self.state = "RECENTER"
        return self._Action.NO_OP

    def _step_recenter(self, hud, obs):
        if self.recenter_backoff > 0:
            self.recenter_backoff -= 1
            if self.recenter_backoff == 0:
                return self._start_survey_scan()
            return self._Action.MOVE_BACK

        # 들어온 방향이 우연히 "문" 방향이면(실측 확인: 스폰 헤딩이 그런
        # 경우가 흔함) "막힐 때까지 전진"은 벽을 안 만나고 다음 방까지
        # 계속 뚫고 들어가버린다(예전 버그, dev_log.md). 그래서 고정 스텝
        # 대신 agent.geometry로 입구에서 측정한 정면 클리어런스의 절반쯤
        # (=대충 방 중앙 방향)까지만 걷는다 — 방 크기에 맞게 적응하면서도
        # 반대쪽 문까지 뚫고 들어가진 않는다(사용자 피드백: 입구에서 너무
        # 조금만 걷고 멈춰서 어색해 보임).
        canon = self._canonicalize(hud.room_name) if hud.room_name else None
        if canon is not None and canon != self.recenter_room:
            # 실수로 문을 넘어간 것 자체가 "이 방향에 문이 있다"는 확실한
            # 증거다 — 근데 예전엔 새 방 쪽 기록만 하고 이전 방(스폰 방일
            # 때가 많음) 노드에는 이 방향을 기록 안 해서, 그 방이 서베이를
            # 아예 한 번도 못 받고 나머지 3방향이 영원히 unknown으로 남는
            # 버그가 있었다(실측 확인, dev_log.md — 8~12개 방 중 3개만
            # 발견하고 끝난 사례). 방을 나가기 전에 이 방향을 이전 방
            # 노드에도 남긴다.
            old_node = self.scene.nodes.get(self.recenter_room)
            if old_node is not None and hud.ok and hud.heading is not None:
                cardinal = min(CARDINAL_HEADINGS,
                                key=lambda h: abs(_heading_diff(h, hud.heading)))
                if old_node.exits.get(cardinal) == "unknown":
                    old_node.exits[cardinal] = "open"
                    old_node.exit_leads_to[cardinal] = canon
            return self._enter_room(hud, entry_heading=hud.heading, parent=self.recenter_room)

        if hint_banner_active(obs):
            # 배너 중엔 깊이 측정이 무의미 — 예전 방식(소량 고정 전진)으로 대체.
            if self._last_action_was_forward and self.prev_frame is not None:
                if is_blocked(self.prev_frame, obs):
                    self.recenter_backoff = _RECENTER_BACKOFF
                    self._last_action_was_forward = False
                    return self._Action.MOVE_BACK
            self.recenter_steps += 1
            if self.recenter_steps > _RECENTER_STEPS:
                return self._start_survey_scan()
            self._last_action_was_forward = True
            return self._Action.MOVE_FORWARD

        clearance = geo.cone_clearance(geo.depth_profile(obs), 0.0, _SEEK_CONE_HALF_DEG)
        if self.recenter_target_clearance is None:
            # 첫 틱: 입구에서 보이는 클리어런스의 절반을 목표로 삼는다.
            # 문 방향이라 아주 멀리(열린 걸로) 보일 수 있어 상한을 둔다.
            self.recenter_target_clearance = max(
                _SEEK_SAFE_CLEARANCE, min(clearance / 2.0, 4.0))

        if clearance <= self.recenter_target_clearance:
            return self._start_survey_scan()

        if self._last_action_was_forward and self.prev_frame is not None:
            if is_blocked(self.prev_frame, obs):
                self.recenter_backoff = _RECENTER_BACKOFF
                self._last_action_was_forward = False
                return self._Action.MOVE_BACK

        self.recenter_steps += 1
        if self.recenter_steps > _RECENTER_MAX_STEPS:  # 안전 상한
            return self._start_survey_scan()
        self._last_action_was_forward = True
        return self._Action.MOVE_FORWARD

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
        # 방 하나를 다 훑을 때마다, 지금까지 본 방들로 힌트가 풀리는지
        # 다시 확인한다(Phase 4) — 풀리면 DFS를 잠시 멈추고 그 방으로.
        goto = self._maybe_start_goto(canon)
        if goto is not None:
            return goto
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

        # 정면이 막힘 — 옆으로 비켜갈 수 있는지 가까운 오프셋부터 확인.
        for off in _SIDESTEP_OFFSETS_DEG[1:]:
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

    # --- GOTO_HINT: 열쇠 방 -> 잠긴 문 (기존 이동 프리미티브만 재사용) ---
    def _locked_door_room(self):
        """잠긴 문이 있는 방. 힌트 배너를 실제로 띄운 위치가 1순위 근거이고,
        없으면 exits에 "locked"로 확정된 방을 찾는다."""
        if self.scene.key_hint_room:
            return self.scene.key_hint_room
        for name, node in self.scene.nodes.items():
            if any(st == "locked" for st in node.exits.values()):
                return name
        return None

    def _locked_door_heading(self, room):
        if room == self.scene.key_hint_room and self.scene.key_hint_heading is not None:
            return self.scene.key_hint_heading
        node = self.scene.nodes.get(room)
        if node is None:
            return None
        for heading, status in node.exits.items():
            if status == "locked":
                return heading
        return None

    def _resolve_key_room(self):
        """힌트 문구 + 지금까지의 관측 -> 열쇠가 있는 방 이름(없으면 None).

        CV로 이미 잰 방 벽 색을 ContentDB에 먼저 채워 넣는다 — VLM 응답이
        아직 안 온 방도 "…walls" 템플릿 힌트에는 바로 답할 수 있게.
        """
        if self.content_db is None or not self.scene.key_hint_text:
            return None
        for name, node in list(self.scene.nodes.items()):
            if not node.wall_color:
                continue
            rec = self.content_db.get_or_create(name)
            if rec.wall_color.value is None:
                rec.wall_color.value = node.wall_color
                rec.wall_color.confidence = 0.9
        return resolve_hint_room(self.scene.key_hint_text, self.content_db)

    def _maybe_start_goto(self, canon):
        """서베이 직후 호출 — 지금 GOTO_HINT로 전환할 이유가 있으면 그
        첫 액션을, 없으면 None을 반환한다."""
        if self.door_unlocked:
            return None
        # 이미 열쇠를 들고 있다 -> 잠긴 문으로 가서 연다.
        if self.key_was_held and not self._goto_door_attempted:
            door_room = self._locked_door_room()
            if (door_room and self._locked_door_heading(door_room) is not None
                    and self.scene.shortest_path(canon, door_room) is not None):
                return self._start_goto(door_room, "door")
        # 아직 열쇠가 없다 -> 힌트가 가리키는 방이 특정되면 그리로.
        if not self.key_was_held and not self._goto_key_attempted:
            target = self._resolve_key_room()
            if target and self.scene.shortest_path(canon, target) is not None:
                return self._start_goto(target, "key")
        return None

    def _start_goto(self, target_room, leg):
        self.goto_target_room = target_room
        self.goto_leg = leg
        self.goto_ticks = 0
        self.goto_sweep_ticks = 0
        self.goto_prev_room = None
        self._wander_heading = None
        if leg == "key":
            self._goto_key_attempted = True
        else:
            self._goto_door_attempted = True
        self._reset_nav_state()
        self.target_heading = None
        self.state = "GOTO_HINT"
        return self._Action.NO_OP

    def _step_goto_hint(self, hud, obs):
        self.goto_ticks += 1
        if self.goto_ticks > _GOTO_MAX_TICKS or self.goto_target_room is None:
            return self._resume_after_goto(hud)

        canon = self._canonicalize(hud.room_name) if hud.room_name else None
        if canon is None:
            return self._Action.MOVE_FORWARD  # 복도 — 계속 전진

        if canon not in self.scene.nodes:
            # 처음 보는 방 = 방금 잠긴 문을 열고 들어왔거나, 이동 중 우연히
            # 새 방을 찾은 것. 어느 쪽이든 그래프에 정식으로 등록하고
            # 평소 탐험 루틴(RECENTER->SURVEY)으로 넘긴다 — "잠긴 문 뒤에
            # 뭐가 있었나"를 답하려면 이 등록이 반드시 필요하다.
            parent = self.goto_prev_room
            entry = None
            if hud.heading is not None:
                entry = min(CARDINAL_HEADINGS,
                            key=lambda h: abs(_heading_diff(h, hud.heading)))
            self.goto_leg = None
            self.goto_target_room = None
            # 그래프를 가로질러 온 뒤라 DFS 스택이 실제 위치와 어긋나 있다.
            # 새 방을 push하기 전에 "루트 -> 직전 방"으로 다시 세워야
            # 이후 backtrack이 엉뚱한 방으로 되돌아가지 않는다.
            if parent and parent in self.scene.nodes:
                self._rebuild_stack(parent)
            return self._enter_room(hud, entry_heading=entry, parent=parent)

        self.goto_prev_room = canon

        # 문이 열린 뒤 목적지 방을 벗어났으면 여정 끝 — 탐험으로 복귀.
        if self.goto_leg == "door" and self.door_unlocked and canon != self.goto_target_room:
            return self._resume_after_goto(hud)

        if canon == self.goto_target_room:
            if self.goto_leg == "key":
                if self.key_was_held:
                    door_room = self._locked_door_room()
                    if (door_room and self._locked_door_heading(door_room) is not None
                            and self.scene.shortest_path(canon, door_room) is not None):
                        return self._start_goto(door_room, "door")
                    return self._resume_after_goto(hud)
                # 열쇠 방에 도착 — 열쇠를 밟으려면 방 안을 돌아다녀야 한다
                # (README: 획득은 자동, 그 위를 지나가기만 하면 됨).
                self.goto_sweep_ticks += 1
                if self.goto_sweep_ticks > _GOTO_SWEEP_MAX_TICKS:
                    return self._resume_after_goto(hud)
                return self._default_wander(hud, obs)
            return self._step_goto_door(hud, obs)

        path = self.scene.shortest_path(canon, self.goto_target_room)
        if not path:
            return self._resume_after_goto(hud)
        heading = path[0]
        if heading != self.target_heading:
            self.target_heading = heading
            self._reset_nav_state()
        return self._navigate_step(hud, obs, heading,
                                    lambda: self._resume_after_goto(hud))

    def _step_goto_door(self, hud, obs):
        """잠긴 문이 있는 방에 도착 — 그 방향으로 걸어 들어가면 자동으로
        열린다(README). 열린 뒤에도 계속 전진해서 그 너머 방까지 들어간다."""
        heading = self._locked_door_heading(self.goto_target_room)
        if heading is None:
            return self._resume_after_goto(hud)
        if heading != self.target_heading:
            self.target_heading = heading
            self._reset_nav_state()
        return self._navigate_step(hud, obs, heading,
                                    lambda: self._resume_after_goto(hud))

    def _resume_after_goto(self, hud):
        """여정 종료(성공/포기 무관) — DFS 스택을 지금 위치 기준으로 다시
        세우고 평소 탐험으로 복귀한다."""
        self.goto_leg = None
        self.goto_target_room = None
        self._reset_nav_state()
        canon = self._canonicalize(hud.room_name) if hud.room_name else None
        if canon and canon in self.scene.nodes:
            self._rebuild_stack(canon)
            node = self.scene.nodes[canon]
            if node.done:
                return self._start_backtrack(canon)
            return self._start_seek(canon, node)
        # 복도 등 — 다음 틱에 방에 도착하면 INIT이 정상 등록한다.
        self.state = "INIT"
        return self._Action.MOVE_FORWARD

    def _rebuild_stack(self, canon):
        """GOTO_HINT로 그래프를 가로질러 이동한 뒤에는 DFS 스택(백트랙
        경로)이 실제 위치와 어긋난다. 루트에서 지금 방까지의 최단 경로로
        스택을 다시 만들어, 이후 backtrack이 정상적인 부모 방으로
        되돌아가게 한다."""
        root = self.scene.visited_order[0] if self.scene.visited_order else canon
        stack = [(root, None)]
        path = self.scene.shortest_path(root, canon) or []
        cur = root
        for heading in path:
            nxt = self.scene.nodes[cur].exit_leads_to.get(heading)
            if nxt is None:
                break
            stack.append((nxt, heading))
            cur = nxt
        if stack[-1][0] != canon:
            stack.append((canon, None))
        self.scene.stack = stack

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
            node = self.scene.nodes[canon]
            if node.done:
                return self._start_backtrack(canon)
            if node.wall_color is None:
                # RECENTER 중 실수로 문을 넘어가서 이 방이 서베이(벽색
                # 기록)를 한 번도 못 받은 경우 — 되돌아온 김에 지금 마저
                # 받는다(실측 확인된 문제, dev_log.md 참고).
                self.recenter_room = canon
                return self._start_survey_scan()
            return self._start_seek(canon, node)

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

    def _flee_cv_aim(self, hud, obs):
        """CV로 적을 찾아 이번 틱의 행동(공격/회전/전진)을 정한다. 못 찾으면 None.

        1순위는 근접 전용 방위 추정(enemies.close_range_bearing). 실측
        확인(dev_log.md, seed=7): strike 진입 시점엔 적이 이미 1.3~1.5m
        코앞이라 몸통이 화면을 거의 다 채워서, detect()의 사람형 비율/
        바닥접점 판정이 구조적으로 0개만 반환한다. close_range_bearing은
        사람형 판정 없이 "벽도 바닥도 아닌 큰 덩어리"만 보므로 이 거리에서도
        먹힌다(이미 HP가 깎여 진입한 상태라 오탐 위험도 낮음). 2순위는
        중간 거리용 사람형 탐지 — 적이 아직 근접하기 전이거나 방금 물러난
        경우다.
        """
        wall_rgb = self._current_wall_rgb(hud)
        distance = None
        bearing = EN.close_range_bearing(obs, wall_rgb=wall_rgb)
        streak_cap = _CLOSE_RANGE_ATTACK_STREAK_MAX
        if bearing is None:
            mobs = EN.detect(obs, wall_rgb=wall_rgb)
            best = next((m for m in mobs if m.score >= _MOB_MIN_SCORE
                         and m.distance <= _MOB_MAX_COMBAT_DIST_M), None)
            if best is None:
                self.flee_cv_turn_streak = 0
                return None
            bearing, distance = best.bearing, best.distance
            streak_cap = _FASTPATH_ATTACK_STREAK_MAX
        if self.flee_fastpath_attack_streak >= streak_cap:
            # 여기 걸리면 "연속으로 이만큼 때렸는데 안 죽는다" = 지금 보고
            # 있는 덩어리가 적이 아니다(문틈 너머 풍경, 벽 등). 이번 틱은
            # CV를 접고 탐색(scan/sweep)에 넘긴다.
            #
            # 실측으로 발견한 버그(seed=0, trace_fight): 이 카운터가 전투
            # 내내 누적되기만 하고 안 끊겨서, 초반에 오탐을 8번 때린 순간
            # 그 전투 내내 CV 조준이 통째로 꺼졌다. 그 뒤 회전하다 진짜 적이
            # 정면 1.4m(cr=8.2도, 콘 안)에 들어왔는데도 공격 대신 그냥
            # 지나쳐 돌았고, 결국 못 죽이고 두 번 더 맞았다. 지금은 공격이
            # 아닌 행동을 한 틱이라도 하면 호출자가 0으로 되돌린다 —
            # 그래서 이 상한은 "연속 헛스윙"만 센다.
            return None

        if abs(bearing) <= _ATTACK_CONE_HALF_DEG:
            if distance is not None and distance > _ATTACK_RANGE_M:
                # 방향(콘)은 이미 맞는데 사거리(3.0m) 밖 — 실측으로 확인된
                # 버그: 이 경우도 "정렬 안 됨"으로 보고 회전만 시켰더니,
                # 회전은 거리를 못 좁히니 bearing 부호가 살짝씩 뒤집히며
                # 좌우로 영원히 진동만 하고 한 번도 공격을 못 했다
                # (dev_log.md). 전진해서 거리부터 좁힌다.
                self._last_action_was_forward = True
                return self._Action.MOVE_FORWARD
            self.flee_ever_attacked = True
            self.flee_aimed_attacks += 1
            self.flee_fastpath_attack_streak += 1
            self.flee_cv_turn_streak = 0
            return self._Action.ATTACK

        # 콘 밖 — 정확한 방위로 회전한다. 다만 계속 따라 돌기만 하고 한 번도
        # 콘에 못 넣으면(대개 문틈/벽 같은 오탐 덩어리를 쫓는 중) 끊고
        # scan/sweep에 넘긴다.
        if self.flee_cv_turn_streak >= _FLEE_CV_TURN_STREAK_MAX:
            return None
        self.flee_cv_turn_streak += 1
        return self._Action.TURN_LEFT if bearing > 0 else self._Action.TURN_RIGHT

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

        # 1) CV 조준 — scan/sweep 중에도 매 틱 돌린다. 예전엔 sweep으로
        #    한 번 넘어가면 CV를 다시 안 봐서, 적이 회전 중에 시야에
        #    들어와도 계속 눈감고 휘두르기만 했다. CV는 1~2ms라 매 틱
        #    돌려도 공짜에 가깝다.
        action = self._flee_cv_aim(hud, obs)
        if action is not None:
            if action != self._Action.ATTACK:
                self.flee_fastpath_attack_streak = 0
            return action
        # CV 조준이 이번 틱에 아무 것도 못 냈다 = 이제부터 하는 건 조준이
        # 아니라 탐색이다. "연속 공격" 카운터는 여기서 끊어준다 — 아래
        # _flee_cv_aim 주석 참고(실측으로 발견한 버그).
        self.flee_fastpath_attack_streak = 0

        # 2) 훑기(scan) 패스: 공격 없이 15도씩 돌면서 매 틱 위 CV로 찾는다
        #    (_FLEE_SCAN_HEADINGS 주석 참고 — 같은 각도를 sweep보다 3배
        #    빨리 훑고, 찾는 즉시 조준 공격으로 넘어간다).
        scan_limit = (_FLEE_SCAN_QUICK_HEADINGS if self.flee_aimed_attacks
                      else _FLEE_SCAN_HEADINGS)
        if self.flee_scan_headings_done < scan_limit:
            self.flee_scan_headings_done += 1
            return self._Action.TURN_RIGHT

        if not self.flee_sweep_active:
            # 한 바퀴 다 훑고도 CV가 아무것도 못 찾음 → 눈감고 휘두르는
            # 안전망(VLM 조준은 안 씀 — 위 주석 참고).
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

    def _resume_after_flee(self):
        self.flee_phase = None
        self.flee_sweep_active = False
        self.flee_scan_headings_done = 0
        self.flee_cv_turn_streak = 0
        self.flee_aimed_attacks = 0
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
