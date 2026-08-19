"""Exploration policy: a rule-based state machine that tries to cover as many rooms as possible using pixels alone.

States:
  INIT     — On the first frame, registers the spawn room as the graph's root.
  RECENTER — Right after entering a room (standing just inside the
             doorway), walks forward about half the front clearance
             measured at the entrance with agent.geometry (roughly
             toward the room's center) -- so the agent isn't left
             standing awkwardly right at the threshold, and moves into
             the room proper (user feedback: stopping and turning right
             at the entrance looked awkward). This also cuts down on
             VLM room-content captures getting contaminated by the
             neighboring room bleeding through the doorway (see
             dev_log.md).
  SURVEY   — From inside the room (where RECENTER left off), turns
             through the 4 cardinal directions (0/90/180/270) and
             records wall color. Doors only ever exist along these 4
             directions (guaranteed by world/layout.py's grid layout,
             and matches the README's heading<->wall table), so there's
             no need to probe arbitrary angles. agent.py buffers frames
             while in this state to send to the VLM later. Once each
             direction is faced head-on, if there's budget left it also
             calls agent.vlm.classify_heading() once to ask ahead of
             time whether it "looks like a door" (per user direction:
             check with the VLM before moving, every time). If the VLM
             is confident it's "not a door," that direction's exits
             entry is marked "wall" right there, skipping SEEK's
             physical approach (bumping into it, turning 180, and the
             rest of the CV safety net) -- the goal being to cut down
             on wasted attempts at an obvious wall. If it "might be a
             door," it's only added to vlm_door_hint and exits is left
             "unknown" (whether it's really a door, and where it leads,
             can only be known by actually walking there) -- SEEK tries
             hinted directions first when picking a candidate. If the
             VLM fails/times out/runs out of budget, behavior is
             identical to before (the VLM is the primary signal, but
             physical CV confirmation is the final safety net -- per
             user confirmation: "VLM leads, CV is the safety net").
  SEEK     — Walks toward one candidate direction (hinted directions
             first). If blocked (wall or object, no distinction),
             retries with small turns; if that still fails, records
             "no door this direction" and moves to the next candidate.
             A room-name change means a door was found.
  RETURN   — Either we connected to an already-visited room, or we
             finished checking all 4 directions of this room and need
             to go back to the previous one (DFS backtrack).
  FLEE     — Triggered with top priority the moment HP actually drops
             (meaning an enemy is nearby), going straight into strike
             (attack) mode. The original design tried "back up to gain
             distance, then counter-attack," but direct measurement
             (see dev_log.md) showed the enemy's chase speed is nearly
             identical to our backward speed, so no matter how much we
             backed up, the distance to the enemy (always 1.3-1.5 m,
             inside its own attack range) never actually increased --
             we just kept getting hit. In other words, "backing up
             creates distance" simply doesn't hold in this game
             (confirmed directly). So we respond right where we got
             hit, without backing up: the enemy is already always
             closer than our own attack range (3.0 m) at 1.3-1.5 m, so
             distance isn't the problem -- we only need to face the
             right direction. Aiming happens every tick with a two-tier
             safety net:
               1) agent.enemies.detect(): uses the saturation boundary
                  of the floor checkerboard to get a real measured
                  distance in each direction (agent.geometry -- the
                  camera is fixed at domain_rand=False, so pixels-to-
                  meters is exact), and identifies mobs by raw color
                  saturation, floor contact, and height (1.15-2.45 m) to
                  produce an exact bearing/distance in degrees. At
                  ~1-2 ms per frame, this is effectively instant -- most
                  ticks end here. Attack immediately if inside the
                  cone/range, otherwise turn to face exactly that direction.
               2) If it can't find anything (including failure/budget
                  exhaustion), switch to a deterministic sweep -- since
                  the attack cone is +/-20 deg in front (40 deg total)
                  and one turn is 15 deg (confirmed by measurement/
                  source), repeating "attack twice, then turn 15 deg" 24
                  times (360 deg) is geometrically guaranteed to catch
                  the enemy in the cone at some point, no matter what
                  angle it's at.
             The VLM (agent.vlm.locate_enemy) was dropped from the
             aiming decision -- confirmed by measurement: its 3-tier
             left/center/right judgment marked "center" using a much
             narrower band than the actual attack cone, so it kept
             judging "not center" even when we were already inside the
             cone, oscillating left-right for over 76 ticks without
             ever landing a hit while continuing to take damage (see
             dev_log.md). Once we've already landed a hit during the
             sweep, the VLM is only used as a secondary check for
             "is it still alive" (just an existence question, not a
             bearing judgment, so this problem doesn't apply there).
             Once switched into the sweep, it runs to completion (if hit
             again mid-sweep, it does not restart from scratch). If it
             completes the full sweep (or exceeds the safety tick cap)
             without a target, it returns to whatever it was doing --
             and if HP drops again afterward (meaning it's still alive,
             or there's a second enemy), the HP-drop detection at the
             top of step() triggers a fresh FLEE automatically.
  GOTO_HINT— Once the hint text's "room with the key" is narrowed down
             to one of the rooms we've already visited, DFS exploration
             pauses and we navigate the shortest path there via
             scene.shortest_path() (leg="key"). Once the key is actually
             picked up (the [KEY] tag), the same mechanism sends us back
             to the room with the locked door and walks into it to open
             it (leg="door") -- necessary to be able to answer "what was
             behind the locked door" (a README QA category). No new
             perception logic is used at all -- it just reuses the
             existing _navigate_step movement primitive. If the tick
             budget is exceeded or there's no path, it quietly returns
             to normal DFS exploration.
  DONE     — Every reachable new room has been found. Rather than
             standing still, it keeps wandering via _default_wander (the
             chance of stumbling onto a missed door/key in the remaining
             time is worth more for the QA score than the risk of dying
             to an enemy while standing around).

Action priority (_arbitrate): flee/counter-attack (FLEE) > proactive
attack > moving toward the key/door (GOTO_HINT) > exploration
(INIT/RECENTER/SURVEY/SEEK/HINT_CAPTURE/RETURN) > default wander
(_default_wander).
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
_MOB_MIN_SCORE = 0.6              # minimum confidence for an agent.enemies detection
_ATTACK_CONE_HALF_DEG = 20.0      # half-angle of the env's attack cone (confirmed from source)
_ATTACK_RANGE_M = 3.0             # the env's attack range (confirmed from source)
# During combat, CV detection ignores any candidate farther than this.
# Measured: our WALL_PALETTE/wall images include a lot of fairly
# saturated colors (mustard at 155, etc.), and a distant wall or picture
# really did get misdetected as "a human-height mob 20 m away" (this
# reproduced from the very first frame of an episode). A genuine enemy
# during combat is always within 1.3-1.5 m (confirmed by measurement),
# so a candidate outside this range is just a long-range false positive
# we don't need anyway.
_MOB_MAX_COMBAT_DIST_M = 4.0
# Confirmed by measurement: CV can mistake a 3D prop like a tree (whose
# height isn't filtered out, since it's close to human height -- see
# dev_log.md) for an enemy and keep insisting it's "in the cone and in
# range." A real enemy has 1-4 HP and should die within a few hits, so
# once attacks keep landing this many times in a row while satisfying
# the cone/range condition, we treat it as a false positive/miss and
# fall through to the sweep instead.
_FASTPATH_ATTACK_STREAK_MAX = 4
# close_range_bearing() is only used once we're already in FLEE strike
# (meaning we've already been hit, so the enemy is right in front of
# us), which makes false positives much less of a risk. Measured (see
# dev_log.md, seed 7): with the enemy's HP at 1-4, hitting this cap of 4
# kept forcing us into the sweep every time, so even when close-range
# detection succeeded, total combat time barely dropped (it kept
# re-entering the sweep every time and stayed around 39 ticks). Since
# this is a situation where a real enemy is essentially guaranteed, the
# cap is raised generously so the fast path alone gets a real chance to
# finish the kill (double the max HP, so a few misses are still fine).
_CLOSE_RANGE_ATTACK_STREAK_MAX = 8

# Front clearance (agent.geometry depth profile) checked before moving
# forward. AGENT_RADIUS (0.4 m) + one step (0.15 m) + margin. This turns
# "try it and react if blocked" into "already know before trying,"
# eliminating movement that visibly bumps into walls (per user
# direction, based on measured feedback that the movement looked awkward).
_SEEK_SAFE_CLEARANCE = 0.75
_SEEK_CONE_HALF_DEG = 15.0
# When we run into a local obstacle like a tree or prop, before giving
# up on the target direction and marking it a wall, first check whether
# we can slip past it by sidestepping slightly (per user direction:
# treat each obstacle as having a boundary and go around it). The FOV is
# +/-37.5 deg (agent.geometry.FOV_X_DEG), so anything outside that range
# isn't visible in this frame anyway -- only probe sideways within that range.
_SIDESTEP_OFFSETS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)
_SIDESTEP_CONE_HALF_DEG = 8.0
_SIDESTEP_MAX_PROBE_DEG = 35.0  # only trust up to about half of FOV_X_DEG (~75)

_TURN_TOLERANCE = 7        # within this many degrees of the target heading counts as "aligned"
_RECOVER_MAX_ATTEMPTS = 8  # if still blocked after this many turns, record "no door this direction"
# Even if recover_attempts gets reset by a successful forward step in
# between, we separately track "how many times, total, has this target
# heading hit the full 180-degree give-up (cornered)" -- this keeps it
# from circling the room forever (found by measurement, see dev_log.md).
_MAX_CORNERED_PER_TARGET = 3
# RECENTER: measures the wall clearance in all four directions
# (0/90/180/270) and computes the exact distance to the room's true
# center (in each axis, "this-direction distance minus opposite-
# direction distance" divided by 2 tells us how far off-center we are,
# so we move exactly that far). Beyond this cap, we treat the reading as
# a door/opening rather than a wall (that direction's distance can't be
# trusted) and estimate the center from the opposite wall alone.
# Measured (confirmed against ground truth, rooms are 10x10): 6.0 was
# too low and misclassified a genuine wall (~9.9 m) as "uncertain" --
# raised to 15.0.
_RECENTER_MEASURE_CAP_M = 15.0

# HINT_CAPTURE: door_touch_radius=1.5 m (confirmed by measurement in
# difficulty.yaml). Being "cornered" close enough to see the padlock
# panel usually means we're already inside that radius, or within a
# step or two (forward_step=0.15 m), so 20 steps (up to 3.0 m of forward
# attempts) is more than enough -- and if it isn't, collision stops us first anyway.
_HINT_CAPTURE_MAX_APPROACH_TICKS = 20

# The attack cone is +/-20 deg in front (40 deg total), one turn is 15
# deg (confirmed via _ATTACK_CONE_HALF_RAD/turn_step from source) --
# turning in 15-degree steps and attacking at every position means the
# cones overlap enough that any angle is guaranteed to be caught at
# least once. This is the basis for the sweep safety net's numbers.
_FLEE_SWEEP_HEADINGS = 24              # 360 / 15 degrees
_FLEE_SWEEP_ATTACKS_PER_HEADING = 2    # enemy HP is 1-4, so attack twice in case one hit isn't a kill
_FLEE_SWEEP_RECHECK_EVERY = 4          # re-check with the VLM "is it still there" every this many headings
# Measured (see dev_log.md, seed 7): if close_range_bearing() already
# landed a hit (flee_ever_attacked=True) and the very next tick suddenly
# can't find anything, that's usually because the enemy's HP (1-4) means
# it died and disappeared. In that case, doing the full 24-direction
# sweep (up to 72 ticks) just in case "it was only just out of view" is
# wasteful. Check briefly, and if nothing's there, return immediately.
_FLEE_SWEEP_QUICK_CHECK_HEADINGS = 4
_FLEE_STRIKE_MAX_TICKS = (             # safety cap accounting for a failed VLM aim plus a full sweep cycle
    _FLEE_SWEEP_HEADINGS * (_FLEE_SWEEP_ATTACKS_PER_HEADING + 1) + 4
)
_FLEE_VLM_MAX_CALLS = VLM_MAX_CALLS_PER_EPISODE  # combat aiming and blocked-direction VLM checks share this budget


# States _arbitrate groups as "exploration" and hands off to _dispatch.
_EXPLORE_STATES = ("INIT", "RECENTER", "SURVEY", "SEEK", "HINT_CAPTURE", "RETURN")

# GOTO_HINT safety cap: if one trip toward the key/door takes more ticks
# than this, something's wrong (a blocked path or a misread room name),
# so quietly return to exploration.
_GOTO_MAX_TICKS = 400
# After arriving in the key's room, the budget for sweeping the room to
# actually step on the key. Picking it up is automatic (per the README)
# -- we just need to walk over it.
_GOTO_SWEEP_MAX_TICKS = 120


def _heading_diff(a: int, b: int) -> int:
    """Normalizes b - a into [-180, 180]. TURN_RIGHT decreases heading and
    TURN_LEFT increases it (confirmed by measurement), so a positive
    result (target is larger) means TURN_LEFT gets closer, and a
    negative result means TURN_RIGHT does."""
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
        self.flee_ever_attacked = False  # whether we've landed at least one ATTACK this fight
        self.flee_sweep_active = False   # whether CV failed to find the enemy (or it wasn't dying) and we switched to the deterministic sweep
        self.flee_sweep_headings_done = 0
        self.flee_sweep_attacks_done = 0
        self.flee_sweep_heading_limit = _FLEE_SWEEP_HEADINGS
        self.flee_fastpath_attack_streak = 0  # how many consecutive attacks CV aiming has fired at the same spot
        self.vlm_calls_used = 0      # number of locate_enemy() calls made during strike (budget cap)
        self._proactive_attack_streak = 0  # how many consecutive ticks of proactive (non-FLEE) attack -- prevents an infinite loop

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
        self._nav_bias_deg = 0.0  # temporary heading offset used to steer around a local obstacle during SEEK/RETURN
        self._nav_fail_count = 0     # how many times we've given up on the current target as "completely blocked" (drives the escalating recovery)
        self._nav_action_queue: list = []  # a pre-planned action sequence used during staged recovery (backing up, sidestepping, etc.)
        self._nav_probed_center = False  # whether we've actually tried moving forward at least once since aligning to this direction
        self._nav_cornered_count = 0  # how many times, total, we've hit the full 180-degree give-up for this target heading
        self._hud_cache_key = None
        self._hud_cache_value = None

        # HINT_CAPTURE: on spotting a padlock panel, briefly drops into
        # this state to approach the door (into the touch radius),
        # trigger and read the hint banner, then back off toward where
        # we were. seek_origin_room/target_heading are inherited as-is
        # from SEEK and don't need to be stored separately.
        self._hint_capture_phase = None  # None | "approach" | "retreat"
        self._hint_capture_forward_steps = 0
        self._hint_capture_retreat_remaining = 0
        self._hint_capture_return_state = None
        self._hint_capture_confirmed_locked = False  # whether the banner actually appeared
        self._hint_captured_this_door = False  # prevents retrying the same door

        # GOTO_HINT (key room -> locked door): agent/__init__.py plugs
        # ContentIngest's ContentDB in here. If None (e.g. running
        # standalone), this feature is simply inert -- exploration
        # itself is entirely unaffected.
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
        # This tick's enemy detections (reused by agent.content_ingest
        # when deciding whether to include this frame in the batch it
        # sends the VLM as "an enemy is visible" -- so EN.detect isn't
        # run twice on the same frame).
        self.last_mobs = []
        self._wander_heading = None

    # HUD OCR (glyph template matching) was an overwhelming bottleneck
    # at ~140 ms per frame (measured against EN.detect's 1.6 ms and
    # depth_profile's 1.4 ms -- 100x). But the HUD bar's pixels are
    # completely unchanged on most ticks where the room name/HP/heading
    # don't change (heading only changes on a turn, and time only ticks
    # once a second). The reference repo's perception.py hashes and
    # caches the HUD bar for the same reason -- the same approach is
    # used here, only re-running OCR when the frame actually changed.
    def _read_hud_cached(self, obs):
        bar_key = hash(obs[:28, :, :].tobytes())
        if bar_key != self._hud_cache_key:
            self._hud_cache_value = read_hud(obs)
            self._hud_cache_key = bar_key
        return self._hud_cache_value

    # Main entry point
    def step(self, obs):
        hud = self._read_hud_cached(obs)
        self._note_hud_transitions(hud)
        action = self._arbitrate(hud, obs)
        self.prev_frame = obs
        return int(action)

    # A single function that lays out the action priority at a glance.
    # Each branch is a thin shell calling already-validated existing
    # logic -- FLEE/attack/exploration behave exactly as before the
    # refactor; the only two things that changed are (1) GOTO_HINT was
    # newly inserted at priority 3, and (2) DONE falls through to
    # wandering instead of turning in place.
    def _arbitrate(self, hud, obs):
        # 1) If we got hit (HP dropped), counter-attack immediately, right where we are -- unconditional top priority.
        if self.state == "FLEE":
            return self._step_flee(hud, obs)

        # 2) Proactive attack: if an enemy is visible within the cone/range,
        #    steal just this one tick without disturbing whatever else we were doing.
        attack = self._attack_should_engage(hud, obs)
        if attack is not None:
            return attack

        # 3) Couldn't read the HUD, or we're in a corridor (no room name) -- move forward until we reach a room.
        if not hud.ok or hud.room_name is None:
            return self._Action.MOVE_FORWARD

        # 4) Moving toward the key/locked door takes priority over exploration.
        if self.state == "GOTO_HINT":
            return self._step_goto_hint(hud, obs)

        # 5) The exploration (DFS) family.
        if self.state in _EXPLORE_STATES:
            return self._dispatch(hud, obs)

        # 6) Anything else (DONE) -- if the key/door quest isn't finished
        #    yet, check every tick whether it can be resolved now.
        #    _maybe_start_goto was originally only called right after
        #    _finish_survey (finishing a new room's survey), but that
        #    turned out to be a real problem: if the hint only becomes
        #    resolvable after leaving the room where its locked door was
        #    last surveyed, or once every room with a candidate has
        #    already been explored, no further survey ever happens
        #    afterward -- so that one-shot trigger would never fire
        #    again, and we'd never transition into GOTO_HINT at all
        #    (just wandering forever in DONE). On top of that, content_db
        #    is filled in asynchronously by a background thread
        #    (agent/content_ingest.py), so right when we enter DONE, the
        #    last room's VLM analysis might not have finished yet -- so
        #    instead of a single check, retrying every tick catches it
        #    once that data does arrive. This is pure local string/dict
        #    matching (no VLM call), so calling it every tick costs
        #    essentially nothing.
        if self.state == "DONE":
            canon = self._canonicalize(hud.room_name) if hud.room_name else None
            if canon is not None:
                goto = self._maybe_start_goto(canon)
                if goto is not None:
                    return goto

        # 7) Anything else -- keep wandering around, avoiding walls.
        return self._default_wander(hud, obs)

    def _note_hud_transitions(self, hud):
        """Common to every step: detect an HP drop (triggers FLEE) and record [KEY] tag transitions.

        Since backing up doesn't create distance from the enemy
        (confirmed by measurement -- see the note at the top of this
        file), we respond right where we got hit, with aiming. If we get
        hit again while already in FLEE (the fight is still ongoing), we
        don't reset the sweep/aiming state currently in progress --
        resetting every time used to be a bug where the sweep could
        never finish a full cycle (see dev_log.md).
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
            self.flee_sweep_headings_done = 0
            self.flee_sweep_attacks_done = 0
            self.flee_sweep_heading_limit = _FLEE_SWEEP_HEADINGS
            self.flee_fastpath_attack_streak = 0
        self.last_hp = hud.hp

        # The [KEY] tag going on then off means the key was used to open the door (per the README).
        if self.key_was_held and not hud.has_key:
            self.door_unlocked = True
        self.key_was_held = hud.has_key

    def _attack_should_engage(self, hud, obs):
        """Action.ATTACK if an enemy is within the cone/range, otherwise None.

        Proactive attack: until now, combat only started once HP
        actually dropped (i.e. we'd already been hit), which produced
        feedback (based on measured behavior) that we'd "see an enemy
        and not kill it, only reacting after getting hit" (per user
        direction). Since agent.enemies already runs at ~1-2 ms per
        frame, running it every tick barely touches the budget, so we
        check every tick even outside of FLEE, and if an enemy is within
        the cone/range, steal just that one tick for an attack while
        leaving whatever we were doing (the exploration state) untouched.
        A bug found by measurement: without a safeguard, unconditionally
        stealing the tick every time this condition holds meant that
        running into a target that never dies (a false positive, or one
        we keep missing) made us repeat ATTACK forever in the exact same
        spot, freezing exploration entirely (seed 1: position/angle
        locked for over 60 ticks). Since enemy HP maxes out at 4, it
        should normally die within a few hits, so once this streak goes
        on longer than that we treat it as a false positive/miss and
        yield this tick back to whatever we were doing (exploring) -- if
        it's a real enemy, we'll get hit again eventually and go through
        the proper FLEE path (including the sweep).

        We tried "stop and watch an out-of-range enemy, attack once it
        approaches" and reverted it based on measurement (see
        dev_log.md) -- this game already confirmed that backing up
        doesn't create distance, so whether we stand and wait or keep
        exploring, the timing of "it eventually gets close enough to hit
        us" barely changes either way. Meanwhile, standing and watching
        completely halted exploration progress, so the agent got stuck
        in enemy-heavy rooms (e.g. Coral Vault) and died from repeated
        engagements (seed 7: died in 640 steps with this logic on, vs.
        surviving the full 4000 with it off -- confirmed directly, A/B).
        """
        mobs = EN.detect(obs, wall_rgb=self._current_wall_rgb(hud) if hud.ok else None)
        self.last_mobs = mobs
        best = next((m for m in mobs if m.score >= _MOB_MIN_SCORE
                     and m.distance <= _MOB_MAX_COMBAT_DIST_M), None)
        if (best is not None and abs(best.bearing) <= _ATTACK_CONE_HALF_DEG
                and best.distance <= _ATTACK_RANGE_M
                and self._proactive_attack_streak < _FASTPATH_ATTACK_STREAK_MAX):
            self._proactive_attack_streak += 1
            self._last_action_was_forward = False  # so this doesn't corrupt whatever state's is_blocked calculation was in progress
            return self._Action.ATTACK
        self._proactive_attack_streak = 0
        return None

    # Default wander (lowest priority)
    # The fallback once exploration is finished (DONE). It used to just
    # turn in place forever, which is equivalent to throwing away all
    # the remaining time. Setting the target heading to "whatever
    # direction we're currently facing" and running the already-
    # validated _navigate_step (forward -> sidestep around an obstacle
    # -> turn 180 if cornered) as-is gives us "keep walking, avoiding
    # walls" with no new perception logic. If completely blocked, the
    # target rotates 90 degrees and tries again.
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
        """Reset the temporary state _navigate_step uses (when picking a new target)."""
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
        # DONE/GOTO_HINT are handled directly by _arbitrate (never reach here).
        return self._enter_room(hud, entry_heading=None, parent=None)

    # Room-name canonicalization (merges ellipsis-truncated variants, see dev_log.md)
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

    # Entering a room
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

    # RECENTER: leave the doorway and walk to the room's "true" center
    # This used to just walk forward half the front clearance seen at
    # the entrance (a guess based on one axis, one direction), which
    # ended up far from the true center whenever the room's shape was
    # even slightly asymmetric (e.g. the door isn't centered on its
    # wall) -- and doing SURVEY's 360-degree scan from that off-center
    # spot was the direct cause of the "it's not turning at the room's
    # center" feedback (per user direction). Now we measure all four
    # directions (0/90/180/270) directly (turning in place +
    # cone_clearance), and for each axis (0-180, 90-270) walk exactly
    # half of "this-direction distance minus opposite-direction
    # distance," landing geometrically at the true center (verified
    # against ground truth: across 4 room entries, error to the true
    # center was 0.00-0.20 m).
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
                # We just walked in along this axis -- measuring the
                # opposite side (through the door we just came from,
                # which is currently open and looks straight into the
                # next room) would wrongly pick up the next room's wall
                # as "this room's" distance. Instead, we use the fact
                # that we know we're only a short walk in from the
                # doorway, and target half of the forward clearance.
                d_fwd = c.get(entry, _RECENTER_MEASURE_CAP_M)
                dist = min(d_fwd, _RECENTER_MEASURE_CAP_M) / 2.0
                heading = entry
            else:
                pos_open = d_pos >= _RECENTER_MEASURE_CAP_M
                neg_open = d_neg >= _RECENTER_MEASURE_CAP_M
                if pos_open or neg_open:
                    continue  # if either side is uncertain (a door/opening), skip this axis entirely
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
        canon = self._canonicalize(hud.room_name) if hud.room_name else None
        if canon is not None and canon != self.recenter_room:
            # Accidentally crossing into a new room is itself solid
            # proof "there's a door this direction" -- but the old code
            # only recorded this on the new room's side, not on the
            # previous room's node (usually the spawn room), leaving
            # that room's survey never triggered and its other 3
            # directions permanently "unknown" (confirmed by
            # measurement, see dev_log.md -- a case where only 3 of 8-12
            # rooms were ever found). Before leaving, record this
            # direction on the previous room's node too.
            old_node = self.scene.nodes.get(self.recenter_room)
            if old_node is not None and hud.ok and hud.heading is not None:
                cardinal = min(CARDINAL_HEADINGS,
                                key=lambda h: abs(_heading_diff(h, hud.heading)))
                if old_node.exits.get(cardinal) == "unknown":
                    old_node.exits[cardinal] = "open"
                    old_node.exit_leads_to[cardinal] = canon
            return self._enter_room(hud, entry_heading=hud.heading, parent=self.recenter_room)

        if hint_banner_active(obs):
            return A.NO_OP  # depth measurement is meaningless while the banner is up -- just wait this tick out.

        if self.recenter_phase == "measure":
            headings = (0, 90, 180, 270)
            target = headings[self.recenter_measure_idx]
            diff = _heading_diff(hud.heading, target) if hud.ok else 0
            if abs(diff) > _TURN_TOLERANCE:
                return A.TURN_LEFT if diff > 0 else A.TURN_RIGHT
            clearance = geo.cone_clearance(geo.depth_profile(obs), 0.0, _SEEK_CONE_HALF_DEG)
            self.recenter_clearances[target] = min(clearance, _RECENTER_MEASURE_CAP_M)
            self.recenter_measure_idx += 1
            if self.recenter_measure_idx >= 4:
                self._plan_recenter_moves()
                self.recenter_phase = "move"
            return A.NO_OP

        if not self.recenter_move_plan:
            return self._start_survey_scan()
        target_heading, steps_remaining = self.recenter_move_plan[0]
        diff = _heading_diff(hud.heading, target_heading) if hud.ok else 0
        if abs(diff) > _TURN_TOLERANCE:
            return A.TURN_LEFT if diff > 0 else A.TURN_RIGHT
        if steps_remaining <= 0:
            self.recenter_move_plan.pop(0)
            self._last_action_was_forward = False
            return A.NO_OP
        if self._last_action_was_forward and self.prev_frame is not None:
            if is_blocked(self.prev_frame, obs):
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

    # SURVEY: turn through the 4 directions and record wall color
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
                elif (node.exits.get(target) == "unknown"
                        and self.recenter_clearances.get(target, _RECENTER_MEASURE_CAP_M)
                        < _RECENTER_MEASURE_CAP_M):
                    # If the VLM says "not a door," cross-check it
                    # against the clearance we already measured for the
                    # same direction, from the same spot (the room
                    # center), during RECENTER (recenter_clearances).
                    # If that measurement was "confidently a nearby
                    # wall" (below the cap), trust the VLM and mark it
                    # "wall" right away, skipping SEEK's physical
                    # approach (back to the original speed). If the
                    # measurement was at or above the cap (meaning the
                    # far side is open, possibly a door/opening), don't
                    # commit to "wall" just because the VLM said so --
                    # leave it "unknown" so SEEK still confirms it
                    # physically. Found by measurement (per user
                    # direction): a single VLM misjudgment could
                    # permanently lose a door, but forcing every single
                    # direction to be physically bumped into (the
                    # previous fix) slowed exploration down enough that
                    # we saw fewer rooms, took more combat exposure, and
                    # died earlier.
                    node.exits[target] = "wall"
        self.survey_queue.pop(0)
        if self.survey_queue:
            return self._Action.NO_OP
        return self._finish_survey(canon, node)

    def _finish_survey(self, canon, node):
        # The single most-common color across the whole frame was too
        # easily swallowed by the neighboring wall color and
        # misjudged as "wall" in the door's own direction (confirmed by
        # measurement, see dev_log.md). So here we only record the
        # room's wall color (for later QA/memory), and let SEEK confirm
        # door-vs-wall by actually walking there (is_blocked) -- slower, but reliable.
        colors = [c for c in self.survey_samples.values() if c]
        if colors and node.wall_color is None:
            node.wall_color = max(set(colors), key=colors.count)
        # Every time we finish surveying a room, re-check whether the
        # rooms seen so far are enough to resolve the hint (Phase 4) --
        # if so, pause DFS and head there.
        goto = self._maybe_start_goto(canon)
        if goto is not None:
            return goto
        return self._start_seek(canon, node)

    # SEEK: walk toward a candidate direction, avoid obstacles if blocked, enter if a door is found
    def _start_seek(self, canon, node):
        candidates = node.unknown_headings()
        if not candidates:
            node.done = True
            return self._start_backtrack(canon)
        # Try whichever direction the VLM thought "looks like a door"
        # during SURVEY first -- so we don't waste time physically
        # bumping into the candidates that probably aren't doors first.
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

    # If 8+ attempts using only physical collision checks (is_blocked)
    # still haven't gotten us through, escalate to progressively
    # stronger recovery (per user direction: reproduced and confirmed by
    # measurement a problem where repeatedly bumping into a wall led to
    # death, see dev_log.md -- stuck at the same spot/angle, oscillating
    # for 70+ ticks without moving an inch while in RETURN, and dying to
    # an enemy in the meantime):
    #   1st failure: back up briefly (2 steps), then realign from
    #                scratch -- retry with a slightly different approach angle.
    #   2nd failure: sidestep (turn 90 degrees, move 3 steps, then
    #                return to the original direction) -- backing up
    #                only creates distance "along the same line," but if
    #                we're standing slightly off to one side of the door
    #                (per user direction: "the door needs to be dead
    #                center in front of me to walk straight out"),
    #                backing up alone can't fix that. Found by
    #                measurement: there were cases where a real door was
    #                physically blocked because the approach angle was
    #                off, and SEEK wrongly marked it "wall" and missed
    #                the whole room entirely (see dev_log.md).
    #   3rd failure: back up much farther (5 steps) and re-approach from an entirely new position.
    #   4th failure: check once with the VLM, "is there really a
    #                door/gap visible?" -- if so, bias toward that
    #                direction, back up, and retry.
    #   Still no luck: call on_final_give_up() (SEEK marks it a wall and
    #                moves to the next candidate; RETURN gives up on
    #                this target and backtracks one more step).
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

    # Shared movement logic run every tick: align to the target heading
    # -> if blocked straight ahead, sidestep around it (closest offset
    # first) -> once past it, return to the original target. Used by
    # both SEEK and RETURN (per user direction: treat a local obstacle
    # like a tree as "having a boundary to go around," but return to the
    # original direction once past it -- contourner).
    def _navigate_step(self, hud, obs, target_heading, on_exhausted):
        # If there's a pre-planned action sequence left over from staged
        # recovery (_escalate_nav_failure), drain that first --
        # it takes priority over the usual align/move decision.
        if self._nav_action_queue:
            self._last_action_was_forward = False
            return self._nav_action_queue.pop(0)

        effective_target = (target_heading + self._nav_bias_deg) % 360
        diff = _heading_diff(hud.heading, effective_target)
        if abs(diff) > _TURN_TOLERANCE:
            self._last_action_was_forward = False
            self._nav_probed_center = False
            return self._Action.TURN_LEFT if diff > 0 else self._Action.TURN_RIGHT

        # Pre-emptively avoiding obstacles via the depth profile is only
        # ever a secondary aid to reduce how often we visibly bump into
        # things -- the final judgment of whether we actually moved
        # always comes from is_blocked (the frame diff). A bug found by
        # measurement: standing in front of a wall-hung picture, the
        # depth profile kept insisting "3.43 m of open space" (the wall
        # color's saturation sat right at the threshold, throwing the
        # boundary detection off slightly), while we were, in fact,
        # completely stuck in place (confirmed even by checking the
        # position coordinates -- see dev_log.md) -- recover_attempts
        # never incremented even once, and we just repeated
        # MOVE_FORWARD for 400+ steps. Even if depth says "fine," if the
        # last forward attempt didn't actually work, we always treat it as blocked.
        if self._last_action_was_forward and self.prev_frame is not None:
            if is_blocked(self.prev_frame, obs):
                self.recover_attempts += 1
                if self.recover_attempts > _RECOVER_MAX_ATTEMPTS:
                    self._nav_bias_deg = 0.0
                    return on_exhausted()
                # Treat the recovery turn as the same kind of bias as a
                # sidestep -- if we just returned a bare TURN_RIGHT, the
                # alignment check at the top would see "doesn't match
                # the target" on the very next tick and immediately turn
                # us right back, so recover_attempts could never
                # accumulate (a real bug, confirmed by measurement, see
                # dev_log.md). Moving the bias along with it makes this
                # new direction "the target for now," which is what
                # actually gives it a chance to be tried.
                self._nav_bias_deg = geo.wrap180(self._nav_bias_deg - 15.0)
                self._last_action_was_forward = False
                return self._Action.TURN_RIGHT
            # The last forward step actually worked -- reset the
            # recovery count, and ease the bias back 15 degrees toward
            # the original target (per user direction: once past the
            # obstacle, return to the original direction). There was a
            # bug where, based purely on the depth-profile prediction,
            # the bias got reset to 0 in a single tick -- but since we
            # actually hadn't moved from that spot, immediately
            # realigning to the target heading -> getting blocked again
            # -> resetting again... produced an infinite oscillation
            # (confirmed by measurement, see dev_log.md). Only easing
            # the bias back by 15 degrees "when a step forward actually
            # succeeded" avoids this.
            self.recover_attempts = 0
            if self._nav_bias_deg > 0:
                self._nav_bias_deg = max(0.0, self._nav_bias_deg - 15.0)
            elif self._nav_bias_deg < 0:
                self._nav_bias_deg = min(0.0, self._nav_bias_deg + 15.0)

        if hint_banner_active(obs):
            # Depth measurement is meaningless while the banner is up -- proceed on the is_blocked judgment alone.
            self._last_action_was_forward = True
            return self._Action.MOVE_FORWARD

        profile = geo.depth_profile(obs)
        # The front safety check keeps using the originally validated
        # 15-degree cone (_SEEK_CONE_HALF_DEG) -- reusing the 8-degree
        # sidestep cone here turned out to be a bug found by
        # measurement: a cone that narrow can misjudge a gap actually
        # narrower than the agent's own width (AGENT_RADIUS 0.4 m) as "open."
        center_clear = geo.cone_clearance(
            profile, 0.0, _SEEK_CONE_HALF_DEG) >= _SEEK_SAFE_CLEARANCE
        # A newly found bug (see dev_log.md): when facing a door/passage
        # head-on from a few meters away, if the door is narrower than
        # the 15-degree cone, both edges of the cone land on the wall
        # beside the door frame, and floor_boundary()'s heuristic ("if
        # the floor isn't visible all the way to the bottom of the
        # frame, treat it as a very close obstacle," the NEAR_RANGE
        # fallback) flattens that reading down to 0.6 m regardless of
        # the wall's actual distance -- so even in a direction the VLM
        # had *just* confirmed was a real door, we'd never attempt a
        # single step forward and immediately gave up as blocked. The
        # depth profile is only ever a secondary, pre-emptive-avoidance
        # signal -- it should never be the sole reason to refuse to move
        # forward at all. If we've aligned to this direction but haven't
        # actually tried moving forward even once yet, we make one real
        # attempt regardless of what depth says, and physically confirm
        # with is_blocked() (if it really is blocked, recover_attempts
        # increments normally on the very next tick and flows into the
        # existing recovery logic -- the safety net is still fully intact).
        if center_clear or not self._nav_probed_center:
            # Easing the bias back toward the original target only
            # happens "when a step forward actually succeeded" (at the
            # is_blocked check above) -- doing it here based purely on
            # the depth prediction is what caused the infinite
            # oscillation bug mentioned above.
            self._last_action_was_forward = True
            self._nav_probed_center = True
            return self._Action.MOVE_FORWARD

        # Blocked straight ahead -- check whether we can slip past sideways, closest offset first.
        for off in _SIDESTEP_OFFSETS_DEG[1:]:
            clearance = geo.cone_clearance(profile, off, _SIDESTEP_CONE_HALF_DEG)
            if clearance >= _SEEK_SAFE_CLEARANCE:
                self._nav_bias_deg = geo.wrap180(self._nav_bias_deg + off)
                self._last_action_was_forward = False
                return self._Action.TURN_LEFT if off > 0 else self._Action.TURN_RIGHT

        # Blocked straight ahead AND to both sides (+/-15, +/-30) -- we're
        # cornered (confirmed by measurement: this looked like wandering
        # the room for hundreds of ticks in fine 15-degree turns looking
        # for a gap -- user feedback: "it's weirdly turning right...
        # keeps bumping into the wall"). Measurement showed this "stuck"
        # state was sometimes actually a locked door (a padlock panel)
        # -- no amount of retrying gets through without the key, so if a
        # padlock is visible, give up immediately instead of wasting
        # time turning and retrying. That said, the color heuristic
        # (lock_visible) alone isn't enough to commit "locked" to exits
        # -- there's a false-positive risk. Instead, we briefly pause
        # SEEK (HINT_CAPTURE) and approach into the door's touch radius,
        # doing a final confirmation via whether the hint banner
        # (hint_banner_active, 100% reliable) actually appears -- if it
        # does, we commit "locked" and also grab the hint text (per user
        # direction: go up to the door, read the hint, then back off to
        # roughly where we were). If we already tried this exact door
        # once (_hint_captured_this_door), we don't retry -- give up right away.
        if geo.lock_visible(obs):
            if self.state == "SEEK" and not self._hint_captured_this_door:
                return self._start_hint_capture()
            self._nav_bias_deg = 0.0
            return on_exhausted()

        # If it's not a padlock, per user direction, don't keep making
        # small turns -- react with an immediate 180-degree turn-around
        # instead. At minimum this guarantees we actually leave the
        # cornered spot we're stuck in, so the next judgment resolves
        # much faster (fewer ticks).
        self.recover_attempts += 1
        # A newly found infinite loop (see dev_log.md): while turning
        # 180 degrees and wandering a wide loop around the room, "a
        # forward step actually succeeding" can happen several times
        # along the way, and each time resets recover_attempts to 0
        # (that's the intent of the is_blocked branch right above it --
        # clear the count once we get past a local obstacle like a
        # tree). The problem: in front of a narrow door where
        # door_visible=True but lock_visible=False (not a padlock, but
        # narrower than the 15-degree safety cone, so depth keeps
        # misjudging it as "blocked"), wandering the wide loop
        # eventually brings the bias back to 0, re-approaching the same
        # door -> colliding again -> another 180-degree turn, forever,
        # with recover_attempts never once exceeding 8 (measured: circled
        # the same door for 600+ ticks with zero room transitions). So
        # "how many times, total, has this target heading hit this
        # 180-degree give-up" is tracked in a separate counter that
        # doesn't get reset by a successful step in between.
        self._nav_cornered_count += 1
        if (self.recover_attempts > _RECOVER_MAX_ATTEMPTS
                or self._nav_cornered_count > _MAX_CORNERED_PER_TARGET):
            self._nav_bias_deg = 0.0
            return on_exhausted()
        self._nav_bias_deg = geo.wrap180(self._nav_bias_deg + 180.0)
        self._last_action_was_forward = False
        return self._Action.TURN_RIGHT

    # HINT_CAPTURE: approach the locked door, read the hint banner, then back off
    def _start_hint_capture(self):
        self._hint_capture_phase = "approach"
        self._hint_capture_forward_steps = 0
        self._hint_capture_confirmed_locked = False
        self._hint_captured_this_door = True  # only try once, regardless of success
        self._hint_capture_return_state = self.state  # usually "SEEK"
        self.state = "HINT_CAPTURE"
        return self._Action.NO_OP

    def _step_hint_capture(self, hud, obs):
        A = self._Action
        if self._hint_capture_phase == "approach":
            if hint_banner_active(obs):
                self._hint_capture_confirmed_locked = True
                # Only the first captured banner text is kept -- even
                # with multiple locked doors, the first hint is the one
                # that counts (not overwritten on a later revisit).
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
                # Never got into the touch radius (either a color false
                # positive, or the key was already held and the door
                # opened outright instead of showing the banner) -- give up and back off.
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

    # GOTO_HINT: key room -> locked door (reuses only the existing movement primitives)
    def _locked_door_room(self):
        """The room with the locked door. The location where we actually
        triggered the hint banner is the primary evidence; failing that,
        find whichever room has an exit confirmed as "locked"."""
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
        """Hint text + observations so far -> the name of the room with the key (None if unresolved).

        Fills any CV-measured wall color we already have into ContentDB
        first, so a room whose VLM response hasn't come back yet can
        still be matched against a "...walls" template hint.
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
        """Called right after a survey finishes (and every tick in DONE
        -- see _arbitrate) -- returns the first action if there's a
        reason to switch into GOTO_HINT right now, otherwise None."""
        if self.door_unlocked:
            return None
        # Already holding the key -> go to the locked door and open it.
        if self.key_was_held and not self._goto_door_attempted:
            door_room = self._locked_door_room()
            if (door_room and self._locked_door_heading(door_room) is not None
                    and self.scene.shortest_path(canon, door_room) is not None):
                return self._start_goto(door_room, "door")
        # Don't have the key yet -> if the hint resolves to a specific room, head there.
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
            return self._Action.MOVE_FORWARD  # in a corridor -- keep moving forward

        if canon not in self.scene.nodes:
            # A room we've never seen before -- either we just opened
            # the locked door and walked through, or we stumbled onto a
            # new room by chance while navigating. Either way, register
            # it properly in the graph and hand off to the normal
            # exploration routine (RECENTER->SURVEY) -- this registration
            # is essential for being able to answer "what was behind the locked door."
            parent = self.goto_prev_room
            entry = None
            if hud.heading is not None:
                entry = min(CARDINAL_HEADINGS,
                            key=lambda h: abs(_heading_diff(h, hud.heading)))
            self.goto_leg = None
            self.goto_target_room = None
            # Having crossed the graph, the DFS stack is now out of sync
            # with our real position. Before pushing the new room, we
            # need to rebuild it as "root -> ... -> the room we just
            # came from," so later backtracking doesn't return to the wrong room.
            if parent and parent in self.scene.nodes:
                self._rebuild_stack(parent)
            return self._enter_room(hud, entry_heading=entry, parent=parent)

        self.goto_prev_room = canon

        # If the door has been unlocked and we've left the target room, the trip is over -- return to exploration.
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
                # Arrived in the key's room -- need to walk around the
                # room to actually step on the key (per the README,
                # pickup is automatic, we just need to pass over it).
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
        """Arrived in the room with the locked door -- walking toward it
        opens it automatically (per the README). Keep walking forward
        afterward, into the room beyond it."""
        heading = self._locked_door_heading(self.goto_target_room)
        if heading is None:
            return self._resume_after_goto(hud)
        if heading != self.target_heading:
            self.target_heading = heading
            self._reset_nav_state()
        return self._navigate_step(hud, obs, heading,
                                    lambda: self._resume_after_goto(hud))

    def _resume_after_goto(self, hud):
        """End of the trip (success or give-up, doesn't matter) -- rebuild
        the DFS stack based on where we actually are now, and return to normal exploration."""
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
        # In a corridor, etc. -- once we reach a room next tick, INIT registers it normally.
        self.state = "INIT"
        return self._Action.MOVE_FORWARD

    def _rebuild_stack(self, canon):
        """After crossing the graph via GOTO_HINT, the DFS stack (the
        backtrack path) is out of sync with our real position. Rebuild
        it as the shortest path from the root to our current room, so
        later backtracking returns to the correct parent room."""
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
            # Connected to an already-visited room -- head back to where we came from.
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

        # A new room!
        return self._enter_room(hud, entry_heading=self.target_heading,
                                 parent=self.seek_origin_room)

    # RETURN: head back to a specific room (also doubles as the backtrack)
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
                # We accidentally crossed a door during RECENTER, so this
                # room never got a survey (wall color recorded) at all --
                # since we're back here anyway, finish that now (a
                # problem confirmed by measurement, see dev_log.md). Our
                # current position is right at the threshold/doorway we
                # just arrived at via RETURN, so calling
                # _start_survey_scan() directly here would be the bug
                # where the 360-degree turn happens right at the door
                # instead of the room's center (reproduced and confirmed
                # per user direction) -- route through RECENTER first, to
                # actually walk to the true center, before moving on to SURVEY.
                return self._start_recenter(canon, entry_heading=hud.heading)
            return self._start_seek(canon, node)

        def on_final_give_up():
            # If even multiple staged attempts can't find a path that's
            # supposed to be open (a bug reproduced and confirmed per
            # user direction: stuck at the same spot, oscillating angle
            # only, not moving an inch for 70+ ticks, then dying to an
            # enemy -- see dev_log.md), give up on this target. Calling
            # _start_backtrack directly again here risks popping the
            # stack a second time (if we got stuck partway through a
            # backtrack) -- instead, call _start_seek fresh, based on
            # the room we're actually in right now: if there's still an
            # unvisited direction, it tries that; if not, _start_seek
            # backtracks normally on its own (the same, already-
            # validated path, so the stack never gets corrupted).
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

    # FLEE: attack (strike) right where we got hit -- CV aiming, falls back to a sweep on failure
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
        # Since backing up doesn't create distance (confirmed by
        # measurement -- see the note at the top of this file), we
        # respond right where we got hit. Priority 1 is agent.enemies's
        # pure CV detection (~1-2 ms, no round-trip delay, an exact
        # bearing in degrees) -- attack immediately if inside the
        # cone/range, otherwise turn to face exactly that direction. If
        # it can't find anything, switch that instant to the
        # deterministic sweep (attack twice -> turn 15 degrees, 24 times
        # = 360 degrees) -- since the attack cone (+/-20 degrees) is
        # wider than the turn interval (15 degrees), this is
        # geometrically guaranteed to catch it.
        #
        # The VLM (locate_enemy) was dropped from the aiming decision --
        # a bug confirmed by measurement: its 3-tier left/center/right
        # judgment marked "center" using a much narrower band than the
        # actual attack cone (+/-20 degrees), so it kept misjudging
        # "not center" even while already inside the cone, oscillating
        # between two headings for 76+ ticks without ever landing a hit
        # (see dev_log.md). CV gives an exact bearing in degrees with
        # none of that ambiguity, and when it can't find anything, we go
        # straight to the 100%-guaranteed sweep instead of waiting
        # longer on a VLM judgment. Once switched into the sweep, even
        # if we get hit again mid-sweep (step() doesn't reset it if
        # we're already in FLEE), it continues from where it left off
        # rather than starting over.
        self.flee_ticks += 1
        if self.flee_ticks > _FLEE_STRIKE_MAX_TICKS:
            return self._resume_after_flee()

        if not self.flee_sweep_active:
            # Priority 1: close-range-only bearing estimate
            # (enemies.close_range_bearing). Confirmed by measurement
            # (see dev_log.md, seed 7): by the time we enter strike, the
            # enemy is already 1.3-1.5 m away, right in front of us, so
            # its body fills almost the whole frame -- detect()'s
            # humanoid-ratio/floor-contact judgment structurally returns
            # zero candidates at that range, and every fight dropped
            # straight into the 24-direction sweep (up to 72 ticks) --
            # the reason kills stopped feeling as instant as before.
            # close_range_bearing skips the humanoid-shape judgment
            # entirely and just looks for "a big blob that's neither
            # wall nor floor," so it still works at this range (and
            # since we've already taken damage to get here, the false-
            # positive risk is low to begin with).
            bearing = EN.close_range_bearing(
                obs, wall_rgb=self._current_wall_rgb(hud),
                require_bottom_band=self.flee_ever_attacked)
            if bearing is not None and self.flee_fastpath_attack_streak < _CLOSE_RANGE_ATTACK_STREAK_MAX:
                if abs(bearing) <= _ATTACK_CONE_HALF_DEG:
                    self.flee_ever_attacked = True
                    self.flee_fastpath_attack_streak += 1
                    return self._Action.ATTACK
                return self._Action.TURN_LEFT if bearing > 0 else self._Action.TURN_RIGHT

            # Priority 2: mid-range humanoid detection (if priority 1
            # failed -- e.g. the enemy hasn't closed in yet, or just backed off).
            mobs = EN.detect(obs, wall_rgb=self._current_wall_rgb(hud))
            best = next((m for m in mobs if m.score >= _MOB_MIN_SCORE
                         and m.distance <= _MOB_MAX_COMBAT_DIST_M), None)
            if best is not None and self.flee_fastpath_attack_streak < _FASTPATH_ATTACK_STREAK_MAX:
                if abs(best.bearing) <= _ATTACK_CONE_HALF_DEG:
                    if best.distance <= _ATTACK_RANGE_M:
                        self.flee_ever_attacked = True
                        self.flee_fastpath_attack_streak += 1
                        return self._Action.ATTACK
                    # Already facing the right direction (in the cone)
                    # but out of range (3.0 m) -- a bug found by
                    # measurement: treating this case as "not aligned"
                    # too and only turning made the bearing's sign flip
                    # back and forth slightly (turning doesn't close the
                    # distance), oscillating left-right forever without
                    # ever landing an attack (see dev_log.md). Move
                    # forward instead, to close the distance first.
                    self._last_action_was_forward = True
                    return self._Action.MOVE_FORWARD
                return self._Action.TURN_LEFT if best.bearing > 0 else self._Action.TURN_RIGHT

            # Neither found anything -- go straight to the sweep (no VLM aiming -- see the note above).
            self.flee_sweep_active = True
            self.flee_sweep_headings_done = 0
            self.flee_sweep_attacks_done = 0
            # If we'd already landed a hit via close/mid-range detection
            # and suddenly can't find anything, it's likely dead and
            # gone (see the _FLEE_SWEEP_QUICK_CHECK_HEADINGS note above)
            # -- only check briefly in that case.
            self.flee_sweep_heading_limit = (
                _FLEE_SWEEP_QUICK_CHECK_HEADINGS if self.flee_ever_attacked
                else _FLEE_SWEEP_HEADINGS
            )

        # --- Deterministic sweep: attack N times -> turn 15 degrees,
        # repeat 24 times (or fewer, if we already landed a hit and then lost it) ---
        if self.flee_sweep_headings_done >= self.flee_sweep_heading_limit:
            return self._resume_after_flee()
        if self.flee_sweep_attacks_done < _FLEE_SWEEP_ATTACKS_PER_HEADING:
            self.flee_sweep_attacks_done += 1
            self.flee_ever_attacked = True
            return self._Action.ATTACK
        self.flee_sweep_attacks_done = 0
        self.flee_sweep_headings_done += 1

        # If we've already landed at least one hit, re-check with the
        # VLM every few headings whether it's "still there," to cut down
        # on wastefully attacking a corpse through the rest of the
        # sweep. Check with CV first (free), and only fall back to the
        # VLM if CV also can't find anything. This early give-up is only
        # allowed once we've swept at least half (180 degrees) -- a bug
        # found by measurement: flee_ever_attacked only means "we issued
        # an ATTACK," not "we definitely landed one" (we might have
        # attacked a false positive). If the enemy genuinely started out
        # of view (e.g. behind us), giving up early with "not seeing it
        # must mean it's dead" after only a few directions could mean we
        # never find a genuinely still-alive enemy and cut the sweep
        # short. We only trust this judgment once at least half the
        # sweep is done.
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
        self.flee_sweep_headings_done = 0
        self.flee_sweep_attacks_done = 0
        self.flee_sweep_heading_limit = _FLEE_SWEEP_HEADINGS
        self.flee_fastpath_attack_streak = 0
        self.state = self.pre_flee_state or "SURVEY"
        self.target_heading = self.pre_flee_target
        if self.state == "RECENTER" and self.recenter_phase == "measure":
            # RECENTER's "measure" phase does nothing but turn in place
            # through the 4 directions (up to a few dozen ticks), never
            # reacting to a threat at all -- a real problem confirmed by
            # measurement (reproduced per user direction): if an enemy
            # is still in the room when we return from FLEE into
            # RECENTER, standing still to finish measuring means getting
            # hit again within a few ticks and dropping right back into
            # FLEE, over and over, unable to escape the room and dying
            # (seed 0: died in just 216 steps). Right after combat,
            # survival/continuing to explore matters more than a
            # precise room-center calculation, so if we were still in
            # the "measure" phase (i.e. hadn't even planned the move
            # yet), we skip finishing the measurement, accept our
            # current position as good enough, and go straight to
            # SURVEY. If we were already in the "move" phase (actually
            # walking toward the center), that movement is itself a form
            # of evasion, so we just let it continue.
            return self._start_survey_scan()
        self._need_align = True
        self._last_action_was_forward = False
        # If we were sidestepping around a local obstacle before combat
        # started, that bias is still set -- if we don't reset it, we'd
        # try to return not to the original target heading
        # (pre_flee_target) but to some biased, unrelated direction --
        # the cause of the measured feedback "after killing the enemy,
        # it doesn't go back to where it was" (see dev_log.md).
        self._nav_bias_deg = 0.0
        self._nav_probed_center = False
        return self._Action.NO_OP
