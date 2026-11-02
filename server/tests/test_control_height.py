"""제어 스레드의 키보드 높이(2-5): 속도 × 시간, 타임아웃, 상자 끝, 손 없을 때, 조종 시작 전 무시, IK와 변화량 제한."""

import numpy as np
import pytest

from server import config
from server.control import ControlWorker
from server.ik import ArmIK
from server.mapping import HandMapper

Z_LO, Z_HI = config.WORKSPACE_Z_M
SPEED = config.HEIGHT_SPEED_M_S
SLOW = config.HEIGHT_SLOW_SPEED_M_S
TICK_S = 1 / config.CONTROL_HZ
CENTER_Z = config.RELATIVE_START_POSITION_M[2]


class FakeSim:
    def __init__(self):
        self.ctrl_log = []
        self.snapshot = None

    def set_ctrl(self, ctrl):
        self.ctrl_log.append(np.asarray(ctrl).copy())


def hand(x=0.5, y=0.5, scale=0.13, detected=True):
    if not detected:
        return {"type": "hand", "t": 0.0, "detected": False, "wrist": None, "index_tip": None, "thumb_tip": None,
                "pinch": None, "pinch_ratio": None, "palm_open": None, "hand_scale": None}
    return {"type": "hand", "t": 0.0, "detected": True, "wrist": [x, y, 0.0], "index_tip": [x, y, 0.0],
            "thumb_tip": [x, y, 0.0], "pinch": 0.1, "pinch_ratio": 0.8, "palm_open": True, "hand_scale": scale}


@pytest.fixture(scope="module")
def shared_ik():
    return ArmIK()


def make(shared_ik, mapper=None):
    sim = FakeSim()
    clock = {"t": 0.0}
    cw = ControlWorker(
        sim,
        ik_factory=lambda: shared_ik,
        mapper=mapper or HandMapper(mode="relative", start_frames=3, z_from_hand_scale=config.Z_FROM_HAND_SCALE),
        clock=lambda: clock["t"],
    )
    shared_ik.reset()
    cw.prepare()
    cw.clock = clock
    return cw, sim


def tick(cw, msg=None):
    """한 틱: (있으면) 손 메시지를 넣고 시계를 1/30초 진행하며 tick."""
    cw.clock["t"] += TICK_S
    if msg is not None:
        cw.submit_hand(msg, cw.clock["t"])
    cw.tick()


def clutch(cw, x=0.5, y=0.5, scale=0.13):
    for _ in range(3):
        tick(cw, hand(x, y, scale))


def target_z(cw):
    return float(cw._mapper.last_target_m[2])


def hold(cw, direction, slow, seconds, repeat_hz=20, hand_msg=None):
    """누르고(첫 메시지), 20Hz로 반복 메시지를 보내고, seconds 뒤에 뗀다(dir 0). 틱은 30Hz."""
    clock = cw.clock
    t0 = clock["t"]
    cw.submit_height(direction, slow, t0)
    next_repeat = t0 + 1 / repeat_hz
    end = t0 + seconds
    while clock["t"] + TICK_S < end - 1e-12:
        clock["t"] += TICK_S
        while next_repeat <= clock["t"] + 1e-12:
            cw.submit_height(direction, slow, next_repeat)
            next_repeat += 1 / repeat_hz
        if hand_msg is not None:
            cw.submit_hand(hand_msg, clock["t"])
        cw.tick()
    clock["t"] = end
    cw.submit_height(0, False, end)
    cw.tick()


def test_config_defaults_and_where_to_change_speed():
    assert (config.HEIGHT_SPEED_M_S, config.HEIGHT_SLOW_SPEED_M_S, config.HEIGHT_COMMAND_TIMEOUT_S) == (0.08, 0.02, 0.3)
    assert config.Z_FROM_HAND_SCALE is False
    cw = ControlWorker(FakeSim())
    assert cw._mapper.z_from_hand_scale is False  # 앱 기본은 높이를 키보드로
    assert cw._height.speed_m_s == config.HEIGHT_SPEED_M_S  # 속도는 config 값에서 온다
    assert cw._height.slow_speed_m_s == config.HEIGHT_SLOW_SPEED_M_S


def test_height_before_the_first_clutch_is_ignored(shared_ik):
    cw, sim = make(shared_ik)
    hold(cw, -1, False, 1.0)
    state = cw.state()
    assert sim.ctrl_log == []  # 로봇은 가만히
    assert state["target_m"] is None and state["target_z_m"] is None
    assert state["height_active"] is False
    # 그 뒤 조종을 시작해도 이전에 누른 키는 반영되지 않는다
    clutch(cw)
    assert target_z(cw) == pytest.approx(CENTER_Z, abs=1e-9)


def test_pressing_before_the_clutch_does_not_move_the_robot_even_while_held(shared_ik):
    cw, sim = make(shared_ik)
    for _ in range(40):  # 키를 계속 누르고 있다(브라우저는 반복해서 보낸다)
        cw.submit_height(-1, False, cw.clock["t"])
        tick(cw)
        assert cw.state()["height_active"] is False
    assert sim.ctrl_log == []  # 로봇은 가만히
    # 조종 시작 전 명령은 기록째로 버려서 state에도 정지(0)로 보인다
    assert cw.state()["height_dir"] == 0
    assert cw.state()["target_z_m"] is None


@pytest.mark.parametrize("seconds", [0.25, 0.5, 0.6])
def test_down_moves_speed_times_time(shared_ik, seconds):
    cw, sim = make(shared_ik)
    clutch(cw)
    z0 = target_z(cw)
    hold(cw, -1, False, seconds)
    assert target_z(cw) == pytest.approx(z0 - SPEED * seconds, abs=1e-9)
    assert z0 - target_z(cw) < z0 - Z_LO  # 상자 끝에 안 닿는 범위에서 검사


def test_up_moves_speed_times_time(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw)
    z0 = target_z(cw)
    hold(cw, +1, False, 0.5)
    assert target_z(cw) == pytest.approx(z0 + SPEED * 0.5, abs=1e-9)


def test_slow_moves_slow_speed_times_time(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw)
    z0 = target_z(cw)
    hold(cw, +1, True, 1.0)
    assert target_z(cw) == pytest.approx(z0 + SLOW * 1.0, abs=1e-9)
    hold(cw, -1, True, 2.0)
    assert target_z(cw) == pytest.approx(z0 + SLOW * 1.0 - SLOW * 2.0, abs=1e-9)


def test_the_tick_rate_does_not_matter(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw)
    z0 = target_z(cw)
    t0 = cw.clock["t"]
    cw.submit_height(-1, False, t0)
    end = t0 + 0.4
    sent = t0
    t = t0
    while t < end - 1e-12:
        t = min(end, t + 0.017)  # 틱이 30Hz가 아니라 약 59Hz여도
        while sent + 0.05 <= t + 1e-12:
            sent += 0.05
            cw.submit_height(-1, False, sent)
        cw.clock["t"] = t
        cw.tick()
    cw.submit_height(0, False, end)
    cw.tick()
    assert z0 - target_z(cw) == pytest.approx(SPEED * 0.4, abs=1e-9)


def test_stops_by_itself_when_messages_stop(shared_ik):
    """브라우저가 죽어서 키를 떼었다는 메시지가 오지 않아도 0.3초 뒤 멈춘다."""
    cw, sim = make(shared_ik)
    clutch(cw)
    z0 = target_z(cw)
    cw.submit_height(-1, False, cw.clock["t"])  # 한 번 보내고 끝
    zs = []
    for _ in range(60):  # 2초
        tick(cw)
        zs.append(target_z(cw))
    assert z0 - zs[-1] == pytest.approx(SPEED * config.HEIGHT_COMMAND_TIMEOUT_S, abs=1e-9)  # 0.3초 × 0.08 = 2.4 cm
    assert zs[-1] == zs[-30]  # 그 뒤로는 안 움직인다
    assert cw.state()["height_dir"] == 0
    assert cw.state()["height_active"] is False


def test_timeout_is_extended_by_each_repeat(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw)
    z0 = target_z(cw)
    t0 = cw.clock["t"]
    for k in range(5):  # 0.2초 동안 20Hz로 반복
        cw.submit_height(-1, False, t0 + 0.05 * k)
    for _ in range(60):
        tick(cw)
    assert z0 - target_z(cw) == pytest.approx(SPEED * (0.2 + config.HEIGHT_COMMAND_TIMEOUT_S), abs=1e-9)


def test_does_not_leave_the_workspace(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw)
    hold(cw, +1, False, 3.0)
    assert target_z(cw) == pytest.approx(Z_HI)
    assert cw.state()["height_limit"] == "top"
    hold(cw, -1, False, 5.0)
    assert target_z(cw) == pytest.approx(Z_LO)
    assert cw.state()["height_limit"] == "bottom"
    for ctrl in sim.ctrl_log:
        assert np.all(np.isfinite(ctrl))
    hold(cw, +1, False, 0.1)  # 반대 키는 바로 반응한다(죽은 구간 없음)
    assert target_z(cw) == pytest.approx(Z_LO + SPEED * 0.1, abs=1e-9)


def test_full_travel_time_matches_the_speed(shared_ik):
    """맨 위에서 맨 아래까지: (0.12 - 0.015) / 0.08 = 1.3125초."""
    cw, sim = make(shared_ik)
    clutch(cw)
    hold(cw, +1, False, 3.0)
    top_time = cw.clock["t"]
    cw.submit_height(-1, False, top_time)
    t = top_time
    reached = None
    while t < top_time + 3.0:
        t += TICK_S
        cw.submit_height(-1, False, t)  # 키를 계속 누르고 있다
        cw.clock["t"] = t
        cw.tick()
        if reached is None and target_z(cw) <= Z_LO + 1e-9:
            reached = t - top_time
    assert reached == pytest.approx((Z_HI - Z_LO) / SPEED, abs=TICK_S)


def test_works_while_the_hand_is_not_visible(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw, 0.4, 0.45)
    tick(cw, hand(0.45, 0.45))
    xy = cw._mapper.last_target_m[:2].copy()
    for _ in range(5):
        tick(cw, hand(detected=False))
    assert cw.state()["controlling"] is False
    z0 = target_z(cw)
    hold(cw, -1, False, 0.4)
    assert target_z(cw) == pytest.approx(z0 - SPEED * 0.4, abs=1e-9)
    np.testing.assert_array_equal(cw._mapper.last_target_m[:2], xy)  # 손이 없으니 x, y는 그대로
    assert cw.state()["controlling"] is False


def test_hand_size_does_not_change_the_height_by_default(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw, 0.5, 0.5, 0.13)
    for scale in (0.07, 0.19, 0.25, 0.05):
        tick(cw, hand(0.5, 0.5, scale))
        assert target_z(cw) == pytest.approx(CENTER_Z, abs=1e-6)


def test_hand_size_changes_height_when_z_from_hand_scale_is_on(shared_ik):
    cw, sim = make(shared_ik, mapper=HandMapper(mode="relative", start_frames=3, z_from_hand_scale=True))
    clutch(cw, 0.5, 0.5, 0.13)
    for _ in range(40):
        tick(cw, hand(0.5, 0.5, 0.18))
    assert target_z(cw) < CENTER_Z - 0.03  # 손이 가까워져서 내려갔다
    z_hand = target_z(cw)
    hold(cw, +1, False, 0.2)  # 키보드 높이가 더해진다
    assert target_z(cw) == pytest.approx(z_hand + SPEED * 0.2, abs=2e-3)
    assert cw.state()["z_from_hand_scale"] is True


def test_keyboard_height_goes_through_ik_and_the_step_limit(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw)
    for _ in range(100):  # 처음 목표(상자 중심)에 도착할 때까지
        tick(cw)
    previous = sim.ctrl_log[-1].copy()
    n_before = len(sim.ctrl_log)
    hold(cw, -1, False, 0.6)
    for ctrl in sim.ctrl_log[n_before:]:
        assert np.all(np.abs(ctrl - previous)[:5] <= config.MAX_JOINT_STEP_RAD + 1e-9)  # 튀지 않는다
        previous = ctrl
    for _ in range(30):
        tick(cw)
    # ctrl(관절 각도)을 순방향 계산에 넣었을 때 집게 끝 높이가 목표 높이에 있다
    shared_ik.set_arm(sim.ctrl_log[-1][:5])
    assert shared_ik.tip_position_m()[2] == pytest.approx(target_z(cw), abs=0.005)
    assert len(sim.ctrl_log) > n_before


def test_ik_failure_while_pressing_keeps_the_goal_and_counts(shared_ik, monkeypatch):
    from server.ik import IKResult

    cw, sim = make(shared_ik)
    clutch(cw)
    for _ in range(60):
        tick(cw)
    last = sim.ctrl_log[-1].copy()
    monkeypatch.setattr(shared_ik, "solve", lambda t: IKResult(np.full(5, np.nan), False, float("nan"), 1, 0.1, 0.0, "boom"))
    hold(cw, -1, False, 0.3)
    state = cw.state()
    assert state["ik_fail_count"] >= 5 and state["ik_last_failure"] == "boom"
    np.testing.assert_allclose(sim.ctrl_log[-1], last, atol=1e-9)  # goal을 바꾸지 않았다


def test_hand_message_and_height_in_the_same_tick_solve_ik_once(shared_ik, monkeypatch):
    cw, sim = make(shared_ik)
    clutch(cw)
    calls = []
    real = shared_ik.solve
    monkeypatch.setattr(shared_ik, "solve", lambda target: calls.append(1) or real(target))
    t = cw.clock["t"] + TICK_S
    cw.clock["t"] = t
    cw.submit_height(-1, False, t - 0.01)
    cw.submit_hand(hand(0.45, 0.5), t)
    cw.tick()
    assert len(calls) == 1


def test_state_fields(shared_ik):
    cw, sim = make(shared_ik)
    state = cw.state()
    for key in ("height_dir", "height_slow", "height_active", "height_limit", "target_z_m", "z_from_hand_scale"):
        assert key in state
    assert state["height_dir"] == 0 and state["height_active"] is False and state["target_z_m"] is None
    clutch(cw)
    cw.submit_height(-1, True, cw.clock["t"])
    tick(cw)
    state = cw.state()
    assert state["height_dir"] == -1 and state["height_slow"] is True and state["height_active"] is True
    assert state["target_z_m"] == pytest.approx(target_z(cw), abs=1e-4)
    assert state["height_limit"] is None
    assert state["z_from_hand_scale"] is False


def test_hand_tracking_still_works_while_a_key_is_held(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw)
    base_y = cw._mapper.last_target_m[1]
    hold(cw, -1, False, 0.4, hand_msg=hand(0.4, 0.5))  # 손을 오른쪽으로 옮기며 키도 누른다
    assert cw._mapper.last_target_m[1] != pytest.approx(base_y)
    assert target_z(cw) == pytest.approx(CENTER_Z - SPEED * 0.4, abs=1e-6)
