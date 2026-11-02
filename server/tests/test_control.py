"""제어 스레드: 시작 조건, 처음 인식 때 안 튐, 손 없을 때 유지, IK 실패 처리, 집게 따라가기, 측정값."""

import time

import numpy as np
import pytest

from server import config
from server.control import ControlWorker, std_mm
from server.ik import ArmIK, IKResult
from server.mapping import HandMapper


class FakeSim:
    """SimWorker 대신: set_ctrl로 받은 값을 기록한다."""

    def __init__(self):
        self.ctrl_log = []
        self.snapshot = None

    def set_ctrl(self, ctrl):
        self.ctrl_log.append(np.asarray(ctrl).copy())


def hand(x=0.5, y=0.5, scale=0.13, pinch_ratio=0.8, detected=True):
    if not detected:
        return {"type": "hand", "t": 0.0, "detected": False, "wrist": None, "index_tip": None, "thumb_tip": None,
                "pinch": None, "pinch_ratio": None, "palm_open": None, "hand_scale": None}
    return {"type": "hand", "t": 0.0, "detected": True, "wrist": [x, y, 0.0], "index_tip": [x, y, 0.0],
            "thumb_tip": [x, y, 0.0], "pinch": 0.1, "pinch_ratio": pinch_ratio, "palm_open": True, "hand_scale": scale}


@pytest.fixture(scope="module")
def shared_ik():
    return ArmIK()


def make(shared_ik, start_frames=3):
    sim = FakeSim()
    clock = {"t": 0.0}
    cw = ControlWorker(
        sim,
        ik_factory=lambda: shared_ik,
        mapper=HandMapper(start_frames=start_frames),
        clock=lambda: clock["t"],
    )
    shared_ik.reset()
    cw.prepare()
    cw.clock = clock
    return cw, sim


def feed(cw, msg):
    """메시지 한 개를 넣고 한 틱 돌린다. 틱마다 시계를 1/CONTROL_HZ초 진행한다."""
    cw.submit_hand(msg, cw.clock["t"])
    cw.tick()
    cw.clock["t"] += 1 / config.CONTROL_HZ


def idle(cw, n):
    for _ in range(n):
        cw.tick()
        cw.clock["t"] += 1 / config.CONTROL_HZ


def home_ctrl(cw):
    return cw._home_ctrl()


def test_no_hand_never_moves_the_robot(shared_ik):
    cw, sim = make(shared_ik)
    for _ in range(10):
        feed(cw, hand(detected=False))
    assert sim.ctrl_log == []
    assert cw.state()["controlling"] is False
    assert cw.state()["target_m"] is None


def test_start_needs_three_consecutive_frames(shared_ik):
    assert config.START_CONSECUTIVE_FRAMES == 3
    cw, sim = make(shared_ik)
    feed(cw, hand())
    feed(cw, hand())
    assert sim.ctrl_log == []  # 2프레임: 아직 안 움직인다
    assert cw.state()["consecutive_frames"] == 2
    feed(cw, hand())
    assert len(sim.ctrl_log) == 1  # 3프레임째에 시작
    assert cw.state()["controlling"] is True


def test_one_stray_frame_does_not_move_the_robot(shared_ik):
    cw, sim = make(shared_ik)
    feed(cw, hand())  # 잘못 인식된 손 1프레임
    feed(cw, hand(detected=False))
    feed(cw, hand())
    feed(cw, hand(detected=False))
    idle(cw, 5)
    assert sim.ctrl_log == []


def test_first_recognition_does_not_jump(shared_ik):
    """손이 작업 상자 한쪽 끝에서 처음 인식돼도 ctrl은 한 틱에 정해진 변화량 이상 튀지 않는다."""
    cw, sim = make(shared_ik)
    start = home_ctrl(cw)
    for _ in range(3):
        feed(cw, hand(0.2, 0.2, 0.18))  # 상자 한쪽 모서리 쪽
    previous = start
    assert len(sim.ctrl_log) >= 1
    for ctrl in sim.ctrl_log:
        step = np.abs(ctrl - previous)
        assert np.all(step[:5] <= config.MAX_JOINT_STEP_RAD + 1e-9)
        assert step[5] <= config.MAX_GRIPPER_STEP_RAD + 1e-9
        previous = ctrl
    # 그 뒤에도 한 틱씩만 움직이면서 목표에 간다
    for _ in range(80):
        feed(cw, hand(0.2, 0.2, 0.18))
    for ctrl in sim.ctrl_log:
        step = np.abs(ctrl - start)
        start = ctrl
        assert np.all(step[:5] <= config.MAX_JOINT_STEP_RAD + 1e-9)
    assert not np.array_equal(sim.ctrl_log[-1][:5], home_ctrl(cw)[:5])


def test_speed_limit_does_not_depend_on_message_timing(shared_ik):
    """메시지가 1/CONTROL_HZ초보다 훨씬 자주 와도(여기서는 4배) 속도 상한(rad/s)은 같다."""
    cw, sim = make(shared_ik)
    dt_s = 1 / config.CONTROL_HZ / 4
    for _ in range(3):  # 시작 조건
        cw.submit_hand(hand(0.2, 0.2, 0.18), cw.clock["t"])
        cw.tick()
        cw.clock["t"] += dt_s
    previous = home_ctrl(cw)
    start = previous.copy()
    for _ in range(120):  # 0.9초 남짓
        cw.submit_hand(hand(0.2, 0.2, 0.18), cw.clock["t"])
        cw.tick()
        cw.clock["t"] += dt_s
    # 조종을 시작하는 첫 걸음만 "지난 걸음"이 없어서 한 틱 분량(MAX_JOINT_STEP_RAD)을 움직인다. 그 뒤는 시간에 비례한다.
    first, rest = sim.ctrl_log[0], sim.ctrl_log[1:]
    assert np.all(np.abs(first - previous)[:5] <= config.MAX_JOINT_STEP_RAD + 1e-9)
    previous = first
    for ctrl in rest:
        step = np.abs(ctrl - previous)
        previous = ctrl
        assert np.all(step[:5] <= config.MAX_JOINT_STEP_RAD * config.CONTROL_HZ * dt_s + 1e-9)
    elapsed_s = 120 * dt_s
    max_travel_rad = config.MAX_JOINT_STEP_RAD * (1 + config.CONTROL_HZ * elapsed_s)
    assert np.all(np.abs(sim.ctrl_log[-1][:5] - start[:5]) <= max_travel_rad + 1e-9)


def test_thread_reacts_to_a_message_without_waiting_for_the_next_period():
    """새 메시지는 다음 주기 틱(최대 33ms)을 기다리지 않고 바로 처리된다. 중앙값으로 본다(기계가 바쁜 순간 한두 번은 느릴 수 있다)."""
    sim = FakeSim()
    cw = ControlWorker(sim, ik_factory=ArmIK, mapper=HandMapper(start_frames=1))
    cw.start()
    try:
        delays_ms = []
        for i in range(25):
            before = len(sim.ctrl_log)
            sent_s = time.monotonic()
            cw.submit_hand(hand(0.3 + 0.01 * (i % 2), 0.4, 0.12), sent_s)
            while len(sim.ctrl_log) == before and time.monotonic() - sent_s < 0.5:
                time.sleep(0.0005)
            delays_ms.append((time.monotonic() - sent_s) * 1000)
            time.sleep(0.037 + 0.003 * (i % 5))  # 주기(33ms)와 어긋나는 간격으로 보낸다
        assert float(np.median(delays_ms)) < 8.0, delays_ms
    finally:
        cw.stop()


def test_ctrl_reaches_ik_goal_and_matches_target(shared_ik):
    cw, sim = make(shared_ik)
    for _ in range(120):
        feed(cw, hand(0.4, 0.45, 0.12))
    target = np.array(cw.state()["target_m"])
    # ctrl(관절 각도)를 순방향 계산에 넣었을 때 집게 끝이 목표 가까이에 있어야 한다.
    shared_ik.set_arm(sim.ctrl_log[-1][:5])
    assert np.linalg.norm(shared_ik.tip_position_m() - target) < 0.005


def test_hand_loss_holds_last_target_and_stops(shared_ik):
    cw, sim = make(shared_ik)
    for _ in range(100):
        feed(cw, hand(0.45, 0.5, 0.12))
    target_before = cw.state()["target_m"]
    ctrl_before = sim.ctrl_log[-1].copy()
    n_before = len(sim.ctrl_log)
    for _ in range(30):
        feed(cw, hand(detected=False))
    assert cw.state()["target_m"] == target_before
    assert cw.state()["controlling"] is False
    np.testing.assert_allclose(sim.ctrl_log[-1], ctrl_before)
    assert len(sim.ctrl_log) == n_before  # 이미 도착해 있으니 새로 보낼 것도 없다


def test_reappearing_hand_waits_for_start_condition(shared_ik):
    cw, sim = make(shared_ik)
    for _ in range(100):
        feed(cw, hand(0.45, 0.5, 0.12))
    before = sim.ctrl_log[-1].copy()
    feed(cw, hand(detected=False))
    feed(cw, hand(0.7, 0.7, 0.1))
    feed(cw, hand(0.7, 0.7, 0.1))
    np.testing.assert_allclose(sim.ctrl_log[-1], before)  # 2프레임까지는 그대로
    feed(cw, hand(0.7, 0.7, 0.1))
    feed(cw, hand(0.7, 0.7, 0.1))
    assert not np.allclose(sim.ctrl_log[-1], before)


def test_gripper_follows_pinch_with_hysteresis(shared_ik):
    cw, sim = make(shared_ik)
    for _ in range(40):
        feed(cw, hand(pinch_ratio=0.8))
    assert cw.state()["gripper_closed"] is False
    assert sim.ctrl_log[-1][5] == pytest.approx(config.GRIPPER_OPEN_RAD)
    for _ in range(40):
        feed(cw, hand(pinch_ratio=0.15))
    assert cw.state()["gripper_closed"] is True
    assert sim.ctrl_log[-1][5] == pytest.approx(config.GRIPPER_CLOSED_RAD)
    for _ in range(40):
        feed(cw, hand(pinch_ratio=0.4))  # 두 기준 사이: 닫힘 유지
    assert cw.state()["gripper_closed"] is True
    assert sim.ctrl_log[-1][5] == pytest.approx(config.GRIPPER_CLOSED_RAD)
    for _ in range(40):
        feed(cw, hand(pinch_ratio=0.7))
    assert cw.state()["gripper_closed"] is False
    assert sim.ctrl_log[-1][5] == pytest.approx(config.GRIPPER_OPEN_RAD)


def test_ik_failure_keeps_ctrl_and_counts(shared_ik, monkeypatch):
    cw, sim = make(shared_ik)
    for _ in range(100):
        feed(cw, hand(0.45, 0.5, 0.12))
    ctrl_before = sim.ctrl_log[-1].copy()
    n_before = len(sim.ctrl_log)

    def failing(target_m):
        return IKResult(np.full(5, np.nan), False, float("nan"), 3, 0.5, 0.0, "boom")

    monkeypatch.setattr(shared_ik, "solve", failing)
    for _ in range(20):
        feed(cw, hand(0.7, 0.3, 0.1))  # 다른 곳으로 움직였지만 IK가 계속 실패
    state = cw.state()
    assert state["ik_fail_count"] == 20  # 새 메시지마다 한 번씩
    assert state["ik_last_failure"] == "boom"
    np.testing.assert_allclose(sim.ctrl_log[-1], ctrl_before)  # 직전 값 유지
    assert len(sim.ctrl_log) == n_before


def test_ik_failure_before_first_goal_does_not_move_robot(shared_ik, monkeypatch):
    cw, sim = make(shared_ik)
    monkeypatch.setattr(
        shared_ik, "solve", lambda t: IKResult(np.zeros(5), False, float("nan"), 1, 0.1, 0.0, "nan")
    )
    for _ in range(10):
        feed(cw, hand())
    assert sim.ctrl_log == []
    assert cw.state()["ik_fail_count"] == 8  # 3프레임째부터 새 목표가 생긴다


def test_nan_hand_target_does_not_publish_nan(shared_ik):
    """IK 안쪽에서 NaN을 막으므로(ArmIK.solve) 어떤 경우에도 ctrl에는 NaN이 가지 않는다."""
    cw, sim = make(shared_ik)
    for _ in range(100):
        feed(cw, hand(0.5, 0.5, 0.12))
    for ctrl in sim.ctrl_log:
        assert np.all(np.isfinite(ctrl))


def test_latency_and_ik_time_are_recorded(shared_ik):
    cw, sim = make(shared_ik)
    for _ in range(30):
        feed(cw, hand())
    state = cw.state()
    assert state["ik_ms"]["n"] > 0 and state["ik_ms"]["mean"] is not None
    assert state["latency_ms"]["n"] > 0
    assert state["latency_ms"]["max"] >= state["latency_ms"]["mean"] >= 0


def test_jitter_is_reported_for_still_wrist(shared_ik):
    cw, sim = make(shared_ik)
    for _ in range(40):
        feed(cw, hand(0.5, 0.5, 0.13))
    jitter = cw.state()["jitter_mm"]
    assert jitter["raw"] == pytest.approx(0.0, abs=1e-6)
    assert jitter["filtered"] == pytest.approx(0.0, abs=1e-6)


def test_jitter_grows_with_noise(shared_ik):
    rng = np.random.default_rng(0)
    cw, sim = make(shared_ik)
    for _ in range(40):
        feed(cw, hand(0.5 + rng.normal(0, 0.01), 0.5 + rng.normal(0, 0.01), 0.13))
    jitter = cw.state()["jitter_mm"]
    assert jitter["raw"] > 1.0
    assert jitter["filtered"] < jitter["raw"]


def test_std_mm_needs_enough_samples():
    few = [np.zeros(3)] * (config.JITTER_MIN_SAMPLES - 1)
    assert std_mm(few) is None
    assert std_mm([np.array([0.0, 0.0, 0.0]), np.array([0.002, 0.0, 0.0])] * 5) == pytest.approx(1.0)


def test_thread_survives_exceptions_and_keeps_running():
    """계산 중 예상 못 한 예외가 나도 제어 스레드가 죽지 않고 loop_error_count만 올라간다."""

    class ExplodingIK(ArmIK):
        def solve(self, target_m):
            raise RuntimeError("unexpected")

    sim = FakeSim()
    cw = ControlWorker(sim, ik_factory=ExplodingIK, mapper=HandMapper(start_frames=1))
    cw.start()
    try:
        for _ in range(5):
            cw.submit_hand(hand(), time.monotonic())
            time.sleep(0.05)
        assert cw.is_alive
        assert cw.state()["loop_error_count"] >= 1
        assert sim.ctrl_log == []
    finally:
        cw.stop()
    assert not cw.is_alive
