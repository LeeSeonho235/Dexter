"""제어 스레드의 상대 모드(2-4): 보기 방위각 반영, 첫 조종 시작, 재진입, state 값."""

import numpy as np
import pytest

from server import config
from server.control import ControlWorker
from server.ik import ArmIK
from server.mapping import HandMapper
from server.view import CameraView

CENTER = np.array(config.RELATIVE_START_POSITION_M)


class FakeSim:
    """SimWorker 대신: ctrl을 기록하고, 보기 방향(view)과 자유 카메라 여부를 바꿀 수 있다."""

    def __init__(self, azimuth_deg=0.0, free_camera=True):
        self.ctrl_log = []
        self.snapshot = None
        self.view = CameraView(azimuth_deg, -35.0, 0.8)
        self.uses_free_camera = free_camera

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


def make(shared_ik, sim=None, mapper=None):
    sim = sim or FakeSim()
    clock = {"t": 0.0}
    cw = ControlWorker(
        sim,
        ik_factory=lambda: shared_ik,
        mapper=mapper or HandMapper(mode="relative", start_frames=3),
        clock=lambda: clock["t"],
    )
    shared_ik.reset()
    cw.prepare()
    cw.clock = clock
    return cw, sim


def feed(cw, msg):
    cw.submit_hand(msg, cw.clock["t"])
    cw.tick()
    cw.clock["t"] += 1 / config.CONTROL_HZ


def clutch(cw, x=0.5, y=0.5, scale=0.13):
    for _ in range(3):
        feed(cw, hand(x, y, scale))


def target(cw):
    return np.array(cw.state()["target_m"])


def test_app_default_mapper_follows_the_config_mode():
    cw = ControlWorker(FakeSim())
    assert config.MAPPING_MODE == "relative"
    assert cw._mapper.mode == config.MAPPING_MODE


def test_explicit_absolute_mapper_is_still_possible(shared_ik):
    cw, sim = make(shared_ik, mapper=HandMapper(start_frames=3))
    clutch(cw, 0.4, 0.4)
    assert cw.state()["mapping_mode"] == "absolute"
    assert cw.state()["reference_hand"] is None  # 절대 모드에는 기준점이 없다


def test_first_clutch_does_not_jump_and_starts_at_the_workspace_center(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw, 0.2, 0.8, 0.18)  # 영상 가장자리 근처에서 처음 잡아도
    np.testing.assert_allclose(target(cw), CENTER, atol=1e-6)  # 목표는 상자 중심에서 시작한다
    start = cw._home_ctrl()
    previous = start
    for ctrl in sim.ctrl_log:
        step = np.abs(ctrl - previous)
        previous = ctrl
        assert np.all(step[:5] <= config.MAX_JOINT_STEP_RAD + 1e-9)  # ctrl도 한 틱 변화량 제한 안에서만 움직인다


def test_state_reports_mode_reference_and_azimuth(shared_ik):
    cw, sim = make(shared_ik, sim=FakeSim(azimuth_deg=105.0))
    feed(cw, hand())
    assert cw.state()["reference_hand"] is None  # 시작 조건 전
    feed(cw, hand())
    feed(cw, hand(0.5, 0.5))
    state = cw.state()
    assert state["mapping_mode"] == "relative"
    assert state["reference_hand"] == pytest.approx([0.5, 0.5])
    assert state["view_azimuth_deg"] == 105.0
    assert state["controlling"] is True
    feed(cw, hand(detected=False))
    assert cw.state()["reference_hand"] is None  # 손이 사라지면 기준점도 없다
    assert cw.state()["controlling"] is False


@pytest.mark.parametrize("az,axis,sign", [(0.0, 1, -1), (90.0, 0, 1), (180.0, 1, 1), (270.0, 0, -1)])
def test_hand_right_follows_the_current_view_azimuth(shared_ik, az, axis, sign):
    """손을 오른쪽(영상 x 감소)으로 조금 움직이면 목표가 그 보기에서의 화면 오른쪽(월드 축 하나)으로 간다."""
    cw, sim = make(shared_ik, sim=FakeSim(azimuth_deg=az))
    clutch(cw)
    base = target(cw)
    for k in range(1, 11):
        feed(cw, hand(0.5 - 0.004 * k, 0.5))
    moved = target(cw) - base
    assert sign * moved[axis] > 0.005
    other = 1 - axis
    assert abs(moved[other]) < 0.1 * abs(moved[axis])


def test_fixed_camera_uses_the_top_view_direction(shared_ik):
    cw, sim = make(shared_ik, sim=FakeSim(azimuth_deg=90.0, free_camera=False))
    clutch(cw)
    base = target(cw)
    for k in range(1, 11):
        feed(cw, hand(0.5 - 0.004 * k, 0.5))
    moved = target(cw) - base
    assert moved[1] < -0.005 and abs(moved[0]) < 0.1 * abs(moved[1])  # 고정 top 카메라: 오른쪽 = -y (방위각 0°와 같다)


def test_simulation_without_a_view_attribute_means_azimuth_zero(shared_ik):
    class NoViewSim:
        snapshot = None

        def set_ctrl(self, ctrl):
            pass

    cw, _ = make(shared_ik, sim=NoViewSim())
    clutch(cw)
    base = target(cw)
    for k in range(1, 11):
        feed(cw, hand(0.5 - 0.004 * k, 0.5))
    assert (target(cw) - base)[1] < -0.005


def test_dragging_the_view_does_not_move_the_target_or_the_robot(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw)
    for k in range(1, 6):
        feed(cw, hand(0.5 - 0.01 * k, 0.5))
    for _ in range(60):  # 손이 가만히 있어서 로봇이 목표에 도착할 때까지
        feed(cw, hand(0.45, 0.5))
    before_target = target(cw)
    before_ctrl = sim.ctrl_log[-1].copy()
    n_before = len(sim.ctrl_log)
    for az in np.linspace(0, 180, 40):  # 보기를 계속 돌린다(드래그), 손은 그대로
        sim.view = CameraView(float(az), -35.0, 0.8)
        feed(cw, hand(0.45, 0.5))
    np.testing.assert_allclose(target(cw), before_target, atol=1e-9)
    np.testing.assert_allclose(sim.ctrl_log[-1], before_ctrl, atol=1e-9)
    # 드래그 중에 전달된 ctrl도 모두 직전 값 그대로(손이 같은 자리면 메시지마다 IK가 1e-9 rad 수준으로만 달라진다)
    for ctrl in sim.ctrl_log[n_before:]:
        assert np.max(np.abs(ctrl - before_ctrl)) < 1e-6


def test_reentry_takes_a_new_reference_and_robot_does_not_jump(shared_ik):
    cw, sim = make(shared_ik)
    clutch(cw, 0.5, 0.5)
    for k in range(1, 9):
        feed(cw, hand(0.5 - 0.01 * k, 0.5))
    for _ in range(40):
        feed(cw, hand(0.42, 0.5))
    held = target(cw)
    ctrl_before = sim.ctrl_log[-1].copy()
    for _ in range(10):  # 손이 화면 밖: 정지
        feed(cw, hand(detected=False))
    np.testing.assert_allclose(target(cw), held, atol=1e-12)
    np.testing.assert_allclose(sim.ctrl_log[-1], ctrl_before, atol=1e-9)
    # 중앙에서 다시 들어온다: 3프레임 연속 전에는 그대로, 그 순간부터 새 기준점
    feed(cw, hand(0.5, 0.5))
    feed(cw, hand(0.5, 0.5))
    np.testing.assert_allclose(target(cw), held, atol=1e-12)
    feed(cw, hand(0.5, 0.5))
    state = cw.state()
    assert state["reference_hand"] == pytest.approx([0.5, 0.5])
    np.testing.assert_allclose(target(cw), held, atol=1e-6)  # 다른 손 위치에서 시작했는데도 목표가 튀지 않는다
    assert np.max(np.abs(sim.ctrl_log[-1] - ctrl_before)) <= config.MAX_JOINT_STEP_RAD + 1e-9  # ctrl도 한 걸음 안


def test_reach_the_workspace_corner_with_hand_motion_near_the_image_center(shared_ik):
    """영상 중앙 근처(가장자리 10% 여유 안쪽)의 손동작만으로 상자 모서리에 목표가 닿는다."""
    cw, sim = make(shared_ik)
    clutch(cw)
    # 상자 중심 → 한쪽 모서리까지: 위에서 보기에서 y는 영상 x에, x는 영상 y에 대응한다
    dy_m = config.WORKSPACE_Y_M[1] - CENTER[1]  # 왼쪽(= 영상 x 증가) 방향 0.18 m
    dx_m = config.WORKSPACE_X_M[1] - CENTER[0]  # 앞쪽(= 영상 y 감소) 방향 0.05 m
    hx = 0.5 + dy_m / config.RELATIVE_GAIN_XY_M_PER_WIDTH
    hy = 0.5 - dx_m / config.RELATIVE_GAIN_XY_M_PER_WIDTH * config.IMAGE_ASPECT
    assert 0.1 <= hx <= 0.9 and 0.1 <= hy <= 0.9
    for k in range(1, 61):
        feed(cw, hand(0.5 + (hx - 0.5) * k / 60, 0.5 + (hy - 0.5) * k / 60))
    for _ in range(60):
        feed(cw, hand(hx, hy))
    np.testing.assert_allclose(target(cw)[:2], [config.WORKSPACE_X_M[1], config.WORKSPACE_Y_M[1]], atol=2e-3)
    # IK가 그 모서리를 실제로 풀어서 ctrl(관절 각도)이 집게 끝을 그 근처로 보낸다
    shared_ik.set_arm(sim.ctrl_log[-1][:5])
    assert np.linalg.norm(shared_ik.tip_position_m()[:2] - target(cw)[:2]) < 0.01
