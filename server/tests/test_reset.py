"""블록 리셋: 순서 기계(순수), 시뮬레이션의 블록 놓기, 제어 스레드와 시뮬레이션을 합친 실제 리셋."""

import time

import numpy as np
import pytest

from server import config
from server.control import ControlWorker
from server.reset import BlockReset, ResetAction, ResetPhase
from server.seeds import block_xy_for_seed
from server.sim_worker import SimWorker

PARK = np.array([0.0, -1.74, 1.2, -1.2, 0.0, 0.9])


def make_reset(**overrides):
    params = dict(tolerance_rad=0.05, hold_s=0.2, timeout_s=4.0, settle_s=0.3, block_tolerance_m=0.003, block_half_m=0.015)
    params.update(overrides)
    return BlockReset(**params)


def wait_until(condition, timeout_s=8.0, interval_s=0.02):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval_s)
    return condition()


# --- 순서 기계(순수) ---


def test_does_not_place_before_the_arm_arrives_and_holds():
    reset = make_reset()
    reset.start(0.0, 4, (0.22, -0.05))
    far = PARK + 0.5
    assert reset.update(0.1, far, PARK, None).action == ResetAction.NONE
    assert reset.update(0.5, PARK + 0.04, PARK, None).action == ResetAction.NONE  # 도착, 아직 유지 시간 전
    assert reset.update(0.6, PARK + 0.04, PARK, None).action == ResetAction.NONE
    assert reset.update(0.7, PARK + 0.04, PARK, None).action == ResetAction.PLACE  # 0.2 초 유지


def test_leaving_the_park_pose_restarts_the_hold():
    reset = make_reset()
    reset.start(0.0, 4, (0.22, -0.05))
    reset.update(0.5, PARK, PARK, None)
    assert reset.update(0.6, PARK + 0.2, PARK, None).action == ResetAction.NONE  # 벗어남
    assert reset.update(0.65, PARK, PARK, None).action == ResetAction.NONE
    assert reset.update(0.8, PARK, PARK, None).action == ResetAction.NONE  # 다시 센 지 0.15 초
    assert reset.update(0.86, PARK, PARK, None).action == ResetAction.PLACE


def test_timeout_fails_without_placing():
    reset = make_reset(timeout_s=4.0)
    reset.start(0.0, 4, (0.22, -0.05))
    assert reset.update(3.9, PARK + 1.0, PARK, None).action == ResetAction.NONE
    step = reset.update(4.1, PARK + 1.0, PARK, None)
    assert step.action == ResetAction.FAILED and "park" in step.reason
    assert reset.phase == ResetPhase.FAILED and not reset.active


def test_done_only_after_settle_and_block_at_seed_position():
    reset = make_reset()
    reset.start(0.0, 4, (0.22, -0.05))
    reset.placed(1.0)
    assert reset.phase == ResetPhase.SETTLING and reset.active
    block = np.array([0.22, -0.05, 0.015])
    assert reset.update(1.1, PARK, PARK, block).action == ResetAction.NONE  # 가라앉는 중
    assert reset.update(1.3, PARK, PARK, block).action == ResetAction.DONE
    assert reset.phase == ResetPhase.DONE and not reset.active


def test_fails_if_block_moved_or_unknown_after_settling():
    for block in (np.array([0.24, -0.05, 0.015]), np.array([0.22, -0.05, 0.05]), None):
        reset = make_reset()
        reset.start(0.0, 4, (0.22, -0.05))
        reset.placed(1.0)
        assert reset.update(1.5, PARK, PARK, block).action == ResetAction.FAILED


# --- 시뮬레이션의 블록 놓기 ---


@pytest.fixture
def worker():
    w = SimWorker(
        config.SCENE_PATH,
        keyframe=config.START_KEYFRAME,
        camera=config.CAMERA,
        width_px=config.RENDER_WIDTH_PX,
        height_px=config.RENDER_HEIGHT_PX,
        frame_fps=config.FRAME_FPS,
        jpeg_quality=config.JPEG_QUALITY,
        max_catch_up_s=config.MAX_CATCH_UP_S,
        shadow=config.RENDER_SHADOW,
        reflection=config.RENDER_REFLECTION,
        tick_hz=config.SIM_TICK_HZ,
    )
    w.start()
    yield w
    w.stop()


def test_place_block_moves_block_and_it_stays_put(worker):
    x, y = block_xy_for_seed(4)
    result = worker.place_block(x, y)
    assert result.ok, result.reason
    time.sleep(0.5)
    block = worker.snapshot["block_m"]
    assert block[0] == pytest.approx(x, abs=0.002) and block[1] == pytest.approx(y, abs=0.002)
    assert block[2] == pytest.approx(config.BLOCK_HALF_M, abs=0.002)


def test_place_block_zeroes_velocity(worker):
    """블록이 움직이는 중에 놓아도 속도가 0이 되어 제자리에 있다(굴러가지 않는다)."""
    assert worker.place_block(0.22, 0.0).ok
    time.sleep(0.3)
    first = worker.snapshot["block_m"].copy()
    time.sleep(0.5)
    np.testing.assert_allclose(worker.snapshot["block_m"], first, atol=1e-3)


def test_place_block_refuses_to_overlap_the_robot_and_keeps_the_block(worker):
    time.sleep(0.2)
    before = worker.snapshot["block_m"].copy()
    # 어깨의 충돌 geom(so101.xml, contype 1)이 있는 자리. 베이스 메시는 충돌이 꺼져 있어서 (0, 0)은 겹침으로 안 잡힌다.
    result = worker.place_block(0.06, 0.0)
    assert not result.ok and "robot" in result.reason
    time.sleep(0.3)
    np.testing.assert_allclose(worker.snapshot["block_m"], before, atol=1e-3)


def test_only_one_place_block_can_be_pending(worker):
    """시뮬레이션 스레드가 집어가기 전에 두 번째 요청이 오면 거절한다(락 안에서 검사)."""
    first = worker.place_block(0.21, 0.0)
    assert first.ok
    assert worker.place_block(0.22, 0.0).ok  # 앞 것이 끝났으니 다시 가능


# --- 제어 스레드 + 시뮬레이션 ---


def hand_msg(x=0.5, y=0.5):
    return {"type": "hand", "t": 0.0, "detected": True, "wrist": [x, y, 0.0], "index_tip": [x, y, 0.0],
            "thumb_tip": [x, y, 0.0], "pinch": 0.1, "pinch_ratio": 0.9, "palm_open": True, "hand_scale": 0.13}


def test_full_reset_parks_arm_opens_gripper_places_block_and_clears_control(worker):
    control = ControlWorker(worker)
    control.start()
    try:
        # 손으로 팔을 작업 상자 쪽으로 움직여 둔다.
        for _ in range(60):
            control.submit_hand(hand_msg(), time.monotonic())
            time.sleep(1 / 30)
        assert control.state()["controlling"] is True
        moved = worker.snapshot["joints_rad"].copy()
        assert np.abs(moved[:5] - control._home_ctrl()[:5]).max() > 0.3  # home에서 꽤 멀어졌다

        done = []
        seed = 7
        reason = control.request_reset(seed, block_xy_for_seed(seed), lambda s, ok, why: done.append((s, ok, why)))
        assert reason is None and control.resetting
        # 리셋 중 들어오는 손 입력은 무시된다.
        for _ in range(10):
            control.submit_hand(hand_msg(0.3, 0.3), time.monotonic())
            time.sleep(1 / 30)
        assert wait_until(lambda: done, timeout_s=8.0), f"reset did not finish, phase={control.reset_phase}"
        assert done == [(seed, True, "")]
        assert not control.resetting and control.reset_phase == "done"

        joints = worker.snapshot["joints_rad"]
        np.testing.assert_allclose(joints[:5], control._home_ctrl()[:5], atol=config.RESET_PARK_TOLERANCE_RAD)
        assert joints[5] > 0.8  # 집게 열림(0.9)
        x, y = block_xy_for_seed(seed)
        block = worker.snapshot["block_m"]
        assert block[0] == pytest.approx(x, abs=0.004) and block[1] == pytest.approx(y, abs=0.004)

        state = control.state()
        assert state["controlling"] is False and state["target_m"] is None  # 이전 목표를 지웠다
        time.sleep(0.5)
        np.testing.assert_allclose(worker.snapshot["joints_rad"][:5], joints[:5], atol=0.02)  # 팔이 되돌아가지 않는다
    finally:
        control.stop()


def test_second_reset_request_while_running_is_refused(worker):
    control = ControlWorker(worker)
    control.start()
    try:
        assert control.request_reset(1, block_xy_for_seed(1), lambda *a: None) is None
        assert "already" in control.request_reset(2, block_xy_for_seed(2), lambda *a: None)
        assert wait_until(lambda: not control.resetting)
    finally:
        control.stop()
