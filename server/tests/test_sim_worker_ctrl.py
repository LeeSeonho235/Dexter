"""시뮬레이션 스레드(2-2): 30Hz 틱, 렌더링은 15fps + 연결 없으면 안 함, ctrl 반영, 집게 끝 위치 읽기."""

import time

import numpy as np
import pytest

from server import config
from server.sim_worker import SimWorker

JPEG_START = b"\xff\xd8"


def wait_until(condition, timeout_s=5.0, interval_s=0.02):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval_s)
    return condition()


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


def test_tick_is_30hz_but_frames_stay_15fps(worker):
    assert config.SIM_TICK_HZ == 30 and config.FRAME_FPS == 15
    assert worker._render_every == 2


def test_no_viewer_still_means_no_frames_at_30hz(worker):
    """연결이 없으면 렌더링을 멈추는 2-1 동작이 틱을 올려도 그대로여야 한다."""
    assert wait_until(lambda: worker.sim_time_s > 0.5)
    assert worker.frames_rendered == 0
    assert worker.latest_frame is None


def test_frames_start_and_stop_with_viewers(worker):
    worker.set_viewer_count(1)
    assert wait_until(lambda: worker.frames_rendered >= 3)
    assert worker.latest_frame[1].startswith(JPEG_START)
    worker.set_viewer_count(0)
    time.sleep(0.3)
    stopped_at = worker.frames_rendered
    time.sleep(0.5)
    assert worker.frames_rendered == stopped_at


def test_frame_rate_is_still_about_15fps(worker):
    worker.set_viewer_count(1)
    assert wait_until(lambda: worker.frames_rendered >= 1)
    start = worker.frames_rendered
    time.sleep(2.0)
    fps = (worker.frames_rendered - start) / 2.0
    assert 8 <= fps <= config.FRAME_FPS + 1


def test_physics_follows_wall_clock_at_30hz(worker):
    start_sim_s = worker.sim_time_s
    start_wall_s = time.monotonic()
    time.sleep(1.0)
    assert abs((worker.sim_time_s - start_sim_s) - (time.monotonic() - start_wall_s)) < 0.15


def test_snapshot_reports_tip_joints_and_block(worker):
    assert wait_until(lambda: worker.snapshot is not None)
    snap = worker.snapshot
    assert snap["tip_m"].shape == (3,) and snap["joints_rad"].shape == (6,) and snap["block_m"].shape == (3,)
    np.testing.assert_allclose(snap["tip_m"], [0.028, -0.001, 0.354], atol=0.01)  # home 자세
    np.testing.assert_allclose(snap["block_m"][:2], [0.20, 0.0], atol=0.005)


def test_set_ctrl_moves_the_joint(worker):
    ctrl = np.array([0.4, -1.74, 1.2, -1.2, 0.0, 0.0])
    worker.set_ctrl(ctrl)
    assert wait_until(lambda: abs(worker.snapshot["joints_rad"][0] - 0.4) < 0.02, timeout_s=5.0)


@pytest.mark.parametrize("bad", [[0.0] * 5, [0.0] * 7, [np.nan] + [0.0] * 5, [np.inf] + [0.0] * 5])
def test_set_ctrl_rejects_bad_values(worker, bad):
    with pytest.raises(ValueError):
        worker.set_ctrl(np.array(bad))
