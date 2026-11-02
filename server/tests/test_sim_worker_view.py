"""시뮬레이션 스레드(2-3): view를 바꾸면 렌더링 각도가 바뀐다. 연결이 없으면 프레임을 만들지 않는다."""

import io
import time

import numpy as np
import pytest
from PIL import Image

from server import config
from server.sim_worker import SimWorker
from server.view import CameraView, default_view

JPEG_START = b"\xff\xd8"


def wait_until(condition, timeout_s=5.0, interval_s=0.02):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval_s)
    return condition()


def make_worker(camera=config.CAMERA):
    return SimWorker(
        config.SCENE_PATH,
        keyframe=config.START_KEYFRAME,
        camera=camera,
        width_px=config.RENDER_WIDTH_PX,
        height_px=config.RENDER_HEIGHT_PX,
        frame_fps=config.FRAME_FPS,
        jpeg_quality=config.JPEG_QUALITY,
        max_catch_up_s=config.MAX_CATCH_UP_S,
        shadow=config.RENDER_SHADOW,
        reflection=config.RENDER_REFLECTION,
        tick_hz=config.SIM_TICK_HZ,
    )


@pytest.fixture
def worker():
    w = make_worker()
    w.start()
    yield w
    w.stop()


def test_camera_setting_is_the_free_camera():
    assert config.CAMERA == "free"


def test_initial_view_is_the_default(worker):
    assert worker.view == default_view()


def test_set_view_changes_the_rendering_angle(worker):
    worker.set_viewer_count(1)
    assert wait_until(lambda: worker.rendered_view == default_view())
    first_jpeg = worker.latest_frame[1]

    other = CameraView(-60.0, -20.0, 0.6)
    worker.set_view(other)
    assert worker.view == other
    assert wait_until(lambda: worker.rendered_view == other)  # 실제로 그 각도로 그린 프레임이 나왔다
    seq = worker.latest_frame[0]
    assert wait_until(lambda: worker.latest_frame[0] > seq)
    assert worker.latest_frame[1] != first_jpeg  # 그림이 달라졌다
    assert worker.latest_frame[1].startswith(JPEG_START)


def mean_abs_diff(jpeg_a: bytes, jpeg_b: bytes) -> float:
    a = np.asarray(Image.open(io.BytesIO(jpeg_a)).convert("RGB"), dtype=float)
    b = np.asarray(Image.open(io.BytesIO(jpeg_b)).convert("RGB"), dtype=float)
    return float(np.abs(a - b).mean())


def test_angle_change_is_a_big_picture_change_but_same_angle_is_not(worker):
    """같은 각도의 연속 프레임은 거의 같고(팔과 블록이 아주 조금 움직일 뿐), 각도를 바꾸면 크게 달라진다.
    위 테스트의 "달라졌다"가 우연이 아님을 숫자로 확인한다(JPEG 바이트가 같을 것이라는 가정은 틀렸다: 시뮬레이션이 완전히 멈춰 있지 않다)."""
    worker.set_viewer_count(1)
    assert wait_until(lambda: worker.frames_rendered >= 2)
    a = worker.latest_frame
    assert wait_until(lambda: worker.latest_frame[0] > a[0] + 1)
    same_angle_diff = mean_abs_diff(a[1], worker.latest_frame[1])

    worker.set_view(CameraView(-60.0, -20.0, 0.6))
    assert wait_until(lambda: worker.rendered_view == CameraView(-60.0, -20.0, 0.6))
    changed_angle_diff = mean_abs_diff(a[1], worker.latest_frame[1])
    assert same_angle_diff < 1.0, same_angle_diff
    assert changed_angle_diff > 5.0, changed_angle_diff


def test_no_viewer_means_no_frames_even_after_set_view(worker):
    worker.set_view(CameraView(10.0, -50.0, 0.7))
    assert wait_until(lambda: worker.sim_time_s > 0.5)
    assert worker.frames_rendered == 0
    assert worker.latest_frame is None
    assert worker.rendered_view is None  # 아직 한 장도 안 그렸다


def test_view_applied_when_a_viewer_connects_later(worker):
    other = CameraView(45.0, -60.0, 1.0)
    worker.set_view(other)
    worker.set_viewer_count(1)
    assert wait_until(lambda: worker.rendered_view == other)


def test_fixed_camera_still_works_and_ignores_view():
    w = make_worker(camera="top")
    w.start()
    try:
        w.set_viewer_count(1)
        assert wait_until(lambda: w.frames_rendered >= 2)
        w.set_view(CameraView(0.0, -30.0, 0.5))
        assert wait_until(lambda: w.frames_rendered >= 4)
        assert w.rendered_view is None  # 고정 카메라는 보기 방향을 쓰지 않는다
        assert w.latest_frame[1].startswith(JPEG_START)
    finally:
        w.stop()
