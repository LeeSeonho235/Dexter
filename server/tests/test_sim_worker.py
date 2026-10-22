"""시뮬레이션 스레드: 실제 시간 진행, 화면 보는 브라우저가 있을 때만 렌더링."""

import time

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
    frames = []
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
        on_frame=lambda seq, jpeg: frames.append((seq, jpeg)),
    )
    w.callback_frames = frames
    w.start()
    yield w
    w.stop()


def test_no_viewer_means_no_frames_but_sim_runs(worker):
    assert wait_until(lambda: worker.sim_time_s > 0.4)
    assert worker.frames_rendered == 0
    assert worker.latest_frame is None
    assert worker.callback_frames == []


def test_viewer_starts_and_stops_rendering(worker):
    worker.set_viewer_count(1)
    assert wait_until(lambda: worker.frames_rendered >= 3)
    seq, jpeg = worker.latest_frame
    assert seq >= 3
    assert jpeg.startswith(JPEG_START)
    assert worker.callback_frames[0][1].startswith(JPEG_START)

    worker.set_viewer_count(0)
    time.sleep(0.2)  # 진행 중이던 렌더링 한 장이 끝날 시간
    stopped_at = worker.frames_rendered
    time.sleep(0.5)
    assert worker.frames_rendered == stopped_at

    worker.set_viewer_count(2)
    assert wait_until(lambda: worker.frames_rendered > stopped_at)


def test_sim_time_follows_wall_clock(worker):
    start_sim_s = worker.sim_time_s
    start_wall_s = time.monotonic()
    time.sleep(1.0)
    elapsed_sim_s = worker.sim_time_s - start_sim_s
    elapsed_wall_s = time.monotonic() - start_wall_s
    # 틱 주기(1/15 s)만큼의 오차는 허용
    assert abs(elapsed_sim_s - elapsed_wall_s) < 0.15


def test_frame_rate_close_to_config(worker):
    worker.set_viewer_count(1)
    assert wait_until(lambda: worker.frames_rendered >= 1)
    start = worker.frames_rendered
    time.sleep(2.0)
    fps = (worker.frames_rendered - start) / 2.0
    # 렌더+JPEG가 약 50ms라 15fps 근처. CPU가 바쁘면 떨어질 수 있으므로 넉넉히 본다.
    assert 8 <= fps <= config.FRAME_FPS + 1


def test_stop_ends_thread(worker):
    assert worker.is_alive
    worker.stop()
    assert not worker.is_alive


def test_bad_scene_raises_on_start(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<mujoco><worldbody><bad/></worldbody></mujoco>")
    w = SimWorker(
        bad,
        keyframe="home",
        camera="top",
        width_px=64,
        height_px=48,
        frame_fps=15,
        jpeg_quality=80,
        max_catch_up_s=0.25,
    )
    with pytest.raises(RuntimeError):
        w.start()
    w.stop()
