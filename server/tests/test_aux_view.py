"""보조 카메라(server/aux_view.py): 꺼짐 상태, 크기·fps, 보는 연결이 없을 때 멈춤, AUTO 중 정지, 자동 감속, 최신 프레임만 유지."""

import io
import os
import time

import mujoco
import numpy as np
import pytest
from PIL import Image

from server import aux_mesh
from server import aux_view as aux_view_module
from server import aux_view_config as ac
from server import config
from server.aux_view import AuxViewWorker, is_due, next_render_time
from server.render import set_render_flags
from server.view import clamp_view, make_free_camera


@pytest.fixture(scope="module")
def home_qpos():
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    return model.key("home").qpos.copy()


@pytest.fixture
def make_worker():
    workers = []

    def make(**kwargs):
        worker = AuxViewWorker(enabled=True, **kwargs)
        worker.start()
        workers.append(worker)
        return worker

    yield make
    for worker in workers:
        worker.stop()


def wait_for(condition, timeout_s=15.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "시간 안에 조건이 채워지지 않았다"
        time.sleep(0.01)


def jpeg_size(jpeg: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(jpeg)).size


def frame_size(level: str) -> tuple[int, int]:
    """프레임 한 장 = 시점 수 × 시점 너비 (px), 높이."""
    view_w_px, height_px = ac.AUX_SIZE_LEVELS[level]
    return view_w_px * len(ac.AUX_CAMERA_VIEWS), height_px


@pytest.mark.skipif("AUX_ENABLED" in os.environ, reason="AUX_ENABLED 환경변수가 설정돼 있어 기본값을 볼 수 없다")
def test_default_is_off():
    assert ac.AUX_ENABLED is False
    assert AuxViewWorker().enabled is False


def test_disabled_worker_creates_no_thread_and_ignores_qpos(home_qpos):
    worker = AuxViewWorker(enabled=False)
    worker.start()
    worker.set_viewer_count(1)
    worker.offer_qpos(home_qpos)
    time.sleep(0.2)
    assert not worker.is_alive and worker._thread is None and worker._renderer is None
    assert worker.latest_frame is None and worker.state()["frames"] == 0 and worker.state()["enabled"] is False
    worker.stop()


def test_frame_has_configured_size_and_is_a_jpeg(make_worker, home_qpos):
    worker = make_worker(fps=30.0)
    worker.set_viewer_count(1)
    worker.offer_qpos(home_qpos)
    wait_for(lambda: worker.latest_frame is not None)
    seq, jpeg = worker.latest_frame
    assert seq >= 1 and jpeg[:2] == b"\xff\xd8"
    assert jpeg_size(jpeg) == frame_size(ac.AUX_SIZE_DEFAULT) == (320, 120)  # 두 시점을 가로로 이어 붙인 한 장
    image = np.asarray(Image.open(io.BytesIO(jpeg)))
    assert image.std() > 5  # 빈 화면이 아니다
    state = worker.state()
    assert (state["views"], state["view_width_px"], state["width_px"], state["height_px"]) == (2, 160, 320, 120)


def reference_halves(pose: np.ndarray) -> list[np.ndarray]:
    """같은 줄인 모델을 mujoco.Renderer로 시점 하나씩 그린 그림(첫 렌더링은 버린다). 두 시점 합친 그림의 왼쪽, 오른쪽 절반과 비교한다."""
    model = aux_mesh.build_model(config.SCENE_PATH, ac.AUX_MESH_FRACTION)
    data = mujoco.MjData(model)
    data.qpos[:13] = pose
    mujoco.mj_forward(model, data)
    view_w_px, height_px = ac.AUX_SIZE_LEVELS["normal"]
    out = []
    with mujoco.Renderer(model, height_px, view_w_px) as renderer:
        for view in ac.AUX_CAMERA_VIEWS:
            camera = make_free_camera(clamp_view(**view), ac.AUX_CAMERA_LOOKAT_M)
            for _ in range(2):
                renderer.update_scene(data, camera=camera)
                set_render_flags(renderer, config.RENDER_SHADOW, config.RENDER_REFLECTION)
                pixels = renderer.render()
            out.append(pixels)
    return out


def test_left_half_is_the_side_view_and_right_half_the_oblique_view(make_worker, home_qpos):
    pose = home_qpos.copy()
    pose[0], pose[1] = 0.4, -1.2  # 팔을 움직인 자세(두 시점이 서로 다르게 보인다)
    worker = make_worker(fps=30.0)
    worker.set_viewer_count(1)
    worker.offer_qpos(pose)
    wait_for(lambda: worker.state()["frames"] >= 2)
    image = np.asarray(Image.open(io.BytesIO(worker.latest_frame[1])))
    ref_side, ref_oblique = reference_halves(pose)

    def mad(a, b):
        return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())

    # 참조 두 장을 가로로 이어 같은 JPEG 품질로 한 번 왕복시키면 JPEG 손실까지 같아진다(2026-11-13 측정: 차이 0). 그래서 기준이 작다.
    buf = io.BytesIO()
    Image.fromarray(np.hstack([ref_side, ref_oblique])).save(buf, format="JPEG", quality=ac.AUX_JPEG_QUALITY)
    expected = np.asarray(Image.open(io.BytesIO(buf.getvalue())))
    assert image.shape == expected.shape == (120, 320, 3)
    assert mad(image, expected) < 0.5
    # 비교가 빈 것이 아님: 시점 순서를 바꾸거나 위아래가 뒤집히면 크게 달라진다(측정: 맞는 쪽 약 3, 반대쪽 약 66 — JPEG 손실만 비교하면 약 3)
    left, right = image[:, :160], image[:, 160:]
    assert mad(left, ref_oblique) > 20.0 and mad(right, ref_side) > 20.0
    assert mad(left, right) > 20.0  # 두 시점은 다른 그림이다


def test_reduced_mesh_model_is_built_once_at_start_with_the_configured_fraction(monkeypatch, home_qpos):
    calls = []
    real_build = aux_view_module.build_model

    def spy(scene_path, fraction, *args, **kwargs):
        calls.append(fraction)
        return real_build(scene_path, fraction, *args, **kwargs)

    monkeypatch.setattr(aux_view_module, "build_model", spy)
    worker = AuxViewWorker(enabled=True, fps=30.0)
    worker.start()
    try:
        worker.set_viewer_count(1)
        worker.offer_qpos(home_qpos)
        wait_for(lambda: worker.state()["frames"] >= 3)
        assert calls == [ac.AUX_MESH_FRACTION] == [0.10]  # 프레임마다 다시 만들지 않는다
    finally:
        worker.stop()
    custom = AuxViewWorker(enabled=False, mesh_fraction=0.5)
    assert custom._mesh_fraction == 0.5


def test_size_level_change_changes_the_frame_size(make_worker, home_qpos):
    worker = make_worker(fps=30.0, size_level="normal")
    worker.set_viewer_count(1)
    worker.offer_qpos(home_qpos)
    wait_for(lambda: worker.latest_frame is not None)
    worker.set_size_level("small")
    wait_for(lambda: jpeg_size(worker.latest_frame[1]) == frame_size("small"))
    assert worker.state()["size_level"] == "small"
    with pytest.raises(ValueError):
        worker.set_size_level("huge")
    with pytest.raises(ValueError):
        AuxViewWorker(enabled=False, size_level="huge")


def test_no_viewers_means_no_rendering(make_worker, home_qpos):
    worker = make_worker(fps=30.0)
    worker.offer_qpos(home_qpos)
    time.sleep(0.5)
    assert worker.state()["frames"] == 0
    worker.set_viewer_count(1)
    wait_for(lambda: worker.state()["frames"] >= 1)
    worker.set_viewer_count(0)
    time.sleep(0.2)  # 진행 중이던 한 장이 끝날 시간
    frames = worker.state()["frames"]
    time.sleep(0.5)
    assert worker.state()["frames"] == frames


def test_no_qpos_yet_means_no_rendering(make_worker):
    worker = make_worker(fps=30.0)
    worker.set_viewer_count(1)
    time.sleep(0.4)
    assert worker.state()["frames"] == 0


def simulate_frames(fps: float, duration_s: float, render_s: float = 0.0, step_s: float = 1 / 64) -> int:
    """가짜 시계로 _loop의 판단(is_due, next_render_time)만 돌려 장수를 센다. 실제 시간이 흐르지 않는다."""
    period_s = 1.0 / fps
    now_s, next_s, frames = 0.0, 0.0, 0
    while now_s < duration_s:
        if is_due(now_s, next_s):
            frames += 1
            now_s += render_s  # 렌더에 걸린 가짜 시간
            next_s = next_render_time(next_s, period_s, now_s)
        else:
            now_s += step_s
    return frames


def test_fps_setting_limits_the_frame_rate_with_fake_clock():
    assert simulate_frames(4.0, 2.0) == 8  # 0, 0.25, ... 1.75 s: 정확히 8장
    assert simulate_frames(1.0, 3.0) == 3
    assert simulate_frames(5.0, 2.0, render_s=0.05) == 10  # 렌더가 주기보다 짧으면 주기는 그대로


def test_overrun_render_does_not_burst_to_catch_up():
    assert next_render_time(next_s=0.0, period_s=0.25, after_render_s=10.0) == 10.0  # 밀렸으면 지금으로 맞춘다
    assert next_render_time(next_s=1.0, period_s=0.25, after_render_s=1.05) == 1.25  # 안 밀렸으면 주기대로
    # 렌더가 주기(0.25 s)보다 오래(0.5 s) 걸리면 쉬지 않고 그리되 한 번에 몰아 그리지는 않는다: 2초에 4장
    assert simulate_frames(4.0, 2.0, render_s=0.5) == 4
    assert is_due(1.0, 1.0) and not is_due(0.99, 1.0)


def test_real_thread_renders_at_most_about_the_configured_rate(make_worker, home_qpos):
    worker = make_worker(fps=4.0)
    worker.set_viewer_count(1)
    worker.offer_qpos(home_qpos)
    wait_for(lambda: worker.state()["frames"] >= 1)  # 실제 시간 의존은 느슨하게: 한 장은 나온다
    time.sleep(1.0)
    assert worker.state()["frames"] <= 10  # 설정 fps(4)를 크게 넘어 그리지 않는다


def test_default_is_not_to_pause_during_auto(make_worker, home_qpos):
    """교정은 AUTO 중에 봐야 의미가 있어서 기본은 멈추지 않는다."""
    assert ac.AUX_PAUSE_DURING_AUTO is False
    worker = make_worker(fps=30.0)
    worker.set_viewer_count(1)
    worker.offer_qpos(home_qpos, auto_running=True)
    wait_for(lambda: worker.state()["frames"] >= 2)
    frames = worker.state()["frames"]
    for _ in range(5):
        worker.offer_qpos(home_qpos, auto_running=True)
        time.sleep(0.05)
    assert not worker.state()["paused"]
    wait_for(lambda: worker.state()["frames"] > frames)  # AUTO 중에도 계속 그린다


def test_pauses_during_auto_and_resumes_after_when_configured(make_worker, home_qpos, monkeypatch):
    monkeypatch.setattr(ac, "AUX_PAUSE_DURING_AUTO", True)  # 멈추는 장치 자체는 그대로 동작한다
    worker = make_worker(fps=30.0)
    worker.set_viewer_count(1)
    worker.offer_qpos(home_qpos)
    wait_for(lambda: worker.state()["frames"] >= 2)
    worker.offer_qpos(home_qpos, auto_running=True)
    time.sleep(0.3)  # 진행 중이던 한 장이 끝날 시간
    frames = worker.state()["frames"]
    assert worker.state()["paused"]
    for _ in range(10):
        worker.offer_qpos(home_qpos, auto_running=True)
        time.sleep(0.05)
    assert worker.state()["frames"] == frames
    worker.offer_qpos(home_qpos, auto_running=False)
    wait_for(lambda: worker.state()["frames"] > frames)
    assert not worker.state()["paused"]


def test_keeps_only_the_latest_frame_for_late_viewers(make_worker, home_qpos):
    worker = make_worker(fps=30.0)
    worker.set_viewer_count(1)
    worker.offer_qpos(home_qpos)
    wait_for(lambda: worker.state()["frames"] >= 3)
    seq, _ = worker.latest_frame
    assert seq == worker.state()["frames"]  # 쌓아 두지 않고 가장 최근 한 장만 들고 있다


def test_slow_render_halves_fps_down_to_the_floor_and_recovers_for_new_viewers():
    worker = AuxViewWorker(enabled=False, fps=8.0)  # 스레드 없이 판단 메서드만 직접 부른다(렌더 시간은 가짜 숫자)
    limit_ms = ac.AUX_MAX_RENDER_MS
    worker._note_render_ms(limit_ms)  # 한계와 같으면 느린 것이 아니다
    assert worker.state()["fps"] == 8.0 and not worker.state()["slowed"]
    assert worker.state()["last_render_ms"] == limit_ms
    worker._note_render_ms(limit_ms + 1.0)
    assert worker.state()["fps"] == 4.0 and worker.state()["slowed"]
    for _ in range(10):
        worker._note_render_ms(limit_ms * 10)
    assert worker.state()["fps"] == ac.AUX_MIN_FPS  # 바닥 밑으로 내려가지 않는다
    worker.set_viewer_count(0)
    worker.set_viewer_count(1)  # 새로 보기 시작하면 설정 fps에서 다시 시작한다
    assert worker.state()["fps"] == 8.0 and not worker.state()["slowed"]


def test_start_failure_raises_and_is_shown_in_state(tmp_path):
    worker = AuxViewWorker(scene_path=tmp_path / "없는_장면.xml", enabled=True)
    with pytest.raises(RuntimeError):
        worker.start()
    assert not worker.is_alive
    assert worker.state()["error"] is not None and worker.state()["running"] is False
    worker.stop()


def test_thread_death_after_start_is_shown_in_state(home_qpos, monkeypatch):
    real_make = AuxViewWorker._make_renderer
    calls = []

    def flaky_make(self, model, level):
        calls.append(level)
        if len(calls) >= 2:  # 처음(시작)은 통과, 크기를 바꿔 다시 만들 때 실패
            raise RuntimeError("EGL 컨텍스트 생성 실패(테스트용)")
        return real_make(self, model, level)

    monkeypatch.setattr(AuxViewWorker, "_make_renderer", flaky_make)
    worker = AuxViewWorker(enabled=True, fps=30.0, size_level="normal")
    worker.start()
    try:
        assert worker.state()["error"] is None
        worker.set_viewer_count(1)
        worker.offer_qpos(home_qpos)
        wait_for(lambda: worker.state()["frames"] >= 1)
        worker.set_size_level("small")
        wait_for(lambda: not worker.is_alive)
        state = worker.state()
        assert state["running"] is False
        assert "EGL 컨텍스트 생성 실패" in state["error"] and state["error"].startswith("RuntimeError")
        # 죽은 뒤에도 물리 스레드가 부르는 함수들은 예외 없이 돌아온다
        worker.offer_qpos(home_qpos)
        worker.set_viewer_count(2)
        assert worker.latest_frame is not None  # 죽기 전 마지막 프레임은 남아 있다
    finally:
        worker.stop()  # 이미 죽었어도 예외 없이 끝난다
    worker.stop()


def test_offer_qpos_is_cheap_for_the_physics_thread(make_worker, home_qpos):
    worker = make_worker(fps=30.0)
    worker.set_viewer_count(1)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        worker.offer_qpos(home_qpos)
    mean_us = (time.perf_counter() - t0) / n * 1e6
    assert mean_us < 200, mean_us  # 렌더 중에도 락은 qpos 복사 동안만 잡으므로 틱이 늘어나지 않는다
