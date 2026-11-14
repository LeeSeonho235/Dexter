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
from server.aux_view import AuxViewWorker, _MultiViewRenderer, build_aux_model, is_due, make_aux_cameras, next_render_time
from server.episode_store import EpisodeStore
from server.render import StateRenderer, set_render_flags
from server.sim_worker import TIP_SITE
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
    return view_w_px * len(ac.AUX_CAMERAS), height_px


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
    assert jpeg_size(jpeg) == frame_size(ac.AUX_SIZE_DEFAULT) == (400, 150)  # 두 시점을 가로로 이어 붙인 한 장
    image = np.asarray(Image.open(io.BytesIO(jpeg)))
    assert image.std() > 5  # 빈 화면이 아니다
    state = worker.state()
    assert (state["views"], state["view_width_px"], state["width_px"], state["height_px"]) == (2, 200, 400, 150)


def reference_halves(pose: np.ndarray) -> list[np.ndarray]:
    """같은 줄인 모델을 mujoco.Renderer로 시점 하나씩 그린 그림(첫 렌더링은 버린다). 두 시점 합친 그림의 왼쪽, 오른쪽 절반과 비교한다."""
    model = build_aux_model(config.SCENE_PATH, ac.AUX_MESH_FRACTION)  # 줄인 메시와 망원 시야각(AUX_FOVY_DEG)이 같은 모델
    data = mujoco.MjData(model)
    data.qpos[:13] = pose
    mujoco.mj_forward(model, data)
    view_w_px, height_px = ac.AUX_SIZE_LEVELS["normal"]
    out = []
    with mujoco.Renderer(model, height_px, view_w_px) as renderer:
        for camera in make_aux_cameras():
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
    assert image.shape == expected.shape == (150, 400, 3)
    assert mad(image, expected) < 0.5
    # 비교가 빈 것이 아님: 시점 순서를 바꾸거나 위아래가 뒤집히면 크게 달라진다(측정: 맞는 쪽 약 3, 반대쪽 약 66 — JPEG 손실만 비교하면 약 3)
    left, right = image[:, :200], image[:, 200:]
    assert mad(left, ref_oblique) > 20.0 and mad(right, ref_side) > 20.0
    assert mad(left, right) > 20.0  # 두 시점은 다른 그림이다


def moved_pose(home_qpos: np.ndarray) -> np.ndarray:
    pose = home_qpos.copy()
    pose[0], pose[1] = 0.4, -1.2
    return pose


def test_no_black_band_at_the_top_of_either_view(make_worker, home_qpos):
    """이전 카메라(시야 45°, 고도 -10°)는 위쪽 27%가 하늘(검정)이었다. 지금은 고도가 시야 반각(10°)보다 가파라서 하늘이 화면 밖이다."""
    worker = make_worker(fps=30.0)
    worker.set_viewer_count(1)
    for pose in (home_qpos, moved_pose(home_qpos)):
        worker.offer_qpos(pose)
        start = worker.state()["frames"]
        wait_for(lambda: worker.state()["frames"] >= start + 3)
        image = np.asarray(Image.open(io.BytesIO(worker.latest_frame[1])))
        for name, tile in (("옆에서", image[:, :200]), ("비스듬히", image[:, 200:])):
            top = tile[:8]  # 위쪽 8줄(칸 높이 150의 약 5%)
            assert (top.max(axis=(1, 2)) > 40).all(), f"{name}: 위쪽에 검은 줄이 있다"
            assert top.mean() > 60, name

    # 검사가 빈 것이 아님: 이전 카메라와 시야각(45°)에서는 같은 검사가 검은 띠를 잡아낸다
    model = aux_mesh.build_model(config.SCENE_PATH, ac.AUX_MESH_FRACTION)  # 시야각을 안 바꾼 모델(45°)
    data = mujoco.MjData(model)
    data.qpos[:13] = home_qpos
    mujoco.mj_forward(model, data)
    old_camera = make_free_camera(clamp_view(azimuth_deg=105.0, elevation_deg=-10.0, distance_m=0.8), (0.23, 0.0, 0.10))
    with mujoco.Renderer(model, 150, 200) as renderer:
        for _ in range(2):
            renderer.update_scene(data, camera=old_camera)
            old = renderer.render()
    assert (old[:8].max(axis=(1, 2)) < 12).all(), "이전 카메라에서 검은 띠가 재현되지 않으면 이 검사는 의미가 없다"


def test_main_model_fovy_is_unchanged_and_only_the_aux_model_is_telephoto(make_worker):
    assert 0.0 < ac.AUX_FOVY_DEG < 45.0
    aux_model = build_aux_model(config.SCENE_PATH, ac.AUX_MESH_FRACTION)
    assert aux_model.vis.global_.fovy == ac.AUX_FOVY_DEG == 20.0
    # 메시만 줄이는 aux_mesh.build_model은 시야각을 건드리지 않는다
    assert aux_mesh.build_model(config.SCENE_PATH, ac.AUX_MESH_FRACTION).vis.global_.fovy == 45.0
    # aux 워커가 돌아간 뒤에 새로 읽은 시뮬레이션 모델과 학습용 StateRenderer의 모델은 config(scene.xml)의 45° 그대로다
    worker = make_worker(fps=30.0)
    worker.set_viewer_count(1)
    wait_for(lambda: worker.state()["running"])
    assert mujoco.MjModel.from_xml_path(str(config.SCENE_PATH)).vis.global_.fovy == 45.0
    with StateRenderer() as state_renderer:
        assert state_renderer.model.vis.global_.fovy == 45.0


HAVE_RAW = (config.RECORD_DIR / "episode_000019.npz").exists()
BLOCK_HEIGHT_MIN_FRACTION = 0.125  # 사용자가 메인 화면을 직접 확대한 모습의 블록 높이(칸 높이 대비 중앙값 12.5%)를 기준으로 삼았다


def demo_work_frames(n: int = 120) -> np.ndarray:
    """시연 24개에서 집게 끝과 블록이 15cm 안인 프레임을 고정 시드로 n개 뽑는다(qpos 13개). data/raw는 읽기만 한다."""
    base = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    data = mujoco.MjData(base)
    site_id, block_id = base.site(TIP_SITE).id, base.body("block").id
    store = EpisodeStore()
    frames, near = [], []
    for meta in store.scan():
        for qpos in store.load_arrays(meta["episode_index"])["qpos"][::3]:
            data.qpos[:13] = qpos
            mujoco.mj_kinematics(base, data)
            frames.append(qpos)
            near.append(np.linalg.norm(data.site_xpos[site_id] - data.xpos[block_id]) < 0.15)
    frames, near = np.array(frames), np.array(near)
    return frames[np.random.default_rng(0).choice(np.nonzero(near)[0], size=n, replace=False)]


def block_height_fractions(model: mujoco.MjModel, cameras: list, frames: np.ndarray, view_w_px: int, view_h_px: int) -> list[list[float | None]]:
    """칸마다, 프레임마다 블록(빨간 픽셀 상자)의 높이 / 칸 높이. 안 보이면 None. 실제 aux 렌더러(_MultiViewRenderer)로 그린다."""
    data = mujoco.MjData(model)
    renderer = _MultiViewRenderer(model, view_w_px, view_h_px, len(cameras))
    out: list[list[float | None]] = [[] for _ in cameras]
    try:
        for index, qpos in enumerate([frames[0], *frames]):  # 첫 렌더링은 버린다(docs/07)
            data.qpos[:13] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            image = renderer.render(data, cameras)
            if index == 0:
                continue
            for i in range(len(cameras)):
                tile = image[:, i * view_w_px : (i + 1) * view_w_px]
                ys, _ = np.nonzero((tile[..., 0] > 120) & (tile[..., 1] < 90) & (tile[..., 2] < 90))
                out[i].append(None if len(ys) < 4 else float(ys.max() - ys.min() + 1) / view_h_px)
    finally:
        renderer.close()
    return out


@pytest.mark.skipif(not HAVE_RAW, reason="data/raw가 이 컴퓨터에 없음")
def test_block_is_at_least_12_5_percent_of_the_tile_height_on_demo_frames():
    """시연 작업 프레임 120개: 두 칸 모두 블록 높이의 중앙값이 칸 높이의 12.5% 이상이고, 블록이 보이는 프레임이 85% 이상이다.
    측정(2026-11-13, 200x150): 옆에서 중앙값 15.3%(보임 100%), 비스듬히 12.7%(보임 87.5%). 기준 화면(사용자 메인 확대)은 중앙값 12.5%, 보임 99%.
    비스듬히는 중앙값 여유가 0.2%p뿐이다(거리 0.8 m이면 15%, 방위각 70°·거리 0.8 m이면 보임 100%). aux_view_config.AUX_CAMERAS."""
    frames = demo_work_frames()
    view_w_px, view_h_px = ac.AUX_SIZE_LEVELS["normal"]
    model = build_aux_model(config.SCENE_PATH, ac.AUX_MESH_FRACTION)
    fractions = block_height_fractions(model, make_aux_cameras(), frames, view_w_px, view_h_px)
    for name, per_frame in zip(("옆에서", "비스듬히"), fractions, strict=True):
        visible = [f for f in per_frame if f is not None]
        assert len(visible) / len(per_frame) >= 0.85, (name, len(visible) / len(per_frame))
        assert float(np.median(visible)) >= BLOCK_HEIGHT_MIN_FRACTION, (name, float(np.median(visible)))

    # 검사가 빈 것이 아님: 이전 카메라(시야 45°, 거리 0.8 m)에서는 블록이 칸 높이의 약 6~8%라서 같은 기준에 못 미친다
    plain = aux_mesh.build_model(config.SCENE_PATH, ac.AUX_MESH_FRACTION)
    old_cameras = [
        make_free_camera(clamp_view(azimuth_deg=105.0, elevation_deg=-10.0, distance_m=0.8), (0.23, 0.0, 0.10)),
        make_free_camera(clamp_view(azimuth_deg=45.0, elevation_deg=-25.0, distance_m=0.8), (0.23, 0.0, 0.10)),
    ]
    old = block_height_fractions(plain, old_cameras, frames[:40], view_w_px, view_h_px)
    for per_frame in old:
        visible = [f for f in per_frame if f is not None]
        assert float(np.median(visible)) < 0.10


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
