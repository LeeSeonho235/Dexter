"""학습·재생용 렌더링: 크기, 결정적인지, 카메라가 마우스 보기와 별개인지, 그림자·반사 설정이 실시간 화면과 같은 config를 쓰는지."""

import mujoco
import numpy as np
import pytest

from server import config
from server.render import StateRenderer, make_record_camera, record_view, set_render_flags
from server.sim_worker import SimWorker


@pytest.fixture(scope="module")
def home_qpos():
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    return model.key("home").qpos.copy()


@pytest.fixture(scope="module")
def renderer():
    with StateRenderer() as r:
        yield r


def test_image_is_240_by_320_uint8_rgb(renderer, home_qpos):
    img = renderer.render(home_qpos)
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8
    assert (config.RECORD_IMAGE_HEIGHT_PX, config.RECORD_IMAGE_WIDTH_PX) == (240, 320)
    assert img.std() > 5  # 빈 화면이 아니다


def test_same_state_renders_the_same_image(renderer, home_qpos):
    np.testing.assert_array_equal(renderer.render(home_qpos), renderer.render(home_qpos))


def test_block_position_changes_the_image(renderer, home_qpos):
    moved = home_qpos.copy()
    moved[6:9] = (0.24, -0.08, 0.015)
    assert not np.array_equal(renderer.render(home_qpos), renderer.render(moved))


def test_wrong_qpos_size_is_rejected(renderer):
    with pytest.raises(ValueError):
        renderer.render(np.zeros(6))


def test_block_and_target_are_visible_in_segmentation(renderer, home_qpos):
    seg = renderer.render_segmentation(home_qpos)
    model = renderer.model
    ids = seg[..., 0]
    assert (ids == model.geom("block").id).sum() > 20
    assert (ids == model.geom("target").id).sum() > 20


def test_record_camera_values_come_from_the_record_config_not_the_mouse_view():
    cam = make_record_camera()
    assert (cam.azimuth, cam.elevation, cam.distance) == (110.0, -35.0, 0.8)
    np.testing.assert_allclose(cam.lookat, config.VIEW_LOOKAT_M)
    assert record_view().azimuth_deg == config.RECORD_CAMERA_AZIMUTH_DEG


def test_mouse_view_changes_do_not_touch_the_record_camera(monkeypatch):
    """마우스 보기 설정(VIEW_*)이나 SimWorker의 보기 방향이 바뀌어도 학습 카메라는 RECORD_CAMERA_* 값 그대로다."""
    from server.view import CameraView

    before = make_record_camera()
    worker = SimWorker(
        config.SCENE_PATH, keyframe="home", camera="free", width_px=64, height_px=48, frame_fps=15,
        jpeg_quality=80, max_catch_up_s=0.25,
    )
    worker.set_view(CameraView(-60.0, -80.0, 1.9))  # 마우스로 돌린 것과 같다
    monkeypatch.setattr(config, "VIEW_DEFAULT", {"azimuth_deg": 10.0, "elevation_deg": -10.0, "distance_m": 1.5})
    monkeypatch.setattr(config, "VIEW_LOOKAT_M", (9.0, 9.0, 9.0))
    after = make_record_camera()
    assert (after.azimuth, after.elevation, after.distance) == (before.azimuth, before.elevation, before.distance)
    np.testing.assert_allclose(after.lookat, before.lookat)  # RECORD_CAMERA_LOOKAT_M은 import 때 값을 복사해 둔 별개 설정


def test_flags_helper_sets_shadow_and_reflection():
    class Scene:
        flags = {}

    class FakeRenderer:
        scene = Scene()

    fake = FakeRenderer()
    set_render_flags(fake, False, False)
    assert fake.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] is False
    assert fake.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] is False
    set_render_flags(fake, True, True)
    assert fake.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] is True


def test_state_renderer_defaults_use_the_live_view_flags(monkeypatch):
    """실시간 화면과 같은 config(RENDER_SHADOW, RENDER_REFLECTION)를 쓴다 — 어긋나면 학습 이미지와 보는 화면이 달라진다."""
    with StateRenderer(width_px=64, height_px=48) as r:
        assert r._shadow == config.RENDER_SHADOW and r._reflection == config.RENDER_REFLECTION
    assert config.RENDER_SHADOW is False and config.RENDER_REFLECTION is False
