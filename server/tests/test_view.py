"""보기 방향(2-3): 범위 자르기, 설정값, 카메라 방향 규칙."""

import math

import mujoco
import numpy as np
import pytest

from server import config
from server.view import CameraView, clamp_view, default_view, make_free_camera, preset_view, wrap_azimuth_deg


def test_in_range_values_are_unchanged():
    assert clamp_view(30.0, -40.0, 0.9) == CameraView(30.0, -40.0, 0.9)


@pytest.mark.parametrize("elevation,expected", [(-200.0, -89.0), (-89.0, -89.0), (-3.0, -3.0), (0.0, -3.0), (45.0, -3.0)])
def test_elevation_is_clamped(elevation, expected):
    assert clamp_view(0.0, elevation, 0.8).elevation_deg == expected


@pytest.mark.parametrize("distance,expected", [(0.0, 0.3), (-5.0, 0.3), (0.3, 0.3), (2.0, 2.0), (99.0, 2.0), (1e308, 2.0)])
def test_distance_is_clamped(distance, expected):
    assert clamp_view(0.0, -30.0, distance).distance_m == expected


@pytest.mark.parametrize("azimuth,expected", [(0.0, 0.0), (90.0, 90.0), (180.0, -180.0), (270.0, -90.0), (360.0, 0.0), (-190.0, 170.0), (725.0, 5.0)])
def test_azimuth_wraps_instead_of_clamping(azimuth, expected):
    assert wrap_azimuth_deg(azimuth) == pytest.approx(expected)
    assert clamp_view(azimuth, -30.0, 0.8).azimuth_deg == pytest.approx(expected)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_is_rejected(bad):
    for args in ((bad, -30.0, 0.8), (0.0, bad, 0.8), (0.0, -30.0, bad)):
        with pytest.raises(ValueError):
            clamp_view(*args)


def test_default_and_presets_are_inside_the_allowed_ranges():
    lo_el, hi_el = config.VIEW_ELEVATION_RANGE_DEG
    lo_d, hi_d = config.VIEW_DISTANCE_RANGE_M
    assert set(config.VIEW_PRESETS) == {"default", "top", "side"}
    for name, preset in config.VIEW_PRESETS.items():
        assert lo_el <= preset["elevation_deg"] <= hi_el, name
        assert lo_d <= preset["distance_m"] <= hi_d, name
        # 자르지 않아도 되는 값이어야 한다(설정한 값이 그대로 쓰인다)
        assert preset_view(name).elevation_deg == preset["elevation_deg"]
        assert preset_view(name).distance_m == preset["distance_m"]
    assert default_view() == preset_view("default")


def test_as_dict_is_rounded_and_json_friendly():
    d = CameraView(12.3456, -33.3333, 0.81234).as_dict()
    assert d == {"azimuth_deg": 12.35, "elevation_deg": -33.33, "distance_m": 0.812}


def camera_axes(view):
    """자유 카메라를 만들어 MuJoCo가 실제로 정하는 카메라 축(월드 좌표)을 구한다: (화면 오른쪽, 화면 위쪽)."""
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=100)
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None, make_free_camera(view), mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    gl = scene.camera[0]
    forward = np.array(gl.forward)
    up = np.array(gl.up)
    right = np.cross(forward, up)
    return right, up, forward


def test_side_preset_has_robot_front_on_screen_right():
    """옆에서 프리셋: 로봇 앞쪽(+x)이 화면 오른쪽을 향한다."""
    right, _, _ = camera_axes(preset_view("side"))
    assert right[0] > 0.9  # +x 쪽


def test_top_preset_matches_the_fixed_top_camera_orientation():
    """위에서 프리셋: 화면 위쪽 = +x, 화면 오른쪽 = -y (scene.xml의 고정 top 카메라와 같다)."""
    right, up, _ = camera_axes(preset_view("top"))
    assert up[0] > 0.9
    assert right[1] < -0.9


def test_default_view_looks_down_at_the_workspace_center():
    _, _, forward = camera_axes(preset_view("default"))
    assert forward[2] < -0.4  # 아래를 향한다(고도 -35° → 약 -0.57)


def test_camera_lookat_is_the_configured_point():
    camera = make_free_camera(preset_view("default"))
    np.testing.assert_allclose(camera.lookat, config.VIEW_LOOKAT_M)
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE


def test_workspace_center_is_near_the_lookat_point():
    center_xy = [sum(config.WORKSPACE_X_M) / 2, sum(config.WORKSPACE_Y_M) / 2]
    np.testing.assert_allclose(config.VIEW_LOOKAT_M[:2], center_xy, atol=0.03)
