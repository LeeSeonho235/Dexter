"""화면 기준 매핑(2-4): 보기 방위각에서 손 오른쪽/위가 어느 월드 방향인지. MuJoCo 실제 카메라와 대조한다."""

import math

import mujoco
import numpy as np
import pytest

from server import config
from server.mapping import screen_step_to_world, view_basis
from server.view import CameraView, make_free_camera

GAIN = 1.0  # m / 영상 너비
ASPECT = 4 / 3


def hand_right(az, amount=0.1):
    """손이 오른쪽으로 갔다: 영상 x는 줄어든다(좌우 뒤집지 않은 원본)."""
    return screen_step_to_world(-amount, 0.0, az, GAIN, ASPECT)


def hand_up(az, amount=0.1):
    """손이 위로 갔다: 영상 y는 줄어든다."""
    return screen_step_to_world(0.0, -amount, az, GAIN, ASPECT)


def unit(v):
    return v / np.linalg.norm(v)


@pytest.mark.parametrize(
    "az,expected_right,expected_forward",
    [
        (0, (0, -1), (1, 0)),  # 위에서 보기: 화면 위 = +x, 화면 오른쪽 = -y
        (90, (1, 0), (0, 1)),  # 화면 오른쪽 = +x
        (180, (0, 1), (-1, 0)),
        (270, (-1, 0), (0, -1)),
        (-90, (-1, 0), (0, -1)),
    ],
)
def test_view_basis_at_quarter_turns(az, expected_right, expected_forward):
    forward, right = view_basis(az)
    np.testing.assert_allclose(right, expected_right, atol=1e-12)
    np.testing.assert_allclose(forward, expected_forward, atol=1e-12)


@pytest.mark.parametrize("az", [0, 90, 105, 180, 270, -45, 725, -1000.5])
def test_basis_is_orthonormal_and_right_handed(az):
    forward, right = view_basis(az)
    assert np.linalg.norm(forward) == pytest.approx(1.0)
    assert np.linalg.norm(right) == pytest.approx(1.0)
    assert forward @ right == pytest.approx(0.0, abs=1e-12)
    # 위에서 내려다볼 때 forward가 위, right가 오른쪽이 되려면 right = forward를 시계 방향으로 90° 돌린 것
    assert forward[0] * right[1] - forward[1] * right[0] == pytest.approx(-1.0)


@pytest.mark.parametrize("az", [0, 90, 105, 180, 270])
def test_hand_right_and_up_go_right_and_forward_on_screen(az):
    forward, right = view_basis(az)
    r = hand_right(az)
    u = hand_up(az)
    np.testing.assert_allclose(unit(r), right, atol=1e-12)  # 손 오른쪽 → 화면 오른쪽
    np.testing.assert_allclose(unit(u), forward, atol=1e-12)  # 손 위 → 화면 안쪽
    assert r @ forward == pytest.approx(0.0, abs=1e-12)
    assert u @ right == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "az,expected_right_world,expected_up_world",
    [
        (0, (0, -1), (1, 0)),
        (90, (1, 0), (0, 1)),
        (105, (0.9659, 0.2588), (-0.2588, 0.9659)),  # "옆에서" 프리셋: 화면 오른쪽이 대략 +x
        (180, (0, 1), (-1, 0)),
        (270, (-1, 0), (0, -1)),
    ],
)
def test_expected_world_directions(az, expected_right_world, expected_up_world):
    np.testing.assert_allclose(unit(hand_right(az)), expected_right_world, atol=1e-3)
    np.testing.assert_allclose(unit(hand_up(az)), expected_up_world, atol=1e-3)


def test_presets_match_the_documented_screen_directions():
    top = config.VIEW_PRESETS["top"]["azimuth_deg"]
    side = config.VIEW_PRESETS["side"]["azimuth_deg"]
    assert unit(hand_up(top))[0] == pytest.approx(1.0)  # 위에서: 손 위 → +x
    assert unit(hand_right(top))[1] == pytest.approx(-1.0)  # 손 오른쪽 → -y
    assert unit(hand_right(side))[0] > 0.9  # 옆에서: 손 오른쪽 → 대략 +x


def test_vertical_hand_motion_is_in_image_width_units():
    """영상 높이 기준 y를 가로 길이 단위로 바꾼다: 같은 비율(0.1)이라도 세로는 높이/너비(0.75)배만큼만 간다."""
    right_m = np.linalg.norm(hand_right(0, 0.1))
    up_m = np.linalg.norm(hand_up(0, 0.1))
    assert right_m == pytest.approx(0.1 * GAIN)
    assert up_m == pytest.approx(0.1 * GAIN / ASPECT)
    # 같은 화소 수(= 같은 물리적 손 이동)를 움직이면 같은 거리: 가로 64px = 0.1 너비, 세로 64px = 0.1333 높이
    assert np.linalg.norm(hand_up(0, 0.1 * ASPECT)) == pytest.approx(right_m)


def test_no_motion_gives_no_step_and_scales_with_gain():
    assert np.allclose(screen_step_to_world(0, 0, 33, 1.0, ASPECT), 0)
    np.testing.assert_allclose(
        screen_step_to_world(-0.1, -0.05, 40, 2.0, ASPECT), 2.0 * screen_step_to_world(-0.1, -0.05, 40, 1.0, ASPECT)
    )


# --- MuJoCo 실제 카메라와 대조: 어느 고도각에서든(-89° 포함) 같다 ---


@pytest.fixture(scope="module")
def mj():
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def actual_screen_axes_xy(mj, azimuth_deg, elevation_deg):
    """MuJoCo가 실제로 만든 카메라의 (화면 오른쪽, 화면 위쪽) 방향을 바닥(xy)에 투영한 단위 벡터."""
    model, data = mj
    scene = mujoco.MjvScene(model, maxgeom=100)
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        make_free_camera(CameraView(azimuth_deg, elevation_deg, 0.8)), mujoco.mjtCatBit.mjCAT_ALL, scene,
    )
    gl = scene.camera[0]
    forward = np.array(gl.forward)
    up = np.array(gl.up)
    right = np.cross(forward, up)
    return unit(right[:2]), unit(up[:2])


@pytest.mark.parametrize("elevation", [-89.0, -75.0, -60.0, -35.0, -10.0, -3.0])
@pytest.mark.parametrize("azimuth", [0, 90, 105, 180, 270, -45, 30.5])
def test_matches_the_real_mujoco_camera(mj, azimuth, elevation):
    right_actual, up_actual = actual_screen_axes_xy(mj, azimuth, elevation)
    forward, right = view_basis(azimuth)
    assert right @ right_actual > math.cos(math.radians(0.5))
    assert forward @ up_actual > math.cos(math.radians(0.5))
    # 그래서 손 오른쪽/위가 실제 화면에서도 오른쪽/위
    assert unit(hand_right(azimuth)) @ right_actual > 0.999
    assert unit(hand_up(azimuth)) @ up_actual > 0.999


def test_near_vertical_view_is_stable(mj):
    """고도각 -89°처럼 거의 수직일 때도 방위각만으로 계산하므로 흔들리지 않는다(방위각을 1°씩 바꾸면 방향도 1°씩)."""
    previous = None
    for az in np.arange(0, 360, 1.0):
        direction = unit(hand_right(az))
        assert np.all(np.isfinite(direction))
        if previous is not None:
            angle = math.degrees(math.acos(np.clip(direction @ previous, -1, 1)))
            assert angle == pytest.approx(1.0, abs=1e-6)
        previous = direction
    right_actual, _ = actual_screen_axes_xy(mj, 17.0, -89.0)
    assert unit(hand_right(17.0)) @ right_actual > 0.9999
