"""상대 이동(클러치) 매핑(2-4): 기준점, 이득, 재진입, 튀지 않음, clamp, 보기 방위각, 중앙 영역만으로 상자 전체."""

import re
from pathlib import Path

import numpy as np
import pytest

from server import config
from server.mapping import HandMapper, MappingParams, clamp_to_workspace, screen_step_to_world, view_basis

ASPECT = config.IMAGE_ASPECT
GAIN_XY = config.RELATIVE_GAIN_XY_M_PER_WIDTH
GAIN_Z = config.RELATIVE_GAIN_Z_M_PER_SCALE
CENTER = np.array(config.RELATIVE_START_POSITION_M)
SCALE_MID = 0.13
SAFE = (0.10, 0.90)  # 영상 가장자리 10% 여유를 뺀 안전 구간
# 필터를 사실상 끄는 값(이득과 방향 계산만 정확히 보려고)
NO_FILTER = {"min_cutoff_hz": 1e6, "beta": 0.0, "d_cutoff_hz": 1e6}


def hand(x=0.5, y=0.5, scale=SCALE_MID, detected=True):
    if not detected:
        return {"type": "hand", "t": 0.0, "detected": False, "wrist": None, "index_tip": None, "thumb_tip": None,
                "pinch": None, "pinch_ratio": None, "palm_open": None, "hand_scale": None}
    return {"type": "hand", "t": 0.0, "detected": True, "wrist": [x, y, 0.0], "index_tip": [x, y, 0.0],
            "thumb_tip": [x, y, 0.0], "pinch": 0.1, "pinch_ratio": 0.8, "palm_open": True, "hand_scale": scale}


def relative(filtered=False, **kw):
    extra = {} if filtered else NO_FILTER
    return HandMapper(mode="relative", start_frames=kw.pop("start_frames", 3), **extra, **kw)


class Feeder:
    """프레임마다 시각을 1/30초 진행하며 메시지를 넣는 도우미."""

    def __init__(self, mapper, az=0.0):
        self.mapper, self.az, self.t = mapper, az, 0.0

    def feed(self, x=0.5, y=0.5, scale=SCALE_MID, detected=True, az=None):
        self.t += 1 / 30
        return self.mapper.update(hand(x, y, scale, detected), self.t, self.az if az is None else az)

    def clutch(self, x=0.5, y=0.5, scale=SCALE_MID):
        result = None
        for _ in range(self.mapper.start_frames):
            result = self.feed(x, y, scale)
        return result


# --- 설정 ---


def test_defaults_modes():
    assert config.MAPPING_MODE == "relative"
    assert HandMapper().mode == "absolute"  # 코드상 기본은 absolute(2-2 동작). 설정값은 ControlWorker가 넘긴다


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        HandMapper(mode="mouse")


def test_gains_are_the_documented_defaults():
    # 영상 너비의 35% 움직임으로 작업 상자의 가장 긴 변(y, 0.36 m)을 지나간다
    assert config.RELATIVE_FULL_TRAVEL_FRACTION == 0.35
    assert GAIN_XY * 0.35 == pytest.approx(0.36)
    assert GAIN_XY == pytest.approx(1.0286, abs=1e-3)
    # 높이는 절대 모드와 같은 감도: hand_scale 범위 전체(0.12)가 상자 높이(0.105 m)
    assert GAIN_Z == pytest.approx(-0.875)
    assert GAIN_Z * (config.HAND_SCALE_RANGE[1] - config.HAND_SCALE_RANGE[0]) == pytest.approx(
        -(config.WORKSPACE_Z_M[1] - config.WORKSPACE_Z_M[0])
    )


def test_server_aspect_matches_the_webcam_request():
    """hand 메시지에는 영상 크기가 없다. 서버 설정의 가로세로비가 웹캠 요청 해상도(web/src/hand/config.js)와 같아야 한다."""
    js = (Path(__file__).resolve().parents[2] / "web" / "src" / "hand" / "config.js").read_text(encoding="utf8")
    width = int(re.search(r"VIDEO_WIDTH_PX = (\d+)", js).group(1))
    height = int(re.search(r"VIDEO_HEIGHT_PX = (\d+)", js).group(1))
    assert config.IMAGE_ASPECT == pytest.approx(width / height)


# --- 기준점(클러치) ---


def test_no_target_before_the_start_condition():
    f = Feeder(relative())
    for _ in range(2):
        r = f.feed(0.4, 0.4)
        assert r.target_m is None and not r.controlling and r.reference_hand is None


def test_first_clutch_starts_at_workspace_center_without_a_jump():
    f = Feeder(relative())
    r = f.clutch(0.3, 0.7, 0.15)  # 손이 영상 어디에서 시작하든
    assert r.controlling
    np.testing.assert_allclose(r.target_m, CENTER, atol=1e-9)  # 처음 목표는 작업 상자 중심(home 끝이 아니다)
    assert r.reference_hand == pytest.approx((0.3, 0.7))


def test_reference_hand_is_the_mean_of_the_start_frames():
    f = Feeder(relative())
    f.feed(0.40, 0.50)
    f.feed(0.43, 0.52)
    r = f.feed(0.46, 0.51)
    assert r.reference_hand == pytest.approx((0.43, 0.51))
    # 평균과 마지막 프레임의 차이만큼만 첫 목표가 기준에서 벗어난다(몇 mm)
    assert np.linalg.norm(r.target_m - CENTER) < 0.05


def test_reference_hand_stays_until_the_hand_is_lost():
    f = Feeder(relative())
    f.clutch(0.5, 0.5)
    r = f.feed(0.6, 0.6)
    assert r.reference_hand == pytest.approx((0.5, 0.5))
    assert f.feed(detected=False).reference_hand is None


# --- 이득 ---


def test_horizontal_gain_hand_right_moves_target_right_on_a_top_view():
    f = Feeder(relative(), az=0.0)
    f.clutch(0.5, 0.5)
    r = f.feed(0.4, 0.5)  # 손이 영상 너비의 10% 오른쪽으로(x 감소)
    delta = r.raw_target_m - CENTER
    assert delta[1] == pytest.approx(-GAIN_XY * 0.1)  # 위에서 보기: 화면 오른쪽 = -y
    assert delta[0] == pytest.approx(0.0, abs=1e-12)
    assert delta[2] == pytest.approx(0.0, abs=1e-12)
    # 영상 너비 10%(640 px에서 64 px)에 로봇 약 10.3 cm
    assert abs(delta[1]) == pytest.approx(0.1029, abs=1e-3)


def test_vertical_gain_uses_the_same_pixel_to_meter_ratio():
    f = Feeder(relative(), az=0.0)
    f.clutch(0.5, 0.5)
    # 손이 위로 영상 높이의 4%(= 너비 3%와 같은 화소 수)만큼. 상자 x 폭이 0.10 m뿐이라 목표가 상자 안에 머무는 작은 움직임으로 본다.
    r = f.feed(0.5, 0.5 - 0.03 * ASPECT)
    delta = r.raw_target_m - CENTER
    assert delta[0] == pytest.approx(GAIN_XY * 0.03)  # 위에서 보기: 손 위 = +x, 같은 화소 수면 가로와 같은 거리
    assert delta[1] == pytest.approx(0.0, abs=1e-12)


def test_height_gain_hand_closer_goes_down():
    f = Feeder(relative())
    f.clutch(0.5, 0.5, SCALE_MID)
    r = f.feed(0.5, 0.5, SCALE_MID + 0.05)  # 손이 가까워졌다(hand_scale 증가)
    assert (r.raw_target_m - CENTER)[2] == pytest.approx(GAIN_Z * 0.05)
    assert (r.raw_target_m - CENTER)[2] < 0
    assert (r.raw_target_m - CENTER)[2] == pytest.approx(-0.04375)


def test_gains_are_configurable():
    f = Feeder(relative(gain_xy_m_per_width=2.0, gain_z_m_per_scale=-2.0))
    f.clutch(0.5, 0.5)
    delta = f.feed(0.45, 0.5, SCALE_MID + 0.01).raw_target_m - CENTER
    assert delta[1] == pytest.approx(-2.0 * 0.05)
    assert delta[2] == pytest.approx(-0.02)


def test_total_movement_equals_gain_times_hand_displacement_at_constant_azimuth():
    rng = np.random.default_rng(0)
    for az in (0.0, 105.0, 200.0):
        f = Feeder(relative(), az=az)
        f.clutch(0.5, 0.5)
        x, y, s = 0.5, 0.5, SCALE_MID
        for _ in range(40):
            x += rng.normal(0, 0.004)
            y += rng.normal(0, 0.004)
            s += rng.normal(0, 0.001)
            r = f.feed(x, y, s)
            expected_xy = screen_step_to_world(x - 0.5, y - 0.5, az, GAIN_XY, ASPECT)
            expected = CENTER + np.array([expected_xy[0], expected_xy[1], GAIN_Z * (s - SCALE_MID)])
            np.testing.assert_allclose(r.raw_target_m, expected, atol=1e-9)  # 누적해도 오차가 쌓이지 않는다


# --- 보기 방위각 ---


def test_hand_direction_follows_the_view_azimuth():
    for az in (0.0, 90.0, 105.0, 180.0, 270.0):
        f = Feeder(relative(), az=az)
        f.clutch(0.5, 0.5)
        right_step = f.feed(0.45, 0.5).raw_target_m - CENTER
        _, right = view_basis(az)
        assert right_step[:2] @ right > 0
        assert np.linalg.norm(right_step[:2] - (right_step[:2] @ right) * right) < 1e-9  # 오른쪽 방향으로만


def test_changing_the_view_does_not_move_the_target():
    f = Feeder(relative(), az=0.0)
    f.clutch(0.5, 0.5)
    before = f.feed(0.45, 0.5).raw_target_m.copy()
    for az in (10.0, 45.0, 90.0, 135.0):  # 손은 그대로, 보기만 돌린다(드래그)
        r = f.feed(0.45, 0.5, az=az)
        np.testing.assert_allclose(r.raw_target_m, before, atol=1e-12)


def test_new_view_applies_only_to_later_hand_motion():
    f = Feeder(relative(), az=0.0)
    f.clutch(0.5, 0.5)
    first = f.feed(0.45, 0.5, az=0.0).raw_target_m.copy()  # 위에서 보기: 오른쪽 = -y
    assert first[1] < CENTER[1]
    f.feed(0.45, 0.5, az=90.0)  # 보기를 90°로 돌림(손은 그대로)
    later = f.feed(0.40, 0.5, az=90.0).raw_target_m  # 손을 더 오른쪽으로
    step = later - first
    assert step[0] > 0 and abs(step[1]) < 1e-12  # 방위각 90°: 화면 오른쪽 = +x
    assert first[1] < CENTER[1] and later[1] == pytest.approx(first[1])  # 이미 간 -y 위치는 그대로


# --- 손이 사라졌다 다시 들어옴 ---


def test_hand_loss_holds_the_last_target_and_resets_the_clutch():
    f = Feeder(relative())
    f.clutch(0.5, 0.5)
    last = f.feed(0.4, 0.45).target_m.copy()
    for _ in range(5):
        r = f.feed(detected=False)
        np.testing.assert_array_equal(r.target_m, last)
        assert not r.controlling and r.reference_hand is None and not r.new_sample


def test_reentry_needs_the_start_frames_and_takes_a_new_reference_without_a_jump():
    f = Feeder(relative(), az=0.0)
    f.clutch(0.5, 0.5)
    last = f.feed(0.35, 0.40).target_m.copy()
    f.feed(detected=False)
    # 영상 반대편 가장자리 근처로 다시 들어온다
    r1 = f.feed(0.8, 0.8)
    r2 = f.feed(0.8, 0.8)
    for r in (r1, r2):
        np.testing.assert_array_equal(r.target_m, last)  # 시작 조건 전에는 그대로
        assert not r.controlling
    r3 = f.feed(0.8, 0.8)
    assert r3.controlling
    assert r3.reference_hand == pytest.approx((0.8, 0.8))  # 새 기준점
    np.testing.assert_allclose(r3.target_m, last, atol=1e-9)  # 다른 곳에서 들어왔는데도 목표가 튀지 않는다
    # 이후에는 새 기준에서 상대적으로 움직인다
    r4 = f.feed(0.79, 0.8)  # 손이 영상 너비의 1% 오른쪽으로(마지막 목표가 상자 한계 근처라 작게)
    np.testing.assert_allclose(r4.raw_target_m - last, [0.0, -GAIN_XY * 0.01, 0.0], atol=1e-9)


def test_reentry_in_the_middle_of_the_start_frames_restarts_the_count():
    f = Feeder(relative())
    f.feed()
    f.feed()
    f.feed(detected=False)
    assert f.feed().target_m is None
    assert f.feed().target_m is None
    assert f.feed().controlling


def test_second_clutch_base_is_the_last_target_not_the_workspace_center():
    f = Feeder(relative())
    f.clutch(0.5, 0.5)
    for _ in range(10):
        last = f.feed(0.3, 0.5).target_m
    assert abs(last[1] - CENTER[1]) > 0.05
    f.feed(detected=False)
    r = f.clutch(0.6, 0.6)
    np.testing.assert_allclose(r.target_m, last, atol=1e-9)


# --- clamp ---


def test_target_is_clamped_to_the_workspace_in_world_coordinates():
    f = Feeder(relative(), az=0.0)
    f.clutch(0.5, 0.5)
    x = 0.5
    for _ in range(30):
        x -= 0.02  # 계속 오른쪽으로(= -y)
        r = f.feed(x, 0.5)
    assert r.target_m[1] == pytest.approx(config.WORKSPACE_Y_M[0])
    assert r.raw_target_m[1] == pytest.approx(config.WORKSPACE_Y_M[0])
    for value, (lo, hi) in zip(r.target_m, (config.WORKSPACE_X_M, config.WORKSPACE_Y_M, config.WORKSPACE_Z_M)):
        assert lo <= value <= hi


def test_overshoot_beyond_the_box_is_discarded_so_coming_back_responds_immediately():
    f = Feeder(relative(), az=0.0)
    f.clutch(0.5, 0.5)
    x = 0.5
    for _ in range(40):  # 상자 끝을 한참 넘게 뻗는다
        x -= 0.02
        f.feed(x, 0.5)
    at_limit = f.feed(x, 0.5).raw_target_m[1]
    back = f.feed(x + 0.05, 0.5).raw_target_m[1]  # 손을 조금 되돌리면 바로 로봇도 되돌아온다(죽은 구간이 없다)
    assert back > at_limit + 0.04


@pytest.mark.parametrize("az", [0.0, 105.0, 180.0, 270.0])
def test_filtered_output_stays_inside_the_box_for_wild_motion(az):
    rng = np.random.default_rng(3)
    f = Feeder(relative(filtered=True), az=az)
    f.clutch(0.5, 0.5)
    x = y = 0.5
    for _ in range(300):
        x += rng.normal(0, 0.05)
        y += rng.normal(0, 0.05)
        r = f.feed(x, y, SCALE_MID + rng.normal(0, 0.03))
        for value, (lo, hi) in zip(r.target_m, (config.WORKSPACE_X_M, config.WORKSPACE_Y_M, config.WORKSPACE_Z_M)):
            assert lo <= value <= hi


# --- 필터 ---


def test_one_euro_filter_still_applies_in_relative_mode():
    rng = np.random.default_rng(1)
    f = Feeder(relative(filtered=True))
    f.clutch(0.5, 0.5)
    raws, outs = [], []
    for _ in range(300):
        r = f.feed(0.5 + rng.normal(0, 0.004), 0.5 + rng.normal(0, 0.004), SCALE_MID + rng.normal(0, 0.002))
        raws.append(r.raw_target_m)
        outs.append(r.target_m)
    assert np.all(np.std(outs[60:], axis=0) < 0.7 * np.std(raws[60:], axis=0))


# --- 중앙 영역만으로 작업 상자 전체에 닿는가 ---


def box_corners():
    xs, ys, zs = config.WORKSPACE_X_M, config.WORKSPACE_Y_M, config.WORKSPACE_Z_M
    return [np.array([x, y, z]) for x in xs for y in ys for z in zs]


def hand_delta_for(world_delta, az):
    """원하는 월드 이동(m)을 만들려면 영상에서 손이 얼마나 움직여야 하는가: (dx_img, dy_img, dscale)."""
    forward, right = view_basis(az)
    along_right = world_delta[:2] @ right
    along_forward = world_delta[:2] @ forward
    return (-along_right / GAIN_XY, -along_forward * ASPECT / GAIN_XY, world_delta[2] / GAIN_Z)


@pytest.mark.parametrize("az", [0.0, 90.0, 105.0, 180.0, 270.0, 33.0])
def test_whole_workspace_is_reachable_from_the_image_center_region(az):
    """기준 로봇 위치와 기준 손이 모두 중앙일 때, 모서리 8개 모두 영상 가장자리 10% 여유 안쪽의 손동작만으로 닿는다."""
    for corner in box_corners():
        f = Feeder(relative(), az=az)
        f.clutch(0.5, 0.5)
        dx, dy, _ = hand_delta_for(corner - CENTER, az)
        assert SAFE[0] <= 0.5 + dx <= SAFE[1] and SAFE[0] <= 0.5 + dy <= SAFE[1], (az, corner, dx, dy)
        # 실제로 매퍼에 넣어서도 닿는지: 손을 직선으로 움직인다(높이는 따로 확인)
        for k in range(1, 21):
            r = f.feed(0.5 + dx * k / 20, 0.5 + dy * k / 20, SCALE_MID)
        np.testing.assert_allclose(r.raw_target_m[:2], corner[:2], atol=1e-6)


@pytest.mark.parametrize("az", [0.0, 90.0, 105.0, 180.0])
def test_any_start_to_any_corner_fits_inside_the_safe_region(az):
    """기준 로봇 위치가 상자 어디(모서리 포함)에 있어도, 반대쪽 끝까지 가는 손 이동량이 안전 구간(80%) 안에 들어간다.
    (그러면 시작 손 위치를 적당히 잡아 가장자리를 건드리지 않고 갈 수 있다.)"""
    span = SAFE[1] - SAFE[0]
    for start in box_corners():
        for goal in box_corners():
            dx, dy, _ = hand_delta_for(goal - start, az)
            assert abs(dx) <= span and abs(dy) <= span, (az, start, goal, dx, dy)


def test_longest_side_needs_about_35_percent_of_the_image_width():
    dx, dy, _ = hand_delta_for(np.array([0.0, 0.36, 0.0]), 0.0)
    assert abs(dx) == pytest.approx(0.35)
    assert dy == pytest.approx(0.0, abs=1e-12)


def test_full_height_range_needs_the_whole_hand_scale_range():
    """높이는 절대 모드와 같은 감도라, 상자 높이 전체는 hand_scale 범위 전체(0.07~0.19)가 필요하다. 한 번의 클러치로
    닿는 높이는 시작할 때 손이 얼마나 먼지에 달려 있다(손 크기가 한계에 가까우면 한 번 빼고 다시 잡아야 한다)."""
    _, _, dscale = hand_delta_for(np.array([0.0, 0.0, config.WORKSPACE_Z_M[1] - config.WORKSPACE_Z_M[0]]), 0.0)
    assert abs(dscale) == pytest.approx(config.HAND_SCALE_RANGE[1] - config.HAND_SCALE_RANGE[0])


def test_height_corners_are_reachable_when_the_hand_starts_far_or_near_enough():
    # 상자 맨 위(z 0.12)에서 클러치: 손이 가장 먼(0.07)에서 시작해 가장 가까이(0.19)로 오면 맨 아래(z 0.015)에 닿는다
    f = Feeder(relative(start_position_m=(0.23, 0.0, config.WORKSPACE_Z_M[1])))
    f.clutch(0.5, 0.5, config.HAND_SCALE_RANGE[0])
    r = f.feed(0.5, 0.5, config.HAND_SCALE_RANGE[1])
    assert r.raw_target_m[2] == pytest.approx(config.WORKSPACE_Z_M[0], abs=1e-9)


def test_workspace_params_are_used_for_the_clamp():
    params = MappingParams(workspace_x_m=(0.2, 0.25), workspace_y_m=(-0.05, 0.05), workspace_z_m=(0.03, 0.06))
    f = Feeder(HandMapper(mode="relative", params=params, start_frames=1, start_position_m=(0.5, 0.5, 0.5), **NO_FILTER))
    r = f.clutch(0.5, 0.5)
    np.testing.assert_allclose(r.target_m, clamp_to_workspace(np.array([0.5, 0.5, 0.5]), params))
