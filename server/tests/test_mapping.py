"""손 → 목표 위치 매핑: 방향, 높이, clamp, 손이 없을 때 유지, 시작 조건, 필터."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from server import config
from server.mapping import HandMapper, MappingParams, clamp_to_workspace, hand_to_raw_target, hand_to_target, lerp

P = MappingParams()
HAND_SCALE_MID = sum(config.HAND_SCALE_RANGE) / 2


def hand(x=0.5, y=0.5, scale=HAND_SCALE_MID, detected=True, pinch_ratio=0.8):
    if not detected:
        return {"type": "hand", "t": 0.0, "detected": False, "wrist": None, "index_tip": None, "thumb_tip": None,
                "pinch": None, "pinch_ratio": None, "palm_open": None, "hand_scale": None}
    return {"type": "hand", "t": 0.0, "detected": True, "wrist": [x, y, 0.0], "index_tip": [x, y, 0.0],
            "thumb_tip": [x, y, 0.0], "pinch": 0.1, "pinch_ratio": pinch_ratio, "palm_open": True, "hand_scale": scale}


@pytest.fixture(scope="module")
def top_camera_axes():
    """sim/scene.xml의 top 카메라가 보는 방향: (화면 오른쪽, 화면 위쪽) 단위 벡터(월드 좌표)."""
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    xmat = data.cam_xmat[model.camera("top").id].reshape(3, 3)
    return xmat[:, 0], xmat[:, 1]


def test_lerp_basic():
    assert lerp(0.5, (0.0, 1.0), (10.0, 20.0)) == 15.0
    assert lerp(0.5, (0.0, 1.0), (20.0, 10.0)) == 15.0
    assert lerp(2.0, (0.0, 1.0), (0.0, 10.0)) == 20.0  # 구간 밖은 자르지 않고 늘린다


def test_center_of_image_is_center_of_workspace():
    target = hand_to_target(0.5, 0.5, HAND_SCALE_MID)
    assert target[0] == pytest.approx(sum(config.WORKSPACE_X_M) / 2)
    assert target[1] == pytest.approx(sum(config.WORKSPACE_Y_M) / 2)
    assert target[2] == pytest.approx(sum(config.WORKSPACE_Z_M) / 2, abs=0.01)


def test_hand_right_moves_gripper_right_on_top_view(top_camera_axes):
    """원본 영상에서 손을 오른쪽으로 가면 이미지 x는 줄어든다. 화면 속 집게도 오른쪽으로 가야 한다."""
    screen_right, _ = top_camera_axes
    left_hand = hand_to_target(0.6, 0.5, HAND_SCALE_MID)
    right_hand = hand_to_target(0.4, 0.5, HAND_SCALE_MID)  # x가 줄었다 = 손이 오른쪽으로 갔다
    assert np.dot(right_hand - left_hand, screen_right) > 0
    assert right_hand[0] == pytest.approx(left_hand[0])  # 앞뒤는 안 바뀐다


def test_hand_up_moves_gripper_up_on_top_view(top_camera_axes):
    """손을 위로(이미지 y 감소) 올리면 화면 속 집게도 위(로봇 앞쪽)로 간다."""
    _, screen_up = top_camera_axes
    low_hand = hand_to_target(0.5, 0.6, HAND_SCALE_MID)
    high_hand = hand_to_target(0.5, 0.4, HAND_SCALE_MID)
    assert np.dot(high_hand - low_hand, screen_up) > 0
    assert high_hand[1] == pytest.approx(low_hand[1])


def test_closer_hand_lowers_gripper():
    far = hand_to_target(0.5, 0.5, config.HAND_SCALE_RANGE[0] + 0.01)
    near = hand_to_target(0.5, 0.5, config.HAND_SCALE_RANGE[1] - 0.01)
    assert near[2] < far[2]


def test_height_direction_follows_config_sign():
    flipped = replace(P, robot_z_for_scale_m=P.robot_z_for_scale_m[::-1])
    far = hand_to_target(0.5, 0.5, config.HAND_SCALE_RANGE[0] + 0.01, flipped)
    near = hand_to_target(0.5, 0.5, config.HAND_SCALE_RANGE[1] - 0.01, flipped)
    assert near[2] > far[2]


@pytest.mark.parametrize("x,y,scale", [(-5, -5, 0.0), (5, 5, 5.0), (0.0, 1.0, 0.01), (1.0, 0.0, 1.0)])
def test_target_is_clamped_to_workspace(x, y, scale):
    t = hand_to_target(x, y, scale)
    assert config.WORKSPACE_X_M[0] <= t[0] <= config.WORKSPACE_X_M[1]
    assert config.WORKSPACE_Y_M[0] <= t[1] <= config.WORKSPACE_Y_M[1]
    assert config.WORKSPACE_Z_M[0] <= t[2] <= config.WORKSPACE_Z_M[1]


def test_clamp_hits_box_faces_exactly():
    raw = hand_to_raw_target(-5, 5, 5.0)
    clamped = clamp_to_workspace(raw)
    assert clamped[0] == config.WORKSPACE_X_M[0]  # 이미지 y가 아주 크다(손이 아래) → 로봇 x는 아주 작아져 하한에 걸린다
    assert clamped[1] == config.WORKSPACE_Y_M[0]  # 이미지 x가 아주 작다(손이 오른쪽) → 로봇 y는 하한
    assert clamped[2] == config.WORKSPACE_Z_M[0]


def test_block_and_green_target_are_inside_workspace():
    """sim/scene.xml의 블록(0.20, 0)과 초록 목표(0.20, 0.12)는 작업 상자 안이어야 손으로 닿을 수 있다."""
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    for xy in (model.body("block").pos[:2], model.geom("target").pos[:2]):
        assert config.WORKSPACE_X_M[0] <= xy[0] <= config.WORKSPACE_X_M[1]
        assert config.WORKSPACE_Y_M[0] <= xy[1] <= config.WORKSPACE_Y_M[1]


# --- HandMapper: 상태가 있는 부분 ---


def mapper(**kw):
    return HandMapper(start_frames=kw.pop("start_frames", 1), **kw)


def test_no_hand_before_start_means_no_target():
    m = mapper()
    r = m.update(hand(detected=False), 0.0)
    assert r.target_m is None and not r.controlling


def test_last_target_is_held_when_hand_disappears():
    m = mapper()
    r1 = m.update(hand(0.4, 0.4), 0.0)
    held = m.update(hand(detected=False), 0.1)
    again = m.update(None, 0.2)
    assert r1.controlling and not held.controlling
    np.testing.assert_array_equal(held.target_m, r1.target_m)
    np.testing.assert_array_equal(again.target_m, r1.target_m)
    assert not held.new_sample


def test_start_requires_consecutive_frames():
    m = HandMapper(start_frames=3)
    first = m.update(hand(), 0.0)
    second = m.update(hand(), 0.03)
    assert first.target_m is None and second.target_m is None
    assert not first.controlling and not second.controlling
    third = m.update(hand(), 0.06)
    assert third.controlling and third.target_m is not None


def test_dropout_resets_the_start_count():
    m = HandMapper(start_frames=3)
    m.update(hand(), 0.0)
    m.update(hand(), 0.03)
    m.update(hand(detected=False), 0.06)  # 잠깐 사라짐
    assert m.update(hand(), 0.09).target_m is None  # 다시 처음부터 센다
    assert m.update(hand(), 0.12).target_m is None
    assert m.update(hand(), 0.15).controlling


def test_after_dropout_old_target_is_held_until_restart_condition():
    m = HandMapper(start_frames=2)
    m.update(hand(0.4, 0.5), 0.0)
    first = m.update(hand(0.4, 0.5), 0.03)
    m.update(hand(detected=False), 0.1)
    moved = m.update(hand(0.7, 0.5), 0.2)  # 다른 곳에서 나타났지만 아직 1프레임
    np.testing.assert_array_equal(moved.target_m, first.target_m)
    assert not moved.controlling


def test_filter_restarts_after_dropout():
    """손이 다른 곳에서 다시 나타나면 필터가 옛 위치를 끌고 오지 않고 새 위치에서 시작한다."""
    m = mapper()
    m.update(hand(0.3, 0.5), 0.0)
    m.update(hand(detected=False), 1.0)
    r = m.update(hand(0.7, 0.5), 2.0)
    np.testing.assert_allclose(r.target_m, hand_to_target(0.7, 0.5, HAND_SCALE_MID))


def test_filter_reduces_jitter_of_still_hand():
    rng = np.random.default_rng(1)
    m = mapper()
    raws, filtered = [], []
    for i in range(300):
        x = 0.5 + rng.normal(0, 0.004)
        y = 0.5 + rng.normal(0, 0.004)
        scale = HAND_SCALE_MID + rng.normal(0, 0.002)  # 높이(z)에도 잡음을 넣어 세 축을 모두 본다
        r = m.update(hand(x, y, scale), i / 30)
        raws.append(r.raw_target_m)
        filtered.append(r.target_m)
    raw_std = np.std(raws[60:], axis=0)
    filt_std = np.std(filtered[60:], axis=0)
    assert np.all(filt_std < 0.7 * raw_std)


def test_filtered_target_stays_inside_workspace():
    m = mapper()
    for i in range(30):
        r = m.update(hand(-3.0 + 0.2 * i, 5.0), i / 30)
        assert config.WORKSPACE_X_M[0] <= r.target_m[0] <= config.WORKSPACE_X_M[1]
        assert config.WORKSPACE_Y_M[0] <= r.target_m[1] <= config.WORKSPACE_Y_M[1]
        assert config.WORKSPACE_Z_M[0] <= r.target_m[2] <= config.WORKSPACE_Z_M[1]
