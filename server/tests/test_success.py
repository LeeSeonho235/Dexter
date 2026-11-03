"""성공 판정: 경계값, 0.5초 유지, 설정과 scene.xml의 목표가 같은지."""

import mujoco
import numpy as np
import pytest

from server import config
from server.success import SuccessHold, is_success, success_from_qpos, success_over_time

# 경계가 정확히 0.03이 되도록 중심 0, 반변 0.03, 여유 0으로 고정해서 본다(부동소수점 덧셈 오차를 피한다).
EDGE = dict(target_center_m=(0.0, 0.0), target_half_m=0.03, xy_margin_m=0.0, z_max_m=0.02, gripper_open_min_rad=0.5)


def test_inside_target_on_floor_with_open_gripper_is_success():
    assert is_success((0.2, 0.12, 0.015), 0.9)


def test_xy_boundary_is_inclusive_and_outside_fails():
    assert is_success((0.03, 0.0, 0.015), 0.9, **EDGE)
    assert is_success((-0.03, 0.03, 0.015), 0.9, **EDGE)
    assert not is_success((0.0300001, 0.0, 0.015), 0.9, **EDGE)
    assert not is_success((0.0, -0.0300001, 0.015), 0.9, **EDGE)


def test_margin_extends_the_xy_limit():
    limit = config.TARGET_HALF_M + config.SUCCESS_XY_MARGIN_M
    cx, cy = config.TARGET_CENTER_M
    assert is_success((cx + limit - 1e-6, cy, 0.015), 0.9)
    assert not is_success((cx + limit + 1e-6, cy, 0.015), 0.9)


def test_z_boundary_is_exclusive():
    assert is_success((0.0, 0.0, 0.0199), 0.9, **EDGE)
    assert not is_success((0.0, 0.0, 0.02), 0.9, **EDGE)  # 정확히 기준이면 실패
    assert not is_success((0.0, 0.0, 0.05), 0.9, **EDGE)  # 들려 있다


def test_gripper_boundary_is_inclusive():
    assert is_success((0.0, 0.0, 0.015), 0.5, **EDGE)
    assert not is_success((0.0, 0.0, 0.015), 0.4999, **EDGE)
    assert not is_success((0.0, 0.0, 0.015), -0.1, **EDGE)  # 닫힘


def test_hold_needs_half_a_second_and_resets_when_broken():
    assert config.SUCCESS_HOLD_S == 0.5
    hold = SuccessHold()
    assert hold.update(10.0, True) is False
    assert hold.update(10.2, True) is False
    assert hold.update(10.4999, True) is False
    assert hold.update(10.5, True) is True  # 정확히 0.5 초는 달성
    assert hold.update(10.6, True) is True
    assert hold.update(10.7, False) is False  # 끊기면 사라진다
    assert hold.update(10.8, True) is False  # 다시 처음부터
    assert hold.update(11.3, True) is True


def test_success_over_time_matches_hold():
    times = np.arange(0.0, 2.0, 1 / 30)
    qpos = np.zeros((len(times), 13))
    qpos[:, 5] = 0.9
    qpos[:, 6:9] = (0.2, 0.12, 0.015)
    qpos[times < 1.0, 6:9] = (0.2, 0.0, 0.015)  # 앞 1초는 목표 밖
    ever, first_s, at_end = success_over_time(times, qpos)
    assert ever is True and at_end is True
    assert first_s == pytest.approx(1.0 + 0.5, abs=1 / 30)
    qpos[:, 6:9] = (0.2, 0.0, 0.015)
    assert success_over_time(times, qpos) == (False, None, False)


def test_success_from_qpos_reads_block_and_gripper():
    qpos = np.zeros(13)
    qpos[5] = 0.9
    qpos[6:9] = (0.2, 0.12, 0.015)
    assert success_from_qpos(qpos)
    qpos[5] = 0.0
    assert not success_from_qpos(qpos)


def test_config_target_matches_scene():
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    geom = model.geom("target")
    assert tuple(geom.pos[:2]) == pytest.approx(config.TARGET_CENTER_M)
    assert geom.size[0] == pytest.approx(config.TARGET_HALF_M)
    assert geom.size[1] == pytest.approx(config.TARGET_HALF_M)


def test_block_half_size_matches_scene_and_rests_below_z_limit():
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    assert model.geom("block").size[2] == pytest.approx(config.BLOCK_HALF_M)
    assert config.BLOCK_HALF_M < config.SUCCESS_BLOCK_Z_MAX_M
