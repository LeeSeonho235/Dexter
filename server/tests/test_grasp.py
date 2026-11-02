"""손 없이 하는 집기 시험(server/grasp_check.py): 블록이 실제로 따라 올라오는지, 열림 폭이 블록보다 넓은지."""

import pytest

from server import config
from server.grasp_check import LIFTED_MIN_M, Rig, jaw_gap_m, run

BLOCK_SIDE_M = 0.03


@pytest.fixture(scope="module")
def rig():
    return Rig(config.GRIPPER_OPEN_RAD, config.GRIPPER_CLOSED_RAD)


def test_open_gap_is_wider_than_block_with_margin(rig):
    gap_m = jaw_gap_m(rig, config.GRIPPER_OPEN_RAD)
    assert gap_m > 2 * BLOCK_SIDE_M  # 블록 폭의 2배 이상이면 위치가 1cm 어긋나도 들어간다


def test_closed_gap_is_narrower_than_block(rig):
    assert jaw_gap_m(rig, config.GRIPPER_CLOSED_RAD) < BLOCK_SIDE_M


def test_block_follows_gripper_up_by_5cm_and_stays_for_2s():
    result = run(config.GRIPPER_OPEN_RAD, config.GRIPPER_CLOSED_RAD, config.WORKSPACE_Z_M[0], verbose=False)
    assert result.success, result.log
    assert result.lifted_m >= LIFTED_MIN_M
    hold = [z for label, _, z, _ in result.log if label.startswith("hold")]
    assert len(hold) == 4
    assert max(hold) - min(hold) < 0.002  # 유지하는 2초 동안 흘러내리지 않는다
