"""sim/scene.xml 기본 검사(렌더링 없음)."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

SCENE_PATH = Path(__file__).resolve().parents[2] / "sim" / "scene.xml"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BLOCK_HALF_SIZE_M = 0.015


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def test_scene_has_robot_block_camera(model):
    for name in ARM_JOINTS:
        assert model.joint(name) is not None
        assert model.actuator(name) is not None
    assert model.nu == 6
    assert model.camera("top") is not None
    assert model.body("block") is not None


def test_target_has_no_collision(model):
    target = model.geom("target")
    assert target.contype[0] == 0
    assert target.conaffinity[0] == 0


def test_block_is_about_3cm(model):
    np.testing.assert_allclose(model.geom("block").size, [BLOCK_HALF_SIZE_M] * 3)


def test_block_stays_on_floor_after_1s(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    n_steps = round(1.0 / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    block_z_m = data.body("block").xpos[2]
    assert 0.010 < block_z_m < 0.020
