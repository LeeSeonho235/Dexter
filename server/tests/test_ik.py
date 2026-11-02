"""역기구학: 작업 상자 안 여러 점에서 목표와 실제 집게 위치 차이 1cm 이하, 변화량 제한, 실패 처리."""

import itertools

import mujoco
import numpy as np
import pytest

from server import config
from server.ik import ARM_JOINTS, ArmIK, limit_joint_step

MAX_ERROR_M = 0.01  # 요구 조건: 목표와 실제 집게 위치 차이 1cm 이하
MAX_TILT_RAD = np.radians(30)  # 집게가 수직 아래에서 이만큼 안으로 기울어야 "아래를 향함"으로 본다


@pytest.fixture(scope="module")
def ik():
    return ArmIK()


def workspace_points():
    xs = np.linspace(*config.WORKSPACE_X_M, 4)
    ys = np.linspace(*config.WORKSPACE_Y_M, 5)
    zs = np.linspace(*config.WORKSPACE_Z_M, 4)
    return [np.array(p) for p in itertools.product(xs, ys, zs)]  # 모서리 8개 포함


def random_points(n=60, seed=0):
    rng = np.random.default_rng(seed)
    lo = [config.WORKSPACE_X_M[0], config.WORKSPACE_Y_M[0], config.WORKSPACE_Z_M[0]]
    hi = [config.WORKSPACE_X_M[1], config.WORKSPACE_Y_M[1], config.WORKSPACE_Z_M[1]]
    return [rng.uniform(lo, hi) for _ in range(n)]


def forward_tip(model, q_arm_rad):
    """IK 객체와 별개로, 풀어낸 관절 각도를 MuJoCo 순방향 계산에 넣어 집게 끝 위치를 구한다(m)."""
    data = mujoco.MjData(model)
    for name, q in zip(ARM_JOINTS, q_arm_rad):
        data.qpos[model.joint(name).qposadr[0]] = q
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[model.site(config.IK_SITE).id].copy()


@pytest.mark.parametrize("name,points", [("grid+corners", workspace_points()), ("random", random_points())])
def test_every_workspace_point_reached_within_1cm(ik, name, points):
    for target in points:
        ik.reset()
        result = ik.solve(target)
        assert result.ok, f"{name}: {np.round(target, 3)} failed: {result.reason}"
        actual = forward_tip(ik.model, result.q_rad)
        assert np.linalg.norm(actual - target) <= MAX_ERROR_M, np.round(target, 3)
        assert result.error_m <= MAX_ERROR_M


def test_points_reached_in_sequence_like_a_moving_hand(ik):
    """매번 reset하지 않고 직전 해에서 이어서 풀어도(손이 움직이는 상황) 1cm 이하."""
    ik.reset()
    for target in random_points(80, seed=3):
        result = ik.solve(target)
        assert result.ok, result.reason
        assert np.linalg.norm(forward_tip(ik.model, result.q_rad) - target) <= MAX_ERROR_M


def test_gripper_points_roughly_down_everywhere(ik):
    for target in workspace_points():
        ik.reset()
        result = ik.solve(target)
        assert result.ok
        assert result.tilt_rad <= MAX_TILT_RAD, (np.round(target, 3), np.degrees(result.tilt_rad))


def test_joint_limits_and_wrist_roll_respected(ik):
    for target in random_points(30, seed=5):
        ik.reset()
        result = ik.solve(target)
        assert result.ok
        for name, q in zip(ARM_JOINTS, result.q_rad):
            lo, hi = ik.model.joint(name).range
            assert lo - 1e-6 <= q <= hi + 1e-6, name
        assert result.q_rad[ARM_JOINTS.index("wrist_roll")] == pytest.approx(config.IK_SEED_ARM_RAD[4], abs=1e-9)


def test_unreachable_target_fails_and_keeps_state(ik):
    ik.reset()
    before_q = ik.arm_q_rad
    before_tip = ik.tip_position_m()
    result = ik.solve(np.array([2.0, 0.0, 0.5]))
    assert not result.ok
    assert "converge" in result.reason
    np.testing.assert_allclose(ik.arm_q_rad, before_q)
    np.testing.assert_allclose(ik.tip_position_m(), before_tip)


@pytest.mark.parametrize("bad", [[np.nan, 0.0, 0.1], [0.2, np.inf, 0.1], [0.2, 0.0, -np.inf]])
def test_nan_target_fails_without_exception(ik, bad):
    ik.reset()
    before_q = ik.arm_q_rad
    result = ik.solve(np.array(bad))
    assert not result.ok
    np.testing.assert_allclose(result.q_rad, before_q)
    assert np.all(np.isfinite(ik.arm_q_rad))


def test_solver_exception_is_reported_not_raised(ik, monkeypatch):
    import mink

    def boom(*args, **kwargs):
        raise mink.NoSolutionFound("daqp")

    ik.reset()
    before_q = ik.arm_q_rad
    monkeypatch.setattr(mink, "solve_ik", boom)
    result = ik.solve(np.array([0.23, 0.0, 0.08]))
    assert not result.ok
    assert "NoSolutionFound" in result.reason
    np.testing.assert_allclose(ik.arm_q_rad, before_q)


def test_solver_nan_is_reported(ik, monkeypatch):
    import mink

    monkeypatch.setattr(mink, "solve_ik", lambda *a, **k: np.full(ik.model.nv, np.nan))
    ik.reset()
    result = ik.solve(np.array([0.25, 0.05, 0.06]))
    assert not result.ok
    assert "NaN" in result.reason


# --- 변화량 제한 ---


def test_limit_step_caps_each_joint():
    cur = np.zeros(5)
    goal = np.array([1.0, -1.0, 0.05, 0.0, -0.3])
    out = limit_joint_step(cur, goal, 0.1)
    np.testing.assert_allclose(out, [0.1, -0.1, 0.05, 0.0, -0.1])


def test_limit_step_reaches_goal_when_close_and_is_idempotent_there():
    cur = np.array([0.5, 0.5])
    goal = np.array([0.55, 0.45])
    out = limit_joint_step(cur, goal, 0.08)
    np.testing.assert_allclose(out, goal)
    np.testing.assert_allclose(limit_joint_step(out, goal, 0.08), goal)


def test_limit_step_converges_in_expected_ticks():
    cur = np.zeros(3)
    goal = np.array([0.8, -0.4, 0.0])
    ticks = 0
    while not np.allclose(cur, goal):
        nxt = limit_joint_step(cur, goal, 0.08)
        assert np.max(np.abs(nxt - cur)) <= 0.08 + 1e-12
        cur = nxt
        ticks += 1
        assert ticks < 100
    assert ticks == 10


def test_first_solve_time_is_small(ik):
    ik.reset()
    times = []
    for target in random_points(20, seed=9):
        times.append(ik.solve(target).elapsed_ms)
    assert max(times) < 50  # 30Hz(33ms) 틱 안에 충분히 들어온다
