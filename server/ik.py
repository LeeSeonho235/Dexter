"""역기구학(IK): 집게 끝 목표 위치(m) → 팔 관절 각도(rad). mink를 쓴다.

- 시뮬레이션 스레드의 MjData와 따로, 이 클래스가 만든 모델 복사본으로만 계산한다(스레드 안전).
- SO-101은 팔 관절이 5개(pan, lift, elbow, wrist_flex, wrist_roll)라 위치 3자유도를 우선하고,
  집게가 아래를 향하게 하는 것은 비용을 작게 둔 약한 목표로 넣는다.
- wrist_roll(손목 회전)과 집게 관절, 블록의 자유 관절은 얼려서 IK가 건드리지 않는다.
- 계산이 실패하거나(예외) NaN이 나오거나 목표와 너무 멀면 ok=False로 돌려주고 내부 상태는 이전 값으로 되돌린다.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np

from server import config

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# 집게 끝 site에서 집게가 뻗은 방향 축(site 좌표계)과, 그 축을 향하게 하고 싶은 월드 방향(아래)
FINGER_AXIS_IN_SITE = (1.0, 0.0, 0.0)
DOWN_IN_WORLD = (0.0, 0.0, -1.0)


@dataclass
class IKResult:
    q_rad: np.ndarray  # 팔 관절 5개(ARM_JOINTS 순서). 실패면 직전 값
    ok: bool
    error_m: float  # 목표와 집게 끝의 거리(m). 계산 못 했으면 nan
    iterations: int
    elapsed_ms: float
    tilt_rad: float  # 집게 방향과 수직 아래 방향 사이 각도(rad). 0이면 정확히 아래
    reason: str = ""


def limit_joint_step(current: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    """current에서 goal 쪽으로 가되 원소마다 max_step(관절 단위: rad)보다 많이 바꾸지 않는다."""
    current = np.asarray(current, dtype=float)
    goal = np.asarray(goal, dtype=float)
    return current + np.clip(goal - current, -max_step, max_step)


class ArmIK:
    def __init__(self, scene_path: Path = config.SCENE_PATH, keyframe: str = config.START_KEYFRAME) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._keyframe = keyframe
        self.configuration = mink.Configuration(self.model)
        self._site_id = self.model.site(config.IK_SITE).id
        self._arm_qpos = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])

        self._frame_task = mink.FrameTask(
            config.IK_SITE, "site", position_cost=config.IK_POSITION_COST, orientation_cost=0.0
        )
        self._down_task = mink.AxisAlignTask(
            config.IK_SITE, "site", axis=FINGER_AXIS_IN_SITE, cost=config.IK_DOWN_COST
        )
        self._down_task.set_target(np.array(DOWN_IN_WORLD))
        # 얼릴 자유도: wrist_roll, gripper, 그리고 위 관절 6개 뒤의 나머지 전부(블록 freejoint)
        frozen_joints = ["wrist_roll", "gripper"]
        dof_indices = [int(self.model.joint(n).dofadr[0]) for n in frozen_joints]
        dof_indices += list(range(6, self.model.nv))
        self._freeze = mink.DofFreezingTask(self.model, dof_indices=dof_indices)
        self._limits = [mink.ConfigurationLimit(self.model)]
        # home에서 시작하면 풀이 막히므로, 집게가 아래를 향한 자세(config.IK_SEED_ARM_RAD)를 시작 자세(seed)로 쓴다.
        self.configuration.update_from_keyframe(self._keyframe)
        self._seed_q = self.configuration.q.copy()
        self._seed_q[self._arm_qpos] = config.IK_SEED_ARM_RAD
        self.reset()

    def reset(self) -> None:
        """IK 내부 상태를 시작 자세(seed)로 되돌린다."""
        self.configuration.update(self._seed_q)

    def set_arm(self, q_rad: np.ndarray) -> None:
        q = self.configuration.q.copy()
        q[self._arm_qpos] = q_rad
        self.configuration.update(q)

    @property
    def arm_q_rad(self) -> np.ndarray:
        return self.configuration.q[self._arm_qpos].copy()

    def tip_position_m(self) -> np.ndarray:
        return self.configuration.data.site_xpos[self._site_id].copy()

    def tilt_rad(self) -> float:
        rot = self.configuration.data.site_xmat[self._site_id].reshape(3, 3)
        finger = rot @ np.array(FINGER_AXIS_IN_SITE)
        return float(np.arccos(np.clip(-finger[2], -1.0, 1.0)))

    def _iterate(self, target_m: np.ndarray) -> tuple[bool, int, float, str]:
        """지금 자세에서 시작해 목표 쪽으로 반복해서 푼다. (성공 여부, 반복 수, 오차 m, 실패 이유)를 돌려준다.

        어떤 예외가 나도 밖으로 던지지 않는다. 실패해도 configuration은 호출한 쪽이 되돌린다.
        """
        self._frame_task.set_target(mink.SE3.from_translation(target_m))
        iterations = 0
        error_m = float("inf")
        try:
            for iterations in range(1, config.IK_MAX_ITERATIONS + 1):
                velocity = mink.solve_ik(
                    self.configuration,
                    [self._frame_task, self._down_task],
                    config.IK_DT_S,
                    config.IK_SOLVER,
                    limits=self._limits,
                    constraints=[self._freeze],
                )
                if not np.all(np.isfinite(velocity)):
                    return False, iterations, float("nan"), "solver returned NaN"
                self.configuration.integrate_inplace(velocity, config.IK_DT_S)
                error_m = float(np.linalg.norm(self.tip_position_m() - target_m))
                if error_m < config.IK_TOLERANCE_M:
                    break
        except Exception as err:  # noqa: BLE001 - 어떤 실패든 서버를 죽이지 않고 ok=False로 알린다
            return False, iterations, float("nan"), f"{type(err).__name__}: {err}"

        if not np.all(np.isfinite(self.configuration.q)) or not np.isfinite(error_m):
            return False, iterations, float("nan"), "configuration is not finite"
        if error_m > config.IK_FAIL_ERROR_M:
            return False, iterations, error_m, f"did not converge (error {error_m * 1000:.1f} mm)"
        return True, iterations, error_m, ""

    def solve(self, target_m: np.ndarray) -> IKResult:
        """목표 위치(m)로 IK를 푼다.

        직전 해에서 시작하므로 가까운 목표는 몇 번 반복으로 끝난다. 못 풀면 시작 자세(seed)에서 한 번 더 푼다.
        그래도 실패하면 ok=False로 돌려주고 내부 상태를 풀기 전 값으로 되돌린다(호출한 쪽은 ctrl을 바꾸지 않는다).
        """
        start = time.perf_counter()
        previous_q = self.configuration.q.copy()
        target_m = np.asarray(target_m, dtype=float)

        def elapsed_ms() -> float:
            return (time.perf_counter() - start) * 1000.0

        if not np.all(np.isfinite(target_m)):
            return IKResult(
                previous_q[self._arm_qpos].copy(), False, float("nan"), 0, elapsed_ms(), self.tilt_rad(),
                "target is not finite",
            )

        ok, iterations, error_m, reason = self._iterate(target_m)
        if not ok:
            self.configuration.update(self._seed_q)
            ok, more, error_m, reason = self._iterate(target_m)
            iterations += more
        if not ok:
            self.configuration.update(previous_q)
            return IKResult(
                previous_q[self._arm_qpos].copy(), False, error_m, iterations, elapsed_ms(), self.tilt_rad(), reason
            )
        return IKResult(self.arm_q_rad, True, error_m, iterations, elapsed_ms(), self.tilt_rad())
