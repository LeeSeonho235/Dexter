"""집기 시험 + 집게 폭 측정(손 없이, 화면 없이 시뮬레이션만).

실행:
    cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.grasp_check

순서: IK로 블록 위로 이동 → 내려가기 → 집게 닫기 → 5cm 들어올리기 → 2초 유지.
실제 서버와 같은 경로를 쓴다: IK(server/ik.py) → 한 틱 최대 변화량 제한 → ctrl → mj_step.
블록 높이를 숫자로 찍는다. 저장하는 파일은 없다(웹캠도 쓰지 않는다).
"""

import os

# mujoco를 import하기 전에 정해야 한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from server import config  # noqa: E402
from server.ik import ArmIK, limit_joint_step  # noqa: E402

ARM_DOF = 5
LIFT_M = 0.05  # 들어올리는 높이(m)
LIFT_S = 1.0  # 들어올리는 데 쓰는 시간(s)
HOLD_S = 2.0  # 들어올린 뒤 유지하는 시간(s)
CLOSE_S = 1.0  # 집게를 닫고 기다리는 시간(s)
HOVER_Z_M = 0.10  # 블록 위에서 멈추는 높이(m)
MOVE_TOLERANCE_M = 0.004
MOVE_TIMEOUT_S = 8.0
LIFTED_MIN_M = 0.04  # 이만큼 이상 따라 올라오면 성공으로 본다


@dataclass
class GraspResult:
    block_start_m: np.ndarray
    log: list[tuple[str, float, float, float]] = field(default_factory=list)  # (단계, 시간 s, 블록 z m, 블록-집게끝 거리 m)
    contacts_during_descent: int = 0
    lifted_m: float = 0.0
    success: bool = False


class Rig:
    """실제 서버와 같은 순서(IK → 변화량 제한 → ctrl → 물리)로 시뮬레이션을 돌리는 도구."""

    def __init__(self, open_rad: float, closed_rad: float) -> None:
        self.ik = ArmIK()
        self.model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(config.START_KEYFRAME).id)
        mujoco.mj_forward(self.model, self.data)
        self.ctrl = self.data.ctrl[:6].copy()
        self.open_rad, self.closed_rad = open_rad, closed_rad
        self.tip_id = self.model.site(config.IK_SITE).id
        self.block_id = self.model.body("block").id
        self.block_geom = self.model.geom("block").id
        self.steps_per_tick = max(1, round(1.0 / config.CONTROL_HZ / self.model.opt.timestep))
        self.gripper_goal = open_rad
        self.goal_arm = self.ctrl[:ARM_DOF].copy()

    @property
    def tip_m(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_id].copy()

    @property
    def block_m(self) -> np.ndarray:
        return self.data.xpos[self.block_id].copy()

    def set_target(self, target_m: np.ndarray) -> None:
        result = self.ik.solve(target_m)
        if not result.ok:
            raise RuntimeError(f"IK failed for {np.round(target_m, 3)}: {result.reason}")
        self.goal_arm = result.q_rad

    def tick(self) -> None:
        self.ctrl[:ARM_DOF] = limit_joint_step(self.ctrl[:ARM_DOF], self.goal_arm, config.MAX_JOINT_STEP_RAD)
        self.ctrl[ARM_DOF] = limit_joint_step(self.ctrl[ARM_DOF], self.gripper_goal, config.MAX_GRIPPER_STEP_RAD)
        self.data.ctrl[:6] = self.ctrl
        for _ in range(self.steps_per_tick):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)

    def block_contacts_with_gripper(self) -> int:
        """블록과 로봇(집게 포함) 사이 접촉 개수. 바닥과의 접촉은 센 제외."""
        floor = self.model.geom("floor").id
        count = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.block_geom in pair and floor not in pair:
                count += 1
        return count


def jaw_gap_m(rig: Rig, gripper_rad: float) -> float:
    """집게 각도(rad)에서 두 집게 끝 사이의 가장 가까운 표면 거리(m).

    집게 끝에는 고정 집게와 움직이는 집게에 작은 구가 3개씩 있다(scene의 so101.xml). 그 6개 구 사이 거리 중 최소값이다.
    블록이 닿는 곳은 집게 끝 쪽이라 블록 폭과 비교할 수 있다. 기준점(gripperframe)에서 광선을 쏘는 방법은
    기준점이 집게 사이가 아니라 끝 쪽이라 바닥에 맞아서 못 썼다.
    """
    m, d = rig.model, rig.data
    mujoco.mj_resetDataKeyframe(m, d, m.key(config.START_KEYFRAME).id)
    d.qpos[5] = gripper_rad
    mujoco.mj_forward(m, d)
    fixed = [m.geom(f"fixed_jaw_sph_tip{i}").id for i in (1, 2, 3)]
    moving = [m.geom(f"moving_jaw_sph_tip{i}").id for i in (1, 2, 3)]
    fromto = np.zeros(6)
    return float(min(mujoco.mj_geomDistance(m, d, a, b, 1.0, fromto) for a in fixed for b in moving))


def run(open_rad: float, closed_rad: float, grasp_z_m: float, verbose: bool = True) -> GraspResult:
    rig = Rig(open_rad, closed_rad)
    result = GraspResult(block_start_m=rig.block_m)
    t = 0.0
    dt = 1.0 / config.CONTROL_HZ

    def record(label: str) -> None:
        tip = rig.tip_m
        block = rig.block_m
        result.log.append((label, rig.data.time, float(block[2]), float(np.linalg.norm(block - tip))))
        if verbose:
            print(
                f"{label:<22} t={rig.data.time:5.2f}s  block=({block[0]:.3f},{block[1]:.3f},{block[2]:.4f}) m  "
                f"tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) m  |block-tip|={np.linalg.norm(block - tip) * 1000:5.1f} mm  "
                f"gripper={rig.data.qpos[5]:+.3f} rad"
            )

    def move_to(target: np.ndarray, label: str) -> None:
        rig.set_target(target)
        for _ in range(int(MOVE_TIMEOUT_S / dt)):
            rig.tick()
            converged = np.allclose(rig.ctrl[:ARM_DOF], rig.goal_arm, atol=1e-4)
            if converged and np.linalg.norm(rig.tip_m - target) < MOVE_TOLERANCE_M:
                break
            result.contacts_during_descent += rig.block_contacts_with_gripper() if label == "descend" else 0
        for _ in range(int(0.3 / dt)):  # 흔들림이 가라앉을 시간
            rig.tick()
        record(label)

    block = result.block_start_m
    record("start")
    move_to(np.array([block[0], block[1], HOVER_Z_M]), "above block (open)")
    move_to(np.array([block[0], block[1], grasp_z_m]), "descend")
    before_close = rig.block_m[2]

    rig.gripper_goal = closed_rad
    for _ in range(int(CLOSE_S / dt)):
        rig.tick()
    record("closed")

    grasp_tip = rig.tip_m
    steps = int(LIFT_S / dt)
    for k in range(1, steps + 1):
        rig.set_target(grasp_tip + np.array([0.0, 0.0, LIFT_M * k / steps]))
        rig.tick()
    record("lifted 5 cm")
    for k in range(int(HOLD_S / dt)):
        rig.tick()
        if k % int(0.5 / dt) == int(0.5 / dt) - 1:
            record(f"hold {(k + 1) * dt:.1f}s")

    final_z = rig.block_m[2]
    result.lifted_m = float(final_z - block[2])
    result.success = result.lifted_m >= LIFTED_MIN_M and np.linalg.norm(rig.block_m - rig.tip_m) < 0.03
    if verbose:
        print(f"block z: start {block[2]:.4f} m, before closing {before_close:.4f} m, after hold {final_z:.4f} m")
        print(f"lifted {result.lifted_m * 1000:.1f} mm (need >= {LIFTED_MIN_M * 1000:.0f} mm) -> "
              f"{'SUCCESS' if result.success else 'FAIL'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--open", type=float, default=config.GRIPPER_OPEN_RAD, help="집게 열림 각도(rad)")
    parser.add_argument("--closed", type=float, default=config.GRIPPER_CLOSED_RAD, help="집게 닫힘 각도(rad)")
    parser.add_argument("--grasp-z", type=float, default=config.WORKSPACE_Z_M[0], help="집을 때 집게 끝 높이(m)")
    args = parser.parse_args()

    rig = Rig(args.open, args.closed)
    print("집게 끝 사이 거리(가장 가까운 끝 구 표면 사이):")
    for label, rad in (("열림", args.open), ("닫힘", args.closed)):
        print(f"  {label} {rad:+.3f} rad: {jaw_gap_m(rig, rad) * 1000:.1f} mm")
    print("블록 한 변: 30.0 mm")
    print()
    run(args.open, args.closed, args.grasp_z)


if __name__ == "__main__":
    main()
