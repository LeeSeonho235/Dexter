"""블록 시작 영역(config.BLOCK_SPAWN_*)에서 로봇이 닿는지 확인한다(손 없이, 화면 없이 시뮬레이션만).

실행:
    cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.spawn_check [--margin 0.02] [--step 0.01]

1. IK 격자: 블록 영역을 margin(m)만큼 사방으로 넓힌 격자의 모든 점에서, 집게 끝이 (a) 블록 높이(z=0.015)와 (b) 접근 높이(z=0.08)에
   닿는지 IK로 푼다. 실패 개수, 최대 오차(mm), 집게가 수직 아래에서 기운 최대 각도(도)를 센다. 목표(초록) 위 놓기 자세(z=0.03)도 푼다.
2. 자동 집기: 영역의 네 모서리에 블록을 놓고 server/grasp_check.py의 Rig로 집어서 5 cm 들어 올린다(들렸는지 숫자로).
저장하는 파일은 없다.
"""

import os

# mujoco를 import하기 전에 정해야 한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from server import config  # noqa: E402
from server.grasp_check import (  # noqa: E402
    CLOSE_S,
    HOVER_Z_M,
    LIFT_M,
    LIFT_S,
    LIFTED_MIN_M,
    MOVE_TIMEOUT_S,
    MOVE_TOLERANCE_M,
    Rig,
)
from server.ik import ArmIK  # noqa: E402

GRASP_Z_M = config.BLOCK_HALF_M  # 블록 중심 높이
APPROACH_Z_M = 0.08
PLACE_Z_M = 0.03


def frange(lo: float, hi: float, step: float) -> list[float]:
    n = max(1, int(round((hi - lo) / step)))
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def ik_grid(margin_m: float, step_m: float) -> None:
    ik = ArmIK()
    xs = frange(config.BLOCK_SPAWN_X_M[0] - margin_m, config.BLOCK_SPAWN_X_M[1] + margin_m, step_m)
    ys = frange(config.BLOCK_SPAWN_Y_M[0] - margin_m, config.BLOCK_SPAWN_Y_M[1] + margin_m, step_m)
    print(f"IK 격자: x {xs[0]:.3f}~{xs[-1]:.3f}, y {ys[0]:.3f}~{ys[-1]:.3f} m (영역 + 사방 {margin_m * 100:.0f} cm), {len(xs) * len(ys)}점")
    for label, z in (("블록 높이 z=0.015", GRASP_Z_M), ("접근 높이 z=0.08", APPROACH_Z_M)):
        fails, worst_mm, worst_tilt = [], 0.0, 0.0
        for x in xs:
            for y in ys:
                ik.reset()
                result = ik.solve(np.array([x, y, z]))
                if not result.ok:
                    fails.append((round(x, 3), round(y, 3), result.reason))
                    continue
                worst_mm = max(worst_mm, result.error_m * 1000)
                worst_tilt = max(worst_tilt, float(np.degrees(result.tilt_rad)))
        print(f"  {label}: 실패 {len(fails)}점, 최대 오차 {worst_mm:.2f} mm, 집게 최대 기울기 {worst_tilt:.1f}도")
        for f in fails[:5]:
            print(f"    실패: {f}")
    cx, cy = config.TARGET_CENTER_M
    ik.reset()
    place = ik.solve(np.array([cx, cy, PLACE_Z_M]))
    print(f"  목표 위 놓기 자세 ({cx:.2f}, {cy:.2f}, {PLACE_Z_M}): {'OK' if place.ok else 'FAIL ' + place.reason}, 오차 {place.error_m * 1000:.2f} mm")


def grasp_at(x_m: float, y_m: float) -> tuple[float, bool]:
    """(x, y)에 블록을 놓고 자동으로 집어 5 cm 들어 올린다. (들린 높이 m, 성공 여부)."""
    rig = Rig(config.GRIPPER_OPEN_RAD, config.GRIPPER_CLOSED_RAD)
    adr = rig.model.joint("block_free").qposadr[0]
    rig.data.qpos[adr : adr + 3] = [x_m, y_m, GRASP_Z_M]
    rig.data.qvel[:] = 0.0
    mujoco.mj_forward(rig.model, rig.data)
    dt = 1.0 / config.CONTROL_HZ

    def move_to(target: np.ndarray) -> None:
        rig.set_target(target)
        for _ in range(int(MOVE_TIMEOUT_S / dt)):
            rig.tick()
            if np.allclose(rig.ctrl[:5], rig.goal_arm, atol=1e-4) and np.linalg.norm(rig.tip_m - target) < MOVE_TOLERANCE_M:
                break
        for _ in range(int(0.3 / dt)):
            rig.tick()

    start_z = float(rig.block_m[2])
    move_to(np.array([x_m, y_m, HOVER_Z_M]))
    move_to(np.array([x_m, y_m, config.WORKSPACE_Z_M[0]]))
    rig.gripper_goal = config.GRIPPER_CLOSED_RAD
    for _ in range(int(CLOSE_S / dt)):
        rig.tick()
    tip = rig.tip_m
    steps = int(LIFT_S / dt)
    for k in range(1, steps + 1):
        rig.set_target(tip + np.array([0.0, 0.0, LIFT_M * k / steps]))
        rig.tick()
    for _ in range(int(1.0 / dt)):
        rig.tick()
    lifted = float(rig.block_m[2] - start_z)
    holding_rad = float(rig.data.qpos[5])
    print(f"    집은 채 들었을 때 집게 각도 {holding_rad:+.3f} rad")
    return lifted, lifted >= LIFTED_MIN_M and bool(np.linalg.norm(rig.block_m - rig.tip_m) < 0.03)


def corner_grasps() -> None:
    corners = [(x, y) for x in config.BLOCK_SPAWN_X_M for y in config.BLOCK_SPAWN_Y_M]
    corners.append((sum(config.BLOCK_SPAWN_X_M) / 2, sum(config.BLOCK_SPAWN_Y_M) / 2))
    print("자동 집기(네 모서리 + 중앙):")
    for x, y in corners:
        try:
            lifted, ok = grasp_at(x, y)
            print(f"  ({x:.3f}, {y:+.3f}): 들린 높이 {lifted * 1000:5.1f} mm (기준 {LIFTED_MIN_M * 1000:.0f} mm 이상) -> {'성공' if ok else '실패'}")
        except RuntimeError as err:
            print(f"  ({x:.3f}, {y:+.3f}): 실패 {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--margin", type=float, default=0.02, help="영역을 사방으로 넓힐 여유(m)")
    parser.add_argument("--step", type=float, default=0.01, help="격자 간격(m)")
    parser.add_argument("--skip-grasp", action="store_true", help="자동 집기를 건너뛴다")
    args = parser.parse_args()
    print(f"블록 영역: x {config.BLOCK_SPAWN_X_M}, y {config.BLOCK_SPAWN_Y_M} m; 작업 상자 x {config.WORKSPACE_X_M}, y {config.WORKSPACE_Y_M}")
    ik_grid(args.margin, args.step)
    if not args.skip_grasp:
        corner_grasps()


if __name__ == "__main__":
    main()
