"""1단계 확인용: sim/scene.xml을 화면 없이 렌더링해 PNG로 저장한다.

실행:
    MUJOCO_GL=egl server/.venv/bin/python sim/render_check.py

저장하는 것은 시뮬레이션 화면뿐이다(웹캠 영상 없음).
"""

import os
from pathlib import Path

# MUJOCO_GL은 mujoco를 import하기 전에 정해져 있어야 한다.
# 이 서버에는 GPU가 없지만 Mesa EGL(CPU)로 렌더링된다. osmesa 라이브러리는 설치돼 있지 않다.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

SCENE_PATH = Path(__file__).resolve().parent / "scene.xml"
OUT_DIR = Path("/tmp")
WIDTH_PX, HEIGHT_PX = 640, 480
SETTLE_TIME_S = 1.0

# 관절 각도(rad): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
# shoulder_pan이 +이면 로봇 오른쪽(-y, top 화면 오른쪽)으로 돈다.
POSES_RAD = [
    [0.6, -1.0, 1.2, 0.3, 0.0, 0.0],  # 오른쪽으로 돌린 자세, 그리퍼 닫힘
    [0.0, 0.5, 0.5, 1.0, 0.0, 1.2],  # 블록 쪽으로 내려 뻗은 자세, 그리퍼 열림
    [-0.6, -0.5, 0.3, 0.5, 1.0, 0.6],  # 왼쪽(목표 쪽)으로 돌리고 손목을 회전한 자세
]
ARM_DOF = 6  # qpos 앞 6개가 팔 관절, 뒤 7개는 블록 freejoint(위치 3 + 쿼터니언 4)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    block = data.body("block")
    mujoco.mj_forward(model, data)
    start_z_m = float(block.xpos[2])

    n_steps = round(SETTLE_TIME_S / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    end_z_m = float(block.xpos[2])
    print(f"block z after {data.time:.2f} s: {end_z_m:.4f} m (start {start_z_m:.4f} m)")

    renderer = mujoco.Renderer(model, HEIGHT_PX, WIDTH_PX)
    try:
        renderer.update_scene(data, camera="top")
        _save(renderer.render(), OUT_DIR / "scene_top.png")

        # 자세 확인은 물리 진행 없이 관절 각도만 넣고 위치만 다시 계산한다.
        for i, pose_rad in enumerate(POSES_RAD, start=1):
            data.qpos[:ARM_DOF] = pose_rad
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="top")
            _save(renderer.render(), OUT_DIR / f"pose_{i}.png")
    finally:
        # 직접 닫지 않으면 종료할 때 EGL 에러가 난다.
        renderer.close()


def _save(pixels, path: Path) -> None:
    Image.fromarray(pixels).save(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
