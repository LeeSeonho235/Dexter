"""2-3 확인용: 기본, 위에서, 옆에서 세 각도를 서버와 같은 방법(자유 카메라, 같은 그리기 옵션)으로 그려 PNG로 저장한다.

실행:
    cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.view_check

두 자세로 그린다: home(팔을 세운 자세)와 over(집게를 블록 위로 내린 자세, 팔이 블록을 가리기 쉬운 경우).
저장하는 것은 시뮬레이션 화면뿐이다(웹캠 영상 없음). 파일: /tmp/view_<보기>_<자세>.png
"""

import os

# mujoco를 import하기 전에 정해야 한다.
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from server import config  # noqa: E402
from server.view import make_free_camera, preset_view  # noqa: E402

OUT_DIR = Path("/tmp")
# 관절 각도(rad): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
POSES_RAD = {
    "home": None,  # keyframe 그대로
    "over": [-0.005, -0.123, 0.106, 1.588, 0.0, 0.9],  # 집게가 아래를 향한 채 블록 위(0.23, 0, 0.08)
}


def count_colors(pixels: np.ndarray) -> tuple[int, int]:
    """대략 빨강(블록), 초록(목표) 픽셀 수. 조명에 민감한 거친 값이라 눈으로 보는 확인을 대신하지 못한다."""
    p = pixels.astype(int)
    # 위에서 내려다보면 조명이 밝아 색이 옅어지므로(예: 연한 빨강) 절대 밝기보다 "다른 채널과의 차이"로 센다.
    red = int(((p[..., 0] > 150) & (p[..., 0] - np.maximum(p[..., 1], p[..., 2]) > 60)).sum())
    green = int(((p[..., 1] > 150) & (p[..., 1] - np.maximum(p[..., 0], p[..., 2]) > 40)).sum())
    return red, green


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, config.RENDER_HEIGHT_PX, config.RENDER_WIDTH_PX)
    try:
        for pose_name, joints in POSES_RAD.items():
            mujoco.mj_resetDataKeyframe(model, data, model.key(config.START_KEYFRAME).id)
            if joints is not None:
                data.qpos[:6] = joints
            mujoco.mj_forward(model, data)
            for view_name in ("default", "top", "side"):
                view = preset_view(view_name)
                renderer.update_scene(data, camera=make_free_camera(view))
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = config.RENDER_SHADOW
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = config.RENDER_REFLECTION
                pixels = renderer.render()
                path = OUT_DIR / f"view_{view_name}_{pose_name}.png"
                Image.fromarray(pixels).save(path)
                red, green = count_colors(pixels)
                print(f"saved {path}  ({view.as_dict()}, red~{red}px, green~{green}px)")
    finally:
        # 직접 닫지 않으면 종료할 때 EGL 에러가 난다.
        renderer.close()


if __name__ == "__main__":
    main()
