"""학습·재생 카메라(RECORD_CAMERA_*, 240x320) 확인: 블록과 목표가 보이는지, 블록이 몇 픽셀인지, 집게가 블록을 가리는지.

실행:
    cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.record_camera_check [--out /tmp/record_camera_check]

블록이 시드 영역의 네 모서리에 있을 때 세 자세(① 팔 올림(home) ② 집게가 블록 위 6 cm ③ 집게가 블록 높이까지 내려옴)를 그린다.
물체 번호 영상(segmentation)으로 블록·목표의 보이는 픽셀 수와 가로·세로를 세고, 집게가 있을 때 블록이 얼마나 남는지(①에 대한 비율)를 낸다.
저장하는 것은 시뮬레이션 화면(PNG)뿐이다. 웹캠 영상은 쓰지 않는다.
"""

import os

# mujoco를 import하기 전에 정해야 한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from server import config  # noqa: E402
from server.ik import ARM_JOINTS, ArmIK  # noqa: E402
from server.render import StateRenderer  # noqa: E402

HOVER_ABOVE_M = 0.06
LOW_Z_M = 0.02


def object_pixels(seg: np.ndarray, geom_id: int) -> dict:
    mask = (seg[..., 0] == geom_id) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"area": 0, "w": 0, "h": 0}
    return {"area": int(mask.sum()), "w": int(xs.max() - xs.min() + 1), "h": int(ys.max() - ys.min() + 1)}


def make_qpos(renderer: StateRenderer, arm_rad: np.ndarray, gripper_rad: float, block_xy: tuple[float, float]) -> np.ndarray:
    model = renderer.model
    qpos = model.key(config.START_KEYFRAME).qpos.copy()
    for name, value in zip(ARM_JOINTS, arm_rad, strict=True):
        qpos[model.joint(name).qposadr[0]] = value
    qpos[model.joint("gripper").qposadr[0]] = gripper_rad
    adr = model.joint("block_free").qposadr[0]
    qpos[adr : adr + 7] = [block_xy[0], block_xy[1], config.BLOCK_HALF_M, 1, 0, 0, 0]
    return qpos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("/tmp/record_camera_check"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ik = ArmIK()
    corners = {
        "x_lo_y_lo": (config.BLOCK_SPAWN_X_M[0], config.BLOCK_SPAWN_Y_M[0]),
        "x_lo_y_hi": (config.BLOCK_SPAWN_X_M[0], config.BLOCK_SPAWN_Y_M[1]),
        "x_hi_y_lo": (config.BLOCK_SPAWN_X_M[1], config.BLOCK_SPAWN_Y_M[0]),
        "x_hi_y_hi": (config.BLOCK_SPAWN_X_M[1], config.BLOCK_SPAWN_Y_M[1]),
    }
    print(f"카메라: 방위각 {config.RECORD_CAMERA_AZIMUTH_DEG}도, 고도 {config.RECORD_CAMERA_ELEVATION_DEG}도, 거리 {config.RECORD_CAMERA_DISTANCE_M} m, "
          f"바라보는 점 {config.RECORD_CAMERA_LOOKAT_M}, 이미지 {config.RECORD_IMAGE_HEIGHT_PX}x{config.RECORD_IMAGE_WIDTH_PX}(세로x가로), "
          f"그림자 {config.RENDER_SHADOW}, 반사 {config.RENDER_REFLECTION}")
    header = f"{'모서리':<10} {'자세':<16} {'블록 px(면적)':>12} {'가로x세로':>9} {'①대비 남음':>10} {'목표 px':>8}"
    print(header)
    tiles: list[Image.Image] = []
    with StateRenderer() as renderer:
        block_id = renderer.model.geom("block").id
        target_id = renderer.model.geom("target").id
        home_arm = np.array(renderer.model.key(config.START_KEYFRAME).qpos[:5])
        for name, (bx, by) in corners.items():
            base_area = None
            for pose, z, gripper in (("1 팔 올림(home)", None, 0.0), ("2 블록 위 6cm", config.BLOCK_HALF_M + HOVER_ABOVE_M, config.GRIPPER_OPEN_RAD),
                                     ("3 블록 높이로", LOW_Z_M, config.GRIPPER_OPEN_RAD)):
                if z is None:
                    arm = home_arm
                else:
                    ik.reset()
                    result = ik.solve(np.array([bx, by, z]))
                    if not result.ok:
                        print(f"{name:<10} {pose:<16} IK 실패: {result.reason}")
                        continue
                    arm = result.q_rad
                qpos = make_qpos(renderer, arm, gripper, (bx, by))
                rgb = renderer.render(qpos)
                seg = renderer.render_segmentation(qpos)
                block = object_pixels(seg, block_id)
                target = object_pixels(seg, target_id)
                if base_area is None:
                    base_area = block["area"]
                kept = block["area"] / base_area if base_area else 0.0
                print(f"{name:<10} {pose:<16} {block['area']:>12d} {block['w']:>4d}x{block['h']:<4d} {kept * 100:>9.0f}% {target['area']:>8d}")
                Image.fromarray(rgb).save(args.out / f"{name}_{pose.split()[0]}.png")
                tile = Image.fromarray(rgb).copy()
                # PIL 기본 폰트는 한글이 깨지므로 라벨은 영문(pose 1=팔 올림, 2=블록 위 6cm, 3=블록 높이)
                ImageDraw.Draw(tile).text((4, 4), f"{name} pose{pose.split()[0]}", fill=(255, 255, 0))
                tiles.append(tile)
    h, w = config.RECORD_IMAGE_HEIGHT_PX, config.RECORD_IMAGE_WIDTH_PX
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
    sheet.save(args.out / "contact_sheet.png")
    print(f"저장: {args.out}/ (개별 {len(tiles)}장 + contact_sheet.png)")


if __name__ == "__main__":
    main()
