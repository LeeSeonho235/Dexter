"""LeRobot 파이프라인 동작 확인용 스크립트 (3-1). 학습 품질이 아니라 "끝까지 도는지"만 본다.

사용법 (저장소 밖 임시 폴더를 쓴다. root는 아직 없는 폴더여야 한다):
  train/.venv/bin/python -I train/smoke_test.py make  <root>          # 가짜 데이터셋 만들기
  train/.venv/bin/python -I train/smoke_test.py infer <root> <ckpt>   # 학습된 정책으로 select_action 호출

데이터는 전부 합성이다(회색 배경에 움직이는 사각형). 웹캠 영상이나 사람 얼굴은 쓰지 않는다.
단위: 관절 각도는 라디안(rad), 시간은 초(s).
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)  # 공용 서버: 스레드 2개로 제한

REPO_ID = "local/handover_smoke"
FPS = 30
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TASK = "pick the red block and place it on the green target"
IMAGE_HW = (64, 64)  # 확인용 작은 이미지(높이, 너비) px. 실제 해상도는 3-2 시작 전에 확정한다.
NUM_EPISODES = 2
FRAMES_PER_EPISODE = 30
IMAGE_KEY = "observation.images.top"


def make_features() -> dict:
    h, w = IMAGE_HW
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        IMAGE_KEY: {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
    }


def fake_image(rng: np.random.Generator, t: int) -> np.ndarray:
    h, w = IMAGE_HW
    img = np.full((h, w, 3), 150, dtype=np.uint8)
    x = int(8 + t * (w - 24) / FRAMES_PER_EPISODE)
    y = int(rng.integers(8, h - 16))
    img[y : y + 8, x : x + 8] = (217, 38, 38)  # 빨간 사각형 = 가짜 블록
    return img


def make(root: Path) -> None:
    from lerobot.datasets import LeRobotDataset

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=make_features(),
        root=root,
        robot_type="so101_sim",
        use_videos=True,
    )
    for _ in range(NUM_EPISODES):
        state = rng.uniform(-0.5, 0.5, size=6).astype(np.float32)
        for t in range(FRAMES_PER_EPISODE):
            action = (state + rng.normal(0, 0.02, size=6)).astype(np.float32)
            ds.add_frame(
                {
                    "observation.state": state,
                    "action": action,
                    IMAGE_KEY: fake_image(rng, t),
                    "task": TASK,
                }
            )
            state = action
        ds.save_episode()
    ds.finalize()
    print(f"dataset made in {time.perf_counter() - t0:.2f}s: episodes={ds.num_episodes} frames={ds.num_frames}")


def infer(root: Path, ckpt: Path) -> None:
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(ckpt))
    policy.to("cpu")
    policy.reset()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ckpt))
    print("chunk_size", policy.config.chunk_size, "n_action_steps", policy.config.n_action_steps)

    rng = np.random.default_rng(1)
    times = []
    for t in range(3):
        obs = {
            "observation.state": rng.uniform(-0.5, 0.5, size=6).astype(np.float32),
            IMAGE_KEY: fake_image(rng, t),  # (H, W, 3) uint8
        }
        t0 = time.perf_counter()
        action = predict_action(obs, policy, torch.device("cpu"), pre, post, use_amp=False, task=TASK)
        times.append(time.perf_counter() - t0)
        print(f"step {t}: action shape={tuple(action.shape)} dtype={action.dtype} {times[-1] * 1000:.1f} ms")
        print("   ", np.round(action.numpy().ravel(), 3))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "make":
        make(Path(sys.argv[2]))
    elif mode == "infer":
        infer(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(f"unknown mode: {mode}")
