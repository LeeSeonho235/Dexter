"""3-3 검증: 만든 LeRobotDataset을 다시 불러서 raw 데이터와 한 칸도 안 어긋나는지, 이미지가 맞는지 확인하고 요약을 출력한다.

- 상태/행동: 데이터셋 k번째 프레임의 observation.state, action이 npz의 k번째(qpos[k][:6], ctrl[k][:6], float32)와 정확히 같은지(array_equal).
- 이미지: 영상은 손실 압축이라 정확히 같을 수 없다. 에피소드마다 몇 프레임을 디코딩해서 frames의 PNG와 비교하고,
  "같은 k의 PNG가 이웃(k±3)의 PNG보다 더 가깝다"를 확인한다(한 칸이라도 어긋나면 걸린다). 블록(빨강) 픽셀의 색 오차도 따로 잰다.
- 실패하면 종료 코드 1.

사용: train/.venv/bin/python -I train/verify_dataset.py [--root data/lerobot/이름]
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 로컬 데이터셋만 읽는다. 못 찾아도 Hub로 조회하지 않는다.

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_config as cfg  # noqa: E402
from build_dataset import frame_file, state_action_for  # noqa: E402
from episode_select import load_arrays, load_metas, select  # noqa: E402

RED_MIN = 150  # 블록 픽셀로 보는 기준(PNG에서 R이 이 이상이고 G, B가 낮음)
RED_MAX_OTHER = 80


def load_dataset(root: Path):
    from lerobot.datasets import LeRobotDataset

    return LeRobotDataset(cfg.REPO_ID, root=root)


def expected_indices(raw_dir: Path, exclude: set[int] = frozenset()) -> list[int]:
    return [r["index"] for r in select(load_metas(raw_dir), exclude) if r["included"]]


def check_state_action(ds, raw_dir: Path, indices: list[int]) -> list[str]:
    """데이터셋 에피소드 e(0부터) = raw 에피소드 indices[e]. 모든 프레임의 state/action이 npz와 정확히 같은지 본다. 문제 목록."""
    problems: list[str] = []
    table = ds.hf_dataset.with_format("numpy")
    state = np.asarray(table["observation.state"])
    action = np.asarray(table["action"])
    ep = np.asarray(table["episode_index"]).reshape(-1)
    frame = np.asarray(table["frame_index"]).reshape(-1)
    index = np.asarray(table["index"]).reshape(-1)
    timestamp = np.asarray(table["timestamp"]).reshape(-1)
    if ds.num_episodes != len(indices):
        problems.append(f"에피소드 수 {ds.num_episodes} != 기대 {len(indices)}")
    if not np.array_equal(index, np.arange(len(index))):
        problems.append("index 열이 0..N-1 연속이 아님")
    for e, raw_index in enumerate(indices):
        mask = ep == e
        exp_state, exp_action = state_action_for(load_arrays(raw_dir, raw_index))
        n = len(exp_state)
        if int(mask.sum()) != n:
            problems.append(f"에피소드 {e}(raw {raw_index}): 프레임 수 {int(mask.sum())} != {n}")
            continue
        if not np.array_equal(frame[mask], np.arange(n)):
            problems.append(f"에피소드 {e}(raw {raw_index}): frame_index가 0..n-1이 아님")
        if not np.array_equal(state[mask], exp_state):
            bad = int(np.argmax((state[mask] != exp_state).any(axis=1)))
            problems.append(f"에피소드 {e}(raw {raw_index}): observation.state가 npz와 다름 (첫 불일치 k={bad})")
        if not np.array_equal(action[mask], exp_action):
            bad = int(np.argmax((action[mask] != exp_action).any(axis=1)))
            problems.append(f"에피소드 {e}(raw {raw_index}): action이 npz와 다름 (첫 불일치 k={bad})")
        if not np.allclose(timestamp[mask], np.arange(n) / cfg.FPS, atol=1e-6):
            problems.append(f"에피소드 {e}(raw {raw_index}): timestamp가 frame_index/30과 다름")
    return problems


def png_array(frames_dir: Path, raw_index: int, k: int) -> np.ndarray:
    return np.asarray(Image.open(frame_file(frames_dir, raw_index, k)).convert("RGB"), dtype=np.uint8)


def decoded_array(ds, global_index: int) -> np.ndarray:
    """데이터셋이 돌려주는 이미지(C,H,W float [0,1])를 (H,W,3) uint8로."""
    img = ds[global_index][cfg.IMAGE_KEY].numpy()
    return np.clip(np.rint(img.transpose(1, 2, 0) * 255.0), 0, 255).astype(np.uint8)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def sample_frames(n: int, count: int) -> list[int]:
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, count)})


def check_images(ds, frames_dir: Path, indices: list[int], per_episode: int = cfg.VERIFY_FRAMES_PER_EPISODE) -> dict:
    """샘플 프레임마다 (같은 k의 PNG와의 오차, 이웃 PNG와의 오차). 문제 목록과 통계를 돌려준다."""
    table = ds.hf_dataset.with_format("numpy")
    ep = np.asarray(table["episode_index"]).reshape(-1)
    starts = {e: int(np.argmax(ep == e)) for e in range(len(indices))}
    own, block, strict, total, static, problems = [], [], 0, 0, 0, []
    for e, raw_index in enumerate(indices):
        n = int((ep == e).sum())
        for k in sample_frames(n, per_episode):
            decoded = decoded_array(ds, starts[e] + k)
            ref = png_array(frames_dir, raw_index, k)
            own_err = mae(decoded, ref)
            own.append(own_err)
            red = (ref[..., 0] >= RED_MIN) & (ref[..., 1] <= RED_MAX_OTHER) & (ref[..., 2] <= RED_MAX_OTHER)
            if red.any():
                block.append(float(np.abs(decoded.astype(np.int16) - ref.astype(np.int16))[red].mean()))
            better_or_equal, strictly, judged = True, True, 0
            for off in cfg.VERIFY_NEIGHBOR_OFFSETS:
                j = k + off
                if not 0 <= j < n:
                    continue
                other_png = png_array(frames_dir, raw_index, j)
                changed = np.abs(ref.astype(np.int16) - other_png.astype(np.int16)).max(axis=2) > cfg.VERIFY_CHANGED_PIXEL_DELTA
                if int(changed.sum()) < cfg.VERIFY_MIN_CHANGED_PX:
                    continue  # 이 이웃과는 거의 같은 그림이라 가릴 수 없다
                judged += 1
                own_region = float(np.abs(decoded.astype(np.int16) - ref.astype(np.int16))[changed].mean())
                other_region = float(np.abs(decoded.astype(np.int16) - other_png.astype(np.int16))[changed].mean())
                better_or_equal &= own_region <= other_region
                strictly &= own_region < other_region
            if judged == 0:
                static += 1  # 어느 이웃과도 구별되지 않는 정지 장면(판정에서 뺀다)
                continue
            total += 1
            strict += int(strictly)
            if not better_or_equal:
                problems.append(f"에피소드 {e}(raw {raw_index}) k={k}: 디코딩된 이미지가 k±3의 PNG와 더 비슷함(어긋남 의심)")
    return {
        "problems": problems,
        "frames_checked": total,
        "frames_static_skipped": static,
        "strictly_closest": strict,
        "mae_all": own,
        "mae_block": block,
    }


def print_summary(ds, root: Path) -> None:
    print(f"에피소드 수: {ds.num_episodes}")
    print(f"총 프레임 수: {ds.num_frames}")
    print(f"fps: {ds.fps}, robot_type: {ds.meta.robot_type}, tasks: {list(ds.meta.tasks.index)}")
    print("features:")
    for key, ft in ds.meta.features.items():
        print(f"  {key}: dtype={ft['dtype']} shape={tuple(ft['shape'])}")
    first = ds[0]
    print("첫 프레임:")
    for key in ("observation.state", "action"):
        print(f"  {key} = {np.round(first[key].numpy(), 5).tolist()}")
    img = first[cfg.IMAGE_KEY]
    print(f"  {cfg.IMAGE_KEY}: shape={tuple(img.shape)} dtype={img.dtype} min={float(img.min()):.3f} max={float(img.max()):.3f}")
    print(f"  timestamp={float(first['timestamp']):.4f} frame_index={int(first['frame_index'])} episode_index={int(first['episode_index'])} task={first['task']!r}")
    print("stats (정규화 통계):")
    for key in ("observation.state", "action"):
        s = ds.meta.stats[key]
        for stat in ("mean", "std", "min", "max"):
            print(f"  {key}.{stat} = {np.round(np.asarray(s[stat]), 4).tolist()}")
    s = ds.meta.stats[cfg.IMAGE_KEY]
    for stat in ("mean", "std", "min", "max"):
        print(f"  {cfg.IMAGE_KEY}.{stat} (RGB, 0~1) = {np.round(np.asarray(s[stat]).reshape(-1), 4).tolist()}")
    size = sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
    print(f"데이터셋 폴더 크기: {size / 1e6:.2f} MB")
    for sub in ("data", "videos", "meta"):
        sz = sum(p.stat().st_size for p in (Path(root) / sub).rglob("*") if p.is_file())
        print(f"  {sub}/: {sz / 1e6:.2f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cfg.DATASET_ROOT)
    parser.add_argument("--raw", type=Path, default=cfg.RAW_DIR)
    parser.add_argument("--frames", type=Path, default=cfg.FRAMES_DIR)
    parser.add_argument("--exclude", default="")
    args = parser.parse_args()
    indices = expected_indices(args.raw, {int(x) for x in args.exclude.split(",") if x.strip()})

    ds = load_dataset(args.root)
    print_summary(ds, args.root)

    problems = check_state_action(ds, args.raw, indices)
    print(f"\n[상태/행동 정확 일치] 에피소드 {len(indices)}개, 프레임 {ds.num_frames}개 -> {'통과' if not problems else '실패'}")
    img = check_images(ds, args.frames, indices)
    problems += img["problems"]
    mae_all, mae_block = np.array(img["mae_all"]), np.array(img["mae_block"])
    print(f"[이미지 대응] 샘플 {img['frames_checked'] + img['frames_static_skipped']}프레임 중 판정 {img['frames_checked']}개 "
          f"(이웃 PNG와 바뀐 픽셀이 {cfg.VERIFY_MIN_CHANGED_PX}개 미만인 정지 장면 {img['frames_static_skipped']}개는 판정에서 뺌): "
          f"바뀐 영역에서 같은 k의 PNG가 이웃(k±3)보다 더 가까움 {img['strictly_closest']}/{img['frames_checked']}, 어긋남 의심 {len(img['problems'])}개")
    print(f"  디코딩 vs PNG 평균 절대 오차(0~255): 전체 평균 {mae_all.mean():.3f} (최대 {mae_all.max():.3f}), "
          f"블록 픽셀 평균 {mae_block.mean():.3f} (최대 {mae_block.max():.3f}, {len(mae_block)}프레임)")
    for p in problems:
        print("문제:", p)
    print("결과:", "통과" if not problems else f"실패 {len(problems)}건")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
