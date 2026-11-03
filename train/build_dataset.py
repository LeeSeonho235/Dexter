"""3-3(b): data/raw(npz+json)와 data/frames(PNG)로 LeRobotDataset(v3.0)을 만든다. train/.venv에서 실행한다(mujoco 필요 없음).

규칙(한 칸도 어긋나면 안 된다)
  샘플 k: observation.state = qpos[k][:6], action = ctrl[k][:6], 이미지 = frames의 k번째(같은 k의 qpos 13개로 렌더링한 것)
- 학습에 쓰는 것은 성공(success=True) 에피소드. 추가로 뺄 것은 --exclude 번호로만 정한다(조용히 버리지 않는다).
- 시작 전에 frames가 지금의 npz와 짝인지 검사한다(프레임 수, qpos 해시, 렌더링 설정, 장면 해시). 다르면 멈춘다.
- 데이터셋은 data/lerobot/<이름>/ (data/는 .gitignore). 이미 있으면 만들지 않고 멈춘다(덮어쓰지 않는다).

사용: train/.venv/bin/python -I train/build_dataset.py [--exclude 5,12] [--crf 18] [--out data/lerobot/이름]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_config as cfg  # noqa: E402
from episode_select import load_arrays, load_metas, select  # noqa: E402

NICE_LEVEL = 19
FRAME_FILE = "frame_{k:06d}.png"


def qpos_sha256(qpos: np.ndarray) -> str:
    """server/render_frames.py와 같은 계산(float64 바이트의 SHA-256)."""
    return hashlib.sha256(np.ascontiguousarray(qpos, dtype=np.float64).tobytes()).hexdigest()


def frame_file(frames_dir: Path, index: int, k: int) -> Path:
    return Path(frames_dir) / f"ep_{index:06d}" / FRAME_FILE.format(k=k)


def make_features(height_px: int, width_px: int, image_mode: bool = False) -> dict:
    names = list(cfg.JOINT_NAMES)
    return {
        "observation.state": {"dtype": "float32", "shape": (cfg.N_JOINTS,), "names": names},
        "action": {"dtype": "float32", "shape": (cfg.N_JOINTS,), "names": names},
        cfg.IMAGE_KEY: {"dtype": "image" if image_mode else "video", "shape": (height_px, width_px, 3), "names": ["height", "width", "channels"]},
    }


def state_action_for(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """npz → (state, action). state[k] = qpos[k][:6], action[k] = ctrl[k][:6], float32. 검증 테스트도 같은 함수를 쓴다."""
    return (
        arrays["qpos"][:, : cfg.N_JOINTS].astype(np.float32),
        arrays["ctrl"][:, : cfg.N_JOINTS].astype(np.float32),
    )


def check_frames(raw_dir: Path, frames_dir: Path, metas_by_index: dict[int, dict], indices: list[int]) -> list[str]:
    """frames가 지금의 raw와 짝인지 검사한다. 문제 목록(비어 있으면 통과)."""
    manifest_path = Path(frames_dir) / "manifest.json"
    if not manifest_path.is_file():
        return [f"{manifest_path} 가 없습니다. 먼저 server/.venv에서 python -m server.render_frames 를 실행하세요"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings, problems = manifest["settings"], []
    for i in indices:
        meta = metas_by_index[i]
        entry = manifest["episodes"].get(str(i))
        if entry is None:
            problems.append(f"에피소드 {i}: manifest에 없음(프레임이 렌더링되지 않았다)")
            continue
        arrays = load_arrays(raw_dir, i)
        n = meta["n_ticks"]
        if entry["n_frames"] != n or len(arrays["qpos"]) != n:
            problems.append(f"에피소드 {i}: 프레임 수 {entry['n_frames']} != n_ticks {n}")
        if entry["qpos_sha256"] != qpos_sha256(arrays["qpos"]):
            problems.append(f"에피소드 {i}: 프레임을 그린 뒤 qpos가 바뀌었다(해시 불일치)")
        if meta["scene_sha256"] != settings["scene_sha256"] or meta["robot_model_sha256"] != settings["robot_model_sha256"]:
            problems.append(f"에피소드 {i}: 녹화 때와 프레임을 그릴 때의 장면/로봇 모델이 다르다")
        rc = meta["record_camera"]
        if (rc["height_px"], rc["width_px"]) != (settings["height_px"], settings["width_px"]) or rc["azimuth_deg"] != settings["camera"]["azimuth_deg"] \
                or rc["elevation_deg"] != settings["camera"]["elevation_deg"] or rc["distance_m"] != settings["camera"]["distance_m"]:
            problems.append(f"에피소드 {i}: 녹화 때 카메라 설정과 프레임의 카메라 설정이 다르다")
        folder = Path(frames_dir) / f"ep_{i:06d}"
        have = sum(1 for _ in folder.glob("frame_*.png")) if folder.is_dir() else 0
        if have != n:
            problems.append(f"에피소드 {i}: 프레임 파일 {have}개 != {n}")
    return problems


def build(
    raw_dir: Path,
    frames_dir: Path,
    out_root: Path,
    exclude: set[int],
    crf: int,
    *,
    image_mode: bool = False,
    codec: str | None = None,
) -> dict:
    """image_mode=True면 영상 대신 PNG 이미지로 저장한다(Colab에서 AV1 4:4:4 디코딩이 안 될 때의 대안, 용량 약 540 MB).
    codec은 영상 코덱(기본 설정의 libaom-av1). 다른 코덱의 예: "h264"(yuv444p, 더 널리 디코딩됨, 실측 블록 오차 4.4)."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    metas = load_metas(raw_dir)
    metas_by_index = {m["episode_index"]: m for m in metas}
    rows = select(metas, exclude)
    indices = [r["index"] for r in rows if r["included"]]
    if not indices:
        raise SystemExit("포함할 에피소드가 없습니다")
    problems = check_frames(raw_dir, frames_dir, metas_by_index, indices)
    if problems:
        raise SystemExit("프레임 검사 실패:\n  " + "\n  ".join(problems))
    if Path(out_root).exists():
        raise SystemExit(f"{out_root} 가 이미 있습니다. 덮어쓰지 않습니다. 다른 --out을 쓰거나 직접 지우세요")

    settings = json.loads((Path(frames_dir) / "manifest.json").read_text(encoding="utf-8"))["settings"]
    features = make_features(settings["height_px"], settings["width_px"], image_mode)
    encoder = None
    if not image_mode:
        vcodec = codec or cfg.VIDEO_CODEC
        # libsvtav1은 yuv444p를 열지 못한다(3-3 실측). 코덱 전용 옵션(cpu-used 등)은 기본 코덱일 때만 쓴다.
        pix_fmt = "yuv420p" if vcodec == "libsvtav1" else cfg.VIDEO_PIX_FMT
        extra = dict(cfg.VIDEO_EXTRA_OPTIONS) if vcodec == cfg.VIDEO_CODEC else {"threads": cfg.VIDEO_EXTRA_OPTIONS.get("threads", "2")}
        encoder = RGBEncoderConfig(vcodec=vcodec, pix_fmt=pix_fmt, crf=crf, g=cfg.VIDEO_GOP, extra_options=extra)
    ds = LeRobotDataset.create(
        repo_id=cfg.REPO_ID, fps=cfg.FPS, features=features, root=out_root, robot_type=cfg.ROBOT_TYPE,
        use_videos=not image_mode, rgb_encoder=encoder,
    )
    start = time.perf_counter()
    total = sum(metas_by_index[i]["n_ticks"] for i in indices)
    done = 0
    for n, i in enumerate(indices, start=1):
        arrays = load_arrays(raw_dir, i)
        state, action = state_action_for(arrays)
        for k in range(len(state)):
            image = np.asarray(Image.open(frame_file(frames_dir, i, k)).convert("RGB"), dtype=np.uint8)
            ds.add_frame({"observation.state": state[k], "action": action[k], cfg.IMAGE_KEY: image, "task": cfg.TASK})
        ds.save_episode()
        done += len(state)
        elapsed = time.perf_counter() - start
        print(f"[{n}/{len(indices)}] 에피소드 {i} -> 데이터셋 {n - 1}번: {len(state)}프레임 | 경과 {elapsed:.0f}s, 남은 시간 약 {elapsed / done * (total - done):.0f}s", flush=True)
    ds.finalize()
    return {"episodes": indices, "frames": total, "seconds": time.perf_counter() - start}


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=cfg.RAW_DIR)
    parser.add_argument("--frames", type=Path, default=cfg.FRAMES_DIR)
    parser.add_argument("--out", type=Path, default=cfg.DATASET_ROOT)
    parser.add_argument("--exclude", default="", help="실패 말고 추가로 뺄 에피소드 번호(쉼표). 기본은 성공 전부 포함")
    parser.add_argument("--crf", type=int, default=cfg.VIDEO_CRF, help="AV1 화질(낮을수록 좋고 큼)")
    parser.add_argument("--image-mode", action="store_true", help="영상 대신 PNG 이미지로 저장(Colab에서 영상 디코딩이 안 될 때의 대안, 약 540 MB)")
    parser.add_argument("--codec", choices=["libaom-av1", "h264", "hevc", "libsvtav1"], help=f"영상 코덱(기본 {cfg.VIDEO_CODEC})")
    args = parser.parse_args()
    try:
        os.nice(NICE_LEVEL)
    except OSError as err:
        print(f"nice 설정 실패(계속 진행): {err}", file=sys.stderr)
    exclude = {int(x) for x in args.exclude.split(",") if x.strip()}
    result = build(args.raw, args.frames, args.out, exclude, args.crf, image_mode=args.image_mode, codec=args.codec)
    print(f"완료: 에피소드 {len(result['episodes'])}개 {result['episodes']}, 프레임 {result['frames']}, {result['seconds']:.0f}s")
    print(f"데이터셋 크기: {dir_size_bytes(args.out) / 1e6:.1f} MB ({args.out})")


if __name__ == "__main__":
    main()
