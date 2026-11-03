"""3-3(a): 저장된 시연 에피소드(data/raw)의 qpos로 시뮬레이션 화면을 다시 그려 data/frames/에 PNG로 저장한다.

- 기본은 성공(success=True)한 에피소드 전부. --episodes 1,2 로 번호를 직접 정할 수 있다(실패 포함).
- 카메라·해상도·그림자는 server/render.py의 StateRenderer(= server/config.py의 RECORD_*)를 그대로 쓴다. 블록 qpos까지 써서 장면을 복원한다.
- 공용 서버라 시작할 때 nice 19, 렌더 스레드 2개로 제한한다(MUJOCO_GL=egl은 CPU 렌더링).
- 에피소드 단위로 임시 폴더에 쓴 뒤 이름을 바꾼다. 중간에 끊겨도 다시 실행하면 끝난 에피소드는 건너뛴다.
- data/raw는 읽기만 한다. 웹캠 영상은 쓰지 않는다(시뮬레이션 화면뿐).

사용:
  server/.venv/bin/python -m server.render_frames                       # 성공 에피소드 전부
  server/.venv/bin/python -m server.render_frames --size-compare --episodes 7   # PNG/JPEG 용량만 재기(저장 안 함)
"""

import os

# mujoco를 import하기 전에 정해야 한다.
RENDER_THREADS = 2
NICE_LEVEL = 19
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LP_NUM_THREADS", str(RENDER_THREADS))  # llvmpipe(CPU EGL) 스레드 수
os.environ.setdefault("OMP_NUM_THREADS", str(RENDER_THREADS))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from server import config  # noqa: E402
from server.episode_store import ROBOT_MODEL_PATH, EpisodeStore, file_sha256, stem_of  # noqa: E402
from server.render import StateRenderer  # noqa: E402

FRAMES_DIR = config.REPO_ROOT / "data" / "frames"
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
IMAGE_FORMAT = "png"
JPEG_QUALITY_FOR_COMPARE = 95


def episode_dir_name(index: int) -> str:
    return f"ep_{index:06d}"


def frame_path(frames_dir: Path, index: int, k: int) -> Path:
    return Path(frames_dir) / episode_dir_name(index) / f"frame_{k:06d}.{IMAGE_FORMAT}"


def qpos_sha256(qpos: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(qpos, dtype=np.float64).tobytes()).hexdigest()


def render_settings() -> dict:
    """프레임이 같은 조건으로 그려졌는지 비교하는 값(manifest에 저장)."""
    return {
        "image_format": IMAGE_FORMAT,
        "height_px": config.RECORD_IMAGE_HEIGHT_PX,
        "width_px": config.RECORD_IMAGE_WIDTH_PX,
        "camera": {
            "azimuth_deg": config.RECORD_CAMERA_AZIMUTH_DEG,
            "elevation_deg": config.RECORD_CAMERA_ELEVATION_DEG,
            "distance_m": config.RECORD_CAMERA_DISTANCE_M,
            "lookat_m": list(config.RECORD_CAMERA_LOOKAT_M),
        },
        "shadow": config.RENDER_SHADOW,
        "reflection": config.RENDER_REFLECTION,
        "scene_sha256": file_sha256(config.SCENE_PATH),
        "robot_model_sha256": file_sha256(ROBOT_MODEL_PATH),
        "mujoco_version": mujoco.__version__,
    }


def read_manifest(frames_dir: Path) -> dict | None:
    path = Path(frames_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(frames_dir: Path, manifest: dict) -> None:
    path = Path(frames_dir) / MANIFEST_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_png(image: np.ndarray, path: Path) -> int:
    Image.fromarray(image).save(path, format="PNG")
    return path.stat().st_size


def render_episode(renderer: StateRenderer, qpos: np.ndarray, frames_dir: Path, index: int) -> int:
    """에피소드 하나의 프레임을 임시 폴더에 쓰고 이름을 바꾼다. 파일 크기 합(byte)을 돌려준다."""
    final = Path(frames_dir) / episode_dir_name(index)
    tmp = Path(frames_dir) / f"{episode_dir_name(index)}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)  # 이전에 끊긴 임시 폴더(이 스크립트가 만든 것)
    tmp.mkdir(parents=True)
    total_bytes = 0
    for k in range(len(qpos)):
        total_bytes += save_png(renderer.render(qpos[k]), tmp / f"frame_{k:06d}.{IMAGE_FORMAT}")
    if final.exists():
        shutil.rmtree(final)  # 설정이 같은데 다시 그리는 경우(--force)만 여기에 온다
    os.replace(tmp, final)
    return total_bytes


def encoded_sizes(image: np.ndarray) -> tuple[int, int]:
    """(PNG 바이트, JPEG q95 바이트). 메모리에서만 인코딩한다."""
    sizes = []
    for fmt, kwargs in (("PNG", {}), ("JPEG", {"quality": JPEG_QUALITY_FOR_COMPARE})):
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format=fmt, **kwargs)
        sizes.append(buf.tell())
    return sizes[0], sizes[1]


def parse_episode_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def episode_is_done(manifest: dict, frames_dir: Path, index: int, qpos: np.ndarray) -> bool:
    entry = manifest["episodes"].get(str(index))
    if entry is None or entry["qpos_sha256"] != qpos_sha256(qpos) or entry["n_frames"] != len(qpos):
        return False
    folder = Path(frames_dir) / episode_dir_name(index)
    return folder.is_dir() and sum(1 for _ in folder.glob(f"frame_*.{IMAGE_FORMAT}")) == len(qpos)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=config.RECORD_DIR)
    parser.add_argument("--out", type=Path, default=FRAMES_DIR)
    parser.add_argument("--episodes", default="", help="그릴 에피소드 번호(쉼표). 기본은 성공 전부")
    parser.add_argument("--size-compare", action="store_true", help="저장하지 않고 PNG/JPEG 용량만 잰다")
    parser.add_argument("--force", action="store_true", help="끝난 에피소드도 다시 그린다")
    args = parser.parse_args()

    try:
        os.nice(NICE_LEVEL)
    except OSError as err:
        print(f"nice 설정 실패(계속 진행): {err}", file=sys.stderr)

    store = EpisodeStore(args.raw)
    metas = store.scan()
    wanted = parse_episode_list(args.episodes)
    by_index = {m["episode_index"]: m for m in metas}
    if wanted:
        missing = [i for i in wanted if i not in by_index]
        if missing:
            print(f"저장된 에피소드가 아닙니다: {missing}", file=sys.stderr)
            return 2
        targets = [by_index[i] for i in wanted]
    else:
        targets = [m for m in metas if m.get("success") is True]

    settings = render_settings()
    for m in targets:  # 같은 장면·로봇 모델로 녹화된 것만 다시 그린다. 다르면 멈추고 알린다(우회하지 않는다).
        if m["scene_sha256"] != settings["scene_sha256"] or m["robot_model_sha256"] != settings["robot_model_sha256"]:
            print(f"에피소드 {m['episode_index']}: 녹화 때와 장면/로봇 모델 해시가 다릅니다. 중단합니다.", file=sys.stderr)
            return 3

    total_frames = sum(m["n_ticks"] for m in targets)
    print(f"대상 {len(targets)}개 에피소드, 프레임 {total_frames}장, 해상도 {settings['height_px']}x{settings['width_px']}, "
          f"nice {NICE_LEVEL}, 렌더 스레드 {RENDER_THREADS}", flush=True)

    if args.size_compare:
        with StateRenderer() as renderer:
            for m in targets:
                qpos = store.load_arrays(m["episode_index"])["qpos"]
                png = jpg = 0
                t0 = time.perf_counter()
                for k in range(len(qpos)):
                    p, j = encoded_sizes(renderer.render(qpos[k]))
                    png += p
                    jpg += j
                n = len(qpos)
                print(f"에피소드 {m['episode_index']}: {n}장 {time.perf_counter() - t0:.1f}s | PNG 합계 {png / 1e6:.2f} MB "
                      f"(장당 {png / n / 1e3:.1f} KB) | JPEG q{JPEG_QUALITY_FOR_COMPARE} 합계 {jpg / 1e6:.2f} MB (장당 {jpg / n / 1e3:.1f} KB)")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.out)
    if manifest is None:
        manifest = {"manifest_version": MANIFEST_VERSION, "settings": settings, "episodes": {}}
    elif manifest["settings"] != settings:
        print("data/frames/manifest.json의 렌더링 설정이 지금과 다릅니다. 다른 --out 폴더를 쓰거나 확인하세요. 중단합니다.", file=sys.stderr)
        return 4

    start = time.perf_counter()
    done_frames = rendered_frames = 0
    with StateRenderer() as renderer:
        for n, m in enumerate(targets, start=1):
            index = m["episode_index"]
            qpos = store.load_arrays(index)["qpos"]
            if not args.force and episode_is_done(manifest, args.out, index, qpos):
                done_frames += len(qpos)
                print(f"[{n}/{len(targets)}] 에피소드 {index}: 이미 끝남, 건너뜀", flush=True)
                continue
            t0 = time.perf_counter()
            size_bytes = render_episode(renderer, qpos, args.out, index)
            took_s = time.perf_counter() - t0
            done_frames += len(qpos)
            rendered_frames += len(qpos)
            manifest["episodes"][str(index)] = {
                "seed": m["seed"],
                "n_frames": len(qpos),
                "bytes": size_bytes,
                "render_s": round(took_s, 1),
                "qpos_sha256": qpos_sha256(qpos),
                "npz": f"{stem_of(index)}.npz",
            }
            write_manifest(args.out, manifest)
            elapsed_s = time.perf_counter() - start
            rendered_s_per_frame = elapsed_s / rendered_frames  # 건너뛴 에피소드는 속도 계산에서 뺀다
            eta_s = rendered_s_per_frame * (total_frames - done_frames)
            print(f"[{n}/{len(targets)}] 에피소드 {index}: {len(qpos)}장 {took_s:.0f}s ({took_s / len(qpos) * 1000:.0f} ms/장), "
                  f"{size_bytes / 1e6:.1f} MB | 경과 {elapsed_s / 60:.1f}분, 남은 시간 약 {eta_s / 60:.1f}분", flush=True)
    print(f"끝: {time.perf_counter() - start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
