"""저장된 에피소드를 서버에서 다시 그려 mp4(+ contact sheet)로 만든다. 눈으로 확인하는 용도다. 화면이 없는 서버라 창은 띄우지 않는다.

- data/raw의 qpos로 server/render.py의 StateRenderer가 그린다(데이터셋 이미지와 같은 카메라). 성공/실패 상관없이 아무 번호나 가능.
- server/.venv에는 영상 인코더가 없어서 mp4는 train/.venv의 av를 서브프로세스로 불러 만든다(새 설치 없음).
  train/.venv가 없거나 인코딩이 안 되면 contact sheet(PNG 한 장)만 만든다.
- 기본 출력: /tmp/dexter_episode_<번호>.mp4, /tmp/dexter_episode_<번호>_sheet.png. 공용 서버라 nice 19, 렌더 스레드 2개.
- mp4와 PNG는 저장소에 올리지 않는다(.gitignore의 *.mp4. PNG는 /tmp에만 둔다).

사용:
  MUJOCO_GL=egl server/.venv/bin/python -m server.replay_episode --episode 7
  server/.venv/bin/python -m server.replay_episode --episode 7 --step 2 --out /tmp/ep7.mp4   # 2프레임마다 1장(빠르게 만들기)
"""

import os

RENDER_THREADS = 2
NICE_LEVEL = 19
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LP_NUM_THREADS", str(RENDER_THREADS))
os.environ.setdefault("OMP_NUM_THREADS", str(RENDER_THREADS))

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from server import config  # noqa: E402
from server.episode_store import EpisodeStore  # noqa: E402
from server.render import StateRenderer  # noqa: E402

TRAIN_DIR = config.REPO_ROOT / "train"
TRAIN_PYTHON = TRAIN_DIR / ".venv" / "bin" / "python"
ENCODER_SCRIPT = TRAIN_DIR / "encode_mp4.py"


def default_out(episode: int, suffix: str) -> Path:
    return Path("/tmp") / f"dexter_episode_{episode:06d}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", type=int, required=True, help="data/raw의 에피소드 번호(episode_000007 → 7)")
    parser.add_argument("--raw", type=Path, default=config.RECORD_DIR)
    parser.add_argument("--out", type=Path, help="mp4 경로(기본 /tmp/dexter_episode_<번호>.mp4)")
    parser.add_argument("--sheet", type=Path, help="contact sheet PNG 경로")
    parser.add_argument("--step", type=int, default=1, help="N프레임마다 1장만 그린다(기본 1 = 전부)")
    parser.add_argument("--no-mp4", action="store_true", help="contact sheet만 만든다")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step은 1 이상")

    try:
        os.nice(NICE_LEVEL)
    except OSError as err:
        print(f"nice 설정 실패(계속 진행): {err}", file=sys.stderr)

    store = EpisodeStore(args.raw)
    metas = {m["episode_index"]: m for m in store.scan()}
    if args.episode not in metas:
        print(f"에피소드 {args.episode} 가 없습니다. 있는 번호: {sorted(metas)}", file=sys.stderr)
        return 2
    meta = metas[args.episode]
    qpos = store.load_arrays(args.episode)["qpos"]
    frames_idx = list(range(0, len(qpos), args.step))
    out_mp4 = args.out or default_out(args.episode, ".mp4")
    out_sheet = args.sheet or default_out(args.episode, "_sheet.png")

    sys.path.insert(0, str(TRAIN_DIR))
    from encode_mp4 import label_frame, make_sheet, sheet_indices  # noqa: E402 (numpy+Pillow만 쓰는 함수)

    encoder = None
    if not args.no_mp4:
        if not TRAIN_PYTHON.is_file():
            print("train/.venv가 없어 mp4를 만들 수 없습니다. contact sheet만 만듭니다.", file=sys.stderr)
        else:
            with StateRenderer() as probe:
                shape = probe.render(qpos[0]).shape
            fps = max(1, round(config.RECORD_FPS / args.step))
            encoder = subprocess.Popen(
                [str(TRAIN_PYTHON), "-I", str(ENCODER_SCRIPT), "stdin", "--width", str(shape[1]), "--height", str(shape[0]), "--fps", str(fps), str(out_mp4)],
                stdin=subprocess.PIPE,
            )

    wanted = set(sheet_indices(len(frames_idx)))
    kept: dict[int, np.ndarray] = {}
    start = time.perf_counter()
    status = f"에피소드 {args.episode} (시드 {meta['seed']}, {'성공' if meta['success'] else '실패'}, {meta['n_ticks']}틱)"
    print(f"{status}: {len(frames_idx)}프레임 그리는 중 (약 {len(frames_idx) * 0.06:.0f}초 예상)", flush=True)
    with StateRenderer() as renderer:
        for n, k in enumerate(frames_idx):
            image = label_frame(renderer.render(qpos[k]), f"raw ep{args.episode} k={k} t={k / config.RECORD_FPS:.1f}s")
            if encoder is not None:
                encoder.stdin.write(np.ascontiguousarray(image).tobytes())
            if n in wanted:
                kept[n] = image
    if encoder is not None:
        encoder.stdin.close()
        code = encoder.wait()
        print(f"mp4: {out_mp4} (인코더 종료 코드 {code})" if code else f"mp4: {out_mp4}")
        if code:
            return code
    make_sheet([kept[n] for n in sorted(kept)]).save(out_sheet)
    print(f"contact sheet: {out_sheet} ({len(kept)}장) | {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
