"""프레임 → mp4 / contact sheet 도구(눈으로 확인용). train/.venv의 av로 인코딩한다(server/.venv에는 인코더가 없다).

두 가지 입력
  stdin    : 표준 입력으로 (높이, 너비, 3) uint8 RGB 프레임을 이어서 받는다(server/replay_episode.py가 쓴다).
  dataset  : LeRobotDataset에서 에피소드 e의 이미지를 디코딩해서 만든다(데이터셋 안의 영상이 맞는지 보려는 용도).

사용:
  train/.venv/bin/python -I train/encode_mp4.py stdin --width 320 --height 240 --fps 30 /tmp/out.mp4
  train/.venv/bin/python -I train/encode_mp4.py dataset --root data/lerobot/<이름> --episode 0 --out /tmp/ds_ep0.mp4 --sheet /tmp/ds_ep0.png

mp4는 H.264(yuv420p)로 만든다(대부분의 플레이어에서 열린다). 이 파일은 저장소에 올리지 않는다(.gitignore의 *.mp4).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

CRF = 18
SHEET_COLUMNS = 4
SHEET_ROWS = 4


def label_frame(image: np.ndarray, text: str) -> np.ndarray:
    """왼쪽 위에 글자를 얹은 복사본(확인용 영상에만 쓴다. 데이터셋 이미지는 건드리지 않는다)."""
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, 8 + 6 * len(text), 12), fill=(0, 0, 0))
    draw.text((4, 1), text, fill=(255, 255, 255))
    return np.asarray(pil)


def make_sheet(frames: list[np.ndarray], columns: int = SHEET_COLUMNS) -> Image.Image:
    """프레임 목록을 격자 한 장으로 붙인다."""
    h, w, _ = frames[0].shape
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * w, rows * h))
    for n, frame in enumerate(frames):
        sheet.paste(Image.fromarray(frame), ((n % columns) * w, (n // columns) * h))
    return sheet


def sheet_indices(n_frames: int, count: int = SHEET_COLUMNS * SHEET_ROWS) -> list[int]:
    return sorted({int(round(x)) for x in np.linspace(0, n_frames - 1, min(count, n_frames))})


class Mp4Writer:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        import av

        self._av = av
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width, self._stream.height, self._stream.pix_fmt = width, height, "yuv420p"
        self._stream.options = {"crf": str(CRF)}

    def write(self, image: np.ndarray) -> None:
        frame = self._av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def run_stdin(args: argparse.Namespace) -> None:
    size = args.width * args.height * 3
    writer = Mp4Writer(args.out, args.width, args.height, args.fps)
    n = 0
    stream = sys.stdin.buffer
    while True:
        buf = stream.read(size)
        if len(buf) < size:
            break
        writer.write(np.frombuffer(buf, dtype=np.uint8).reshape(args.height, args.width, 3))
        n += 1
    writer.close()
    print(f"mp4 {n}프레임 -> {args.out}")


def run_dataset(args: argparse.Namespace) -> None:
    import dataset_config as cfg
    from verify_dataset import decoded_array, load_dataset

    ds = load_dataset(args.root)
    table = ds.hf_dataset.with_format("numpy")
    ep = np.asarray(table["episode_index"]).reshape(-1)
    idx = np.flatnonzero(ep == args.episode)
    if len(idx) == 0:
        raise SystemExit(f"데이터셋에 에피소드 {args.episode} 가 없습니다 (0~{ds.num_episodes - 1})")
    first = decoded_array(ds, int(idx[0]))
    writer = Mp4Writer(args.out, first.shape[1], first.shape[0], cfg.FPS) if args.out else None
    kept: dict[int, np.ndarray] = {}
    wanted = set(sheet_indices(len(idx))) if args.sheet else set()
    for k, g in enumerate(idx):
        image = decoded_array(ds, int(g))
        labeled = label_frame(image, f"ds ep{args.episode} k={k} t={k / cfg.FPS:.1f}s")
        if writer:
            writer.write(labeled)
        if k in wanted:
            kept[k] = labeled
    if writer:
        writer.close()
        print(f"mp4 {len(idx)}프레임 -> {args.out}")
    if args.sheet:
        make_sheet([kept[k] for k in sorted(kept)]).save(args.sheet)
        print(f"contact sheet {len(kept)}장 -> {args.sheet}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("stdin")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("out", type=Path)
    p = sub.add_parser("dataset")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--episode", type=int, required=True, help="데이터셋 안의 에피소드 번호(0부터)")
    p.add_argument("--out", type=Path)
    p.add_argument("--sheet", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    try:
        os.nice(19)
    except OSError:
        pass
    run_stdin(args) if args.mode == "stdin" else run_dataset(args)


if __name__ == "__main__":
    main()
