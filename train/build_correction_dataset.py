"""4-3: 1호 시연(data/raw 성공 에피소드) + 교정의 HUMAN 구간으로 LeRobotDataset(v3.0)을 만든다. train/.venv에서 실행한다.

- 시연 프레임은 3-3에서 그린 data/frames, 교정 구간 프레임은 server.render_correction_frames가 그린 폴더(--corr-frames)를 쓴다.
- 샘플 k: observation.state = qpos[k][:6], action = ctrl[k][:6], 이미지 = 같은 k의 qpos로 그린 프레임. 교정은 HUMAN 구간만(구간 하나 = 에피소드 하나).
  `is_intervention`은 구간을 자를 때만 쓰고 데이터셋 컬럼으로 넣지 않는다.
- 입력(--raw, --frames, --corrections, --corr-frames)은 읽기만 한다. --out은 직접 적고, 이미 있으면 만들지 않고 멈춘다.
- 시작 전 검사(하나라도 걸리면 만들지 않고 종료 코드 3): 프레임-npz 짝, 구간 배열 정합, 장면/로봇/체크포인트/환경 해시,
  시연+교정 모두 시드와 블록 위치가 평가 시드 1000~1049와 섞이지 않는지(auto_runs.csv 포함). 환경 해시 일부는 --allow-diff로 허락할 수 있다.
- 만든 뒤 <out>/handover_sources.json에 데이터셋 에피소드 번호 ↔ 출처(시연 번호 / 교정 번호, 시드, 샘플 범위)를 남긴다.

사용 (저장소 루트에서):
  train/.venv/bin/python -I train/build_correction_dataset.py --corrections <교정 폴더> --corr-frames <구간 프레임 폴더> \\
      --out data/lerobot/dexter_pipeline_test [--check-only]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(TRAIN_DIR.parent))  # server.* 순수 모듈(numpy만 씀)을 쓰기 위해
import build_dataset as bd  # noqa: E402
import dataset_config as cfg  # noqa: E402
from episode_select import load_arrays, load_metas, select  # noqa: E402

from server import correction_segments as cs  # noqa: E402

NICE_LEVEL = 19
FORBIDDEN_NAMES = {"dexter_pick_place_v1", "dexter_pick_place_v2"}  # 1호 데이터셋과 본 실험용 이름은 쓰지 않는다


def check_corr_frames(corr_frames: Path, segs: list[cs.Segment], corr_dir: Path, demo_settings: dict, metas: list[dict]) -> list[str]:
    """구간 프레임이 지금의 교정 npz와 짝인지, 시연 프레임과 같은 렌더링 설정인지."""
    path = Path(corr_frames) / "manifest.json"
    if not path.is_file():
        return [f"{path} 가 없다. 먼저 server/.venv에서 python -m server.render_correction_frames 를 실행하세요"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if manifest["settings"] != demo_settings:
        problems.append("교정 프레임의 렌더링 설정(해상도·카메라·장면 해시·mujoco)이 시연 프레임과 다르다")
    for meta in metas:
        if meta["scene_sha256"] != manifest["settings"]["scene_sha256"] or meta["robot_model_sha256"] != manifest["settings"]["robot_model_sha256"]:
            problems.append(f"교정 {meta['episode_index']}: 프레임을 그릴 때와 장면/로봇 모델이 다르다")
    if len(manifest["episodes"]) != len(segs):
        problems.append(f"manifest의 구간 수 {len(manifest['episodes'])} != 지금 구간 수 {len(segs)}")
    for n, seg in enumerate(segs):
        entry = manifest["episodes"].get(str(n))
        if entry is None:
            problems.append(f"구간 {n}: manifest에 없다")
            continue
        qpos = cs.segment_qpos(corr_dir, seg)
        if (entry["source_index"], entry["start_sample"], entry["end_sample"]) != (seg.source_index, seg.start_sample, seg.end_sample):
            problems.append(f"구간 {n}: 교정 번호/샘플 범위가 manifest와 다르다")
        if entry["n_frames"] != len(qpos) or entry["qpos_sha256"] != bd.qpos_sha256(qpos):
            problems.append(f"구간 {n}: 프레임을 그린 뒤 qpos가 바뀌었거나 프레임 수가 다르다")
        folder = Path(corr_frames) / f"ep_{n:06d}"
        have = sum(1 for _ in folder.glob("frame_*.png")) if folder.is_dir() else 0
        if have != len(qpos):
            problems.append(f"구간 {n}: 프레임 파일 {have}개 != {len(qpos)}")
    return problems


def plan_build(args) -> dict:
    """검사를 모두 하고 계획(+문제 목록)을 돌려준다. 파일을 쓰지 않는다."""
    demo_metas_all = load_metas(args.raw)
    demo_by_index = {m["episode_index"]: m for m in demo_metas_all}
    demo_rows = select(demo_metas_all, set(args.exclude))
    demo_indices = [r["index"] for r in demo_rows if r["included"]]
    demo_metas = [demo_by_index[i] for i in demo_indices]
    problems = bd.check_frames(args.raw, args.frames, demo_by_index, demo_indices) if demo_indices else ["포함할 시연 에피소드가 없다"]
    demo_settings = json.loads((Path(args.frames) / "manifest.json").read_text(encoding="utf-8"))["settings"] if (Path(args.frames) / "manifest.json").is_file() else {}

    corr_metas = cs.load_correction_metas(args.corrections, set(args.exclude_corrections))
    if not corr_metas:
        problems.append("저장된 교정이 없다")
    segs, seg_problems = cs.all_segments(args.corrections, corr_metas)
    problems += seg_problems
    if not segs:
        problems.append("HUMAN 구간이 하나도 없다")
    if demo_settings:
        problems += check_corr_frames(args.corr_frames, segs, args.corrections, demo_settings, corr_metas)
        problems += cs.check_environment(
            corr_metas, expected_scene_sha256=demo_settings["scene_sha256"], expected_robot_sha256=demo_settings["robot_model_sha256"],
            reference={**cs.reference_from_demos(demo_metas), **({"checkpoint_sha256": args.expect_checkpoint_sha} if args.expect_checkpoint_sha else {})},
            allow_diff=set(args.allow_diff),
        )
    problems += cs.check_seed_separation(corr_metas, "교정") + cs.check_seed_separation(demo_metas, "시연")
    for csv_path in (Path(args.corrections) / "auto_runs.csv",):
        problems += cs.check_auto_runs_csv(csv_path)
    return {"demo_indices": demo_indices, "demo_by_index": demo_by_index, "segments": segs, "problems": problems,
            "demo_seeds": sorted({m["seed"] for m in demo_metas}), "corr_seeds": sorted({m["seed"] for m in corr_metas})}


def sources_table(plan: dict) -> list[dict]:
    """데이터셋 에피소드 번호 ↔ 출처. 시연 먼저(raw 번호 순), 그다음 교정 구간 순."""
    rows = []
    for i in plan["demo_indices"]:
        m = plan["demo_by_index"][i]
        rows.append({"kind": "demo", "source_index": i, "seed": m["seed"], "start_sample": 0, "end_sample": m["n_ticks"] - 1, "n_frames": m["n_ticks"]})
    for s in plan["segments"]:
        rows.append({"kind": "correction_human", "source_index": s.source_index, "seed": s.seed, "start_sample": s.start_sample,
                     "end_sample": s.end_sample, "n_frames": s.n_samples})
    for n, r in enumerate(rows):
        r["dataset_episode"] = n
    return rows


def build(args, plan: dict) -> dict:
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    settings = json.loads((Path(args.frames) / "manifest.json").read_text(encoding="utf-8"))["settings"]
    features = bd.make_features(settings["height_px"], settings["width_px"], args.image_mode)
    encoder = None
    if not args.image_mode:
        vcodec = args.codec or cfg.VIDEO_CODEC
        pix_fmt = "yuv420p" if vcodec == "libsvtav1" else cfg.VIDEO_PIX_FMT
        extra = dict(cfg.VIDEO_EXTRA_OPTIONS) if vcodec == cfg.VIDEO_CODEC else {"threads": cfg.VIDEO_EXTRA_OPTIONS.get("threads", "2")}
        encoder = RGBEncoderConfig(vcodec=vcodec, pix_fmt=pix_fmt, crf=args.crf, g=cfg.VIDEO_GOP, extra_options=extra)
    ds = LeRobotDataset.create(repo_id=f"local/{Path(args.out).name}", fps=cfg.FPS, features=features, root=args.out,
                               robot_type=cfg.ROBOT_TYPE, use_videos=not args.image_mode, rgb_encoder=encoder)
    rows = sources_table(plan)
    seg_by_key = {(s.source_index, s.start_sample): n for n, s in enumerate(plan["segments"])}
    for row in rows:
        if row["kind"] == "demo":
            arrays = load_arrays(args.raw, row["source_index"])
            frames_dir, frame_index, lo, hi = args.frames, row["source_index"], 0, len(arrays["qpos"])
        else:
            arrays = cs.load_correction_arrays(args.corrections, row["source_index"])
            frames_dir, frame_index = args.corr_frames, seg_by_key[(row["source_index"], row["start_sample"])]
            lo, hi = row["start_sample"], row["end_sample"] + 1
        state, action = bd.state_action_for({k: arrays[k][lo:hi] for k in ("qpos", "ctrl")})
        assert len(state) == row["n_frames"]
        for k in range(len(state)):
            image = np.asarray(Image.open(bd.frame_file(frames_dir, frame_index, k)).convert("RGB"), dtype=np.uint8)
            ds.add_frame({"observation.state": state[k], "action": action[k], cfg.IMAGE_KEY: image, "task": cfg.TASK})
        ds.save_episode()
        print(f"데이터셋 {row['dataset_episode']}번 <- {row['kind']} {row['source_index']} (시드 {row['seed']}, {row['n_frames']}프레임)", flush=True)
    ds.finalize()
    (Path(args.out) / "handover_sources.json").write_text(
        json.dumps({"note": "dexter_pipeline_test 같은 파이프라인 시험용 데이터셋이면 본 실험 결과로 쓰지 않는다", "episodes": rows,
                    "demo_seeds": plan["demo_seeds"], "correction_seeds": plan["corr_seeds"], "allowed_env_diff": sorted(args.allow_diff)},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"episodes": len(rows), "frames": sum(r["n_frames"] for r in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=cfg.RAW_DIR)
    parser.add_argument("--frames", type=Path, default=cfg.FRAMES_DIR)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--corr-frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude", type=lambda s: [int(x) for x in s.split(",") if x], default=[], help="뺄 시연 번호(쉼표). 실패는 항상 빠진다")
    parser.add_argument("--exclude-corrections", type=lambda s: [int(x) for x in s.split(",") if x], default=[], help="뺄 교정 번호(쉼표)")
    parser.add_argument("--allow-diff", type=lambda s: [x for x in s.split(",") if x], default=[], help=f"허락할 환경 차이. 가능: {', '.join(cs.ALLOWABLE_DIFF_FIELDS)}")
    parser.add_argument("--expect-checkpoint-sha", default=None)
    parser.add_argument("--crf", type=int, default=cfg.VIDEO_CRF)
    parser.add_argument("--image-mode", action="store_true")
    parser.add_argument("--codec", choices=["libaom-av1", "h264", "hevc", "libsvtav1"])
    parser.add_argument("--check-only", action="store_true", help="검사와 출처 표만 출력하고 만들지 않는다")
    args = parser.parse_args()
    unknown = set(args.allow_diff) - set(cs.ALLOWABLE_DIFF_FIELDS)
    if unknown:
        print(f"허락할 수 없는 항목: {sorted(unknown)}", file=sys.stderr)
        return 2
    if Path(args.out).name in FORBIDDEN_NAMES:
        print(f"{Path(args.out).name}는 쓰지 않는 이름이다(이번 시험은 dexter_pipeline_test)", file=sys.stderr)
        return 2
    try:
        os.nice(NICE_LEVEL)
    except OSError as err:
        print(f"nice 설정 실패(계속 진행): {err}", file=sys.stderr)

    plan = plan_build(args)
    rows = sources_table(plan)
    print(f"시연 {len(plan['demo_indices'])}개 + 교정 구간 {len(plan['segments'])}개 = 에피소드 {len(rows)}개, 프레임 {sum(r['n_frames'] for r in rows)}")
    for r in rows:
        if r["kind"] != "demo":
            print(f"  데이터셋 {r['dataset_episode']}번: 교정 {r['source_index']} 시드 {r['seed']} 샘플 {r['start_sample']}~{r['end_sample']} ({r['n_frames']}개)")
    print(f"시연 시드 {plan['demo_seeds']}, 교정 시드 {plan['corr_seeds']}, 평가 시드 범위 {cs.config.EVAL_SEED_RANGE}")
    if plan["problems"]:
        print("검사 실패. 만들지 않는다:\n  " + "\n  ".join(plan["problems"]), file=sys.stderr)
        return 3
    print("검사 통과(환경 해시, 평가 시드 분리, 프레임-npz 짝)")
    if args.check_only:
        return 0
    if Path(args.out).exists():
        print(f"{args.out} 가 이미 있다. 덮어쓰지 않는다", file=sys.stderr)
        return 2
    result = build(args, plan)
    print(f"완료: 에피소드 {result['episodes']}개, 프레임 {result['frames']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
