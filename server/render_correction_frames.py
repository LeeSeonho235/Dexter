"""4-3: 교정(data/corrections 또는 복사해 온 폴더)의 HUMAN 구간만 qpos로 다시 그려 PNG로 저장한다. server/.venv에서 실행한다.

- 3-3의 server/render_frames.py와 같은 렌더러·설정·manifest 형식을 쓴다. 구간 하나 = `ep_<구간번호>` 폴더(번호는 0부터, 교정 번호와 다르다).
- 입력(--corrections)은 읽기만 한다. 출력(--out)은 반드시 직접 적는다(data/frames에 섞이지 않게 기본값이 없다). 비어 있지 않은 manifest와 설정이 다르면 멈춘다.
- 그리기 전에 검사한다: 구간 배열 정합, 장면/로봇/체크포인트/환경 해시, 시드·블록 위치가 평가 시드 1000~1049와 섞이지 않는지.
  하나라도 걸리면 아무것도 그리지 않고 종료 코드 3으로 멈춘다(우회하지 않는다). 환경 해시 일부는 --allow-diff 이름으로 허락할 수 있다.
- 웹캠 영상은 없다(시뮬레이션 화면뿐). 공용 서버라 nice 19, 렌더 스레드 2개.

사용:
  server/.venv/bin/python -m server.render_correction_frames --corrections <교정 폴더> --out <프레임 폴더> [--check-only]
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LP_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from server import correction_segments as cs  # noqa: E402
from server.render import StateRenderer  # noqa: E402
from server.render_frames import (  # noqa: E402
    MANIFEST_VERSION,
    episode_is_done,
    qpos_sha256,
    read_manifest,
    render_episode,
    render_settings,
    write_manifest,
)

NICE_LEVEL = 19


def plan_segments(corr_dir: Path, exclude: set[int], settings: dict, allow_diff: set[str], expect_checkpoint_sha: str | None = None) -> tuple[list[dict], list[str]]:
    """(구간 계획, 문제 목록). 계획 항목: seg_index, source_index, seed, start_sample, end_sample, n_samples. 문제가 있으면 그리지 않는다."""
    metas = cs.load_correction_metas(corr_dir, exclude)
    if not metas:
        return [], ["저장된 교정이 없다(user_decision == saved인 것)"]
    reference = {"mujoco_version": settings["mujoco_version"]}
    if expect_checkpoint_sha:
        reference["checkpoint_sha256"] = expect_checkpoint_sha
    problems = cs.check_environment(
        metas, expected_scene_sha256=settings["scene_sha256"], expected_robot_sha256=settings["robot_model_sha256"],
        reference=reference, allow_diff=allow_diff,
    )
    problems += cs.check_seed_separation(metas, "교정")
    problems += cs.check_auto_runs_csv(Path(corr_dir) / "auto_runs.csv")
    segs, seg_problems = cs.all_segments(corr_dir, metas)
    problems += seg_problems
    if not segs:
        problems.append("HUMAN 구간이 하나도 없다")
    plan = [
        {"seg_index": n, "source_index": s.source_index, "seed": s.seed, "start_sample": s.start_sample, "end_sample": s.end_sample, "n_samples": s.n_samples}
        for n, s in enumerate(segs)
    ]
    return plan, problems


def render_plan(plan: list[dict], corr_dir: Path, out: Path, settings: dict, *, force: bool = False, log=print) -> int:
    """plan을 out에 그린다. 끝난 구간은 건너뛴다(같은 qpos 해시, 같은 프레임 수). 새로 그린 구간 수를 돌려준다."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(out)
    if manifest is None:
        manifest = {"manifest_version": MANIFEST_VERSION, "settings": settings, "episodes": {}}
    elif manifest["settings"] != settings:
        raise SystemExit(f"{out}/manifest.json의 렌더링 설정이 지금과 다르다. 다른 --out을 쓰세요")
    rendered = 0
    with StateRenderer() as renderer:
        for item in plan:
            seg = cs.Segment(item["source_index"], item["seed"], item["start_sample"], item["end_sample"], item["n_samples"])
            qpos = cs.segment_qpos(corr_dir, seg)
            idx = item["seg_index"]
            entry_ok = episode_is_done(manifest, out, idx, qpos) and all(
                manifest["episodes"][str(idx)].get(k) == item[k] for k in ("source_index", "start_sample", "end_sample")
            )
            if entry_ok and not force:
                log(f"구간 {idx}: 이미 끝남, 건너뜀")
                continue
            t0 = time.perf_counter()
            size_bytes = render_episode(renderer, qpos, out, idx)
            manifest["episodes"][str(idx)] = {
                **{k: item[k] for k in ("seed", "source_index", "start_sample", "end_sample")},
                "n_frames": len(qpos), "bytes": size_bytes, "render_s": round(time.perf_counter() - t0, 1),
                "qpos_sha256": qpos_sha256(qpos), "npz": f"correction_{seg.source_index:06d}.npz",
            }
            write_manifest(out, manifest)
            rendered += 1
            log(f"구간 {idx}(교정 {seg.source_index}, 시드 {seg.seed}, 샘플 {seg.start_sample}~{seg.end_sample}): {len(qpos)}장 {time.perf_counter() - t0:.0f}s")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corrections", type=Path, required=True, help="교정 폴더(읽기만 한다)")
    parser.add_argument("--out", type=Path, required=True, help="구간 프레임 폴더(data/frames가 아닌 새 폴더)")
    parser.add_argument("--exclude", default="", help="뺄 교정 번호(쉼표)")
    parser.add_argument("--allow-diff", default="", help=f"허락할 환경 차이(쉼표). 가능: {', '.join(cs.ALLOWABLE_DIFF_FIELDS)}")
    parser.add_argument("--expect-checkpoint-sha", default=None, help="교정이 기대하는 1호 model.safetensors SHA-256")
    parser.add_argument("--check-only", action="store_true", help="검사와 구간 표만 출력하고 그리지 않는다")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        os.nice(NICE_LEVEL)
    except OSError as err:
        print(f"nice 설정 실패(계속 진행): {err}", file=sys.stderr)

    allow = {x for x in args.allow_diff.split(",") if x}
    unknown = allow - set(cs.ALLOWABLE_DIFF_FIELDS)
    if unknown:
        print(f"허락할 수 없는 항목: {sorted(unknown)}", file=sys.stderr)
        return 2
    settings = render_settings()
    plan, problems = plan_segments(args.corrections, {int(x) for x in args.exclude.split(",") if x}, settings, allow, args.expect_checkpoint_sha)
    if allow:
        print(f"허락한 환경 차이: {sorted(allow)}")
    for item in plan:
        print(f"구간 {item['seg_index']}: 교정 {item['source_index']} 시드 {item['seed']} 샘플 {item['start_sample']}~{item['end_sample']} ({item['n_samples']}개)")
    if problems:
        print("검사 실패. 그리지 않는다:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 3
    print(f"검사 통과: 구간 {len(plan)}개, 프레임 {sum(p['n_samples'] for p in plan)}장")
    if args.check_only:
        return 0
    render_plan(plan, args.corrections, args.out, settings, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
