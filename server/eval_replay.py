"""정답 재생 검사(3-4): 시연의 ctrl을 평가 루프에서 그대로 재생해서, 평가 루프가 녹화와 어긋나 있지 않은지 본다.

시연 k번째 샘플의 ctrl을 평가 샘플 k에 넣고(ReplayPolicy) 시연과 같은 시드의 블록 위치에서 시연 길이만큼 돌린 뒤 비교한다:
성공 재현 여부, 마지막 샘플의 블록 위치 차이(mm), 팔 관절 궤적 차이(rad), 시작 자세 차이.
시작 자세: home(평가 기본)과 recorded(시연의 첫 샘플 자세, 시연 4개가 home이 아닌 자세에서 시작했기 때문)를 둘 다 돌릴 수 있다.
물리 비결정성: 같은 입력이면 mujoco는 같은 결과를 내지만, 시연은 ctrl이 틱 경계에서 바뀌었고(격자와 최대 한 스텝=5 ms 차이) 접촉 물리가 민감해서 완전히 같지는 않을 수 있다.

참고 기준(합격선이 아님): 처음에는 "성공 20개 중 15개 이상 재생 성공, 블록 최종 위치 차이 1 cm 이하"를 합격선으로 정했다. 사용자가 이를 철회했다:
그 기준은 사전에 정한 대략적인 휴리스틱이었고, 열린 고리 재생은 접촉에 민감해서(ctrl에 1e-6 rad 노이즈만 줘도 결과가 바뀐다) 합격선으로 부적절하다.
이 스크립트의 재생 성공률(12/20)은 학습 정책의 성공률과 나란히 보고하는 "열린 고리 기준선"이다(docs/07-lerobot-notes.md 12절). 아래 참고 기준 출력은 그 값과 비교하려고 남겨 두었다.
이 재생은 **시연 시드(0~99)** 의 블록 위치에서 돌린다. 평가 시드(1000~1049)와 다르다.

사용:
  MUJOCO_GL=egl server/.venv/bin/python -m server.eval_replay [--start home|recorded|both] [--boundary grid|floor|ceil] [--json 결과.json]
  server/.venv/bin/python -m server.eval_replay --start home --out experiments/replay_baseline        # 기준선 저장(results.csv, summary.json)
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from server import config  # noqa: E402
from server import eval_config as ec  # noqa: E402
from server.episode_store import EpisodeStore  # noqa: E402
from server.eval_env import EvalEnv, ReplayPolicy, rollout  # noqa: E402
from server.eval_metrics import distribution  # noqa: E402
from server.grid import grid_step_index  # noqa: E402
from server.success import success_over_time  # noqa: E402

MIN_SUCCESSES = 15
MAX_BLOCK_DIFF_MM = 10.0


def replay_episode(env: EvalEnv, arrays: dict[str, np.ndarray], meta: dict, start: str) -> dict:
    """에피소드 하나를 재생하고 시연과 비교한 값을 돌려준다."""
    ctrl, qpos_demo = arrays["ctrl"], arrays["qpos"]
    n = len(ctrl)
    init_arm = qpos_demo[0, :6] if start == "recorded" else None
    res = rollout(env, ReplayPolicy(ctrl), meta["seed"], stop_on_success=False, max_samples=n - 1, init_arm=init_arm, record_trace=True)
    trace = res.trace
    assert trace is not None and trace.shape == qpos_demo.shape, (trace.shape, qpos_demo.shape)
    t_replay = np.array([grid_step_index(k, ec.EVAL_FPS, env.timestep_s) * env.timestep_s for k in range(n)])
    ever, first_s, _ = success_over_time(t_replay, trace)
    demo_first = meta["auto_success"]["first_sim_time_s"]
    demo_first_rel = None if demo_first is None else float(demo_first - arrays["sim_time_s"][0])
    block_diff_mm = np.linalg.norm(trace[:, 6:9] - qpos_demo[:, 6:9], axis=1) * 1000.0
    # 보조 지표: 시연이 처음 성공을 채운 샘플에서의 블록 위치 차이. 최종 차이는 성공 뒤 팔이 물러나며 블록을 건드리는 것에 좌우될 수 있다.
    demo_ever, demo_first_s, _ = success_over_time(arrays["sim_time_s"], qpos_demo)
    k_success = int(np.argmax(arrays["sim_time_s"] >= demo_first_s)) if demo_ever else None
    arm_diff = trace[:, :6] - qpos_demo[:, :6]
    return {
        "episode": meta["episode_index"], "seed": meta["seed"], "start": start, "n_samples": n,
        "demo_success": bool(meta["success"]), "replay_success": bool(ever),
        "replay_first_success_s": first_s, "demo_first_success_s": demo_first_rel,
        "final_block_diff_mm": float(block_diff_mm[-1]),
        "final_block_diff_xy_mm": float(np.linalg.norm(trace[-1, 6:8] - qpos_demo[-1, 6:8]) * 1000.0),
        "block_diff_at_demo_success_mm": None if k_success is None else float(block_diff_mm[k_success]),
        "max_block_diff_mm": float(block_diff_mm.max()),
        "arm_rms_rad": float(np.sqrt((arm_diff**2).mean())),
        "arm_max_rad": float(np.abs(arm_diff).max()),
        "start_arm_diff_rad": float(np.abs(arm_diff[0]).max()),
        # 실패 유형(평가와 같은 분류기)과 집기/놓기 단계를 가르는 값: 블록 최대 높이, 마지막 블록의 목표 중심까지 거리
        "failure": res.failure,
        "max_block_z_m": float(res.max_block_z_m),
        "lifted": bool(res.max_block_z_m >= ec.LIFT_Z_MIN_M),
        "final_block_dist_to_target_mm": float(np.hypot(res.final_block_m[0] - config.TARGET_CENTER_M[0], res.final_block_m[1] - config.TARGET_CENTER_M[1]) * 1000.0),
        "replay_final_block_m": [round(float(v), 5) for v in trace[-1, 6:9]],
        "demo_final_block_m": [round(float(v), 5) for v in qpos_demo[-1, 6:9]],
    }


def summarize(rows: list[dict]) -> dict:
    ok = sum(r["replay_success"] for r in rows)
    diff = [r["final_block_diff_mm"] for r in rows]
    at_success = [r["block_diff_at_demo_success_mm"] for r in rows if r["block_diff_at_demo_success_mm"] is not None]
    return {
        "block_diff_at_demo_success_mm": distribution(at_success),
        "n": len(rows),
        "replay_successes": int(ok),
        "final_block_diff_mm": distribution(diff),
        "n_block_diff_within_10mm": int(sum(d <= MAX_BLOCK_DIFF_MM for d in diff)),
        "arm_rms_rad": distribution([r["arm_rms_rad"] for r in rows]),
        "failure_counts": {k: sum(1 for r in rows if r["failure"] == k) for k in ["success", *ec.FAILURE_KEYS]},
        # 참고 기준을 채웠는지(합격선이 아님. 모듈 설명 참고). 키 이름은 호환을 위해 그대로 둔다.
        "trusted": bool(ok >= MIN_SUCCESSES * len(rows) / 20 and max(diff) <= MAX_BLOCK_DIFF_MM),
    }


REPLAY_CSV_COLUMNS = ["episode", "seed", "replay_success", "replay_first_success_s", "demo_first_success_s", "final_block_diff_mm", "failure",
                      "lifted", "max_block_z_m", "final_block_dist_to_target_mm"]


def write_baseline(out_dir: Path, rows: list[dict], meta: dict) -> dict:
    """열린 고리 재생 기준선을 out_dir/results.csv, summary.json으로 저장한다(이미 있으면 덮어쓰지 않고 예외)."""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "results.csv").exists() or (out_dir / "summary.json").exists():
        raise FileExistsError(f"{out_dir} 에 이미 결과가 있습니다")
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPLAY_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in REPLAY_CSV_COLUMNS})
    summary = {**meta, "summary": summarize(rows)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=config.RECORD_DIR)
    parser.add_argument("--start", choices=["home", "recorded", "both"], default="both")
    parser.add_argument("--boundary", choices=["grid", "floor", "ceil"], default="grid", help="ctrl이 바뀌는 스텝 경계(진단용, 기본은 녹화 격자)")
    parser.add_argument("--episodes", default="", help="raw 에피소드 번호(쉼표). 기본은 성공 전부")
    parser.add_argument("--json", type=Path, help="결과를 JSON으로 저장")
    parser.add_argument("--out", type=Path, help="기준선 폴더에 results.csv, summary.json 저장(--start home 또는 recorded 하나만)")
    args = parser.parse_args()
    if args.out and args.start == "both":
        parser.error("--out은 --start home 또는 recorded 하나와 함께 씁니다")
    try:
        os.nice(ec.NICE_LEVEL)
    except OSError:
        pass

    store = EpisodeStore(args.raw)
    metas = [m for m in store.scan() if m["success"]]
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",") if x.strip()}
        metas = [m for m in store.scan() if m["episode_index"] in wanted]
    starts = ["home", "recorded"] if args.start == "both" else [args.start]
    t0 = time.perf_counter()
    all_rows: dict[str, list[dict]] = {s: [] for s in starts}
    with EvalEnv(with_renderer=False, boundary=args.boundary) as env:
        for start in starts:
            print(f"\n[시작 자세 {start}, 경계 {args.boundary}] 에피소드 {len(metas)}개")
            print("| raw | 시드 | 재생 성공 | 첫 성공(재생/시연 s) | 블록 최종 차이 mm (xy) | 블록 최대 차이 mm | 팔 RMS rad | 시작 팔 차이 rad |")
            print("|---|---|---|---|---|---|---|---|")
            for m in metas:
                row = replay_episode(env, store.load_arrays(m["episode_index"]), m, start)
                all_rows[start].append(row)
                fs = "-" if row["replay_first_success_s"] is None else f"{row['replay_first_success_s']:.2f}"
                print(f"| {row['episode']} | {row['seed']} | {'O' if row['replay_success'] else 'X'} | {fs} / {row['demo_first_success_s']:.2f} | "
                      f"{row['final_block_diff_mm']:.1f} ({row['final_block_diff_xy_mm']:.1f}) | {row['max_block_diff_mm']:.1f} | "
                      f"{row['arm_rms_rad']:.4f} | {row['start_arm_diff_rad']:.4f} |")
    out: dict = {"boundary": args.boundary, "elapsed_s": round(time.perf_counter() - t0, 1), "runs": {}}
    for start, rows in all_rows.items():
        s = summarize(rows)
        out["runs"][start] = {"summary": s, "episodes": rows}
        d = s["final_block_diff_mm"]
        print(f"\n[{start}] 재생 성공 {s['replay_successes']}/{s['n']}, 블록 최종 위치 차이 mm (최소/25%/중앙/75%/최대) "
              f"{d['min']}/{d['p25']}/{d['median']}/{d['p75']}/{d['max']}, 10 mm 이내 {s['n_block_diff_within_10mm']}/{s['n']}, "
              f"팔 RMS 중앙값 {s['arm_rms_rad']['median']} rad (참고 기준 {'충족' if s['trusted'] else '미충족'}, 합격선 아님)")
        a = s["block_diff_at_demo_success_mm"]
        print(f"[{start}] (보조) 시연이 처음 성공한 샘플에서의 블록 위치 차이 mm (최소/25%/중앙/75%/최대) "
              f"{a['min']}/{a['p25']}/{a['median']}/{a['p75']}/{a['max']}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out:
        start = starts[0]
        meta = {
            "run": args.out.name, "kind": "open_loop_replay_baseline", "start": start, "boundary": args.boundary,
            "seeds": "시연 시드(0~99)의 블록 위치. 평가 시드(1000~1049)와 다르다", "demo_seeds": [r["seed"] for r in all_rows[start]],
            "note": "시연의 ctrl을 그대로 재생한 열린 고리 기준선. 접촉에 민감해서 합격선이 아니라 학습 정책 성공률과 나란히 보고하는 기준선이다.",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__},
        }
        write_baseline(args.out, all_rows[start], meta)
        print(f"기준선 저장: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
