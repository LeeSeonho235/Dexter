"""평가 지표: Wilson 신뢰구간, 실패 유형 판정, 요약, 결과 파일(CSV/JSON) 쓰기. numpy와 표준 라이브러리만 쓴다(server/.venv, train/.venv 어디서나).

실패 유형 판정 규칙(위에서부터 먼저 맞는 것):
  success      최대 시간 안에 성공 조건(블록이 목표 안 + 바닥 + 집게 열림)이 0.5초 유지됨
  grasp_fail   블록 높이가 끝까지 LIFT_Z_MIN_M 미만(안 들림). 블록이 PUSHED_XY_M 넘게 밀렸으면 pushed 표시
  timeout      끝날 때 블록이 공중(z >= 성공 높이 기준)이거나, 목표 안 바닥에 있는데 판정이 안 됨(집게 안 열림, 유지 부족)
  target_miss  한 번 들렸고, 끝날 때 바닥에 있으며 목표 중심에서 NEAR_TARGET_M 이내(성공 영역 밖)
  dropped      한 번 들렸고, 끝날 때 바닥에 있으며 목표 중심에서 NEAR_TARGET_M 밖
  other        위에 안 걸림
"""

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from server import config
from server import eval_config as ec
from server.success import is_success

CSV_COLUMNS = [
    "seed", "success", "failure", "pushed", "time_to_success_s", "sim_s", "n_samples", "block_start_x_m", "block_start_y_m",
    "final_block_x_m", "final_block_y_m", "final_block_z_m", "final_gripper_rad", "max_block_z_m", "max_block_move_m",
    "limited_samples", "policy_calls", "render_s", "policy_s", "physics_s",
]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """성공 비율의 Wilson 점수 신뢰구간(기본 95%). n이 0이면 (nan, nan)."""
    if n <= 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def classify_failure(
    success: bool,
    max_block_z_m: float,
    final_block_m: Sequence[float],
    block_start_m: Sequence[float],
    *,
    lift_z_min_m: float = ec.LIFT_Z_MIN_M,
    pushed_xy_m: float = ec.PUSHED_XY_M,
    near_target_m: float = ec.NEAR_TARGET_M,
    max_block_move_m: float | None = None,
) -> tuple[str, bool]:
    """(유형 키, 밀침 여부). 성공이면 ("success", False). 밀침은 grasp_fail일 때만 True가 될 수 있다."""
    if success:
        return "success", False
    if max_block_z_m < lift_z_min_m:
        if max_block_move_m is None:
            moved = math.hypot(final_block_m[0] - block_start_m[0], final_block_m[1] - block_start_m[1])
        else:
            moved = max_block_move_m
        return "grasp_fail", moved >= pushed_xy_m
    on_floor = final_block_m[2] < config.SUCCESS_BLOCK_Z_MAX_M
    if not on_floor:
        return "timeout", False
    if is_success(final_block_m, 10.0):  # 집게 조건을 빼고 위치·높이만 본다: 목표 안 바닥인데 판정이 안 됐다
        return "timeout", False
    dist_m = math.hypot(final_block_m[0] - config.TARGET_CENTER_M[0], final_block_m[1] - config.TARGET_CENTER_M[1])
    return ("target_miss" if dist_m <= near_target_m else "dropped"), False


def distribution(values: Sequence[float]) -> dict[str, float] | None:
    if len(values) == 0:
        return None
    q = np.percentile(np.asarray(values, dtype=float), [0, 25, 50, 75, 100])
    return dict(zip(["min", "p25", "median", "p75", "max"], (round(float(x), 3) for x in q), strict=True))


def summarize(rows: Sequence[dict]) -> dict:
    """에피소드별 결과 행(dict, CSV_COLUMNS 키)을 요약한다."""
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    lo, hi = wilson_interval(len(ok), n)
    failures = {k: 0 for k in ec.FAILURE_KEYS}
    for r in rows:
        if not r["success"]:
            failures[r["failure"]] = failures.get(r["failure"], 0) + 1
    calls = sum(r["policy_calls"] for r in rows)
    samples = sum(r["n_samples"] for r in rows)
    return {
        "n": n,
        "successes": len(ok),
        "success_rate": (len(ok) / n) if n else None,
        "wilson95": [round(lo, 4), round(hi, 4)] if n else None,
        "time_to_success_s": distribution([r["time_to_success_s"] for r in ok]),
        "failure_counts": failures,
        "grasp_fail_pushed": sum(1 for r in rows if r["failure"] == "grasp_fail" and r["pushed"]),
        "limited_sample_fraction": (sum(r["limited_samples"] for r in rows) / samples) if samples else None,
        "timing_s": {
            "render_total": round(sum(r["render_s"] for r in rows), 3),
            "policy_total": round(sum(r["policy_s"] for r in rows), 3),
            "physics_total": round(sum(r["physics_s"] for r in rows), 3),
            "policy_calls": calls,
            "samples": samples,
        },
    }


def write_results(out_dir: Path, rows: Sequence[dict], meta: dict) -> dict:
    """out_dir/results.csv, summary.json을 쓴다. 이미 results.csv가 있으면 덮어쓰지 않고 예외."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / "results.csv", out_dir / "summary.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"{out_dir} 에 이미 결과가 있습니다. 다른 --out을 쓰세요")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k in CSV_COLUMNS})
    summary = {**meta, "summary": summarize(rows)}
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return summary
