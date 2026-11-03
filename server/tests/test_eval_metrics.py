"""평가 지표: Wilson 신뢰구간, 실패 유형 판정, 요약, 결과 파일."""

import csv
import json
import math

import pytest

from server import config
from server import eval_config as ec
from server.eval_metrics import CSV_COLUMNS, classify_failure, distribution, summarize, wilson_interval, write_results

START = (0.22, 0.03, 0.015)
TARGET = (*config.TARGET_CENTER_M, 0.015)


def test_wilson_known_values():
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0 and hi == pytest.approx(0.4345, abs=1e-3)
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    lo, hi = wilson_interval(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-3) and hi == 1.0
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_wilson_is_narrower_for_larger_n_and_contains_the_rate():
    lo50, hi50 = wilson_interval(25, 50)
    lo10, hi10 = wilson_interval(5, 10)
    assert (hi50 - lo50) < (hi10 - lo10)
    assert lo50 < 0.5 < hi50


def test_success_is_success():
    assert classify_failure(True, 0.05, TARGET, START) == ("success", False)


def test_never_lifted_is_grasp_fail_and_pushed_is_flagged():
    assert classify_failure(False, 0.015, START, START) == ("grasp_fail", False)
    pushed_to = (START[0] + 0.05, START[1], 0.015)
    assert classify_failure(False, 0.016, pushed_to, START) == ("grasp_fail", True)
    # 도중에 멀리 밀렸다가 돌아온 경우도 max_block_move_m로 밀침 표시
    assert classify_failure(False, 0.016, START, START, max_block_move_m=0.04) == ("grasp_fail", True)


def test_block_in_the_air_at_the_end_is_timeout():
    assert classify_failure(False, 0.08, (0.2, 0.1, 0.06), START) == ("timeout", False)


def test_block_in_target_but_not_held_is_timeout():
    inside = (config.TARGET_CENTER_M[0] + 0.01, config.TARGET_CENTER_M[1], 0.015)  # 목표 안 바닥(집게 조건만 안 맞은 경우)
    assert classify_failure(False, 0.05, inside, START) == ("timeout", False)


def test_dropped_vs_target_miss_by_distance_to_target():
    near = (config.TARGET_CENTER_M[0], config.TARGET_CENTER_M[1] - 0.05, 0.015)  # 목표 중심에서 5 cm
    far = (config.TARGET_CENTER_M[0], config.TARGET_CENTER_M[1] - 0.15, 0.015)  # 15 cm
    assert ec.NEAR_TARGET_M == 0.06
    assert classify_failure(False, 0.05, near, START)[0] == "target_miss"
    assert classify_failure(False, 0.05, far, START)[0] == "dropped"


def test_every_failure_key_is_known():
    for args in [(False, 0.015, START, START), (False, 0.08, (0.2, 0.1, 0.06), START), (False, 0.05, (0.2, -0.1, 0.015), START)]:
        assert classify_failure(*args)[0] in ec.FAILURE_KEYS


def row(seed, success, failure="success", t=None, limited=0, samples=100):
    return {"seed": seed, "success": success, "failure": failure, "pushed": False, "time_to_success_s": t, "sim_s": 1.0, "n_samples": samples,
            "block_start_x_m": 0.2, "block_start_y_m": 0.0, "final_block_x_m": 0.2, "final_block_y_m": 0.12, "final_block_z_m": 0.015,
            "final_gripper_rad": 0.9, "max_block_z_m": 0.05, "max_block_move_m": 0.1, "limited_samples": limited, "policy_calls": 2,
            "render_s": 0.1, "policy_s": 0.2, "physics_s": 0.05}


def test_summary_counts_rate_interval_and_times():
    rows = [row(1000, True, t=10.0), row(1001, True, t=20.0), row(1002, False, "dropped"), row(1003, False, "timeout", limited=10)]
    s = summarize(rows)
    assert (s["n"], s["successes"], s["success_rate"]) == (4, 2, 0.5)
    assert s["wilson95"][0] < 0.5 < s["wilson95"][1]
    assert s["time_to_success_s"]["median"] == 15.0
    assert s["failure_counts"]["dropped"] == 1 and s["failure_counts"]["timeout"] == 1 and s["failure_counts"]["grasp_fail"] == 0
    assert s["limited_sample_fraction"] == pytest.approx(10 / 400)
    assert s["timing_s"]["policy_calls"] == 8


def test_distribution_empty_is_none():
    assert distribution([]) is None
    assert distribution([1, 2, 3, 4, 5])["median"] == 3.0


def test_write_results_files_and_no_overwrite(tmp_path):
    rows = [row(1000, True, t=12.5), row(1001, False, "grasp_fail")]
    summary = write_results(tmp_path / "run1", rows, {"run": "run1", "policy": "hold"})
    assert summary["run"] == "run1" and summary["summary"]["successes"] == 1
    with (tmp_path / "run1" / "results.csv").open(encoding="utf-8") as f:
        read = list(csv.DictReader(f))
    assert list(read[0].keys()) == CSV_COLUMNS and [r["seed"] for r in read] == ["1000", "1001"]
    assert json.loads((tmp_path / "run1" / "summary.json").read_text(encoding="utf-8"))["summary"]["n"] == 2
    with pytest.raises(FileExistsError):
        write_results(tmp_path / "run1", rows, {})
