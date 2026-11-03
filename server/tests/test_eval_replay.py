"""정답 재생 검사(server/eval_replay.py): 요약·신뢰 판정 규칙, 그리고 재생이 시연과 같은 결과를 내는 에피소드의 회귀 방지.

주의: 20개 전체의 재생은 신뢰 기준(성공 15개 이상, 블록 최종 차이 1 cm 이하)을 **통과하지 못한다**(docs/07 평가 절).
여기서는 기준을 통과하는 척하지 않고, 재생이 시연과 거의 같게 나오는 에피소드 몇 개와 판정 규칙만 확인한다.
"""

import numpy as np
import pytest

from server import config
from server.episode_store import EpisodeStore
from server.eval_env import EvalEnv
from server.eval_replay import MAX_BLOCK_DIFF_MM, MIN_SUCCESSES, replay_episode, summarize

HAVE_RAW = (config.RECORD_DIR / "episode_000019.npz").exists()


def fake_row(success=True, diff=3.0):
    return {"replay_success": success, "final_block_diff_mm": diff, "block_diff_at_demo_success_mm": diff, "arm_rms_rad": 0.01,
            "failure": "success" if success else "grasp_fail"}


def test_trust_rule_needs_15_successes_and_block_diff_within_1cm():
    assert summarize([fake_row() for _ in range(20)])["trusted"]
    assert summarize([fake_row() for _ in range(15)] + [fake_row(False) for _ in range(5)])["trusted"]
    assert not summarize([fake_row() for _ in range(14)] + [fake_row(False) for _ in range(6)])["trusted"]
    assert not summarize([fake_row() for _ in range(19)] + [fake_row(True, MAX_BLOCK_DIFF_MM + 1)])["trusted"]
    assert MIN_SUCCESSES == 15 and MAX_BLOCK_DIFF_MM == 10.0


def test_write_baseline_files_and_no_overwrite(tmp_path):
    import csv
    import json

    from server.eval_replay import REPLAY_CSV_COLUMNS, write_baseline

    rows = [{**fake_row(), "episode": 1, "seed": 1, "replay_first_success_s": 9.5, "demo_first_success_s": 9.5, "failure": "success",
             "lifted": True, "max_block_z_m": 0.05, "final_block_dist_to_target_mm": 3.0, "demo_success": True},
            {**fake_row(False, 50.0), "episode": 2, "seed": 2, "replay_first_success_s": None, "demo_first_success_s": 20.0, "failure": "grasp_fail",
             "lifted": False, "max_block_z_m": 0.02, "final_block_dist_to_target_mm": 150.0, "demo_success": True}]
    summary = write_baseline(tmp_path / "rb", rows, {"run": "rb", "kind": "open_loop_replay_baseline"})
    assert summary["summary"]["replay_successes"] == 1 and summary["summary"]["failure_counts"]["grasp_fail"] == 1
    read = list(csv.DictReader((tmp_path / "rb" / "results.csv").open(encoding="utf-8")))
    assert list(read[0]) == REPLAY_CSV_COLUMNS and [r["failure"] for r in read] == ["success", "grasp_fail"]
    assert json.loads((tmp_path / "rb" / "summary.json").read_text(encoding="utf-8"))["kind"] == "open_loop_replay_baseline"
    with pytest.raises(FileExistsError):
        write_baseline(tmp_path / "rb", rows, {})


@pytest.fixture(scope="module")
def store_env():
    with EvalEnv(with_renderer=False) as env:
        yield EpisodeStore(), env


@pytest.mark.skipif(not HAVE_RAW, reason="data/raw가 이 컴퓨터에 없음")
@pytest.mark.parametrize("episode", [19, 21, 17])  # 재생이 시연과 거의 같게 나오는 에피소드(실측 블록 최종 차이 0.3, 2.1, 2.3 mm)
def test_replay_reproduces_these_demos(store_env, episode):
    store, env = store_env
    meta = next(m for m in store.scan() if m["episode_index"] == episode)
    row = replay_episode(env, store.load_arrays(episode), meta, "home")
    assert row["start_arm_diff_rad"] < 1e-3  # 시작 자세 일치
    assert row["replay_success"] and row["final_block_diff_mm"] < 10.0
    # 주의: 성공한 재생도 블록 최대 높이가 16.6~30.6 mm(바닥 위 중심 15 mm)라 3 cm(LIFT_Z_MIN_M)를 넘지 않는 경우가 대부분이다(docs/07 12절).
    assert row["failure"] == "success" and 0.015 <= row["max_block_z_m"] < 0.06
    assert abs(row["replay_first_success_s"] - row["demo_first_success_s"]) < 0.2  # 성공 시각이 0.2초 이내로 같다
    assert row["arm_rms_rad"] < 0.05


@pytest.mark.skipif(not HAVE_RAW, reason="data/raw가 이 컴퓨터에 없음")
def test_replay_is_deterministic_and_start_pose_variant_is_consistent(store_env):
    store, env = store_env
    meta = next(m for m in store.scan() if m["episode_index"] == 21)
    a = replay_episode(env, store.load_arrays(21), meta, "home")
    b = replay_episode(env, store.load_arrays(21), meta, "home")
    c = replay_episode(env, store.load_arrays(21), meta, "recorded")
    assert a == b
    assert c["start_arm_diff_rad"] == 0.0 and abs(c["final_block_diff_mm"] - a["final_block_diff_mm"]) < 1.0
    assert np.isfinite(a["max_block_diff_mm"])
