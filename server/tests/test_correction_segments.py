"""4-3: HUMAN 구간 자르기, 환경 해시 검사, 평가 시드 분리 검사(server/correction_segments.py)."""

import numpy as np
import pytest
from correction_fakes import CKPT_SHA, ROBOT_SHA, SCENE_SHA, make_correction

from server import config, seeds
from server import correction_config as cc
from server import correction_segments as cs

ENV = {"expected_scene_sha256": SCENE_SHA, "expected_robot_sha256": ROBOT_SHA}


def test_owner_human_code_matches_correction_config():
    assert cs.OWNER_HUMAN == cc.OWNER_HUMAN


def test_human_runs_splits_and_handles_edges():
    flags = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
    assert cs.human_runs(flags) == [(0, 1), (4, 4), (6, 7)]
    assert cs.human_runs(np.zeros(5, dtype=bool)) == []


def test_segments_cut_only_human_samples_and_wait_is_excluded(tmp_path):
    meta = make_correction(tmp_path, 0, 40, [(10, 17), (25, 39)])
    arrays = cs.load_correction_arrays(tmp_path, 0)
    segs, problems = cs.segments_for(meta, arrays)
    assert problems == []
    assert [(s.start_sample, s.end_sample, s.n_samples) for s in segs] == [(10, 17, 8), (25, 39, 15)]
    covered = np.zeros(40, dtype=bool)
    for s in segs:
        covered[s.start_sample : s.end_sample + 1] = True
    assert (arrays["owner"][covered] == cs.OWNER_HUMAN).all()
    assert not covered[9] and not covered[24]  # 바로 앞의 HUMAN_WAIT 샘플은 들어가지 않는다
    assert cs.segment_qpos(tmp_path, segs[0]).shape == (8, 13)


def test_segments_report_mismatches_instead_of_guessing(tmp_path):
    meta = make_correction(tmp_path, 0, 40, [(10, 17)])
    arrays = cs.load_correction_arrays(tmp_path, 0)
    bad_json = {**meta, "interventions": [{"start_sample": 10, "end_sample": 18, "n_samples": 9}]}
    assert any("interventions" in p for p in cs.segments_for(bad_json, arrays)[1])
    bad_flags = {**arrays, "is_intervention": arrays["is_intervention"].copy()}
    bad_flags["is_intervention"][3] = True
    assert any("owner" in p for p in cs.segments_for(meta, bad_flags)[1])
    assert any("n_ticks" in p for p in cs.segments_for({**meta, "n_ticks": 41}, arrays)[1])


def test_load_metas_only_saved_and_requires_npz(tmp_path):
    make_correction(tmp_path, 0, 40, [(5, 9)])
    make_correction(tmp_path, 1, 41, [(5, 9)], decision="discarded")
    make_correction(tmp_path, 2, 42, [(5, 9)])
    assert [m["episode_index"] for m in cs.load_correction_metas(tmp_path)] == [0, 2]
    assert [m["episode_index"] for m in cs.load_correction_metas(tmp_path, exclude={2})] == [0]
    (tmp_path / "correction_000002.npz").unlink()
    with pytest.raises(FileNotFoundError):
        cs.load_correction_metas(tmp_path)


def test_environment_check_passes_when_everything_matches(tmp_path):
    metas = [make_correction(tmp_path, i, 40 + i, [(5, 9)]) for i in range(3)]
    ref = {"mujoco_version": "3.16.0", "config_sha256": "d" * 64, "checkpoint_sha256": CKPT_SHA}
    assert cs.check_environment(metas, **ENV, reference=ref) == []


def test_environment_check_catches_each_difference(tmp_path):
    ok = make_correction(tmp_path, 0, 40, [(5, 9)])
    assert any("scene_sha256" in p for p in cs.check_environment([make_correction(tmp_path, 1, 41, [(5, 9)], scene="9" * 64)], **ENV))
    assert any("robot_model_sha256" in p for p in cs.check_environment([make_correction(tmp_path, 2, 42, [(5, 9)], robot="9" * 64)], **ENV))
    other_ckpt = make_correction(tmp_path, 3, 43, [(5, 9)], ckpt="9" * 64)
    assert any("base_checkpoint" in p for p in cs.check_environment([ok, other_ckpt], **ENV))
    assert any("base_checkpoint" in p for p in cs.check_environment([ok], **ENV, reference={"checkpoint_sha256": "9" * 64}))
    mac = make_correction(tmp_path, 4, 44, [(5, 9)], mujoco_version="3.2.0")
    assert any("mujoco_version" in p for p in cs.check_environment([mac], **ENV, reference={"mujoco_version": "3.16.0"}))
    assert any("config_sha256" in p for p in cs.check_environment([ok], **ENV, reference={"config_sha256": "f" * 64}))


def test_allow_diff_only_lifts_the_named_field_and_never_scene_or_checkpoint(tmp_path):
    mac = make_correction(tmp_path, 0, 40, [(5, 9)], mujoco_version="3.2.0")
    ref = {"mujoco_version": "3.16.0"}
    assert cs.check_environment([mac], **ENV, reference=ref, allow_diff={"mujoco_version"}) == []
    assert cs.check_environment([mac], **ENV, reference={"config_sha256": "f" * 64}, allow_diff={"mujoco_version"}) != []
    bad_scene = make_correction(tmp_path, 1, 41, [(5, 9)], scene="9" * 64)
    assert cs.check_environment([bad_scene], **ENV, allow_diff=set(cs.ALLOWABLE_DIFF_FIELDS)) != []
    assert "scene_sha256" not in cs.ALLOWABLE_DIFF_FIELDS and "checkpoint_sha256" not in cs.ALLOWABLE_DIFF_FIELDS


def test_reference_from_demos_keeps_only_fields_all_demos_agree_on():
    demos = [{"mujoco_version": "3.16.0", "config_sha256": "x"}, {"mujoco_version": "3.16.0", "config_sha256": "y"}]
    assert cs.reference_from_demos(demos) == {"mujoco_version": "3.16.0"}


def test_seed_separation_passes_for_correction_seeds(tmp_path):
    metas = [make_correction(tmp_path, i, 28 + i, [(5, 9)]) for i in range(3)]
    assert cs.check_seed_separation(metas, "교정") == []


def test_seed_separation_catches_eval_seed_out_of_range_and_wrong_position(tmp_path):
    lo, hi = config.EVAL_SEED_RANGE
    assert any("평가 시드 범위" in p for p in cs.check_seed_separation([make_correction(tmp_path, 0, lo + 5, [(5, 9)])], "교정"))
    assert any("밖" in p for p in cs.check_seed_separation([make_correction(tmp_path, 1, hi + 10, [(5, 9)], block_xy=(0.2, 0.0))], "교정"))
    wrong = make_correction(tmp_path, 2, 30, [(5, 9)], block_xy=(0.2, 0.0))  # 시드 30이 만드는 위치가 아니다
    assert any("시드가 만드는 위치와 다르다" in p for p in cs.check_seed_separation([wrong], "교정"))


def test_seed_separation_catches_demo_seed_with_eval_block_position(tmp_path):
    """시드 기록은 정상(40)인데 블록이 평가 시드 1007의 위치에 있는 경우(시드 기록이 틀린 것과 같은 효과)도 잡는다."""
    eval_xy = seeds.block_xy_for_seed(config.EVAL_SEED_RANGE[0] + 7)
    sneaky = make_correction(tmp_path, 0, 40, [(5, 9)], block_xy=eval_xy)
    problems = cs.check_seed_separation([sneaky], "교정")
    assert any("평가 시드 1007의 위치" in p for p in problems)


def test_auto_runs_csv_check(tmp_path):
    header = "seed,kind,outcome\n"
    good, bad = tmp_path / "good.csv", tmp_path / "auto_runs.csv"
    good.write_text(header + "28,correction_saved,success\n30,auto_only,success\n", encoding="utf-8")
    bad.write_text(header + "28,correction_saved,success\n1003,auto_only,success\n", encoding="utf-8")
    assert cs.check_auto_runs_csv(good) == []
    assert any("1003" in p for p in cs.check_auto_runs_csv(bad))
    assert cs.check_auto_runs_csv(tmp_path / "none.csv") == []
