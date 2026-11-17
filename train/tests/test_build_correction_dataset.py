"""4-3: 시연 + 교정 HUMAN 구간 데이터셋 변환(train/build_correction_dataset.py). 가짜 raw/교정/프레임(64x64)으로 만든다.

핵심: 교정 구간의 k번째 프레임이 교정 npz의 start+k번째 qpos/ctrl[:6]과 한 칸도 안 어긋나는지, HUMAN_WAIT/AUTO가 안 들어가는지,
평가 시드가 섞이면 아무것도 만들지 않는지, 출처 표(번호 매핑)가 맞는지.
"""

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(TRAIN_DIR.parent / "server" / "tests"))
import build_correction_dataset as bcd  # noqa: E402
import build_dataset as bd  # noqa: E402
import dataset_config as cfg  # noqa: E402
import test_dataset_correspondence as tdc  # noqa: E402
import verify_dataset as vd  # noqa: E402
from correction_fakes import make_correction  # noqa: E402

from server import config, seeds  # noqa: E402
from server import correction_segments as cs  # noqa: E402


def make_inputs(tmp_path: Path, corr_specs=((0, 28, [(8, 15)]), (1, 29, [(5, 9), (20, 31)])), n=40):
    raw, frames = tdc.make_raw_and_frames(tmp_path)
    for p in raw.glob("episode_*.json"):  # 실제 시연 json에는 block_start_m이 있다
        m = json.loads(p.read_text(encoding="utf-8"))
        m["block_start_m"] = [*seeds.block_xy_for_seed(m["seed"]), 0.0149]
        p.write_text(json.dumps(m), encoding="utf-8")
    corr, corr_frames = tmp_path / "corr", tmp_path / "corr_frames"
    metas = [make_correction(corr, i, seed, runs, n) for i, seed, runs in corr_specs]
    segs, problems = cs.all_segments(corr, metas)
    assert problems == []
    settings = json.loads((frames / "manifest.json").read_text(encoding="utf-8"))["settings"]
    episodes = {}
    for idx, seg in enumerate(segs):
        qpos = cs.segment_qpos(corr, seg)
        folder = corr_frames / f"ep_{idx:06d}"
        folder.mkdir(parents=True)
        for k in range(len(qpos)):
            Image.fromarray(tdc.fake_image(seg.source_index + 7, seg.start_sample + k)).save(folder / f"frame_{k:06d}.png")
        episodes[str(idx)] = {"n_frames": len(qpos), "qpos_sha256": bd.qpos_sha256(qpos), "source_index": seg.source_index,
                              "start_sample": seg.start_sample, "end_sample": seg.end_sample, "seed": seg.seed}
    (corr_frames / "manifest.json").write_text(json.dumps({"settings": settings, "episodes": episodes}), encoding="utf-8")
    args = Namespace(raw=raw, frames=frames, corrections=corr, corr_frames=corr_frames, out=tmp_path / "dexter_pipeline_test",
                     exclude=[], exclude_corrections=[], allow_diff=[], expect_checkpoint_sha=None, crf=cfg.VIDEO_CRF, image_mode=True, codec=None)
    return args, segs


def test_plan_passes_and_sources_table_maps_demos_then_segments(tmp_path):
    args, segs = make_inputs(tmp_path)
    plan = bcd.plan_build(args)
    assert plan["problems"] == []
    rows = bcd.sources_table(plan)
    assert [(r["dataset_episode"], r["kind"], r["source_index"]) for r in rows] == [
        (0, "demo", 0), (1, "demo", 2), (2, "correction_human", 0), (3, "correction_human", 1), (4, "correction_human", 1)]  # 시연 1은 실패라 빠진다
    assert [(r["start_sample"], r["end_sample"], r["n_frames"]) for r in rows[2:]] == [(8, 15, 8), (5, 9, 5), (20, 31, 12)]
    assert [r["seed"] for r in rows[2:]] == [28, 29, 29]


def test_build_segment_frames_match_the_correction_npz_exactly(tmp_path):
    args, segs = make_inputs(tmp_path)
    plan = bcd.plan_build(args)
    result = bcd.build(args, plan)
    n_demo = tdc.LENGTHS[0] + tdc.LENGTHS[2]
    assert result == {"episodes": 5, "frames": n_demo + 8 + 5 + 12}
    ds = vd.load_dataset(args.out)
    assert ds.num_episodes == 5 and ds.num_frames == result["frames"]
    assert set(ds.meta.features) >= {"observation.state", "action", cfg.IMAGE_KEY}
    assert "is_intervention" not in ds.meta.features
    table = ds.hf_dataset.with_format("numpy")
    state, action = np.asarray(table["observation.state"]), np.asarray(table["action"])
    pos = n_demo
    for seg in segs:
        arrays = cs.load_correction_arrays(args.corrections, seg.source_index)
        sl = slice(seg.start_sample, seg.end_sample + 1)
        np.testing.assert_array_equal(state[pos : pos + seg.n_samples], arrays["qpos"][sl, :6].astype(np.float32))
        np.testing.assert_array_equal(action[pos : pos + seg.n_samples], arrays["ctrl"][sl, :6].astype(np.float32))
        assert (arrays["owner"][sl] == cs.OWNER_HUMAN).all()
        first_image = ds[pos][cfg.IMAGE_KEY].numpy()  # (3,H,W) float [0,1]
        expected = np.asarray(Image.open(bd.frame_file(args.corr_frames, segs.index(seg), 0)).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
        np.testing.assert_allclose(first_image, expected, atol=1 / 255)
        pos += seg.n_samples
    assert pos == ds.num_frames
    for start, index in ((0, 0), (tdc.LENGTHS[0], 2)):  # 앞의 시연 두 개도 raw와 그대로 같다(검사 함수는 데이터셋 전체가 시연이라고 가정해서 쓰지 않는다)
        a = tdc.fake_arrays(index, tdc.LENGTHS[index])
        np.testing.assert_array_equal(state[start : start + tdc.LENGTHS[index]], a["qpos"][:, :6].astype(np.float32))
        np.testing.assert_array_equal(action[start : start + tdc.LENGTHS[index]], a["ctrl"][:, :6].astype(np.float32))
    sources = json.loads((args.out / "handover_sources.json").read_text(encoding="utf-8"))
    assert [e["kind"] for e in sources["episodes"]] == ["demo", "demo"] + ["correction_human"] * 3
    assert sources["correction_seeds"] == [28, 29]


def test_eval_seed_or_position_stops_everything_and_writes_nothing(tmp_path):
    lo = config.EVAL_SEED_RANGE[0]
    args, _ = make_inputs(tmp_path, corr_specs=((0, lo + 2, [(8, 15)]),))
    assert any("평가 시드" in p for p in bcd.plan_build(args)["problems"])
    assert not args.out.exists()

    other = tmp_path / "second"
    other.mkdir()
    args2, _ = make_inputs(other, corr_specs=((0, 28, [(8, 15)]),))
    meta_path = args2.corrections / "correction_000000.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["block_start_m"][:2] = list(seeds.block_xy_for_seed(lo + 9))  # 시드는 28인데 블록은 평가 시드 1009 위치
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert any("평가 시드 1009" in p for p in bcd.plan_build(args2)["problems"])


def test_eval_seed_in_demo_data_or_auto_runs_csv_is_caught_too(tmp_path):
    args, _ = make_inputs(tmp_path)
    (args.corrections / "auto_runs.csv").write_text("seed,kind\n28,correction_saved\n1001,auto_only\n", encoding="utf-8")
    assert any("1001" in p for p in bcd.plan_build(args)["problems"])
    (args.corrections / "auto_runs.csv").unlink()
    p = args.raw / "episode_000000.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["seed"], m["block_start_m"] = config.EVAL_SEED_RANGE[0], [*seeds.block_xy_for_seed(config.EVAL_SEED_RANGE[0]), 0.0149]
    p.write_text(json.dumps(m), encoding="utf-8")
    assert any("시연 0" in x and "평가 시드" in x for x in bcd.plan_build(args)["problems"])


def test_environment_mismatch_stops_and_allow_diff_is_explicit(tmp_path):
    args, _ = make_inputs(tmp_path)
    for p in args.raw.glob("episode_*.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        m["mujoco_version"] = "3.16.0"
        p.write_text(json.dumps(m), encoding="utf-8")
    meta_path = args.corrections / "correction_000001.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["mujoco_version"] = "3.2.0"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert any("mujoco_version" in p for p in bcd.plan_build(args)["problems"])
    args.allow_diff = ["mujoco_version"]
    assert bcd.plan_build(args)["problems"] == []


def test_frames_out_of_sync_with_npz_are_caught(tmp_path):
    args, segs = make_inputs(tmp_path)
    arrays = dict(cs.load_correction_arrays(args.corrections, 0))
    arrays["qpos"] = arrays["qpos"] + 1e-3  # 프레임을 그린 뒤 qpos가 바뀐 경우
    np.savez_compressed(args.corrections / "correction_000000.npz", **arrays)
    assert any("qpos가 바뀌었" in p for p in bcd.plan_build(args)["problems"])
    (tmp_path / "x").mkdir()
    args2, _ = make_inputs(tmp_path / "x")
    (args2.corr_frames / "ep_000000" / "frame_000003.png").unlink()
    assert any("프레임 파일" in p for p in bcd.plan_build(args2)["problems"])


def test_main_refuses_forbidden_dataset_names_and_existing_out(tmp_path, monkeypatch):
    args, _ = make_inputs(tmp_path)
    base = ["prog", "--raw", str(args.raw), "--frames", str(args.frames), "--corrections", str(args.corrections), "--corr-frames", str(args.corr_frames)]
    for name in ("dexter_pick_place_v2", "dexter_pick_place_v1"):
        monkeypatch.setattr(sys, "argv", [*base, "--out", str(tmp_path / name)])
        assert bcd.main() == 2
        assert not (tmp_path / name).exists()
    args.out.mkdir()
    monkeypatch.setattr(sys, "argv", [*base, "--out", str(args.out), "--image-mode"])
    assert bcd.main() == 2
    monkeypatch.setattr(sys, "argv", [*base, "--out", str(tmp_path / "chk"), "--check-only"])
    assert bcd.main() == 0 and not (tmp_path / "chk").exists()


def test_inputs_are_not_modified(tmp_path):
    args, _ = make_inputs(tmp_path)
    before = {p: p.read_bytes() for d in (args.raw, args.frames, args.corrections, args.corr_frames) for p in d.rglob("*") if p.is_file()}
    bcd.build(args, bcd.plan_build(args))
    after = {p: p.read_bytes() for d in (args.raw, args.frames, args.corrections, args.corr_frames) for p in d.rglob("*") if p.is_file()}
    assert before == after


def test_allow_diff_rejects_non_allowable_names(tmp_path, monkeypatch):
    args, _ = make_inputs(tmp_path)
    argv = ["prog", "--raw", str(args.raw), "--frames", str(args.frames), "--corrections", str(args.corrections),
            "--corr-frames", str(args.corr_frames), "--out", str(tmp_path / "o"), "--allow-diff", "scene_sha256"]
    monkeypatch.setattr(sys, "argv", argv)
    assert bcd.main() == 2
