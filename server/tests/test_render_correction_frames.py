"""4-3: 교정 HUMAN 구간 프레임 렌더링. 프레임 k가 구간의 qpos[k]로 그려지는지, 검사가 걸리면 그리지 않는지, 이어하기가 맞는지."""

import mujoco
import numpy as np
import pytest
from correction_fakes import make_correction
from PIL import Image

from server import config, seeds
from server import correction_segments as cs
from server.render import StateRenderer
from server.render_correction_frames import plan_segments, render_plan
from server.render_frames import episode_dir_name, frame_path, read_manifest, render_settings


@pytest.fixture(scope="module")
def settings():
    return render_settings()


@pytest.fixture(scope="module")
def home_qpos():
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    return model.key("home").qpos.copy()


def make_real(tmp_path, settings, home_qpos, index=0, seed=28, runs=((4, 8), (14, 17)), n=20, **kwargs):
    """장면 해시가 진짜인 가짜 교정. 블록은 시드가 정한 위치, 집게는 home에서 프레임마다 조금씩 움직인다."""
    qpos = np.tile(home_qpos, (n, 1))
    qpos[:, 6:8] = seeds.block_xy_for_seed(seed)
    qpos[:, 0] = np.linspace(0.0, 0.5, n)
    return make_correction(tmp_path, index, seed, list(runs), n, qpos=qpos, scene=settings["scene_sha256"], robot=settings["robot_model_sha256"],
                           mujoco_version=settings["mujoco_version"], **kwargs)


def test_plan_has_one_item_per_human_segment_numbered_from_zero(tmp_path, settings, home_qpos):
    make_real(tmp_path, settings, home_qpos, 0, 28)
    make_real(tmp_path, settings, home_qpos, 1, 29, runs=((2, 5),))
    plan, problems = plan_segments(tmp_path, set(), settings, set())
    assert problems == []
    assert [(p["seg_index"], p["source_index"], p["start_sample"], p["end_sample"]) for p in plan] == [(0, 0, 4, 8), (1, 0, 14, 17), (2, 1, 2, 5)]


def test_plan_refuses_eval_seed_and_wrong_scene(tmp_path, settings, home_qpos):
    make_real(tmp_path, settings, home_qpos, 0, config.EVAL_SEED_RANGE[0] + 3)
    _, problems = plan_segments(tmp_path, set(), settings, set())
    assert any("평가 시드" in p for p in problems)
    other = tmp_path / "other"
    make_correction(other, 0, 28, [(4, 8)], n=20)  # scene 해시가 진짜가 아니다
    _, problems = plan_segments(other, set(), settings, set())
    assert any("scene_sha256" in p for p in problems)


def test_render_writes_only_segment_frames_matching_qpos_and_resumes(tmp_path, settings, home_qpos):
    corr, out = tmp_path / "corr", tmp_path / "frames"
    make_real(corr, settings, home_qpos)
    plan, problems = plan_segments(corr, set(), settings, set())
    assert problems == []
    assert render_plan(plan, corr, out, settings, log=lambda *_: None) == 2
    manifest = read_manifest(out)
    assert sorted(manifest["episodes"]) == ["0", "1"]
    assert manifest["episodes"]["0"]["n_frames"] == 5 and manifest["episodes"]["1"]["n_frames"] == 4
    assert sorted(p.name for p in out.iterdir()) == [episode_dir_name(0), episode_dir_name(1), "manifest.json"]  # 이미지는 구간 프레임뿐
    seg = cs.Segment(0, 28, 14, 17, 4)
    qpos = cs.segment_qpos(corr, seg)
    with StateRenderer() as renderer:
        for k in range(4):
            expected = renderer.render(qpos[k])
            got = np.asarray(Image.open(frame_path(out, 1, k)).convert("RGB"))
            assert np.abs(got.astype(np.int16) - expected.astype(np.int16)).max() <= 1
    assert render_plan(plan, corr, out, settings, log=lambda *_: None) == 0  # 이어하기: 모두 건너뜀


def test_render_stops_when_existing_manifest_settings_differ(tmp_path, settings, home_qpos):
    corr, out = tmp_path / "corr", tmp_path / "frames"
    make_real(corr, settings, home_qpos)
    plan, _ = plan_segments(corr, set(), settings, set())
    render_plan(plan, corr, out, settings, log=lambda *_: None)
    with pytest.raises(SystemExit):
        render_plan(plan, corr, out, {**settings, "width_px": settings["width_px"] + 1}, log=lambda *_: None)
