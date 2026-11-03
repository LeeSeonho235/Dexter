"""3-3 핵심 테스트: 데이터셋 k번째 프레임의 state/action/이미지가 raw의 k번째와 한 칸도 안 어긋나는지.

가짜 raw(npz+json)와 가짜 프레임(PNG)으로 작은 데이터셋(성공 2개 + 실패 1개)을 만들어 build_dataset.build()와
verify_dataset의 검사 함수를 돌린다. 어긋난 데이터를 일부러 만들어 검사가 잡는지도 본다.

pytest로 돌린다(train/.venv에는 아직 pytest가 없다). 이 파일은 pytest를 import하지 않고 tmp_path 인자만 쓰므로
`python -c`에서 임시 폴더를 넘겨 직접 호출해도 된다.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))
import build_dataset as bd  # noqa: E402
import dataset_config as cfg  # noqa: E402
import verify_dataset as vd  # noqa: E402

H, W = 64, 64
SCENE_SHA, ROBOT_SHA = "a" * 64, "b" * 64
LENGTHS = {0: 14, 1: 20, 2: 17}  # 에피소드 1은 실패, 0과 2는 성공


def fake_arrays(index: int, n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(index)
    qpos = rng.uniform(-1, 1, size=(n, cfg.QPOS_DIM))
    ctrl = rng.uniform(-1, 1, size=(n, cfg.CTRL_DIM))
    t = np.cumsum(np.where(np.arange(n) % 2 == 0, 0.030, 0.035))
    return {"sim_time_s": t, "qpos": qpos, "ctrl": ctrl, "is_intervention": np.zeros(n, dtype=bool)}


def fake_image(index: int, k: int) -> np.ndarray:
    """k마다 다른 그림: 배경 밝기가 k에 따라 변하고 빨간 사각형이 k에 따라 움직인다(이웃 프레임과 구별되게)."""
    img = np.full((H, W, 3), 60 + 6 * k + index, dtype=np.uint8)
    x = 4 + 2 * k
    img[20:30, x : x + 10] = (220, 30, 30)
    return img


def make_raw_and_frames(root: Path) -> tuple[Path, Path]:
    raw, frames = root / "raw", root / "frames"
    raw.mkdir()
    episodes = {}
    for index, n in LENGTHS.items():
        arrays = fake_arrays(index, n)
        np.savez_compressed(raw / f"episode_{index:06d}.npz", **arrays)
        success = index != 1
        meta = {
            "episode_index": index, "seed": index, "success": success, "end_reason": "stop_success" if success else "stop_fail",
            "n_ticks": n, "duration_s": float(arrays["sim_time_s"][-1] - arrays["sim_time_s"][0]), "still_ratio": 0.3,
            "still_longest_s": 1.0, "success_differs_from_auto": False, "scene_sha256": SCENE_SHA, "robot_model_sha256": ROBOT_SHA,
            "record_camera": {"height_px": H, "width_px": W, "azimuth_deg": 110.0, "elevation_deg": -35.0, "distance_m": 0.8},
        }
        (raw / f"episode_{index:06d}.json").write_text(json.dumps(meta), encoding="utf-8")
        folder = frames / f"ep_{index:06d}"
        folder.mkdir(parents=True)
        for k in range(n):
            Image.fromarray(fake_image(index, k)).save(folder / f"frame_{k:06d}.png")
        episodes[str(index)] = {"n_frames": n, "qpos_sha256": hashlib.sha256(np.ascontiguousarray(arrays["qpos"], dtype=np.float64).tobytes()).hexdigest()}
    settings = {
        "scene_sha256": SCENE_SHA, "robot_model_sha256": ROBOT_SHA, "height_px": H, "width_px": W,
        "camera": {"azimuth_deg": 110.0, "elevation_deg": -35.0, "distance_m": 0.8},
    }
    (frames / "manifest.json").write_text(json.dumps({"settings": settings, "episodes": episodes}), encoding="utf-8")
    return raw, frames


def build_small(tmp_path: Path, exclude: set[int] = frozenset()):
    raw, frames = make_raw_and_frames(tmp_path)
    out = tmp_path / "ds"
    result = bd.build(raw, frames, out, set(exclude), crf=cfg.VIDEO_CRF)
    return raw, frames, out, result


def test_only_successes_are_in_the_dataset_and_state_action_match_exactly(tmp_path):
    raw, frames, out, result = build_small(tmp_path)
    assert result["episodes"] == [0, 2] and result["frames"] == LENGTHS[0] + LENGTHS[2]
    ds = vd.load_dataset(out)
    assert ds.num_episodes == 2 and ds.num_frames == LENGTHS[0] + LENGTHS[2]
    assert vd.check_state_action(ds, raw, [0, 2]) == []
    # 데이터셋 k번째 프레임(항목 접근)도 npz의 k번째와 같다
    arrays = fake_arrays(2, LENGTHS[2])
    start = LENGTHS[0]
    for k in (0, 5, LENGTHS[2] - 1):
        item = ds[start + k]
        np.testing.assert_array_equal(item["observation.state"].numpy(), arrays["qpos"][k, :6].astype(np.float32))
        np.testing.assert_array_equal(item["action"].numpy(), arrays["ctrl"][k, :6].astype(np.float32))
        assert item["task"] == cfg.TASK


def test_state_is_qpos_and_action_is_ctrl_not_swapped_or_shifted(tmp_path):
    raw, frames, out, _ = build_small(tmp_path)
    ds = vd.load_dataset(out)
    table = ds.hf_dataset.with_format("numpy")
    state, action = np.asarray(table["observation.state"]), np.asarray(table["action"])
    a0 = fake_arrays(0, LENGTHS[0])
    assert not np.array_equal(state[: LENGTHS[0]], action[: LENGTHS[0]])
    assert not np.array_equal(action[: LENGTHS[0] - 1], a0["ctrl"][1:, :6].astype(np.float32))  # 한 칸 앞으로 밀린 값이 아니다
    np.testing.assert_array_equal(state[: LENGTHS[0]], a0["qpos"][:, :6].astype(np.float32))


def test_checker_catches_a_one_frame_shift(tmp_path):
    raw, frames, out, _ = build_small(tmp_path)
    ds = vd.load_dataset(out)
    # raw의 ctrl을 한 칸 밀어 놓고 데이터셋과 비교하면 문제가 나와야 한다
    shifted = dict(np.load(raw / "episode_000000.npz"))
    shifted["ctrl"] = np.roll(shifted["ctrl"], 1, axis=0)
    np.savez_compressed(raw / "episode_000000.npz", **shifted)
    problems = vd.check_state_action(ds, raw, [0, 2])
    assert any("action" in p and "raw 0" in p for p in problems)


def test_images_match_the_same_k_and_checker_catches_a_shift(tmp_path):
    raw, frames, out, _ = build_small(tmp_path)
    ds = vd.load_dataset(out)
    ok = vd.check_images(ds, frames, [0, 2], per_episode=6)
    assert ok["frames_checked"] >= 8, "판정한 프레임이 너무 적다(정지 장면 제외 규칙이 검사를 비우면 안 된다)"
    assert ok["problems"] == [] and ok["strictly_closest"] == ok["frames_checked"]
    assert max(ok["mae_all"]) < 6.0  # 손실 압축이지만 같은 그림이다
    # 프레임 PNG를 한 칸씩 밀어 놓은 폴더로 비교하면 걸려야 한다
    for index in (0, 2):
        folder = frames / f"ep_{index:06d}"
        n = LENGTHS[index]
        originals = [(folder / f"frame_{k:06d}.png").read_bytes() for k in range(n)]
        for k in range(n):
            (folder / f"frame_{k:06d}.png").write_bytes(originals[(k + 3) % n])
    shifted = vd.check_images(ds, frames, [0, 2], per_episode=6)
    assert shifted["problems"], "3프레임 어긋난 이미지를 검사가 못 잡았다"


def test_extra_exclude_removes_only_the_named_success(tmp_path):
    raw, frames, out, result = build_small(tmp_path, exclude={0})
    assert result["episodes"] == [2]
    assert vd.load_dataset(out).num_episodes == 1


def test_build_refuses_to_overwrite_existing_dataset(tmp_path):
    raw, frames = make_raw_and_frames(tmp_path)
    out = tmp_path / "ds"
    out.mkdir()
    try:
        bd.build(raw, frames, out, set(), crf=cfg.VIDEO_CRF)
    except SystemExit as err:
        assert "이미 있습니다" in str(err)
    else:
        raise AssertionError("기존 폴더를 덮어썼다")


def test_stale_frames_are_rejected(tmp_path):
    raw, frames = make_raw_and_frames(tmp_path)
    changed = dict(np.load(raw / "episode_000000.npz"))
    changed["qpos"] = changed["qpos"] + 0.001  # 프레임을 그린 뒤 qpos가 바뀐 상황
    np.savez_compressed(raw / "episode_000000.npz", **changed)
    metas = {m["episode_index"]: m for m in bd.load_metas(raw)}
    problems = bd.check_frames(raw, frames, metas, [0, 2])
    assert len(problems) == 1 and "해시" in problems[0]


def test_missing_frames_are_rejected(tmp_path):
    raw, frames = make_raw_and_frames(tmp_path)
    (frames / "ep_000002" / "frame_000003.png").unlink()
    metas = {m["episode_index"]: m for m in bd.load_metas(raw)}
    problems = bd.check_frames(raw, frames, metas, [0, 2])
    assert len(problems) == 1 and "프레임 파일" in problems[0]
