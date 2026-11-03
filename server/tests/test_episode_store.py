"""에피소드 저장소: npz + json 저장·읽기·삭제, 해시, 쓰기 실패, 쓰기 스레드가 막혀도 호출이 바로 돌아오는지."""

import hashlib
import json
import threading
import time

import numpy as np
import pytest

from server import config
from server.episode_store import EpisodeStore, EpisodeWriter, config_snapshot, config_sha256, stem_of
from server.recorder import FinishedEpisode

TS = 0.005


def make_episode(index=0, seed=3, n=40, success=True):
    t = 10.0 + np.arange(n) * (1 / 30)
    qpos = np.zeros((n, 13))
    qpos[:, 6:9] = (0.2, 0.0, 0.015)
    qpos[:, 9] = 1.0
    return FinishedEpisode(
        episode_index=index,
        seed=seed,
        block_start_m=(0.21, -0.02, 0.015),
        target_m=config.TARGET_CENTER_M,
        sim_time_s=t,
        qpos=qpos,
        ctrl=np.zeros((n, 6)),
        is_intervention=np.zeros(n, dtype=bool),
        success=success,
        end_reason="stop_success" if success else "stop_fail",
        auto_success_ever=False,
        auto_success_first_s=None,
        auto_success_at_end=False,
        still_ratio=0.25,
        still_longest_s=0.4,
        created_at="2026-11-03T12:00:00+09:00",
    )


def test_save_creates_only_npz_and_json_and_reads_back(tmp_path):
    store = EpisodeStore(tmp_path / "raw")
    ep = make_episode(index=12)
    meta = store.save(ep, TS)
    files = sorted(p.name for p in (tmp_path / "raw").iterdir())
    assert files == ["episode_000012.json", "episode_000012.npz"]  # 이미지·영상 파일 없음, 임시 파일 안 남음

    arrays = store.load_arrays(12)
    assert set(arrays) == {"sim_time_s", "qpos", "ctrl", "is_intervention"}  # 이미지 배열 없음
    np.testing.assert_array_equal(arrays["qpos"], ep.qpos)
    np.testing.assert_array_equal(arrays["sim_time_s"], ep.sim_time_s)
    assert arrays["qpos"].dtype == np.float64 and arrays["is_intervention"].dtype == bool

    on_disk = json.loads((tmp_path / "raw" / "episode_000012.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(meta))


def test_meta_has_the_requested_fields(tmp_path):
    store = EpisodeStore(tmp_path)
    meta = store.save(make_episode(index=2, seed=7, n=61, success=False), TS)
    assert meta["episode_index"] == 2 and meta["seed"] == 7 and meta["source"] == "demo"
    assert meta["success"] is False and meta["n_ticks"] == 61
    assert meta["block_start_m"] == [0.21, -0.02, 0.015] and meta["target_m"] == list(config.TARGET_CENTER_M)
    assert meta["is_intervention_any"] is False
    assert meta["fps"] == 30 and meta["sim_timestep_s"] == TS
    assert meta["still_ratio"] == 0.25 and meta["still_longest_s"] == 0.4
    assert meta["auto_success"] == {"ever": False, "first_sim_time_s": None, "at_end": False}
    assert meta["success_differs_from_auto"] is False  # 내가 실패, 자동도 한 번도 달성 못 함 → 같다
    assert store.save(make_episode(index=3, success=True), TS)["success_differs_from_auto"] is True  # 성공으로 눌렀지만 자동은 아님
    assert meta["created_at"] and meta["mujoco_version"]
    assert meta["duration_s"] == pytest.approx(60 / 30, abs=1e-3)
    assert meta["qpos_layout"][5] == "gripper" and len(meta["qpos_layout"]) == 13
    cam = meta["record_camera"]
    assert (cam["azimuth_deg"], cam["elevation_deg"], cam["distance_m"]) == (110.0, -35.0, 0.8)
    assert (cam["height_px"], cam["width_px"]) == (240, 320)


def test_scene_and_config_hashes(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>")
    store = EpisodeStore(tmp_path / "raw", scene_path=scene)
    meta = store.save(make_episode(), TS)
    assert meta["scene_sha256"] == hashlib.sha256(b"<mujoco/>").hexdigest()
    assert meta["robot_model_sha256"] and len(meta["robot_model_sha256"]) == 64
    assert meta["config_sha256"] == config_sha256(config_snapshot())
    assert meta["config"]["RECORD_FPS"] == 30

    scene.write_text("<mujoco model='changed'/>")
    other = EpisodeStore(tmp_path / "raw2", scene_path=scene).save(make_episode(), TS)
    assert other["scene_sha256"] != meta["scene_sha256"]


def test_config_hash_changes_when_a_setting_changes(monkeypatch):
    before = config_sha256(config_snapshot())
    monkeypatch.setattr(config, "STILL_CTRL_DELTA_RAD", 0.5)
    assert config_sha256(config_snapshot()) != before


def test_config_hash_does_not_depend_on_this_computers_paths():
    snapshot = config_snapshot()
    assert "REPO_ROOT" not in snapshot and "RECORD_DIR" not in snapshot and "SCENE_PATH" not in snapshot


def test_scan_lists_in_order_and_ignores_files_without_json(tmp_path):
    store = EpisodeStore(tmp_path)
    store.save(make_episode(index=10), TS)
    store.save(make_episode(index=2), TS)
    np.savez(tmp_path / "episode_000099.npz", x=np.zeros(1))  # json 없는 npz = 저장 끝나지 않은 것
    (tmp_path / "episode_000050.json").write_text("{not json")  # 깨진 json
    assert [m["episode_index"] for m in store.scan()] == [2, 10]


def test_scan_of_missing_folder_is_empty_and_does_not_create_it(tmp_path):
    store = EpisodeStore(tmp_path / "nope")
    assert store.scan() == []
    assert not (tmp_path / "nope").exists()


def test_delete_removes_both_files(tmp_path):
    store = EpisodeStore(tmp_path)
    store.save(make_episode(index=1), TS)
    store.save(make_episode(index=2), TS)
    store.delete(2)
    assert [m["episode_index"] for m in store.scan()] == [1]
    assert not (tmp_path / "episode_000002.npz").exists()
    store.delete(2)  # 이미 없어도 오류 없음


def test_failed_save_leaves_no_files_and_raises(tmp_path):
    blocker = tmp_path / "raw"
    blocker.write_text("파일이 폴더 자리에 있다")  # 폴더를 만들 수 없다
    store = EpisodeStore(blocker)
    with pytest.raises(OSError):
        store.save(make_episode(), TS)
    assert blocker.read_text() == "파일이 폴더 자리에 있다"


def test_failure_in_the_middle_cleans_up(tmp_path, monkeypatch):
    store = EpisodeStore(tmp_path)

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(OSError):
        store.save(make_episode(index=5), TS)
    assert list(tmp_path.iterdir()) == []


def test_stem():
    assert stem_of(12) == "episode_000012"


# --- 쓰기 스레드 ---


class SlowStore(EpisodeStore):
    """save가 release될 때까지 막힌다(디스크가 멈춘 상황)."""

    def __init__(self, root):
        super().__init__(root)
        self.release = threading.Event()

    def save(self, ep, timestep_s):
        self.release.wait(5.0)
        return super().save(ep, timestep_s)


def test_submit_returns_immediately_even_if_the_disk_is_stuck(tmp_path):
    store = SlowStore(tmp_path)
    saved = []
    writer = EpisodeWriter(store, TS, on_saved=saved.append, on_failed=lambda i, e: None, on_deleted=lambda i, e: None)
    writer.start()
    t0 = time.perf_counter()
    for i in range(5):
        writer.submit_save(make_episode(index=i))
    assert time.perf_counter() - t0 < 0.05  # 디스크가 멈춰도 호출 쪽은 기다리지 않는다
    assert saved == []
    store.release.set()
    writer.close()
    assert [m["episode_index"] for m in saved] == [0, 1, 2, 3, 4]  # 순서대로 모두 저장


def test_writer_reports_failure_and_keeps_working(tmp_path):
    class FlakyStore(EpisodeStore):
        calls = 0

        def save(self, ep, timestep_s):
            FlakyStore.calls += 1
            if FlakyStore.calls == 1:
                raise OSError(28, "No space left on device")
            return super().save(ep, timestep_s)

    saved, failed = [], []
    writer = EpisodeWriter(
        FlakyStore(tmp_path), TS, on_saved=saved.append, on_failed=lambda i, e: failed.append((i, e)),
        on_deleted=lambda i, e: None,
    )
    writer.start()
    writer.submit_save(make_episode(index=0))
    writer.submit_save(make_episode(index=1))
    writer.close()
    assert len(failed) == 1 and failed[0][0] == 0 and "No space left" in failed[0][1]
    assert [m["episode_index"] for m in saved] == [1]  # 실패한 뒤에도 다음 것은 저장된다
    assert [m["episode_index"] for m in EpisodeStore(tmp_path).scan()] == [1]


def test_writer_delete_reports_result(tmp_path):
    store = EpisodeStore(tmp_path)
    store.save(make_episode(index=3), TS)
    deleted = []
    writer = EpisodeWriter(store, TS, on_saved=lambda m: None, on_failed=lambda i, e: None,
                           on_deleted=lambda i, e: deleted.append((i, e)))
    writer.start()
    writer.submit_delete(3)
    writer.close()
    assert deleted == [(3, None)] and store.scan() == []
