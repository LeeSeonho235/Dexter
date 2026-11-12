"""교정 저장소(4-2): npz/json 형식과 정합, auto_runs.csv(kind 열), 저장 실패, 시드 배분표(평가·AUTO 시드 규칙과 충돌 없음), 프로토콜."""

import json

import numpy as np
import pytest

from server import config, correction_config as cc
from server.correction import CorrectionRecording, check_alignment, intervention_segments
from server.correction_store import (
    CSV_FIELDS,
    KIND_AUTO_ONLY,
    KIND_DISCARDED,
    KIND_SAVED,
    CorrectionStore,
    CorrectionWriter,
    checkpoint_identity,
)
from server.protocol import AUTO_ACTIONS, ProtocolError, parse_client_message
from server.seeds import DEMO, EVAL, block_xy_for_seed, seed_kind


def make_recording(seed=30, n=60, human=(20, 40), wait=(15, 20)):
    owner = np.zeros(n, dtype=np.int8)
    owner[wait[0] : wait[1]] = cc.OWNER_HUMAN_WAIT
    owner[human[0] : human[1]] = cc.OWNER_HUMAN
    t = 10.0 + np.arange(n) / 30.0
    qpos = np.zeros((n, 13))
    qpos[:, 6:9] = [0.2, 0.0, 0.015]
    ctrl = np.tile(np.linspace(0, 0.3, 6), (n, 1)) + np.arange(n)[:, None] * 0.001
    if wait[1] > wait[0]:
        ctrl[wait[0] : wait[1]] = ctrl[wait[0] - 1]  # 대기 중 ctrl은 그대로
    return CorrectionRecording(
        seed=seed, block_start_m=(0.2, 0.0, 0.015), sim_time_s=t, qpos=qpos, ctrl=ctrl, owner=owner, outcome="success", end_reason="success",
        time_to_success_s=1.9,
    )


def test_save_writes_aligned_npz_and_complete_json(tmp_path):
    store = CorrectionStore(tmp_path / "corrections")
    rec = make_recording()
    meta = store.save(rec)
    assert meta["episode_index"] == 0
    files = sorted(p.name for p in (tmp_path / "corrections").iterdir())
    assert files == ["correction_000000.json", "correction_000000.npz"]  # 임시 파일이 남지 않는다
    arrays = store.load_arrays(0)
    assert set(arrays) == {"sim_time_s", "qpos", "ctrl", "is_intervention", "owner"}
    assert arrays["qpos"].shape == (60, 13) and arrays["ctrl"].shape == (60, 6) and arrays["owner"].dtype == np.int8
    assert arrays["is_intervention"].dtype == bool
    assert np.array_equal(arrays["is_intervention"], arrays["owner"] == cc.OWNER_HUMAN)  # is_intervention은 HUMAN만
    assert not arrays["is_intervention"][15:20].any()  # HUMAN_WAIT는 개입이 아니다
    saved = json.loads((tmp_path / "corrections" / "correction_000000.json").read_text(encoding="utf-8"))
    for key in ("episode_index", "seed", "block_start_m", "source", "base_checkpoint", "interventions", "intervention_count", "human_time_s", "auto_time_s",
                "total_s", "auto_result", "user_decision", "scene_sha256", "config_sha256", "correction_config_sha256", "created_at", "robot_model_sha256"):
        assert key in saved, key
    assert saved["source"] == "correction" and saved["seed"] == 30 and saved["user_decision"] == "saved"
    assert saved["interventions"] == [{"start_sample": 20, "end_sample": 39, "n_samples": 20}] and saved["intervention_count"] == 1
    assert saved["human_time_s"] == pytest.approx(20 / 30, abs=1e-3)  # 사람 시간은 HUMAN 구간만
    assert saved["human_wait_time_s"] == pytest.approx(5 / 30, abs=1e-3) and saved["auto_time_s"] == pytest.approx(35 / 30, abs=1e-3)
    assert saved["auto_result"] == {"outcome": "success", "end_reason": "success", "time_to_success_s": 1.9, "error": ""}
    assert saved["base_checkpoint"]["model_safetensors_sha256"] is None or len(saved["base_checkpoint"]["model_safetensors_sha256"]) == 64
    assert saved["base_checkpoint"]["model_safetensors_path"].endswith("model.safetensors")
    assert saved["n_ticks"] == 60 and saved["fps"] == 30
    assert intervention_segments(arrays["owner"]) == [(20, 39)]
    assert check_alignment(rec) == []
    assert saved["config"]["MAPPING_MODE"] == config.MAPPING_MODE  # 설정 해시는 server/config.py 값 전체에서 만든다
    assert saved["ctrl_jumps"]["handback_rad"] == [pytest.approx(0.001)]  # 40번 샘플이 HUMAN 다음 첫 AUTO(이 기록에서 ctrl이 샘플마다 0.001씩 변한다)
    assert store.save(rec)["episode_index"] == 1  # 번호는 이어서


def test_data_corrections_is_ignored_by_git_and_separate_from_raw():
    assert cc.CORRECTION_DIR != config.RECORD_DIR and cc.CORRECTION_DIR.parent == config.RECORD_DIR.parent
    lines = (config.REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "data/" in lines  # data/corrections/도 걸린다
    dataset_config = (config.REPO_ROOT / "train" / "dataset_config.py").read_text()
    assert "corrections" not in dataset_config  # build_dataset은 data/raw만 읽는다


def test_save_failure_leaves_no_files_and_raises(tmp_path, monkeypatch):
    store = CorrectionStore(tmp_path / "c")
    rec = make_recording()

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(OSError):
        store.save(rec)
    assert [p.name for p in (tmp_path / "c").iterdir()] == []  # 임시 파일도 npz도 json도 없다
    assert store.scan() == []


def test_writer_reports_save_failure_once_and_a_ledger_only_failure_separately(tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")  # 폴더를 만들 수 없게 파일을 둔다
    store = CorrectionStore(blocker / "corrections")
    saved, failed = [], []
    writer = CorrectionWriter(store, on_saved=lambda r, m: saved.append(m), on_failed=lambda r, e: failed.append(e), on_csv_failed=lambda r, e: failed.append("csv:" + e))
    writer.start()
    writer.submit_save(make_recording())
    writer.close()
    assert saved == [] and len(failed) == 1 and not failed[0].startswith("csv:")  # 저장 실패 안내가 장부 실패 안내에 덮이지 않는다
    writer2 = CorrectionWriter(store, on_saved=lambda r, m: None, on_failed=lambda r, e: None, on_csv_failed=lambda r, e: failed.append("csv:" + e))
    writer2.start()
    writer2.submit_csv(make_recording(), KIND_AUTO_ONLY)  # 장부만 쓰다가 실패한 경우는 알린다
    writer2.close()
    assert len(failed) == 2 and failed[1].startswith("csv:")


def test_csv_has_header_once_and_kind_column(tmp_path):
    store = CorrectionStore(tmp_path / "c")
    store.append_csv(make_recording(seed=31, human=(0, 0), wait=(0, 0)), KIND_AUTO_ONLY)
    store.append_csv(make_recording(seed=32), KIND_DISCARDED)
    store.append_csv(make_recording(seed=33), KIND_SAVED, episode_index=4)
    lines = (tmp_path / "c" / "auto_runs.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == CSV_FIELDS and len(lines) == 4
    rows = store.csv_rows()
    assert [(r["seed"], r["kind"]) for r in rows] == [("31", "auto_only"), ("32", "correction_discarded"), ("33", "correction_saved")]
    assert rows[0]["outcome"] == "success" and float(rows[1]["human_time_s"]) == pytest.approx(20 / 30, abs=1e-2) and rows[2]["episode_index"] == "4"
    assert store.used_seeds() == [31, 32, 33]


def test_checkpoint_identity_hashes_the_model_file(tmp_path):
    folder = tmp_path / "pretrained_model"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"abc")
    info = checkpoint_identity(folder)
    assert info["model_safetensors_sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert checkpoint_identity(tmp_path / "nothing")["model_safetensors_sha256"] is None


# --- 시드 배분 ---


def test_seed_blocks_cover_28_to_99_once_in_alternating_blocks_of_12():
    seen: dict[int, str] = {}
    for start, end, purpose in cc.SEED_BLOCKS:
        assert end - start == 12 and purpose in cc.PURPOSE_LABELS
        for seed in range(start, end):
            assert seed not in seen
            seen[seed] = purpose
    assert sorted(seen) == list(range(28, 100))  # 학습에 쓴 시연 시드(0~27)를 뺀 72개를 빈틈없이
    assert sum(1 for p in seen.values() if p == cc.PURPOSE_CORRECTION) == 36 == sum(1 for p in seen.values() if p == cc.PURPOSE_EXTRA_DEMO)
    purposes = [p for _, _, p in cc.SEED_BLOCKS]
    assert purposes == [cc.PURPOSE_CORRECTION, cc.PURPOSE_EXTRA_DEMO] * 3  # 번갈아


def test_seed_purposes_do_not_clash_with_eval_or_auto_seed_rules():
    for seed in range(*config.EVAL_SEED_RANGE):
        assert cc.purpose_for_seed(seed) is None  # 평가 시드에는 용도 안내가 붙지 않는다
        assert seed_kind(seed) == EVAL
    for seed in range(0, 28):
        assert cc.purpose_for_seed(seed) is None  # 학습에 쓴 시드
    for seed in range(28, 100):
        assert seed_kind(seed) == DEMO and cc.purpose_for_seed(seed) in cc.PURPOSE_LABELS
        block_xy_for_seed(seed)  # 리셋이 낼 수 있는 시드다
    assert cc.purpose_for_seed(None) is None and cc.purpose_for_seed(100) is None and cc.purpose_for_seed(-1) is None
    assert cc.SEED_BLOCKS[0][0] > 27 and cc.SEED_BLOCKS[-1][1] <= config.DEMO_SEED_RANGE[1]  # 시연 시드 범위 안


# --- 프로토콜 ---


@pytest.mark.parametrize("action", ["takeover", "handback", "save_correction", "discard_correction", "start", "stop", "shutdown_policy"])
def test_auto_actions_are_accepted(action):
    assert action in AUTO_ACTIONS
    assert parse_client_message(json.dumps({"type": "auto", "action": action}), 4096)["action"] == action


@pytest.mark.parametrize(
    "msg",
    [{"type": "auto", "action": "take_over"}, {"type": "auto", "action": 1}, {"type": "auto"}, {"type": "auto", "action": "takeover", "x": 1},
     {"type": "auto", "action": None}, {"type": "auto", "action": ["takeover"]}],
)
def test_bad_auto_messages_are_rejected(msg):
    with pytest.raises(ProtocolError):
        parse_client_message(json.dumps(msg), 4096)
