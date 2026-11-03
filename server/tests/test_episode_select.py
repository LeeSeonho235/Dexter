"""3-3 에피소드 선택·품질 점검(train/episode_select.py)과 train/dataset_config.py가 서버 설정과 맞는지.

train/은 패키지가 아니라서 경로를 직접 추가해 불러온다(numpy와 json만 쓰는 모듈이라 server/.venv에서 돈다).
"""

import json
import sys

import numpy as np

from server import config
from server.episode_store import JOINT_NAMES
from server.recorder import CTRL_DIM, QPOS_DIM

sys.path.insert(0, str(config.REPO_ROOT / "train"))
import dataset_config as cfg  # noqa: E402
import episode_select as sel  # noqa: E402


def meta(index, success=True, **over):
    base = {
        "episode_index": index, "seed": index, "success": success,
        "end_reason": "stop_success" if success else "stop_fail", "n_ticks": 100, "duration_s": 3.3 * 3,
        "still_ratio": 0.3, "still_longest_s": 1.0, "success_differs_from_auto": False,
    }
    base.update(over)
    return base


def arrays(n=100, dt_ms=(30.0, 35.0)):
    t = np.concatenate([[0.0], np.cumsum([dt_ms[i % 2] / 1000.0 for i in range(n - 1)])])
    return {"sim_time_s": t, "qpos": np.zeros((n, 13)), "ctrl": np.zeros((n, 6))}


def test_config_matches_server():
    assert cfg.FPS == config.RECORD_FPS
    assert cfg.JOINT_NAMES == JOINT_NAMES
    assert (cfg.QPOS_DIM, cfg.CTRL_DIM) == (QPOS_DIM, CTRL_DIM)
    assert cfg.N_JOINTS == CTRL_DIM
    assert cfg.RAW_DIR == config.RECORD_DIR
    assert cfg.IMAGE_KEY == "observation.images.main"


def test_only_successful_episodes_are_included():
    rows = sel.select([meta(0, False), meta(1), meta(2, False, end_reason="max_length"), meta(3)])
    assert [r["index"] for r in rows if r["included"]] == [1, 3]
    assert "max_length" in rows[2]["reason"]


def test_user_exclude_only_removes_what_is_named():
    rows = sel.select([meta(1), meta(2), meta(3)], exclude={2})
    assert [r["index"] for r in rows if r["included"]] == [1, 3]
    assert rows[1]["reason"] == "사용자가 제외 지정"


def test_success_must_be_exactly_true():
    assert not sel.select([meta(1, success=None)])[0]["included"]


def test_clean_episode_has_no_problems():
    assert sel.check_arrays(meta(1), arrays()) == []


def test_nan_in_ctrl_or_qpos_is_reported():
    a = arrays()
    a["ctrl"][5, 2] = np.nan
    a["qpos"][7, 0] = np.inf
    problems = sel.check_arrays(meta(1), a)
    assert any("ctrl" in p for p in problems) and any("qpos" in p for p in problems)


def test_tick_interval_outside_grid_is_reported():
    a = arrays()
    a["sim_time_s"][50:] += 0.02  # 한 곳만 50ms 간격
    problems = sel.check_arrays(meta(1), a)
    assert len(problems) == 1 and "틱 간격" in problems[0]


def test_shape_mismatch_is_reported():
    a = arrays(n=90)
    assert "shape" in sel.check_arrays(meta(1, n_ticks=100), a)[0]


def test_quality_row_warns_but_does_not_exclude():
    m = meta(1, still_ratio=0.8, still_longest_s=4.0, success_differs_from_auto=True, duration_s=2.0)
    row = sel.quality_row(m, arrays())
    text = " ".join(row["warnings"])
    for word in ("정지 비율", "가장 긴 정지", "자동 판정", "길이"):
        assert word in text
    assert row["data_problems"] == []
    assert sel.select([m])[0]["included"]  # 경고가 있어도 제외하지 않는다


def test_distribution_percentiles():
    d = sel.distribution([1, 2, 3, 4, 5])
    assert d == {"min": 1.0, "p25": 2.0, "median": 3.0, "p75": 4.0, "max": 5.0}


def test_load_metas_reads_json_only_and_sorts(tmp_path):
    for i in (2, 0, 1):
        (tmp_path / f"episode_{i:06d}.json").write_text(json.dumps(meta(i)), encoding="utf-8")
    (tmp_path / "episode_000009.npz").write_bytes(b"")  # json 없는 npz는 저장된 것이 아니다
    assert [m["episode_index"] for m in sel.load_metas(tmp_path)] == [0, 1, 2]
