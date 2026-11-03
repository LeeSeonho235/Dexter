"""녹화기: 틱 정렬(state와 action이 같은 순간), 격자 간격, 최대 길이, 정지 구간, 상태 기계, 샘플링 비용."""

import time

import mujoco
import numpy as np
import pytest

from server import config
from server.grid import grid_step_indices
from server.recorder import END_MAX_LENGTH, END_STOP_FAIL, END_STOP_SUCCESS, EpisodeRecorder, still_stats

TS = 0.005


class Sink:
    def __init__(self):
        self.episodes = []

    def __call__(self, ep):
        self.episodes.append(ep)


def make_recorder(sink, max_s=60.0, timestep_s=TS):
    return EpisodeRecorder(sink=sink, timestep_s=timestep_s, max_s=max_s)


def begin(rec, index=0, seed=0):
    rec.begin(index, seed, (0.2, 0.0, 0.015), config.TARGET_CENTER_M)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))


def new_data(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


def run_physics(model, data, rec, n_steps, ctrl_at):
    """SimWorker._loop와 같은 순서: ctrl을 정하고 → 물리 스텝 직전에 on_step → mj_step. 스텝 직전 (qpos, ctrl) 기준 기록도 같이 돌려준다."""
    reference = []
    for step in range(n_steps):
        data.ctrl[:6] = ctrl_at(step)
        reference.append((data.qpos.copy(), data.ctrl[:6].copy()))
        rec.on_step(data.time, data.qpos, data.ctrl)
        mujoco.mj_step(model, data)
    return reference


def jump_ctrl(step):
    ctrl = np.array([0.0, -1.74, 1.2, -1.2, 0.0, 0.0])
    if step >= 100:
        ctrl[0] = 0.6  # 100 스텝째에 어깨 회전 목표를 크게 바꾼다
    return ctrl


def test_state_and_action_come_from_the_same_instant(model):
    """샘플 k의 (qpos, ctrl)는 독립된 기준 실행의 같은 스텝 직전 값과 정확히 같다. 한 칸 어긋나면 여기서 걸린다."""
    sink = Sink()
    rec = make_recorder(sink)
    data = new_data(model)
    begin(rec)
    reference = run_physics(model, data, rec, 400, jump_ctrl)
    ep = rec.stop(True)
    assert ep is not None

    steps = grid_step_indices(ep.n_ticks, config.RECORD_FPS, TS)
    assert ep.n_ticks > 50
    for k, step in enumerate(steps):
        ref_qpos, ref_ctrl = reference[step]
        np.testing.assert_array_equal(ep.qpos[k], ref_qpos)
        np.testing.assert_array_equal(ep.ctrl[k], ref_ctrl)
        assert ep.sim_time_s[k] == pytest.approx(step * TS, abs=1e-9)


def test_ctrl_change_shows_up_in_qpos_one_sample_later(model):
    """ctrl가 바뀐 샘플에서는 qpos가 아직 그 ctrl의 영향을 받기 전이다(영향은 다음 샘플부터)."""
    rec = make_recorder(Sink())
    data = new_data(model)
    begin(rec)
    run_physics(model, data, rec, 400, jump_ctrl)
    ep = rec.stop(True)
    steps = grid_step_indices(ep.n_ticks, config.RECORD_FPS, TS)
    k_jump = next(k for k, s in enumerate(steps) if s >= 100)
    assert ep.ctrl[k_jump][0] == 0.6 and ep.ctrl[k_jump - 1][0] == 0.0
    before = abs(ep.qpos[k_jump][0] - ep.qpos[k_jump - 1][0])  # 점프 전 구간의 움직임
    after = abs(ep.qpos[k_jump + 1][0] - ep.qpos[k_jump][0])  # 점프 직후 구간의 움직임
    assert after > before + 1e-3


def test_samples_are_on_the_30_hz_grid(model):
    rec = make_recorder(Sink())
    data = new_data(model)
    begin(rec)
    run_physics(model, data, rec, 600, lambda s: np.array([0, -1.74, 1.2, -1.2, 0, 0.0]))  # 3 초
    ep = rec.stop(False)
    spacing_ms = {round(d * 1000) for d in np.diff(ep.sim_time_s)}
    assert spacing_ms == {30, 35}
    assert ep.n_ticks == 90  # 3 초 × 30 Hz
    ideal = ep.sim_time_s[0] + np.arange(ep.n_ticks) / config.RECORD_FPS
    assert np.abs(ep.sim_time_s - ideal).max() <= TS / 2 + 1e-9


def test_nothing_is_recorded_when_not_active(model):
    sink = Sink()
    rec = make_recorder(sink)
    data = new_data(model)
    run_physics(model, data, rec, 50, jump_ctrl)
    assert rec.state()["ticks"] == 0 and sink.episodes == []


def test_max_length_ends_as_failure_and_calls_sink_once(model):
    sink = Sink()
    rec = make_recorder(sink, max_s=1.0)  # 30 샘플
    data = new_data(model)
    begin(rec, index=4, seed=9)
    run_physics(model, data, rec, 400, jump_ctrl)  # 2 초 진행해도 1 초에서 끝난다
    assert not rec.active
    assert len(sink.episodes) == 1
    ep = sink.episodes[0]
    assert ep.n_ticks == 30 and ep.end_reason == END_MAX_LENGTH and ep.success is False
    assert (ep.episode_index, ep.seed) == (4, 9)


def test_stop_reasons_and_inactive_stop(model):
    sink = Sink()
    rec = make_recorder(sink)
    assert rec.stop(True) is None  # 녹화 중이 아니면 아무 일도 없다
    data = new_data(model)
    for success, reason in ((True, END_STOP_SUCCESS), (False, END_STOP_FAIL)):
        begin(rec)
        run_physics(model, data, rec, 100, jump_ctrl)
        ep = rec.stop(success)
        assert ep.success is success and ep.end_reason == reason
    assert len(sink.episodes) == 2


def test_discard_does_not_call_sink(model):
    sink = Sink()
    rec = make_recorder(sink)
    data = new_data(model)
    assert rec.discard() is False
    begin(rec)
    run_physics(model, data, rec, 100, jump_ctrl)
    assert rec.state()["ticks"] > 0
    assert rec.discard() is True
    assert not rec.active and sink.episodes == []
    assert rec.state()["ticks"] == 0


def test_too_short_episode_is_dropped(model):
    sink = Sink()
    rec = make_recorder(sink)
    data = new_data(model)
    begin(rec)
    run_physics(model, data, rec, 3, jump_ctrl)  # 샘플 1개
    assert rec.stop(True) is None
    assert sink.episodes == [] and not rec.active


def test_new_episode_starts_clean(model):
    sink = Sink()
    rec = make_recorder(sink)
    data = new_data(model)
    begin(rec, index=0)
    run_physics(model, data, rec, 100, jump_ctrl)
    rec.stop(True)
    begin(rec, index=1)
    run_physics(model, data, rec, 100, jump_ctrl)
    ep = rec.stop(True)
    assert ep.episode_index == 1
    assert ep.sim_time_s[0] == pytest.approx(data.time - 100 * TS, abs=1e-9)  # 두 번째 에피소드의 첫 샘플 시각


def test_episode_arrays_have_expected_shapes_and_no_intervention(model):
    rec = make_recorder(Sink())
    data = new_data(model)
    begin(rec)
    run_physics(model, data, rec, 300, jump_ctrl)
    ep = rec.stop(True)
    n = ep.n_ticks
    assert ep.sim_time_s.shape == (n,) and ep.qpos.shape == (n, 13) and ep.ctrl.shape == (n, 6)
    assert ep.is_intervention.shape == (n,) and ep.is_intervention.dtype == bool
    assert not ep.is_intervention.any()
    assert ep.sim_time_s.dtype == ep.qpos.dtype == ep.ctrl.dtype == np.float64
    assert ep.source == "demo"


# --- 정지 구간 ---


def feed(rec, ctrl_rows, qpos_rows=None):
    """격자 시각에 맞춰 샘플을 직접 넣는다(물리 없이)."""
    steps = grid_step_indices(len(ctrl_rows), config.RECORD_FPS, TS)
    for k, step in enumerate(steps):
        qpos = np.zeros(13) if qpos_rows is None else qpos_rows[k]
        rec.on_step(10.0 + step * TS, qpos, np.asarray(ctrl_rows[k], dtype=float))


def test_still_stats_counts_small_changes_only():
    still = np.zeros((11, 6))
    ratio, longest_s = still_stats(still, config.STILL_CTRL_DELTA_RAD, 30)
    assert ratio == 1.0 and longest_s == pytest.approx(10 / 30)
    moving = np.zeros((11, 6))
    moving[:, 2] = np.arange(11) * 0.01  # 한 샘플에 0.01 rad
    assert still_stats(moving, config.STILL_CTRL_DELTA_RAD, 30)[0] == 0.0
    assert still_stats(np.zeros((1, 6)), config.STILL_CTRL_DELTA_RAD, 30) == (0.0, 0.0)


def test_recorder_still_ratio_matches_batch_function():
    rows = np.zeros((61, 6))
    rows[20:30, 0] = np.arange(10) * 0.05  # 10 샘플 움직임
    rows[30:, 0] = rows[29, 0]
    rec = make_recorder(Sink())
    begin(rec)
    feed(rec, rows)
    live = rec.state()
    ep = rec.stop(True)
    ratio, longest_s = still_stats(ep.ctrl, config.STILL_CTRL_DELTA_RAD, config.RECORD_FPS)
    assert ep.still_ratio == pytest.approx(ratio) and ep.still_longest_s == pytest.approx(longest_s)
    assert live["still_ratio"] == pytest.approx(ratio, abs=1e-3)
    assert live["still_longest_s"] == pytest.approx(longest_s, abs=0.01)
    assert 0.3 < ep.still_ratio < 0.9  # 1/3은 움직임, 나머지는 정지
    # 뒤쪽 정지: 이웃 쌍 30~60번 = 31개(샘플 29→30도 값이 같아 정지로 센다)
    assert ep.still_longest_s == pytest.approx(31 / config.RECORD_FPS, abs=1e-9)


def test_gripper_only_change_counts_as_moving():
    rows = np.zeros((31, 6))
    rows[:, 5] = np.linspace(0, 1.0, 31)  # 집게만 계속 변한다(샘플당 약 0.033 rad)
    assert still_stats(rows, config.STILL_CTRL_DELTA_RAD, 30)[0] == 0.0


# --- 자동 성공 판정 ---


def test_auto_success_hold_is_tracked_and_reported():
    rec = make_recorder(Sink())
    begin(rec)
    qpos = np.zeros((60, 13))
    qpos[:, 5] = 0.9
    qpos[:, 6:9] = (0.2, 0.0, 0.015)  # 목표 밖
    qpos[30:, 6:9] = (0.2, 0.12, 0.015)  # 1 초 뒤부터 목표 안
    seen_met = []
    steps = grid_step_indices(60, config.RECORD_FPS, TS)
    for k, step in enumerate(steps):
        rec.on_step(5.0 + step * TS, qpos[k], np.zeros(6))
        seen_met.append(rec.state()["success_condition_met"])
    assert seen_met[30] is False  # 막 들어왔을 때는 아직
    assert seen_met[-1] is True  # 0.5 초 이상 유지
    ep = rec.stop(True)
    assert ep.auto_success_ever and ep.auto_success_at_end
    assert ep.auto_success_first_s == pytest.approx(5.0 + 1.0 + 0.5, abs=0.04)


def test_auto_success_differs_from_manual_when_not_reached():
    rec = make_recorder(Sink())
    begin(rec)
    feed(rec, np.zeros((40, 6)))
    ep = rec.stop(True)  # 내가 성공이라고 눌렀지만 조건은 한 번도 안 맞았다
    assert ep.success is True and ep.auto_success_ever is False and ep.auto_success_first_s is None


# --- 비용 ---


def test_on_step_is_cheap_when_idle_and_when_recording():
    rec = make_recorder(Sink())
    qpos, ctrl = np.zeros(13), np.zeros(6)
    t0 = time.perf_counter()
    for i in range(20000):
        rec.on_step(i * TS, qpos, ctrl)
    idle_us = (time.perf_counter() - t0) / 20000 * 1e6
    begin(rec)
    t0 = time.perf_counter()
    for i in range(8000):  # 40 초 분량의 물리 스텝
        rec.on_step(i * TS, qpos, ctrl)
    active_us = (time.perf_counter() - t0) / 8000 * 1e6
    # 한 틱(33 ms)에 스텝이 약 7번이다. 스텝당 비용이 수십 µs 이하면 틱 예산(33,000 µs)에 비해 무시할 수준이다.
    assert idle_us < 20 and active_us < 100
