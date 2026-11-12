"""server/aux_view_bench.py의 측정 없이 확인할 수 있는 부분: 통계 수집, 제안 기준 비교, 인자 검사, 부하 거부, feeder의 qpos 구성.
실제 측정(SimWorker를 띄워 도는 것)은 이 테스트에서 하지 않는다."""

import sys
import time

import numpy as np
import pytest

from server import aux_view_bench as bench
from server.aux_view import QPOS_DIM


def test_recording_tick_stats_keeps_all_intervals_and_resets():
    stats = bench.RecordingTickStats()
    for ms in (30.0, 34.0, 60.0):
        stats.add(ms / 1000.0)
    assert stats.intervals_ms == pytest.approx([30.0, 34.0, 60.0])
    assert stats.summary()["over"]["50"] == 1  # 부모 TickStats의 집계도 그대로
    stats.reset()
    assert stats.intervals_ms == [] and stats.summary()["n"] == 0


def test_pct_handles_empty_and_values():
    assert bench.pct([], 95) is None
    assert bench.pct([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_frame_log_records_only_while_recording():
    class FakeAux:
        def state(self):
            return {"last_render_ms": 12.5}

    log = bench.FrameLog()
    log.aux = FakeAux()
    log(1, b"x" * 100)  # recording이 꺼져 있으면 기록하지 않는다
    assert log.render_ms == []
    log.recording = True
    log(2, b"x" * 2048)
    assert log.render_ms == [12.5] and log.jpeg_bytes == [2048] and len(log.times_s) == 1
    log.clear()
    assert log.render_ms == [] and log.jpeg_bytes == [] and log.times_s == []


def make_result(p95, over50, cpu, aux=None):
    result = {"tick_p95_ms": p95, "tick_over50_pct": over50, "cpu_cores": cpu, "aux_state": {"slowed": False, "error": None}}
    if aux:
        result.update(aux)
    return result


GOOD_AUX = {"aux_frames": 100, "aux_render_p95_ms": 20.0, "aux_jpeg_kb_max": 6.0}


def test_provisional_check_passes_and_flags_each_criterion():
    base = make_result(55.0, 5.0, 1.5)
    ok = bench.provisional_check({"B": base, "C": make_result(56.0, 5.5, 1.7, GOOD_AUX)})
    assert all(ok["C"].values()), ok

    worse = bench.provisional_check({
        "B": base,
        "C": make_result(58.0, 7.0, 2.0, {"aux_frames": 100, "aux_render_p95_ms": 45.0, "aux_jpeg_kb_max": 12.0}),
    })
    assert not any(worse["C"][name] for name in ("tick_p95", "over50", "cpu", "aux_render_p95", "jpeg_kb"))

    slowed = make_result(55.0, 5.0, 1.5, GOOD_AUX)
    slowed["aux_state"] = {"slowed": True, "error": None}
    assert bench.provisional_check({"B": base, "D": slowed})["D"]["aux_not_slowed"] is False


def test_provisional_check_needs_baseline_and_frames():
    assert bench.provisional_check({"C": make_result(56.0, 5.0, 1.6, GOOD_AUX)}) == {}  # B 없음
    assert bench.provisional_check({"B": make_result(55.0, 5.0, 1.5), "C": make_result(56.0, 5.0, 1.6)}) == {}  # aux 프레임 없음


def test_criteria_are_marked_provisional_and_use_the_agreed_proposals():
    c = bench.CRITERIA_PROVISIONAL
    assert (c["tick_p95_ms_over_B"], c["over50_pct_points_over_B"], c["cpu_cores_over_B"], c["jpeg_kb_max"]) == (2.0, 1.0, 0.3, 10.0)


def test_bad_order_is_rejected(monkeypatch):
    for order in ("", "AXB", "AAB"):
        monkeypatch.setattr(sys, "argv", ["aux_view_bench", "--order", order])
        with pytest.raises(SystemExit):
            bench.main()


def test_refuses_to_run_when_load_is_too_high(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aux_view_bench", "--max-load", "0"])
    monkeypatch.setattr(bench.os, "getloadavg", lambda: (3.0, 2.0, 1.0))
    started = []
    monkeypatch.setattr(bench, "SimWorker", lambda *a, **k: started.append(1))  # 불리면 안 된다
    assert bench.main() == 2
    assert started == [] and "측정하지 않는다" in capsys.readouterr().err


def test_feeder_builds_a_13_value_qpos_with_identity_block_rotation():
    class FakeSim:
        snapshot = {"joints_rad": np.arange(6) * 0.1, "block_m": np.array([0.2, -0.05, 0.015])}

    class FakeAux:
        def __init__(self):
            self.offered = []

        def offer_qpos(self, qpos, auto_running=False):
            self.offered.append(np.array(qpos))

    aux = FakeAux()
    feeder = bench.AuxFeeder(FakeSim(), aux)
    feeder.start()
    deadline = time.monotonic() + 3.0
    while not aux.offered and time.monotonic() < deadline:
        time.sleep(0.01)
    feeder.stop()
    feeder.join(2.0)
    assert aux.offered, "feeder가 qpos를 넘기지 않았다"
    qpos = aux.offered[0]
    assert qpos.shape == (QPOS_DIM,)
    np.testing.assert_allclose(qpos[:6], np.arange(6) * 0.1)
    np.testing.assert_allclose(qpos[6:9], [0.2, -0.05, 0.015])
    np.testing.assert_allclose(qpos[9:], [1.0, 0.0, 0.0, 0.0])
