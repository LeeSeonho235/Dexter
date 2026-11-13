"""server/aux_view_bench.py의 측정 없이 확인할 수 있는 부분: 통계 수집, 비교 쌍과 제안 기준, 인자 검사, 부하·체크포인트 거부, 조건표.
실제 측정(SimWorker, aux, 정책 프로세스를 띄워 도는 것)은 이 테스트에서 하지 않는다."""

import os
import sys
import time

import pytest

from server import auto_config
from server import aux_view_bench as bench
from server import aux_view_config as ac


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


class FakeSim:
    def __init__(self) -> None:
        self.tick_stats = bench.RecordingTickStats()
        self._t0 = time.monotonic()

    @property
    def frames_rendered(self) -> int:
        return int((time.monotonic() - self._t0) * 15)  # 15 fps처럼 늘어난다


class FakeAux:
    def state(self):
        return {"slowed": False, "error": None, "frames": 0, "last_render_ms": None}


def test_measure_stops_early_and_reports_the_real_duration_and_main_fps():
    sim = FakeSim()
    for _ in range(10):
        sim.tick_stats.add(0.033)
    started = time.monotonic()
    result = bench.measure(sim, FakeAux(), bench.FrameLog(), seconds=5.0, stop_early=lambda: time.monotonic() - started > 0.3, poll_s=0.02)
    assert 0.25 < result["measured_s"] < 1.0  # AUTO가 일찍 끝나면 거기까지만 잰다
    assert 10 <= result["main_fps"] <= 20
    assert result["ticks"] == 0  # measure 시작에서 틱 통계를 지운다(측정 구간만)
    assert "aux_frames" not in result


def make_result(p95, over50, cpu, fps=15.0, aux=None, slowed=False, error=None):
    result = {
        "tick_p95_ms": p95, "tick_over50_pct": over50, "cpu_cores": cpu, "main_fps": fps,
        "aux_state": {"slowed": slowed, "error": error, "frames": 0},
    }
    if aux:
        result.update(aux)
    return result


GOOD_AUX = {"aux_frames": 150, "aux_render_p95_ms": 15.0, "aux_fps": 9.9}


def test_pairs_reference_known_conditions_and_g_h_differ_only_in_aux():
    assert set(bench.PAIRS) | set(bench.PAIRS.values()) <= set(bench.CONDITIONS)
    g, h = bench.CONDITIONS["G"], bench.CONDITIONS["H"]
    assert g["mode"] == h["mode"] == "auto" and g["aux"] == "off" and h["aux"] == "view" and g["main_viewers"] == h["main_viewers"]
    assert bench.CONDITIONS["F"]["aux"] == "idle" and bench.CONDITIONS["A"]["main_viewers"] == 0
    assert bench.CONDITIONS["E"]["aux"] == "off" and bench.CONDITIONS["D"]["aux"] == "view" and bench.CONDITIONS["D"]["mode"] == bench.CONDITIONS["E"]["mode"] == "record"
    assert bench.PAIRS == {"F": "B", "C": "B", "E": "B", "D": "E", "H": "G"}


def test_check_criteria_passes_and_flags_each_criterion():
    base = make_result(55.0, 5.0, 1.5)
    ok = bench.check_criteria({"B": base, "C": make_result(56.0, 5.5, 1.8, aux=GOOD_AUX)})
    assert all(ok["C"].values()), ok

    edge = bench.check_criteria({"B": base, "C": make_result(58.0, 8.0, 2.0, fps=14.5, aux=GOOD_AUX)})  # 정확히 기준값(+3, +3, -0.5, +0.5)은 통과
    assert all(edge["C"].values()), edge

    worse = bench.check_criteria({
        "B": base,
        "C": make_result(58.1, 8.1, 2.01, fps=14.4, aux={"aux_frames": 100, "aux_render_p95_ms": 40.1, "aux_fps": 7.9}, slowed=True),
    })  # 기준을 조금씩 넘으면 항목마다 실패
    assert worse["C"]["aux_has_frames"] is True  # 프레임은 있었다(그래서 이 항목만 참)
    assert not any(ok for name, ok in worse["C"].items() if name != "aux_has_frames"), worse["C"]


def test_check_criteria_uses_the_right_baseline_for_record_and_auto():
    results = {
        "B": make_result(53.0, 5.0, 1.5),
        "E": make_result(55.5, 5.0, 1.6),  # 녹화 자체의 비용(E-B)
        "D": make_result(58.0, 5.2, 1.9, aux=GOOD_AUX),  # D는 E와 비교한다: B와 비교하면 +5로 기준을 넘지만 E와는 +2.5로 안쪽
        "G": make_result(90.0, 40.0, 1.2),
        "H": make_result(91.0, 41.0, 1.5, aux=GOOD_AUX),
    }
    checks = bench.check_criteria(results)
    assert set(checks) == {"E", "D", "H"}  # F, C는 측정 안 했으니 건너뛴다
    assert all(checks["D"].values()) and all(checks["H"].values()) and all(checks["E"].values())


def test_check_criteria_flags_an_idle_aux_that_rendered():
    base = make_result(55.0, 5.0, 1.5)
    quiet = make_result(55.5, 5.0, 1.5)
    quiet["aux_state"]["frames"] = 0
    assert bench.check_criteria({"B": base, "F": quiet})["F"]["aux_idle_renders_nothing"] is True
    noisy = make_result(55.5, 5.0, 1.5)
    noisy["aux_state"]["frames"] = 12  # 보는 연결이 0인데 그렸다면 끄는 장치가 고장 난 것
    assert bench.check_criteria({"B": base, "F": noisy})["F"]["aux_idle_renders_nothing"] is False


def test_check_criteria_skips_failed_or_missing_measurements():
    ok = make_result(55.0, 5.0, 1.5)
    assert bench.check_criteria({"C": make_result(56.0, 5.0, 1.6, aux=GOOD_AUX)}) == {}  # 대조 없음
    assert bench.check_criteria({"B": ok, "C": {"error": "AUTO가 시작되지 않았다"}}) == {}
    assert bench.check_criteria({"G": {"error": "x"}, "H": make_result(60.0, 5.0, 1.5, aux=GOOD_AUX)}) == {}


def test_criteria_are_the_values_agreed_on_2026_11_13():
    c = bench.CRITERIA
    assert (c["tick_p95_ms_over_base"], c["over50_pct_points_over_base"], c["main_fps_drop_max"], c["cpu_cores_over_base"]) == (3.0, 3.0, 0.5, 0.5)
    assert (c["aux_render_p95_ms_max"], c["aux_fps_min"]) == (40.0, 8.0)
    assert c["aux_render_p95_ms_max"] == ac.AUX_MAX_RENDER_MS  # 감속 기준과 같은 값
    assert ac.AUX_ENABLED is False or "AUX_ENABLED" in os.environ  # 기본 꺼짐 유지


def test_estimate_is_under_five_minutes_for_the_full_forward_and_reverse_orders():
    forward = bench.estimate_seconds("ABFCEDGH", 20.0, 3.0)
    reverse = bench.estimate_seconds("HGDECFBA", 20.0, 3.0)
    assert forward == reverse and 150 < forward < 300
    assert bench.estimate_seconds("AB", 20.0, 3.0) == 2 * 23.0 + 5.0


def test_there_is_no_feeder_anymore():
    assert not hasattr(bench, "AuxFeeder")  # 실제 attach_aux 경로를 쓴다


@pytest.mark.parametrize("order", ["", "AXB", "AAB", "abc"])
def test_bad_order_is_rejected(monkeypatch, order):
    monkeypatch.setattr(sys, "argv", ["aux_view_bench", "--order", order])
    with pytest.raises(SystemExit):
        bench.main()


def test_refuses_to_run_when_load_is_too_high(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aux_view_bench", "--order", "BC", "--max-load", "0"])
    monkeypatch.setattr(bench.os, "getloadavg", lambda: (3.0, 2.0, 1.0))
    started = []
    monkeypatch.setattr(bench, "Bench", lambda *a, **k: started.append(1))  # 불리면 안 된다
    assert bench.main() == 2
    assert started == [] and "측정하지 않는다" in capsys.readouterr().err


def test_auto_conditions_refuse_when_the_real_checkpoint_is_missing_and_never_fall_back(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["aux_view_bench", "--order", "GH"])
    monkeypatch.setattr(auto_config, "POLICY_CHECKPOINT", tmp_path / "없는_체크포인트")
    started = []
    monkeypatch.setattr(bench, "Bench", lambda *a, **k: started.append(1))
    assert bench.main() == 3
    err = capsys.readouterr().err
    assert started == [] and "체크포인트가 없다" in err and "가짜 정책으로 대신하지 않는다" in err


def test_preflight_passes_with_the_real_files_in_this_repo():
    problem = bench.preflight_auto()
    if problem is not None:
        pytest.skip(f"이 컴퓨터에는 AUTO 파일이 없다: {problem}")
    assert problem is None


def test_parse_seeds_accepts_only_distinct_demo_seeds():
    assert bench.parse_seeds("0,1,2") == [0, 1, 2]
    assert bench.parse_seeds("5") == [5]
    for bad in ("", "a,b", "1,1", "-1", "100", "1000"):  # 1000은 평가 시드라서 막는다
        with pytest.raises(ValueError):
            bench.parse_seeds(bad)


def test_bad_auto_seeds_are_rejected_by_the_command_line(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["aux_view_bench", "--order", "GH", "--auto-seeds", "1000"])
    with pytest.raises(SystemExit):
        bench.main()


def test_plan_pairs_g_and_h_per_seed_and_keeps_other_conditions_once():
    forward = bench.build_plan("ABFCEDGH", [0, 1, 2])
    assert forward[:6] == [(key, None) for key in "ABFCED"]
    assert forward[6:] == [("G", 0), ("H", 0), ("G", 1), ("H", 1), ("G", 2), ("H", 2)]
    reverse = bench.build_plan("HGDECFBA", [0, 1, 2])
    assert reverse[:6] == [("H", 0), ("G", 0), ("H", 1), ("G", 1), ("H", 2), ("G", 2)]
    assert reverse[6:] == [(key, None) for key in "DECFBA"]
    assert bench.build_plan("GBH", [0, 1]) == [("G", 0), ("G", 1), ("B", None), ("H", 0), ("H", 1)]  # 떨어져 있으면 따로 묶는다
    assert bench.build_plan("BC", [0, 1, 2]) == [("B", None), ("C", None)]  # AUTO가 없으면 시드는 무관


RAW_1 = {"intervals_ms": [30.0, 30.0, 60.0], "render_ms": [10.0, 20.0], "jpeg_bytes": [1024, 2048], "aux_gaps": 1, "aux_span_s": 0.1, "cpu_s": 1.0, "wall_s": 2.0, "main_frames": 30}
RAW_2 = {"intervals_ms": [31.0, 52.0], "render_ms": [30.0], "jpeg_bytes": [3072], "aux_gaps": 0, "aux_span_s": 0.0, "cpu_s": 3.0, "wall_s": 2.0, "main_frames": 10}
STATE = {"slowed": False, "error": None, "frames": 3}


def test_stats_from_one_raw_and_pooled_raws_use_the_same_calculation():
    single = bench.stats_from_raw(RAW_1, STATE)
    assert single["ticks"] == 3 and single["tick_over50_pct"] == pytest.approx(33.33) and single["main_fps"] == 15.0 and single["cpu_cores"] == 0.5
    assert single["aux_fps"] == 10.0 and single["aux_frames"] == 2

    pooled = bench.stats_from_raw(bench.pool_raw([RAW_1, RAW_2]), STATE)
    assert pooled["ticks"] == 5 and pooled["measured_s"] == 4.0
    assert pooled["tick_over50_pct"] == 40.0  # 60, 52 두 개가 50 ms를 넘는다
    assert pooled["main_fps"] == 10.0 and pooled["cpu_cores"] == 1.0  # (30 + 10) / 4 s, (1 + 3) / 4 s
    assert pooled["aux_fps"] == 10.0 and pooled["aux_frames"] == 3 and pooled["aux_render_p50_ms"] == 20.0
    assert pooled["tick_p95_ms"] == bench.pct([30.0, 30.0, 60.0, 31.0, 52.0], 95)  # 틱은 이어 붙여서 백분위를 낸다(평균의 평균이 아니다)


def test_pool_of_runs_without_aux_has_no_aux_fields():
    plain = dict(RAW_1, render_ms=[], jpeg_bytes=[], aux_gaps=0, aux_span_s=0.0)
    assert "aux_frames" not in bench.stats_from_raw(bench.pool_raw([plain, plain]), STATE)


def test_auto_pair_table_gives_per_seed_differences_and_skips_incomplete_pairs():
    def run(key, seed, p95, over50, fps, cpu, ticks=230):
        return {"key": key, "seed": seed, "result": {"ticks": ticks, "tick_p95_ms": p95, "tick_over50_pct": over50, "main_fps": fps, "cpu_cores": cpu}}

    runs = [
        run("G", 0, 46.0, 1.0, 14.5, 1.7), run("H", 0, 47.5, 2.5, 14.3, 2.0),
        run("G", 1, 45.0, 1.0, 14.6, 1.7), {"key": "H", "seed": 1, "result": {"error": "AUTO가 시작되지 않았다"}},  # 한쪽이 실패한 시드는 뺀다
        run("H", 2, 47.0, 2.0, 14.5, 2.0),  # G가 없는 시드도 뺀다
        run("B", None, 45.0, 0.0, 15.0, 1.7),
    ]
    assert bench.auto_pair_table(runs) == [
        {"seed": 0, "ticks_g": 230, "ticks_h": 230, "d_tick_p95_ms": 1.5, "d_over50_pct_points": 1.5, "d_main_fps": -0.2, "d_cpu_cores": 0.3}
    ]


def test_estimate_counts_every_seed_for_auto_conditions():
    assert bench.estimate_seconds("GH", 20.0, 3.0, n_seeds=3) == 5.0 + 3 * (3 + 20 + 12) + 3 * (3 + 20 + 12 + 1)
    assert bench.estimate_seconds("GH", 20.0, 3.0) == bench.estimate_seconds("GH", 20.0, 3.0, n_seeds=1)
    assert bench.estimate_seconds("BC", 20.0, 3.0, n_seeds=3) == bench.estimate_seconds("BC", 20.0, 3.0)  # AUTO가 없으면 시드 수는 무관
