"""보조 카메라(2-6)가 시뮬레이션 틱과 메인 화면을 얼마나 늦추는지 재는 도구(브라우저·WebSocket 없이 서버 내부 객체를 직접 돌린다).

실행(CPU만 쓴다. 정방향과 역방향을 따로 돌려 순서 효과를 본다. 한 번에 5분 안이다):
    cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.aux_view_bench --order ABFCEDGH --out <폴더>   # 정방향
    cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.aux_view_bench --order HGDECFBA --out <폴더>   # 역방향
    cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.aux_view_bench --order GH --auto-seeds 0,1,2 --out <폴더>   # AUTO만, 시드 3개
--order로 일부만 돌려도 된다(예: --order BC).

실제 서버와 같은 연결을 쓴다: SimWorker.attach_aux(실제 AuxViewWorker)가 틱마다 qpos 사본을 넘긴다(임시 feeder 스레드는 없다).
SimWorker, ControlWorker, RecordingService, AutoRunner를 server/app.py와 같은 방식으로 이어 붙이고 손 입력은 가짜 숫자(원을 그리는 손목, 웹캠 아님).
  A 화면 없음, aux 꺼짐(aux를 붙이지 않음)                    B 메인 화면, aux 꺼짐
  F 메인 화면 + aux 켜짐, 보는 연결 0 (렌더 안 함: B와 같아야 한다)   C 메인 화면 + aux 두 시점, 보는 연결 1
  E 메인 화면 + 녹화, aux 꺼짐 (D의 대조)                        D 메인 화면 + 녹화 + aux 보는 연결 1
  G AUTO(실제 체크포인트), 메인 화면, aux 꺼짐                    H AUTO + aux 보는 연결 1 (AUX_PAUSE_DURING_AUTO 기본 False)
비교 쌍: F-B, C-B, E-B(녹화 자체의 비용), D-E, H-G.
AUTO는 실제 정책 프로세스(server/auto_config.py의 체크포인트, train/.venv)를 쓴다. 가짜 정책은 쓰지 않는다. 체크포인트나 파이썬이 없거나
정책이 시작되지 않으면 이유를 출력하고 그 조건은 error로 남긴다.
AUTO 시드 여러 개(--auto-seeds): 시드마다 G와 H를 짝으로 이어서 돈다(시드0 G,H → 시드1 G,H …. 역방향이면 H,G). 같은 시드라서 정책 동작이 비슷하다.
  조건별 결과는 시드를 모두 합친 틱 간격으로 p50/p95 등을 낸다(AUTO 에피소드 하나는 약 8초라 틱이 약 230개뿐이어서 합친다). 시드별 값도 같이 낸다.
틱 간격 = 시뮬레이션 스레드의 틱 시작 사이 벽시계 시간(ms). 메인 렌더 틱은 약 55 ms라서 최대값은 비교에 쓰지 않고 p50/p95와 50 ms 초과 비율을 본다.
AUTO 중에는 정책을 기다리는 틱이 길어진다(약 140 ms). G와 H에 똑같이 들어가므로 H-G 차이만 본다.
메인 fps = 측정 구간의 SimWorker.frames_rendered 증가 ÷ 벽시계. CPU = 이 프로세스의 CPU 시간 ÷ 벽시계(코어 수).

한계:
  - 정책 프로세스(별도 프로세스, nice 19, 스레드 2개)의 CPU는 CPU 열에 들어가지 않는다. 서버 전체에는 그만큼의 부하가 더 있다.
  - AUTO 조건은 에피소드가 끝나거나 --seconds가 되면 끝난다. 정책이 일찍 성공하면 측정 시간이 짧아진다(measured_s에 실제 값).
  - 시드를 맞추려고 RecordingService의 비공개 값(_next_seed)을 이 도구에서만 직접 바꾼다(제품 코드에는 영향 없음).
저장: --out 폴더에 결과 JSON, 에피소드와 교정 저장은 --out/work 아래 임시 폴더(data/ 의 기존 파일은 건드리지 않는다). 영상은 저장하지 않는다.
"""

import os

# mujoco를 import하기 전에 정해야 한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from server import auto_config  # noqa: E402
from server import aux_view_config as ac  # noqa: E402
from server import config  # noqa: E402
from server.auto import AutoRunner  # noqa: E402
from server.aux_view import AuxViewWorker  # noqa: E402
from server.control import ControlWorker  # noqa: E402
from server.correction_store import CorrectionStore  # noqa: E402
from server.episode_store import EpisodeStore  # noqa: E402
from server.policy_ipc import PolicyClient  # noqa: E402
from server.record_bench import HandFeeder, wait_until  # noqa: E402
from server.recording import RecordingService  # noqa: E402
from server.sim_worker import SimWorker, TickStats  # noqa: E402

# 합격 기준(2026-11-13 사용자와 확정). 비교 쌍마다 "후보가 비교 대상보다 이만큼 이상 나빠지면 안 된다".
CRITERIA = {
    "tick_p95_ms_over_base": 3.0,  # 틱 간격 p95 증가(ms)
    "over50_pct_points_over_base": 3.0,  # 50 ms 초과 비율 증가(%p)
    "main_fps_drop_max": 0.5,  # 메인 fps 감소
    "cpu_cores_over_base": 0.5,  # CPU 증가(코어)
    "aux_render_p95_ms_max": 40.0,  # aux 렌더+JPEG p95(ms). 설정의 AUX_MAX_RENDER_MS(감속 기준)와 같은 값
    "aux_fps_min": 8.0,  # 보는 연결이 있을 때 aux가 실제로 내는 fps(설정 10)
}

# main_viewers: 메인 화면 보는 연결 수 / aux: "off"(붙이지 않음), "idle"(붙임, 보는 연결 0), "view"(붙임, 보는 연결 1) / mode: plain, record, auto
CONDITIONS = {
    "A": {"label": "A 화면 없음, aux 꺼짐", "main_viewers": 0, "aux": "off", "mode": "plain"},
    "B": {"label": "B 메인 화면, aux 꺼짐", "main_viewers": 1, "aux": "off", "mode": "plain"},
    "F": {"label": "F 메인 + aux 켜짐(보는 연결 0)", "main_viewers": 1, "aux": "idle", "mode": "plain"},
    "C": {"label": "C 메인 + aux 두 시점", "main_viewers": 1, "aux": "view", "mode": "plain"},
    "E": {"label": "E 메인 + 녹화, aux 꺼짐", "main_viewers": 1, "aux": "off", "mode": "record"},
    "D": {"label": "D 메인 + 녹화 + aux", "main_viewers": 1, "aux": "view", "mode": "record"},
    "G": {"label": "G AUTO, aux 꺼짐", "main_viewers": 1, "aux": "off", "mode": "auto"},
    "H": {"label": "H AUTO + aux", "main_viewers": 1, "aux": "view", "mode": "auto"},
}
PAIRS = {"F": "B", "C": "B", "E": "B", "D": "E", "H": "G"}  # 후보: 비교 대상
EXTRA_S = {"plain": 0.0, "record": 8.0, "auto": 12.0}  # 리셋, 정책 불러오기, 저장 등 측정 밖에서 드는 시간(추정)


class RecordingTickStats(TickStats):
    """TickStats와 같지만 틱 간격 전체를 따로 모아 p50/p95를 낼 수 있게 한다(SimWorker는 그대로, tick_stats 속성만 바꿔 끼운다)."""

    def __init__(self) -> None:
        super().__init__()
        self.intervals_ms: list[float] = []

    def reset(self) -> None:
        super().reset()
        self.intervals_ms = []

    def add(self, interval_s: float) -> None:
        super().add(interval_s)
        self.intervals_ms.append(interval_s * 1000.0)


class FrameLog:
    """aux의 on_frame 콜백이 부르는 기록기. 측정 구간 동안만 모은다. 콜백은 aux 스레드에서 불린다."""

    def __init__(self) -> None:
        self.aux: AuxViewWorker | None = None
        self.recording = False
        self.render_ms: list[float] = []
        self.jpeg_bytes: list[int] = []
        self.times_s: list[float] = []

    def clear(self) -> None:
        self.render_ms, self.jpeg_bytes, self.times_s = [], [], []

    def __call__(self, seq: int, jpeg: bytes) -> None:
        if not self.recording or self.aux is None:
            return
        self.times_s.append(time.perf_counter())
        self.jpeg_bytes.append(len(jpeg))
        self.render_ms.append(self.aux.state()["last_render_ms"])  # _note_render_ms가 콜백보다 먼저 불린다


def pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 2) if values else None


def stats_from_raw(raw: dict, aux_state: dict) -> dict:
    """원자료(raw) 하나에서 보고용 통계를 만든다. 측정 하나와 합친 여러 측정이 같은 계산을 쓴다."""
    intervals = raw["intervals_ms"]
    wall = raw["wall_s"]
    result = {
        "measured_s": round(wall, 2),
        "ticks": len(intervals),
        "tick_mean_ms": round(float(np.mean(intervals)), 2) if intervals else None,
        "tick_p50_ms": pct(intervals, 50),
        "tick_p95_ms": pct(intervals, 95),
        "tick_p99_ms": pct(intervals, 99),
        "tick_over50_pct": round(100.0 * sum(ms > 50.0 for ms in intervals) / len(intervals), 2) if intervals else None,
        "main_fps": round(raw["main_frames"] / wall, 2),
        "cpu_cores": round(raw["cpu_s"] / wall, 3),
        "aux_state": aux_state,
    }
    if raw["render_ms"]:
        result.update(
            aux_frames=len(raw["render_ms"]),
            aux_fps=round(raw["aux_gaps"] / raw["aux_span_s"], 2) if raw["aux_span_s"] > 0 else None,
            aux_render_p50_ms=pct(raw["render_ms"], 50),
            aux_render_p95_ms=pct(raw["render_ms"], 95),
            aux_jpeg_kb_mean=round(float(np.mean(raw["jpeg_bytes"])) / 1024.0, 2),
            aux_jpeg_kb_max=round(max(raw["jpeg_bytes"]) / 1024.0, 2),
        )
    return result


def pool_raw(raws: list[dict]) -> dict:
    """여러 측정(시드마다 한 번씩)의 원자료를 합친다: 틱 간격, aux 렌더 시간은 이어 붙이고 시간과 개수는 더한다."""
    return {
        "intervals_ms": [v for raw in raws for v in raw["intervals_ms"]],
        "render_ms": [v for raw in raws for v in raw["render_ms"]],
        "jpeg_bytes": [v for raw in raws for v in raw["jpeg_bytes"]],
        "aux_gaps": sum(raw["aux_gaps"] for raw in raws),
        "aux_span_s": sum(raw["aux_span_s"] for raw in raws),
        "cpu_s": sum(raw["cpu_s"] for raw in raws),
        "wall_s": sum(raw["wall_s"] for raw in raws),
        "main_frames": sum(raw["main_frames"] for raw in raws),
    }


def measure(sim: SimWorker, aux: AuxViewWorker, log: FrameLog, seconds: float, stop_early=None, poll_s: float = 0.1) -> dict:
    """seconds 동안(또는 stop_early()가 참이 될 때까지) 틱 간격, 메인 fps, CPU, aux 렌더를 모은다. 결과에 원자료("_raw")가 들어 있다."""
    sim.tick_stats.reset()
    log.clear()
    log.recording = True
    frames0 = sim.frames_rendered
    cpu0, wall0 = time.process_time(), time.monotonic()
    while True:
        elapsed = time.monotonic() - wall0
        if elapsed >= seconds or (stop_early is not None and stop_early()):
            break
        time.sleep(min(poll_s, seconds - elapsed))
    cpu, wall = time.process_time() - cpu0, time.monotonic() - wall0
    log.recording = False
    times = list(log.times_s)
    raw = {
        "intervals_ms": list(sim.tick_stats.intervals_ms),
        "render_ms": list(log.render_ms),
        "jpeg_bytes": list(log.jpeg_bytes),
        "aux_gaps": max(len(times) - 1, 0),
        "aux_span_s": (times[-1] - times[0]) if len(times) > 1 else 0.0,
        "cpu_s": cpu,
        "wall_s": wall,
        "main_frames": sim.frames_rendered - frames0,
    }
    result = stats_from_raw(raw, aux.state())
    result["_raw"] = raw
    return result


def check_criteria(results: dict[str, dict]) -> dict[str, dict[str, bool]]:
    """비교 쌍(PAIRS)마다 확정 기준 확인. 측정이 실패했거나 대조가 없으면 그 쌍은 건너뛴다."""
    c = CRITERIA
    out: dict[str, dict[str, bool]] = {}
    for key, base_key in PAIRS.items():
        r, base = results.get(key), results.get(base_key)
        if not r or not base or "error" in r or "error" in base or r.get("tick_p95_ms") is None or base.get("tick_p95_ms") is None:
            continue
        checks = {
            "tick_p95": r["tick_p95_ms"] - base["tick_p95_ms"] <= c["tick_p95_ms_over_base"],
            "over50": r["tick_over50_pct"] - base["tick_over50_pct"] <= c["over50_pct_points_over_base"],
            "main_fps": base["main_fps"] - r["main_fps"] <= c["main_fps_drop_max"],
            "cpu": r["cpu_cores"] - base["cpu_cores"] <= c["cpu_cores_over_base"],
        }
        if CONDITIONS[key]["aux"] == "view":
            checks["aux_has_frames"] = r.get("aux_frames", 0) > 0
            checks["aux_render_p95"] = r.get("aux_render_p95_ms") is not None and r["aux_render_p95_ms"] <= c["aux_render_p95_ms_max"]
            checks["aux_fps"] = (r.get("aux_fps") or 0.0) >= c["aux_fps_min"]
            checks["aux_not_slowed"] = not r["aux_state"]["slowed"] and r["aux_state"]["error"] is None
        if CONDITIONS[key]["aux"] == "idle":
            checks["aux_idle_renders_nothing"] = r.get("aux_frames", 0) == 0 and r["aux_state"]["frames"] == 0
        out[key] = checks
    return out


def parse_seeds(text: str) -> list[int]:
    """"0,1,2" → [0, 1, 2]. 시연 시드 범위(config.DEMO_SEED_RANGE) 안의 겹치지 않는 정수만 받는다."""
    try:
        seeds = [int(part) for part in text.split(",") if part.strip() != ""]
    except ValueError:
        raise ValueError(f"시드는 쉼표로 구분한 정수여야 한다: {text!r}") from None
    lo, hi = config.DEMO_SEED_RANGE
    if not seeds or len(set(seeds)) != len(seeds) or any(not lo <= s < hi for s in seeds):
        raise ValueError(f"시드는 {lo}~{hi - 1} 안의 겹치지 않는 정수 하나 이상이어야 한다: {text!r}")
    return seeds


def build_plan(order: str, seeds: list[int]) -> list[tuple[str, int | None]]:
    """(조건, 시드) 실행 목록. AUTO가 아닌 조건은 시드가 None이고 한 번. 이어진 AUTO 조건들은 시드마다 묶어서 돈다(시드0 G,H → 시드1 G,H …)."""
    plan: list[tuple[str, int | None]] = []
    i = 0
    while i < len(order):
        if CONDITIONS[order[i]]["mode"] != "auto":
            plan.append((order[i], None))
            i += 1
            continue
        j = i
        while j < len(order) and CONDITIONS[order[j]]["mode"] == "auto":
            j += 1
        for seed in seeds:
            plan.extend((key, seed) for key in order[i:j])
        i = j
    return plan


def auto_pair_table(runs: list[dict]) -> list[dict]:
    """시드마다 H-G 차이(틱 p95 ms, 50ms 초과 %p, 메인 fps, CPU 코어). runs 항목: {"key", "seed", "result"}. 둘 다 측정된 시드만."""
    by = {(run["key"], run["seed"]): run["result"] for run in runs if run["seed"] is not None}
    rows = []
    for seed in sorted({seed for _, seed in by}):
        g, h = by.get(("G", seed)), by.get(("H", seed))
        if not g or not h or "error" in g or "error" in h:
            continue
        rows.append(
            {
                "seed": seed,
                "ticks_g": g["ticks"], "ticks_h": h["ticks"],
                "d_tick_p95_ms": round(h["tick_p95_ms"] - g["tick_p95_ms"], 2),
                "d_over50_pct_points": round(h["tick_over50_pct"] - g["tick_over50_pct"], 2),
                "d_main_fps": round(h["main_fps"] - g["main_fps"], 2),
                "d_cpu_cores": round(h["cpu_cores"] - g["cpu_cores"], 3),
            }
        )
    return rows


def estimate_seconds(order: str, seconds: float, warmup: float, n_seeds: int = 1) -> float:
    """상한에 가까운 예상(s). AUTO는 에피소드가 길어져 --seconds를 다 쓰는 경우를 센다(실제로는 정책이 일찍 성공하면 더 짧다)."""
    total = 5.0
    for key in order:
        spec = CONDITIONS[key]
        runs = n_seeds if spec["mode"] == "auto" else 1
        total += runs * (warmup + seconds + EXTRA_S[spec["mode"]] + (1.0 if spec["aux"] != "off" else 0.0))
    return total


def preflight_auto() -> str | None:
    """AUTO 조건에 필요한 실제 체크포인트와 정책 파이썬이 있는지. 없으면 이유(가짜 정책으로 바꾸지 않는다)."""
    for label, path in (("체크포인트", auto_config.POLICY_CHECKPOINT), ("정책 파이썬", auto_config.POLICY_PYTHON), ("정책 스크립트", auto_config.POLICY_SCRIPT)):
        if not Path(path).exists():
            return f"{label}가 없다: {path}"
    return None


class Bench:
    """서버 안쪽 객체를 app.py와 같은 방식으로 이어 붙이고 조건마다 aux, 녹화, AUTO를 바꿔 가며 잰다."""

    def __init__(self, work_dir: Path, need_auto: bool, seconds: float, warmup: float) -> None:
        self.seconds, self.warmup = seconds, warmup
        self.sim = SimWorker(
            config.SCENE_PATH, keyframe=config.START_KEYFRAME, camera=config.CAMERA, width_px=config.RENDER_WIDTH_PX,
            height_px=config.RENDER_HEIGHT_PX, frame_fps=config.FRAME_FPS, jpeg_quality=config.JPEG_QUALITY,
            max_catch_up_s=config.MAX_CATCH_UP_S, shadow=config.RENDER_SHADOW, reflection=config.RENDER_REFLECTION,
            tick_hz=config.SIM_TICK_HZ,
        )
        self.sim.tick_stats = RecordingTickStats()
        self.sim.start()
        self.control = ControlWorker(self.sim)
        self.control.start()
        corrections = CorrectionStore(work_dir / "corrections")
        self.service = RecordingService(self.sim, self.control, store=EpisodeStore(work_dir / "raw"), extra_seeds=corrections.used_seeds)
        self.service.start()
        self.auto: AutoRunner | None = None
        if need_auto:
            self.auto = AutoRunner(
                self.sim, self.sim.owner, PolicyClient(), self.service, control=self.control, arm_check=self._arm_check,
                max_s=auto_config.AUTO_MAX_S, store=corrections,
            )
            self.service.decision_pending = self.auto.has_pending_decision
        self.hand = HandFeeder(self.control)
        self.hand.start()
        self.log = FrameLog()
        self.off_aux = AuxViewWorker(enabled=False)  # 꺼진 객체: 지표 자리를 채울 뿐 아무것도 만들지 않는다

    def _arm_check(self) -> str | None:
        """server/app.py의 arm_check와 같다: AUTO는 리셋이 끝난 home 자세에서만 시작한다."""
        snapshot = self.sim.snapshot
        if snapshot is None:
            return "시뮬레이션이 아직 준비되지 않았습니다"
        deviation_rad = float(np.max(np.abs(np.asarray(snapshot["joints_rad"]) - self.control.park_ctrl())))
        if deviation_rad > auto_config.AUTO_START_ARM_TOLERANCE_RAD:
            return f"팔이 home 자세에서 {deviation_rad:.2f} rad 벗어나 있습니다"
        return None

    def close(self) -> None:
        self.hand.stop()
        if self.auto is not None:
            self.auto.close()  # 실행 중이면 멈추고 정책 프로세스를 끈다
        self.service.close()
        self.control.stop()
        self.sim.stop()

    # --- 한 조건(AUTO면 한 시드) ---

    def run(self, key: str, seed: int | None = None) -> dict:
        spec = CONDITIONS[key]
        aux = self.off_aux
        if spec["aux"] != "off":
            aux = AuxViewWorker(enabled=True, on_frame=self.log)
            aux.start()
            aux.set_viewer_count(1 if spec["aux"] == "view" else 0)
            self.sim.attach_aux(aux)
        else:
            self.sim.attach_aux(None)
        self.log.aux = aux
        self.sim.set_viewer_count(spec["main_viewers"])
        try:
            if spec["mode"] == "record":
                return self._run_record(aux)
            if spec["mode"] == "auto":
                return self._run_auto(aux, 0 if seed is None else seed)
            time.sleep(self.warmup)
            return measure(self.sim, aux, self.log, self.seconds)
        finally:
            self.sim.attach_aux(None)
            if aux is not self.off_aux:
                aux.stop()

    def _reset_block(self, seed: int | None = None) -> str | None:
        """블록 리셋을 하고 끝나기를 기다린다. 못 하면 이유."""
        self.hand.enabled.clear()  # 리셋 중에는 손 입력이 어차피 무시된다
        if seed is not None:
            with self.service._lock:  # noqa: SLF001 - 시드를 맞추려고 이 도구에서만 직접 바꾼다
                self.service._next_seed = seed  # noqa: SLF001
        reason = self.service.handle_reset()
        if reason is not None:
            return f"리셋 거절: {reason}"
        if not wait_until(lambda: self.service.state()["block_ready"], 20.0):
            return "리셋이 20초 안에 끝나지 않았다"
        return None

    def _run_record(self, aux: AuxViewWorker) -> dict:
        reason = self._reset_block()
        if reason is not None:
            return {"error": reason}
        self.hand.enabled.set()
        time.sleep(self.warmup)
        refusal = self.service.handle_record({"action": "start"})
        if refusal is not None:
            return {"error": f"녹화 시작 거절: {refusal}"}
        result = measure(self.sim, aux, self.log, self.seconds)
        self.service.handle_record({"action": "stop_success"})
        wait_until(lambda: self.service.state()["pending_writes"] == 0, 15.0)
        return result

    def _run_auto(self, aux: AuxViewWorker, seed: int) -> dict:
        assert self.auto is not None
        reason = self._reset_block(seed)
        if reason is not None:
            return {"error": reason}
        refusal = self.auto.handle("start")
        if refusal is not None:
            return {"error": f"AUTO 시작 거절: {refusal}"}
        deadline = time.monotonic() + auto_config.POLICY_START_TIMEOUT_S + 15.0
        while True:  # 정책 프로세스 시작(첫 실행에서만 약 3초)과 에피소드 시작을 기다린다
            state = self.auto.state()
            if state["status"] == "running":
                break
            if state["status"] not in ("idle", "loading") or time.monotonic() > deadline:
                return {"error": f"AUTO가 시작되지 않았다: status={state['status']}, error={state.get('error')!r}"}
            time.sleep(0.1)
        time.sleep(self.warmup)
        result = measure(self.sim, aux, self.log, self.seconds, stop_early=lambda: self.auto.state()["status"] != "running")
        status = self.auto.state()["status"]
        result["auto"] = {"seed": seed, "status_at_window_end": status, "finished_early": status != "running"}
        if status == "running":
            self.auto.handle("stop")
        wait_until(lambda: self.auto.state()["status"] != "running", 15.0)
        result["auto"]["final_status"] = self.auto.state()["status"]
        self.hand.enabled.set()
        return result


def print_row(key: str, label: str, r: dict) -> None:
    if "error" in r:
        print(f"{label:<34} 측정 실패: {r['error']}")
        return
    line = (
        f"{label:<34} 틱 p50 {r['tick_p50_ms']:6.2f}  p95 {r['tick_p95_ms']:6.2f}  >50ms {r['tick_over50_pct']:5.2f}%  "
        f"메인 {r['main_fps']:5.2f}fps  CPU {r['cpu_cores']:.2f}코어  ({r['measured_s']}s, 틱 {r['ticks']})"
    )
    if r.get("aux_frames"):
        line += (
            f" | aux {r['aux_fps']}fps  렌더 p50 {r['aux_render_p50_ms']} / p95 {r['aux_render_p95_ms']} ms  "
            f"JPEG 평균 {r['aux_jpeg_kb_mean']} / 최대 {r['aux_jpeg_kb_max']} KB"
        )
    elif CONDITIONS[key]["aux"] == "view":
        line += " | aux 프레임 0장(측정 실패)"
    if "auto" in r:
        line += f" | AUTO 시드 {r['auto']['seed']} 끝 상태 {r['auto']['final_status']}"
    print(line)


def strip_raw(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "_raw"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=20.0, help="조건마다 재는 시간(s). AUTO는 에피소드가 먼저 끝나면 거기까지")
    parser.add_argument("--warmup", type=float, default=3.0, help="조건을 바꾼 뒤 재기 전에 기다리는 시간(s)")
    parser.add_argument("--order", default="ABFCEDGH", help="돌릴 조건과 순서(A, B, F, C, E, D, G, H 중). 일부만 줘도 된다")
    parser.add_argument("--auto-seeds", default="0", help="G, H가 쓰는 시연 시드(0~99), 쉼표로 여러 개. 시드마다 G와 H를 짝으로 돈다")
    parser.add_argument("--max-load", type=float, default=50.0, help="시작 전 load average(1분)가 이 값 이상이면 돌리지 않는다")
    parser.add_argument("--out", type=Path, default=None, help="결과 JSON과 임시 저장 폴더를 둘 폴더(기본: 임시 폴더)")
    args = parser.parse_args()
    if not args.order or any(key not in CONDITIONS for key in args.order) or len(set(args.order)) != len(args.order):
        parser.error(f"--order는 {''.join(CONDITIONS)} 중에서 겹치지 않게 쓴다(예: ABFCEDGH, HGDECFBA, BC)")
    try:
        seeds = parse_seeds(args.auto_seeds)
    except ValueError as err:
        parser.error(str(err))
    need_auto = any(CONDITIONS[key]["mode"] == "auto" for key in args.order)
    if need_auto:
        problem = preflight_auto()
        if problem is not None:
            print(f"AUTO 조건(G, H)을 돌릴 수 없다: {problem}. 가짜 정책으로 대신하지 않는다. --order에서 G, H를 빼면 나머지는 돈다.", file=sys.stderr)
            return 3

    load_1min = os.getloadavg()[0]
    if load_1min >= args.max_load:
        print(f"load average {load_1min:.1f} >= {args.max_load:g}: 서버가 바빠서 측정하지 않는다.", file=sys.stderr)
        return 2
    os.nice(19)  # 공용 서버: 다른 사람 작업을 방해하지 않게 가장 낮은 우선순위로

    planned_s = estimate_seconds(args.order, args.seconds, args.warmup, len(seeds))
    print(
        f"조건 {args.order}{' 시드 ' + ','.join(map(str, seeds)) if need_auto else ''}, 상한 약 {planned_s:.0f}초 예상. load {load_1min:.2f}. 공칭 틱 {1000 / config.SIM_TICK_HZ:.1f} ms, "
        f"메인 화면 {config.FRAME_FPS} fps, aux {ac.AUX_FPS} fps({len(ac.AUX_CAMERA_VIEWS)}시점), AUX_PAUSE_DURING_AUTO={ac.AUX_PAUSE_DURING_AUTO}"
    )
    if planned_s > 300:
        print("경고: 5분이 넘는다. --order나 --auto-seeds를 줄여 나눠 돌리는 것이 좋다.", file=sys.stderr)

    out_dir = args.out or Path(tempfile.mkdtemp(prefix="aux_bench_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    bench = Bench(out_dir / "work", need_auto, args.seconds, args.warmup)
    runs: list[dict] = []
    try:
        time.sleep(2.0)  # 팔이 움직이기 시작하고 안정될 시간
        for key, seed in build_plan(args.order, seeds):
            result = bench.run(key, seed)
            runs.append({"key": key, "seed": seed, "result": result})
            print_row(key, CONDITIONS[key]["label"] + (f" [시드 {seed}]" if seed is not None else ""), result)
            sys.stdout.flush()
    finally:
        bench.close()

    # 조건별 결과: 같은 조건의 측정(AUTO는 시드마다 한 번씩)을 합친다.
    results: dict[str, dict] = {}
    for key in args.order:
        mine = [run["result"] for run in runs if run["key"] == key]
        good = [r for r in mine if "error" not in r]
        if len(mine) == 1 or not good:
            results[key] = strip_raw(mine[0]) if good else (mine[0] if mine else {"error": "측정 안 됨"})
            continue
        pooled = stats_from_raw(pool_raw([r["_raw"] for r in good]), good[-1]["aux_state"])
        pooled["runs"] = len(good)
        pooled["auto_runs"] = [r["auto"] for r in good]
        pooled["errors"] = [r["error"] for r in mine if "error" in r]
        results[key] = pooled
    if need_auto and len(seeds) > 1:
        print("\n조건별 합친 결과(시드 " + ",".join(map(str, seeds)) + "):")
        for key in args.order:
            if CONDITIONS[key]["mode"] == "auto":
                print_row(key, CONDITIONS[key]["label"] + " [합침]", results[key])
    table = auto_pair_table(runs)
    if table:
        print("\n시드별 H-G 차이 (틱 p95 ms / 50ms 초과 %p / 메인 fps / CPU 코어):")
        for row in table:
            print(f"  시드 {row['seed']}: {row['d_tick_p95_ms']:+.2f} / {row['d_over50_pct_points']:+.2f} / {row['d_main_fps']:+.2f} / {row['d_cpu_cores']:+.3f}  (틱 G {row['ticks_g']}, H {row['ticks_h']})")

    checks = check_criteria(results)
    if checks:
        print("\n확정 기준 대비 O/X — 짝: " + ", ".join(f"{k}-{PAIRS[k]}" for k in checks))
        for key, items in checks.items():
            print(f"  {key}-{PAIRS[key]}: " + "  ".join(f"{name} {'O' if ok else 'X'}" for name, ok in items.items()))
    else:
        print("\n기준 비교는 짝(F-B, C-B, E-B, D-E, H-G)이 같이 측정됐을 때만 나온다.")

    elapsed_s = round(time.monotonic() - started, 1)
    out_path = out_dir / f"aux_bench_{args.order}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "order": args.order, "seconds": args.seconds, "warmup": args.warmup, "auto_seeds": seeds, "load_1min_at_start": load_1min,
        "elapsed_s": elapsed_s,
        "settings": {
            "AUX_FPS": ac.AUX_FPS, "AUX_SIZE_LEVELS": ac.AUX_SIZE_LEVELS, "AUX_JPEG_QUALITY": ac.AUX_JPEG_QUALITY, "AUX_MAX_RENDER_MS": ac.AUX_MAX_RENDER_MS,
            "AUX_MESH_FRACTION": ac.AUX_MESH_FRACTION, "AUX_PAUSE_DURING_AUTO": ac.AUX_PAUSE_DURING_AUTO, "SIM_TICK_HZ": config.SIM_TICK_HZ,
            "FRAME_FPS": config.FRAME_FPS, "POLICY_CHECKPOINT": str(auto_config.POLICY_CHECKPOINT),
        },
        "criteria": CRITERIA, "pairs": PAIRS, "results": results, "checks": checks, "auto_pair_table": table,
        "runs": [{"key": run["key"], "seed": run["seed"], "result": strip_raw(run["result"])} for run in runs],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 파일: {out_path}  (걸린 시간 {elapsed_s} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
