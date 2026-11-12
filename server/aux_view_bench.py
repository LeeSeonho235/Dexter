"""보조 카메라(2-6)가 틱 루프를 얼마나 늦추는지, 보조 화면 한 장이 얼마나 비싼지 재는 도구(브라우저·WebSocket 없이 서버 내부 객체를 직접 돌린다).

실행(사용자가 한다. Claude는 구현만 했고 아직 돌리지 않았다):
    cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.aux_view_bench [--seconds 20] [--warmup 3] [--order ABCD] [--out 폴더]

record_bench.py와 같은 SimWorker/ControlWorker/가짜 손을 쓰고 조건만 바꾼다.
  A: 보는 연결 없음, aux 꺼짐           B: 메인 화면 있음, aux 꺼짐
  C: 메인 화면 + aux normal 크기        D: 메인 화면 + aux small 크기
조건 일부만 / 순서를 바꿔 돌리려면 --order(예: --order DCBA, --order BC). 5분이 넘지 않게 조건을 나눠 돌릴 수 있다.

feeder 스레드: sim_worker.py에는 아직 offer_qpos가 없다(3단계). 그래서 이 도구의 feeder 스레드가 틱 주기(SIM_TICK_HZ)로
SimWorker.snapshot에서 관절 6개와 블록 위치를 읽어 qpos 13개로 만들어 aux.offer_qpos()에 넘긴다. 한계:
  - 이 스레드가 추가로 깨어나는 비용(Python 스레드 하나, 30Hz)이 측정에 들어간다. A, B에서도 같은 feeder를 돌려(aux는 꺼진 객체) 조건 사이 차이에서는 빠지게 했다.
  - 실제 연결에서는 물리 틱 안에서 호출하므로 feeder 없이 락 비용만 든다. 그래서 이 값은 실제보다 약간 높은 쪽(보수적)이다.
  - snapshot에 블록 회전이 없어서 회전은 "없음"(1,0,0,0)으로 채운다. 블록이 기울어도 그림만 다를 뿐 렌더 비용은 거의 같을 것으로 보지만 확인한 것은 아니다.
측정 구간 안에서 aux 렌더 시간, JPEG 크기, 실제 fps는 on_frame 콜백으로 모은다(영상 파일은 저장하지 않는다).
틱 간격 = 시뮬레이션 스레드의 틱 시작 사이 벽시계 시간(ms). 메인 화면 렌더 틱은 약 55 ms라서 최대값은 비교에 쓰지 않고 p50/p95와 50 ms 초과 비율을 본다.
CPU = 프로세스 전체(모든 스레드, EGL 렌더링 스레드 포함)의 CPU 시간 ÷ 벽시계 시간(코어 수).
결과 숫자는 --out 폴더의 aux_bench_<시각>.json에만 쓴다(기본: 시스템 임시 폴더 아래 새 폴더).
"""

import os

# mujoco를 import하기 전에 정해야 한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from server import aux_view_config as ac  # noqa: E402
from server import config  # noqa: E402
from server.aux_view import QPOS_DIM, AuxViewWorker  # noqa: E402
from server.control import ControlWorker  # noqa: E402
from server.record_bench import HandFeeder  # noqa: E402
from server.sim_worker import SimWorker, TickStats  # noqa: E402

# 합격 기준 "제안값". 1차 측정 결과를 본 뒤 사용자가 확정한다(확정 전에는 판정이 아니라 참고 표시).
CRITERIA_PROVISIONAL = {
    "tick_p95_ms_over_B": 2.0,  # C/D의 틱 간격 p95가 B보다 이만큼(ms)보다 더 늘면 안 된다
    "over50_pct_points_over_B": 1.0,  # 50 ms 초과 비율(%)이 B보다 이만큼(%p)보다 더 늘면 안 된다
    "cpu_cores_over_B": 0.3,  # CPU가 B보다 이만큼(코어)보다 더 늘면 안 된다
    "aux_render_p95_ms_max": ac.AUX_MAX_RENDER_MS,  # aux 렌더+JPEG p95가 감속 기준 이하
    "jpeg_kb_max": 10.0,  # 프레임 하나 크기(KB)
}

CONDITIONS = {
    "A": ("A 화면 없음, aux 꺼짐", 0, None),
    "B": ("B 메인 화면, aux 꺼짐", 1, None),
    "C": ("C 메인 화면 + aux normal", 1, "normal"),
    "D": ("D 메인 화면 + aux small", 1, "small"),
}


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


class AuxFeeder(threading.Thread):
    """SimWorker.snapshot → qpos 13개(rad, m) → aux.offer_qpos. sim_worker.py 연결 전의 임시 대역이다(모듈 설명 참고)."""

    def __init__(self, sim: SimWorker, aux: AuxViewWorker) -> None:
        super().__init__(daemon=True)
        self._sim = sim
        self.aux = aux  # 조건이 바뀔 때 main 스레드가 바꿔 끼운다(참조 대입)
        self._stop_event = threading.Event()
        self._qpos = np.zeros(QPOS_DIM)
        self._qpos[9] = 1.0  # 블록 회전 쿼터니언 w (회전 없음)

    def run(self) -> None:
        while not self._stop_event.wait(1.0 / config.SIM_TICK_HZ):
            snap = self._sim.snapshot
            if snap is None:
                continue
            self._qpos[:6] = snap["joints_rad"]  # rad
            self._qpos[6:9] = snap["block_m"]  # m
            self.aux.offer_qpos(self._qpos)

    def stop(self) -> None:
        self._stop_event.set()


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


def measure(sim: SimWorker, aux: AuxViewWorker, log: FrameLog, seconds: float) -> dict:
    sim.tick_stats.reset()
    log.clear()
    log.recording = True
    cpu0, wall0 = time.process_time(), time.monotonic()
    time.sleep(seconds)
    cpu, wall = time.process_time() - cpu0, time.monotonic() - wall0
    log.recording = False
    intervals = list(sim.tick_stats.intervals_ms)
    state = aux.state()
    result = {
        "ticks": len(intervals),
        "tick_mean_ms": round(float(np.mean(intervals)), 2) if intervals else None,
        "tick_p50_ms": pct(intervals, 50),
        "tick_p95_ms": pct(intervals, 95),
        "tick_p99_ms": pct(intervals, 99),
        "tick_over50_pct": round(100.0 * sum(ms > 50.0 for ms in intervals) / len(intervals), 2) if intervals else None,
        "cpu_cores": round(cpu / wall, 3),
        "aux_state": state,
    }
    if log.times_s:
        span_s = log.times_s[-1] - log.times_s[0]
        result.update(
            aux_frames=len(log.times_s),
            aux_fps=round((len(log.times_s) - 1) / span_s, 2) if span_s > 0 else None,
            aux_render_p50_ms=pct(log.render_ms, 50),
            aux_render_p95_ms=pct(log.render_ms, 95),
            aux_jpeg_kb_mean=round(float(np.mean(log.jpeg_bytes)) / 1024.0, 2),
            aux_jpeg_kb_max=round(max(log.jpeg_bytes) / 1024.0, 2),
        )
    return result


def provisional_check(results: dict[str, dict]) -> dict[str, dict[str, bool]]:
    """B 대비 제안 기준 확인. 확정 전의 참고용이라 이름에 provisional을 붙여 출력한다."""
    out: dict[str, dict[str, bool]] = {}
    base = results.get("B")
    for key in ("C", "D"):
        r = results.get(key)
        if r is None or base is None or r["tick_p95_ms"] is None or r.get("aux_frames") is None:
            continue
        c = CRITERIA_PROVISIONAL
        out[key] = {
            "tick_p95": r["tick_p95_ms"] - base["tick_p95_ms"] <= c["tick_p95_ms_over_B"],
            "over50": r["tick_over50_pct"] - base["tick_over50_pct"] <= c["over50_pct_points_over_B"],
            "cpu": r["cpu_cores"] - base["cpu_cores"] <= c["cpu_cores_over_B"],
            "aux_render_p95": r["aux_render_p95_ms"] <= c["aux_render_p95_ms_max"],
            "jpeg_kb": r["aux_jpeg_kb_max"] <= c["jpeg_kb_max"],
            "aux_not_slowed": not r["aux_state"]["slowed"] and r["aux_state"]["error"] is None,
        }
    return out


def print_row(key: str, label: str, r: dict) -> None:
    line = (
        f"{label:<26} 틱 p50 {r['tick_p50_ms']:6.2f}  p95 {r['tick_p95_ms']:6.2f}  >50ms {r['tick_over50_pct']:5.2f}%  "
        f"CPU {r['cpu_cores']:.2f}코어"
    )
    if r.get("aux_frames") is not None:
        line += (
            f" | aux {r['aux_fps']}fps  렌더 p50 {r['aux_render_p50_ms']} / p95 {r['aux_render_p95_ms']} ms  "
            f"JPEG 평균 {r['aux_jpeg_kb_mean']} / 최대 {r['aux_jpeg_kb_max']} KB"
        )
    elif key in ("C", "D"):
        line += " | aux 프레임 0장(측정 실패)"
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=20.0, help="조건마다 재는 시간(s)")
    parser.add_argument("--warmup", type=float, default=3.0, help="조건을 바꾼 뒤 재기 전에 기다리는 시간(s)")
    parser.add_argument("--order", default="ABCD", help="돌릴 조건과 순서(A, B, C, D 중). 일부만 줘도 된다")
    parser.add_argument("--max-load", type=float, default=50.0, help="시작 전 load average(1분)가 이 값 이상이면 돌리지 않는다")
    parser.add_argument("--out", type=Path, default=None, help="결과 JSON을 쓸 폴더(기본: 임시 폴더)")
    args = parser.parse_args()
    if not args.order or any(key not in CONDITIONS for key in args.order) or len(set(args.order)) != len(args.order):
        parser.error("--order는 A, B, C, D를 겹치지 않게 쓴다(예: ABCD, DCBA, BC)")

    load_1min = os.getloadavg()[0]
    if load_1min >= args.max_load:
        print(f"load average {load_1min:.1f} >= {args.max_load:g}: 서버가 바빠서 측정하지 않는다.", file=sys.stderr)
        return 2
    os.nice(19)  # 공용 서버: 다른 사람 작업을 방해하지 않게 가장 낮은 우선순위로

    planned_s = len(args.order) * (args.warmup + args.seconds) + 5.0
    print(f"조건 {args.order}, 약 {planned_s:.0f}초 예상. load {load_1min:.2f}. 공칭 틱 {1000 / config.SIM_TICK_HZ:.1f} ms, 메인 화면 {config.FRAME_FPS} fps, aux {ac.AUX_FPS} fps")
    if planned_s > 300:
        print("경고: 5분이 넘는다. --order로 나눠 돌리는 것이 좋다.", file=sys.stderr)

    sim = SimWorker(
        config.SCENE_PATH, keyframe=config.START_KEYFRAME, camera=config.CAMERA, width_px=config.RENDER_WIDTH_PX,
        height_px=config.RENDER_HEIGHT_PX, frame_fps=config.FRAME_FPS, jpeg_quality=config.JPEG_QUALITY,
        max_catch_up_s=config.MAX_CATCH_UP_S, shadow=config.RENDER_SHADOW, reflection=config.RENDER_REFLECTION,
        tick_hz=config.SIM_TICK_HZ,
    )
    sim.tick_stats = RecordingTickStats()
    sim.start()
    control = ControlWorker(sim)
    control.start()
    hand = HandFeeder(control)
    hand.start()
    log = FrameLog()
    off_aux = AuxViewWorker(enabled=False)  # 꺼진 객체: offer_qpos가 바로 돌아온다(A, B에서도 같은 feeder를 돌리기 위해)
    feeder = AuxFeeder(sim, off_aux)
    feeder.start()
    results: dict[str, dict] = {}
    try:
        time.sleep(2.0)  # 팔이 움직이기 시작하고 안정될 시간
        for key in args.order:
            label, main_viewers, aux_level = CONDITIONS[key]
            aux = off_aux
            if aux_level is not None:
                aux = AuxViewWorker(enabled=True, size_level=aux_level, on_frame=log)
                aux.start()
                aux.set_viewer_count(1)
            log.aux = aux
            feeder.aux = aux
            sim.set_viewer_count(main_viewers)
            try:
                time.sleep(args.warmup)  # 크기 변경·첫 렌더(EGL 준비) 등 시작 비용을 측정에서 뺀다
                results[key] = measure(sim, aux, log, args.seconds)
            finally:
                feeder.aux = off_aux
                if aux is not off_aux:
                    aux.stop()
    finally:
        feeder.stop()
        hand.stop()
        control.stop()
        sim.stop()

    print("\n틱 간격(ms)과 CPU:")
    for key in args.order:
        print_row(key, CONDITIONS[key][0], results[key])
    checks = provisional_check(results)
    if checks:
        print("\n제안 기준(확정 전, 참고) 대비: B가 같이 측정됐을 때만 나온다")
        for key, items in checks.items():
            print(f"  {key}: " + "  ".join(f"{name} {'O' if ok else 'X'}" for name, ok in items.items()))
    else:
        print("\n제안 기준 비교는 B와 C 또는 D를 같이 돌렸을 때만 나온다.")

    out_dir = args.out or Path(tempfile.mkdtemp(prefix="aux_bench_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"aux_bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "order": args.order, "seconds": args.seconds, "warmup": args.warmup, "load_1min_at_start": load_1min,
        "settings": {"AUX_FPS": ac.AUX_FPS, "AUX_SIZE_LEVELS": ac.AUX_SIZE_LEVELS, "AUX_JPEG_QUALITY": ac.AUX_JPEG_QUALITY,
                     "AUX_MAX_RENDER_MS": ac.AUX_MAX_RENDER_MS, "SIM_TICK_HZ": config.SIM_TICK_HZ, "FRAME_FPS": config.FRAME_FPS},
        "criteria_provisional": CRITERIA_PROVISIONAL, "results": results, "provisional_checks": checks,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 파일: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
