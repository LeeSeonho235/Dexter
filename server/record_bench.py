"""녹화가 틱 루프를 늦추는지 재는 도구(브라우저·WebSocket 없이 서버 내부 객체를 직접 돌린다).

실행:
    cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.record_bench [--seconds 20] [--episode-seconds 30] [--out 폴더]

같은 SimWorker/ControlWorker/RecordingService를 쓰고 조건만 바꿔서 틱 간격과 CPU를 잰다. 손 입력은 가짜 숫자(원을 그리는 손목, 웹캠 아님).
  A: 화면 보는 브라우저 없음, 녹화 없음      B: 화면 있음(렌더링 + JPEG), 녹화 없음
  C: 화면 있음 + 녹화                       D: 화면 없음 + 녹화
틱 간격 = 시뮬레이션 스레드의 틱 시작 사이 벽시계 시간(ms). 화면 렌더링이 있는 틱은 약 55 ms 걸려서 B, C의 최대값이 커지는 것은 녹화와 상관없다.
CPU = 프로세스 전체(모든 스레드, EGL 렌더링 스레드 포함)의 CPU 시간 ÷ 벽시계 시간(코어 수). WebSocket 전송은 들어 있지 않다.
저장 파일은 --out(기본: 시스템 임시 폴더 아래 새 폴더)에만 쓰고, 끝날 때 지우지 않는다(크기 확인용).
"""

import os

# mujoco를 import하기 전에 정해야 한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from server import config  # noqa: E402
from server.control import ControlWorker  # noqa: E402
from server.episode_store import EpisodeStore  # noqa: E402
from server.recording import RecordingService  # noqa: E402
from server.sim_worker import SimWorker  # noqa: E402


def fake_hand(t_s: float) -> dict:
    """원을 그리며 움직이고 가끔 집게를 닫는 가짜 손(숫자만)."""
    x = 0.5 + 0.12 * math.sin(2 * math.pi * t_s / 6.0)
    y = 0.5 + 0.08 * math.cos(2 * math.pi * t_s / 6.0)
    pinch_ratio = 0.2 if int(t_s / 3.0) % 2 else 0.9
    return {"type": "hand", "t": t_s, "detected": True, "wrist": [x, y, 0.0], "index_tip": [x, y, 0.0],
            "thumb_tip": [x, y, 0.0], "pinch": 0.1, "pinch_ratio": pinch_ratio, "palm_open": True, "hand_scale": 0.13}


class HandFeeder(threading.Thread):
    def __init__(self, control: ControlWorker) -> None:
        super().__init__(daemon=True)
        self._control = control
        self._stop_event = threading.Event()
        self.enabled = threading.Event()
        self.enabled.set()

    def run(self) -> None:
        t0 = time.monotonic()
        while not self._stop_event.wait(1 / 30):
            if self.enabled.is_set():
                self._control.submit_hand(fake_hand(time.monotonic() - t0), time.monotonic())

    def stop(self) -> None:
        self._stop_event.set()


def wait_until(condition, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def measure(worker: SimWorker, seconds: float) -> dict:
    worker.tick_stats.reset()
    cpu0, wall0 = time.process_time(), time.monotonic()
    time.sleep(seconds)
    cpu, wall = time.process_time() - cpu0, time.monotonic() - wall0
    stats = worker.tick_stats.summary()
    stats["cpu_cores"] = cpu / wall
    return stats


def print_row(label: str, s: dict) -> None:
    over = s["over"]
    print(
        f"{label:<26} n={s['n']:>5}  평균 {s['mean_ms']:6.2f}  최대 {s['max_ms']:7.2f}  "
        f">33.3ms {over['33.3']:>5}  >40ms {over['40']:>5}  >50ms {over['50']:>5}  CPU {s['cpu_cores']:.2f}코어"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=20.0, help="조건 A, B를 재는 시간(s)")
    parser.add_argument("--episode-seconds", type=float, default=30.0, help="조건 C, D에서 녹화하는 시간(s)")
    parser.add_argument("--out", type=Path, default=None, help="에피소드를 저장할 폴더(기본: 임시 폴더)")
    parser.add_argument("--order", default="CED", help="C, E, D 조건을 도는 순서(기본 CED). 순서에 따라 쓰는 시드가 달라진다")
    args = parser.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="record_bench_"))

    worker = SimWorker(
        config.SCENE_PATH, keyframe=config.START_KEYFRAME, camera=config.CAMERA, width_px=config.RENDER_WIDTH_PX,
        height_px=config.RENDER_HEIGHT_PX, frame_fps=config.FRAME_FPS, jpeg_quality=config.JPEG_QUALITY,
        max_catch_up_s=config.MAX_CATCH_UP_S, shadow=config.RENDER_SHADOW, reflection=config.RENDER_REFLECTION,
        tick_hz=config.SIM_TICK_HZ,
    )
    worker.start()
    control = ControlWorker(worker)
    control.start()
    service = RecordingService(worker, control, store=EpisodeStore(out))
    service.start()
    feeder = HandFeeder(control)
    feeder.start()
    results: dict[str, dict] = {}
    try:
        time.sleep(2.0)  # 팔이 움직이기 시작하고 안정될 시간
        worker.set_viewer_count(0)
        results["A 화면 없음, 녹화 없음"] = measure(worker, args.seconds)
        worker.set_viewer_count(1)
        time.sleep(1.0)
        results["B 화면 있음, 녹화 없음"] = measure(worker, args.seconds)

        # E는 C와 똑같이 리셋하고 손을 움직이되 녹화만 하지 않는 대조 조건이다(리셋 직후의 팔 움직임 차이를 녹화와 구분하려고).
        # 시드 위치(블록 자리)에 따라 렌더링 시간이 달라질 수 있어서, 순서를 바꿔 가며 확인할 수 있게 --order로 정한다.
        conditions = {
            "C": ("C 화면 있음 + 녹화", 1, True),
            "E": ("E 화면 있음, 리셋만(대조)", 1, False),
            "D": ("D 화면 없음 + 녹화", 0, True),
        }
        for key in args.order:
            label, viewers, recording = conditions[key]
            worker.set_viewer_count(viewers)
            feeder.enabled.clear()  # 리셋 중에는 손 입력이 어차피 무시된다. 끝난 뒤 다시 켠다.
            assert service.handle_reset() is None
            assert wait_until(lambda: service.state()["block_ready"], 15.0), "reset did not finish"
            feeder.enabled.set()
            time.sleep(1.0)
            if recording:
                refusal = service.handle_record({"action": "start"})
                assert refusal is None, f"start refused: {refusal}"
            results[label] = measure(worker, args.episode_seconds)
            if recording:
                state_before_stop = service.recorder.state()
                assert service.handle_record({"action": "stop_success"}) is None
                assert wait_until(lambda: service.state()["pending_writes"] == 0 and service.state()["saved_total"] >= 1, 15.0)
                results[label]["recorder"] = state_before_stop
        saved = sorted(out.glob("episode_*.npz"))
    finally:
        feeder.stop()
        service.close()
        control.stop()
        worker.stop()

    print(f"\n틱 간격(ms)과 CPU. 공칭 틱 주기 {1000 / config.SIM_TICK_HZ:.1f} ms (설정 {config.SIM_TICK_HZ} Hz), 화면 렌더링 {config.FRAME_FPS} fps")
    for label, stats in results.items():
        print_row(label, stats)
    for label in ("C 화면 있음 + 녹화", "D 화면 없음 + 녹화"):
        r = results[label]["recorder"]  # 위의 틱 통계와 같은 구간에서 센 녹화 상태
        print(f"{label}: 녹화 {r['ticks']}샘플 / {r['elapsed_s']} s, 정지 비율 {r['still_ratio']}, 가장 긴 정지 {r['still_longest_s']} s")

    print("\n저장된 파일:")
    for path in sorted(out.iterdir()):
        print(f"  {path.name:<28} {path.stat().st_size:>9,d} bytes")
    store = EpisodeStore(out)
    for path in saved:
        index = int(path.stem.split("_")[1])
        arrays = store.load_arrays(index)
        t = arrays["sim_time_s"]
        spacing_ms = np.diff(t) * 1000
        print(
            f"  에피소드 {index}: {len(t)}샘플, 시뮬레이션 시간 {t[-1] - t[0]:.2f} s, 샘플 간격 평균 {spacing_ms.mean():.2f} ms "
            f"(최소 {spacing_ms.min():.0f}, 최대 {spacing_ms.max():.0f} ms), 격자 최대 오차 "
            f"{np.abs(t - (t[0] + np.arange(len(t)) / config.RECORD_FPS)).max() * 1000:.2f} ms"
        )
    print(f"\n출력 폴더: {out}")


if __name__ == "__main__":
    main()
