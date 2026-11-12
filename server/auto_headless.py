"""AUTO(4-1) 헤드리스 검증과 측정. 브라우저 없이 서버 객체(SimWorker, ControlWorker, PolicyClient)를 직접 돌린다. 서버 환경(server/.venv)에서 실행.
정책 프로세스는 진짜(train/.venv, 실제 체크포인트)다. 평가 시드(1000~1049)도 여기서는 내부 함수를 직접 불러서 쓸 수 있다(화면의 AUTO는 막혀 있다).

  server/.venv/bin/python -m server.auto_headless --mode compare --seeds 1002,1005,1000,1010,1015 --start exact   # 평가 CSV와 비교
  server/.venv/bin/python -m server.auto_headless --mode compare --seeds 1002,1005,1000 --start live              # 실제 리셋(BlockReset) 경로로 시작
  server/.venv/bin/python -m server.auto_headless --mode load --repeat 3                                          # 정책 프로세스 시작~READY 시간, 메모리
  server/.venv/bin/python -m server.auto_headless --mode orphan --watch both                                      # 서버가 kill -9로 죽으면 정책 프로세스가 끝나는가

--start exact: 평가와 같은 초기화(EvalEnv.reset과 같은 계산)로 시작 → 루프 동일성 검증.
--start live : 화면과 같은 경로. 제어 스레드가 팔을 home으로 올리고(BlockReset) 블록을 시드 위치에 놓은 직후 그 상태 그대로 시작한다.
--viewers 1  : 화면 보는 브라우저가 있는 것처럼 화면 렌더링(15fps, JPEG)도 켠다(틱 간격 측정용).
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from server import auto_config, config  # noqa: E402
from server import policy_ipc as ipc  # noqa: E402
from server.auto import AutoEpisode, AutoResult  # noqa: E402
from server.control import ControlWorker  # noqa: E402
from server.seeds import block_xy_for_seed  # noqa: E402
from server.sim_worker import SimWorker  # noqa: E402

CLK_TCK = os.sysconf("SC_CLK_TCK")
EVAL_CSV = config.REPO_ROOT / "experiments" / "act_v1_demo20" / "results.csv"


# --- /proc로 CPU와 메모리 재기(새 패키지 없이) ---


def cpu_s(pid: int) -> float | None:
    """프로세스(모든 스레드)가 쓴 CPU 시간(s)."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / CLK_TCK  # utime, stime
    except (OSError, IndexError, ValueError):
        return None


def mem_mb(pid: int) -> dict[str, float] | None:
    try:
        out = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                out[line.split(":")[0]] = int(line.split()[1]) / 1024.0
        return out
    except OSError:
        return None


def make_worker(viewers: int) -> SimWorker:
    worker = SimWorker(
        config.SCENE_PATH, keyframe=config.START_KEYFRAME, camera=config.CAMERA, width_px=config.RENDER_WIDTH_PX,
        height_px=config.RENDER_HEIGHT_PX, frame_fps=config.FRAME_FPS, jpeg_quality=config.JPEG_QUALITY,
        max_catch_up_s=config.MAX_CATCH_UP_S, shadow=config.RENDER_SHADOW, reflection=config.RENDER_REFLECTION,
        tick_hz=config.SIM_TICK_HZ, on_frame=(lambda seq, jpeg: None) if viewers else None,
    )
    worker.start()
    worker.set_viewer_count(viewers)
    return worker


def read_eval_rows(path: Path = EVAL_CSV) -> dict[int, dict]:
    with path.open(encoding="utf-8") as f:
        return {int(r["seed"]): r for r in csv.DictReader(f)}


# --- 에피소드 하나 ---


def run_one(worker: SimWorker, client: ipc.PolicyClient, control: ControlWorker | None, seed: int, start: str, max_s: float) -> dict:
    """에피소드를 돌리고 결과와 측정값을 dict로 돌려준다."""
    if start == "live":
        assert control is not None
        done = threading.Event()
        outcome: dict = {}

        def on_reset(s, ok, reason):
            outcome.update(ok=ok, reason=reason)
            done.set()

        reason = control.request_reset(seed, block_xy_for_seed(seed), on_reset)
        assert reason is None, reason
        assert done.wait(30) and outcome["ok"], outcome
        time.sleep(0.05)

    def on_finish(result: AutoResult) -> None:
        worker.set_ctrl(result.final_ctrl)  # AutoRunner와 같은 마무리: 마지막 ctrl을 이어준다
        if control is not None:
            control.request_adopt(result.final_ctrl)
        worker.owner.release_auto()

    assert worker.owner.try_acquire_auto()
    episode = AutoEpisode(seed, client, max_s=max_s, exact_start=(start == "exact"), on_finish=on_finish)
    server_pid, policy_pid = os.getpid(), client.status().get("pid")
    cpu0, pcpu0, wall0 = cpu_s(server_pid), cpu_s(policy_pid) if policy_pid else None, time.perf_counter()
    assert worker.start_auto(episode) is None
    assert episode.done.wait(max_s * 2 + 60), "episode did not finish"
    wall = time.perf_counter() - wall0
    cpu1, pcpu1 = cpu_s(server_pid), cpu_s(policy_pid) if policy_pid else None
    result = episode.result
    ticks = worker.tick_stats.summary()
    return {
        "seed": seed, "result": result, "wall_s": wall, "ticks": ticks,
        "server_cpu_cores": None if cpu0 is None else (cpu1 - cpu0) / wall,
        "policy_cpu_cores": None if pcpu0 is None or pcpu1 is None else (pcpu1 - pcpu0) / wall,
    }


def compare_row(run: dict, eval_row: dict | None) -> dict:
    result: AutoResult = run["result"]
    row = {
        "seed": run["seed"], "live_outcome": result.outcome, "live_time_s": result.time_to_success_s, "live_samples": result.n_samples,
        "live_block_xy_m": (round(result.final_block_m[0], 6), round(result.final_block_m[1], 6)), "error": result.error,
    }
    if eval_row is not None:
        eval_ok = eval_row["success"] == "True"
        eval_t = float(eval_row["time_to_success_s"]) if eval_row["time_to_success_s"] else None
        ex, ey = float(eval_row["final_block_x_m"]), float(eval_row["final_block_y_m"])
        row.update(
            eval_outcome="success" if eval_ok else eval_row["failure"], eval_time_s=eval_t, eval_samples=int(eval_row["n_samples"]),
            eval_block_xy_m=(ex, ey),
            block_diff_mm=math.hypot(result.final_block_m[0] - ex, result.final_block_m[1] - ey) * 1000.0,
            same_success=(result.outcome == "success") == eval_ok,
            time_diff_samples=None if (eval_t is None or result.time_to_success_s is None) else round((result.time_to_success_s - eval_t) * config.RECORD_FPS, 2),
            same_samples=result.n_samples == int(eval_row["n_samples"]),
        )
    return row


def summarize_requests(runs: list[dict]) -> dict:
    reqs = [r for run in runs for r in run["result"].requests]
    if not reqs:
        return {}
    total = [r["render_ms"] + r["request_ms"] for r in reqs]
    return {
        "n_requests": len(reqs),
        "render_ms": (float(np.mean([r["render_ms"] for r in reqs])), float(np.max([r["render_ms"] for r in reqs]))),
        "request_ms(전송+추론+응답)": (float(np.mean([r["request_ms"] for r in reqs])), float(np.max([r["request_ms"] for r in reqs]))),
        "infer_ms(프로세스 안)": (float(np.mean([r["infer_ms"] for r in reqs])), float(np.max([r["infer_ms"] for r in reqs]))),
        "end_to_end_ms(렌더+전송+추론+응답)": (float(np.mean(total)), float(np.max(total))),
        "physics_stall_ms(물리 멈춤)": (float(np.mean([r["stall_ms"] for r in reqs])), float(np.max([r["stall_ms"] for r in reqs]))),
    }


def mode_compare(args) -> int:
    seeds = [int(s) for s in args.seeds.split(",")]
    eval_rows = read_eval_rows()
    client = ipc.PolicyClient()
    t0 = time.perf_counter()
    client.ensure_started()
    print(f"정책 프로세스 준비: {time.perf_counter() - t0:.1f}s (READY까지 {client.load_s:.1f}s, pid {client.status()['pid']})", flush=True)
    worker = make_worker(args.viewers)
    control = None
    if args.start == "live":
        control = ControlWorker(worker)
        control.start()
    runs, rows = [], []
    try:
        for seed in seeds:
            run = run_one(worker, client, control, seed, args.start, args.max_s)
            runs.append(run)
            row = compare_row(run, eval_rows.get(seed))
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
            time.sleep(0.5)
        policy_mem = mem_mb(client.status()["pid"])
        server_mem = mem_mb(os.getpid())
    finally:
        if control is not None:
            control.stop()
        worker.stop()
        client.shutdown()
    print("\n--- 요약 ---")
    print("start:", args.start, "viewers:", args.viewers)
    for key, value in summarize_requests(runs).items():
        print(f"  {key}: {value if not isinstance(value, tuple) else f'평균 {value[0]:.1f} / 최대 {value[1]:.1f}'}")
    for run in runs:
        t = run["ticks"]
        print(f"  시드 {run['seed']}: 틱 n={t['n']} 평균 {t['mean_ms']} 최대 {t['max_ms']} ms, 초과 {t['over']}, 벽시계 {run['wall_s']:.1f}s, "
              f"서버 CPU {run['server_cpu_cores']:.2f}코어, 정책 CPU {run['policy_cpu_cores']:.2f}코어")
    print(f"  메모리 MB: 서버 {server_mem}, 정책 {policy_mem}")
    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "requests": summarize_requests(runs)}, ensure_ascii=False, indent=1, default=str))
    return 0


def mode_load(args) -> int:
    for i in range(args.repeat):
        client = ipc.PolicyClient()
        t0 = time.perf_counter()
        client.ensure_started()
        ready_s = time.perf_counter() - t0
        pid = client.status()["pid"]
        mem_ready = mem_mb(pid)
        # 첫 요청(첫 추론은 느릴 수 있다)과 둘째 요청 시간, 메모리
        try:
            from server.render import StateRenderer

            with StateRenderer() as renderer:
                qpos = np.array(renderer.model.key(config.START_KEYFRAME).qpos)
                renderer.render(qpos)  # 버리는 렌더링(평가와 같게)
                qpos[6:9] = [0.21, -0.02, 0.015]
                image = renderer.render(qpos)
            times = []
            for j in range(3):
                t1 = time.perf_counter()
                client.request(qpos[:6].astype(np.float32), image, reset=(j == 0))
                times.append((time.perf_counter() - t1) * 1000.0)
            mem_after = mem_mb(pid)
        finally:
            client.shutdown()
        print(f"[{i + 1}/{args.repeat}] 프로세스 시작~READY {client.load_s:.2f}s (ensure_started 전체 {ready_s:.2f}s), READY 직후 메모리 {mem_ready}, "
              f"요청 3번 {[round(t, 1) for t in times]} ms, 요청 뒤 메모리 {mem_after}", flush=True)
    return 0


# --- 서버가 kill -9로 죽을 때 ---


def mode_hold_policy(args) -> int:
    """(orphan 모드의 자식) 서버처럼 정책 프로세스를 띄우고 pid를 파일에 적은 뒤 그냥 기다린다."""
    watch = args.watch

    def launcher(socket_path: str, log_path: str):
        cmd = [
            str(auto_config.POLICY_PYTHON), str(auto_config.POLICY_SCRIPT), "--socket", socket_path, "--checkpoint", str(auto_config.POLICY_CHECKPOINT),
            "--idle-timeout-s", str(auto_config.POLICY_IDLE_TIMEOUT_S), "--nice", str(auto_config.POLICY_NICE), "--cleanup-dir", os.path.dirname(socket_path),
        ]
        if watch in ("both", "parent"):
            cmd += ["--parent-pid", str(os.getpid())]
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1")
        with open(log_path, "ab") as log_file:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file, env=env, cwd=str(config.REPO_ROOT))
        # 모델을 읽기 전에 pid를 알린다(읽는 도중에 서버가 죽는 경우도 시험하려고)
        Path(args.pid_file).write_text(json.dumps({"policy_pid": proc.pid, "dir": os.path.dirname(socket_path)}))
        return proc

    client = ipc.PolicyClient(launcher)
    client.ensure_started()
    time.sleep(3600)
    return 0


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except OSError:
        return False


def mode_orphan(args) -> int:
    pid_file = Path(f"/tmp/dexter-orphan-{os.getpid()}.json")
    pid_file.unlink(missing_ok=True)
    child = subprocess.Popen([sys.executable, "-m", "server.auto_headless", "--mode", "hold-policy", "--pid-file", str(pid_file), "--watch", args.watch],
                             cwd=str(config.REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 120
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.2)
        assert child.poll() is None, "server stand-in exited early"
    info = json.loads(pid_file.read_text())
    policy_pid = info["policy_pid"]
    if not args.during_load:
        # 소켓이 열리고 서버 대역이 접속할 때까지 기다린다(= 평소 상태: 정책 프로세스가 연결을 기다리며 떠 있음)
        sock_path = os.path.join(info["dir"], "policy.sock")
        while not os.path.exists(sock_path) and time.monotonic() < deadline:
            time.sleep(0.1)
        time.sleep(1.5)
    print(f"서버 대역 pid {child.pid}, 정책 프로세스 pid {policy_pid} (감시 방식: {args.watch}, "
          f"{'모델 읽는 중에' if args.during_load else '연결된 상태에서'} 서버 kill -9), 소켓 폴더 {info['dir']}")
    assert alive(policy_pid)
    t0 = time.monotonic()
    os.kill(child.pid, signal.SIGKILL)  # kill -9
    child.wait()
    while alive(policy_pid) and time.monotonic() - t0 < 60:
        time.sleep(0.05)
    gone_s = time.monotonic() - t0
    print(f"kill -9 후 정책 프로세스가 사라지기까지 {gone_s:.2f}s (살아 있음: {alive(policy_pid)}), 소켓 폴더 남음: {os.path.exists(info['dir'])}")
    pid_file.unlink(missing_ok=True)
    return 0 if not alive(policy_pid) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["compare", "load", "orphan", "hold-policy"], required=True)
    parser.add_argument("--seeds", default="1002,1005,1000,1010,1015")
    parser.add_argument("--start", choices=["exact", "live"], default="exact")
    parser.add_argument("--viewers", type=int, default=0)
    parser.add_argument("--max-s", type=float, default=auto_config.AUTO_MAX_S)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--watch", choices=["both", "parent", "eof"], default="both")
    parser.add_argument("--pid-file")
    parser.add_argument("--during-load", action="store_true", help="orphan: 정책 프로세스가 모델을 읽는 중에(소켓이 아직 없을 때) 서버를 죽인다")
    parser.add_argument("--out", help="비교 결과 JSON을 쓸 경로")
    args = parser.parse_args(argv)
    try:
        os.nice(5)
    except OSError:
        pass
    return {"compare": mode_compare, "load": mode_load, "orphan": mode_orphan, "hold-policy": mode_hold_policy}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
