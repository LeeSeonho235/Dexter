"""테스트용 가짜 정책 프로세스: 진짜 Unix 소켓과 server/policy_ipc.py 프로토콜로 말하지만 모델은 없다(스레드 하나).
정책 프로세스가 죽을 때, 응답이 늦을 때, 오류를 낼 때, 시작하지 못할 때를 흉내 낸다."""

import os
import socket
import threading
import time

import numpy as np

from server import config, policy_ipc as ipc


class FakeProc:
    """subprocess.Popen처럼 poll/terminate/kill/wait를 가진 가짜."""

    _next_pid = 40000

    def __init__(self) -> None:
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.returncode: int | None = None
        self._done = threading.Event()

    def poll(self):
        return self.returncode

    def exit(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = code
        self._done.set()

    def terminate(self) -> None:
        self.exit(-15)

    def kill(self) -> None:
        self.exit(-9)

    def wait(self, timeout=None):
        if not self._done.wait(timeout):
            import subprocess

            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


def park_chunk(n: int = 100) -> np.ndarray:
    """home 자세 + 집게 열림을 n번(팔이 그대로 있는 행동)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    park = np.array(model.key(config.START_KEYFRAME).ctrl[:6], dtype=float)
    park[5] = config.GRIPPER_OPEN_RAD
    return np.tile(park, (n, 1)).astype(np.float32)


class FakePolicyServer:
    def __init__(self, predict=None, *, delay_s=0.0, die_after=None, error_on=None, ready_delay_s=0.0, fail_start=False, n_action_steps=100):
        """predict(state, image, reset, call_index) -> (n, 6). die_after=N이면 N번째 요청까지 답하고 그 뒤 요청에서 연결을 끊는다(프로세스 사망).
        delay_s: 요청마다 늦게 답한다. error_on=N: N번째 요청에 ERROR 응답. fail_start: 소켓을 열지 않고 종료 코드 3으로 끝난다."""
        self.predict = predict or (lambda state, image, reset, i: park_chunk(n_action_steps))
        self.delay_s = delay_s
        self.die_after = die_after
        self.error_on = error_on
        self.ready_delay_s = ready_delay_s
        self.fail_start = fail_start
        self.n_action_steps = n_action_steps
        self.requests: list[dict] = []  # 받은 요청(state, image, reset)
        self.launch_count = 0
        self.procs: list[FakeProc] = []

    def launcher(self, socket_path: str, log_path: str) -> FakeProc:
        self.launch_count += 1
        proc = FakeProc()
        self.procs.append(proc)
        if self.fail_start:
            with open(log_path, "a") as f:
                f.write("fake start failure\n")
            proc.exit(3)
            return proc
        threading.Thread(target=self._serve, args=(socket_path, proc), daemon=True).start()
        return proc

    def _serve(self, socket_path: str, proc: FakeProc) -> None:
        time.sleep(self.ready_delay_s)
        if proc.returncode is not None:
            return
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            server.bind(socket_path)
        finally:
            os.umask(old)
        server.listen(1)
        server.settimeout(0.1)
        conn = None
        try:
            while proc.returncode is None:
                try:
                    conn, _ = server.accept()
                    break
                except TimeoutError:
                    continue
            if conn is None:
                return
            ipc.send_message(conn, ipc.T_READY, b'{"n_action_steps": %d}' % self.n_action_steps)
            calls = 0
            while proc.returncode is None:
                try:
                    got = ipc.recv_message(conn, 0.1, 5.0)
                except ipc.PolicyError:
                    proc.exit(0)
                    return
                if got is None:
                    continue
                msg_type, payload = got
                if msg_type == ipc.T_PING:
                    ipc.send_message(conn, ipc.T_PONG)
                elif msg_type == ipc.T_SHUTDOWN:
                    ipc.send_message(conn, ipc.T_BYE)
                    proc.exit(0)
                elif msg_type == ipc.T_REQUEST:
                    seq, reset, state, image = ipc.decode_request(payload)
                    self.requests.append({"state": state, "image": image, "reset": reset})
                    calls += 1
                    if self.die_after is not None and calls > self.die_after:
                        proc.exit(-9)  # 갑자기 죽는다: 연결이 끊긴다
                        return
                    if self.delay_s:
                        time.sleep(self.delay_s)
                    if self.error_on == calls:
                        ipc.send_message(conn, ipc.T_ERROR, b"fake error")
                        continue
                    actions = np.asarray(self.predict(state, image, reset, calls - 1), dtype=np.float32)
                    try:
                        ipc.send_message(conn, ipc.T_CHUNK, ipc.encode_chunk(seq, actions, 1.5))
                    except OSError:
                        proc.exit(0)
                        return
        finally:
            if conn is not None:
                conn.close()
            server.close()
            try:
                os.unlink(socket_path)
            except OSError:
                pass


class InProcessPolicy:
    """AutoEpisode/AutoRunner에 바로 꽂는 가짜 PolicyClient(소켓 없이). 상태 표시와 수명까지 흉내 낸다."""

    def __init__(self, predict=None, *, load_s=0.2, n_action_steps=100):
        self.predict = predict or (lambda state, image, reset, i: park_chunk(n_action_steps))
        self.load_s = load_s
        self.started = False
        self.fail_start = ""
        self.fail_request_after: int | None = None
        self.calls = 0
        self.touches = 0
        self.shutdowns = 0

    def ensure_started(self, cancel=None):
        if self.started:
            return
        deadline = time.monotonic() + self.load_s
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                raise ipc.PolicyError("시작이 취소되었습니다")
            time.sleep(0.01)
        if self.fail_start:
            raise ipc.PolicyError(self.fail_start)
        self.started = True

    def request(self, state, image, reset):
        if not self.started:
            raise ipc.PolicyError("정책 프로세스가 꺼져 있습니다")
        if self.fail_request_after is not None and self.calls >= self.fail_request_after:
            self.started = False
            raise ipc.PolicyError("정책 프로세스와 연결이 끊겼습니다")
        actions = np.asarray(self.predict(state, image, reset, self.calls), dtype=np.float64)
        self.calls += 1
        return ipc.ChunkResult(actions=actions, infer_ms=1.0, total_ms=2.0)

    def status(self):
        return {"state": "on" if self.started else "off", "idle_remaining_s": 600.0 if self.started else None, "load_s": self.load_s, "pid": 1}

    def touch(self):
        self.touches += 1

    def shutdown(self, timeout_s=5.0):
        self.started = False
        self.shutdowns += 1
