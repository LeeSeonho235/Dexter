"""정책 프로세스와의 통신(4-1): 메시지 형식 + 서버 쪽 클라이언트. 표준 라이브러리와 numpy만 쓴다(두 환경 모두에서 import 가능).

연결: Unix 소켓(/tmp 아래 0700 폴더 안의 0600 파일). 같은 서버의 다른 사용자는 파일 권한 때문에 접속하지 못한다(TCP 127.0.0.1은 누구나 접속 가능).
정책 프로세스(train/policy_server.py)가 소켓을 열고(모델을 다 읽은 뒤), 서버가 접속해서 READY를 받는다.

메시지: 헤더 12바이트 ">4sBBHI" = (매직 b"DXP1", 종류, 0, 0, 본문 길이) + 본문.
  서버 → 프로세스: REQUEST(관측) / PING / SHUTDOWN
  프로세스 → 서버: READY(JSON) / CHUNK(행동 청크) / PONG / BYE / ERROR(문구)
REQUEST 본문: "<IBHH" (요청 번호, 플래그[1=reset, 2=이미지 있음], 높이, 너비) + state float32[6] + (이미지 uint8 H×W×3)
CHUNK 본문: "<IHHf" (요청 번호, 행동 개수 n, 차원, 프로세스 안에서 잰 추론 시간 ms) + float32[n, dim]

프로세스가 죽거나 응답이 없으면 PolicyError. 서버(시뮬레이션 스레드)는 이 오류를 받아 AUTO를 안전하게 멈춘다.
"""

import json
import logging
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from server import auto_config, config

log = logging.getLogger(__name__)

MAGIC = b"DXP1"
_HEADER = struct.Struct(">4sBBHI")
HEADER_BYTES = _HEADER.size

# 서버 → 프로세스
T_REQUEST, T_PING, T_SHUTDOWN = 1, 2, 3
# 프로세스 → 서버
T_READY, T_CHUNK, T_PONG, T_BYE, T_ERROR = 10, 11, 12, 13, 14

FLAG_RESET = 1
FLAG_IMAGE = 2
STATE_DIM = 6
_REQ_HEAD = struct.Struct("<IBHH")
_CHUNK_HEAD = struct.Struct("<IHHf")
MAX_IMAGE_SIDE_PX = 2048


class PolicyError(RuntimeError):
    """정책 프로세스 문제(꺼짐, 응답 없음, 잘못된 응답, 프로세스가 보낸 오류)."""


class ProtocolViolation(PolicyError):
    """상대가 형식에 안 맞는 바이트를 보냈다."""


class ConnectionClosed(PolicyError):
    """상대가 소켓을 닫았다."""


@dataclass
class ChunkResult:
    actions: np.ndarray  # (n, dim) float64
    infer_ms: float  # 프로세스 안에서 잰 전처리+추론+후처리
    total_ms: float  # 서버에서 잰 요청~응답(전송 포함, 렌더링 제외)


# --- 바이트 주고받기 ---


def send_message(sock: socket.socket, msg_type: int, payload: bytes = b"") -> None:
    if len(payload) > auto_config.MAX_PAYLOAD_BYTES:
        raise ProtocolViolation("payload too large")
    sock.sendall(_HEADER.pack(MAGIC, msg_type, 0, 0, len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int, deadline_s: float | None) -> bytes:
    """n바이트를 읽는다. deadline_s(time.monotonic 기준)를 넘기면 TimeoutError, 상대가 닫으면 ConnectionClosed."""
    chunks: list[bytes] = []
    got = 0
    while got < n:
        if deadline_s is not None:
            remaining = deadline_s - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("receive timed out")
            sock.settimeout(remaining)
        try:
            data = sock.recv(n - got)
        except TimeoutError:
            raise
        except OSError as err:
            raise ConnectionClosed(f"socket error: {err}") from err
        if not data:
            raise ConnectionClosed("peer closed the connection")
        chunks.append(data)
        got += len(data)
    return b"".join(chunks)


def recv_message(sock: socket.socket, timeout_s: float, io_timeout_s: float | None = None) -> tuple[int, bytes] | None:
    """메시지 하나를 읽는다. 첫 바이트가 timeout_s 안에 안 오면 None(아직 아무것도 안 읽음).
    첫 바이트가 온 뒤에는 나머지를 io_timeout_s(기본 timeout_s) 안에 읽어야 하고, 못 읽으면 TimeoutError."""
    io_timeout_s = timeout_s if io_timeout_s is None else io_timeout_s
    try:
        first = _recv_exact(sock, 1, time.monotonic() + timeout_s)
    except TimeoutError:
        return None
    deadline = time.monotonic() + io_timeout_s
    head = first + _recv_exact(sock, HEADER_BYTES - 1, deadline)
    magic, msg_type, _, _, length = _HEADER.unpack(head)
    if magic != MAGIC:
        raise ProtocolViolation("bad magic")
    if length > auto_config.MAX_PAYLOAD_BYTES:
        raise ProtocolViolation(f"payload too large ({length} bytes)")
    payload = _recv_exact(sock, length, deadline) if length else b""
    return msg_type, payload


# --- 본문 만들기/읽기 ---


def encode_request(seq: int, reset: bool, state: np.ndarray, image: np.ndarray | None) -> bytes:
    state = np.asarray(state, dtype="<f4").reshape(-1)
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError("state must be 6 finite numbers")
    flags = FLAG_RESET if reset else 0
    h = w = 0
    body = b""
    if image is not None:
        image = np.ascontiguousarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be uint8 (H, W, 3)")
        h, w = image.shape[:2]
        if not (0 < h <= MAX_IMAGE_SIDE_PX and 0 < w <= MAX_IMAGE_SIDE_PX):
            raise ValueError("image size out of range")
        flags |= FLAG_IMAGE
        body = image.tobytes()
    return _REQ_HEAD.pack(seq, flags, h, w) + state.tobytes() + body


def decode_request(payload: bytes) -> tuple[int, bool, np.ndarray, np.ndarray | None]:
    """(요청 번호, reset 여부, state float32[6], 이미지 또는 None). 형식이 안 맞으면 ProtocolViolation."""
    if len(payload) < _REQ_HEAD.size + STATE_DIM * 4:
        raise ProtocolViolation("request too short")
    seq, flags, h, w = _REQ_HEAD.unpack_from(payload)
    off = _REQ_HEAD.size
    state = np.frombuffer(payload, dtype="<f4", count=STATE_DIM, offset=off).astype(np.float32)
    off += STATE_DIM * 4
    if not np.all(np.isfinite(state)):
        raise ProtocolViolation("state is not finite")
    image = None
    if flags & FLAG_IMAGE:
        if not (0 < h <= MAX_IMAGE_SIDE_PX and 0 < w <= MAX_IMAGE_SIDE_PX) or len(payload) - off != h * w * 3:
            raise ProtocolViolation("image size does not match")
        image = np.frombuffer(payload, dtype=np.uint8, count=h * w * 3, offset=off).reshape(h, w, 3).copy()
    elif len(payload) != off:
        raise ProtocolViolation("unexpected bytes after state")
    return seq, bool(flags & FLAG_RESET), state, image


def encode_chunk(seq: int, actions: np.ndarray, infer_ms: float) -> bytes:
    actions = np.asarray(actions, dtype="<f4")
    if actions.ndim != 2:
        raise ValueError("actions must be (n, dim)")
    return _CHUNK_HEAD.pack(seq, actions.shape[0], actions.shape[1], float(infer_ms)) + actions.tobytes()


def decode_chunk(payload: bytes) -> tuple[int, np.ndarray, float]:
    if len(payload) < _CHUNK_HEAD.size:
        raise ProtocolViolation("chunk too short")
    seq, n, dim, infer_ms = _CHUNK_HEAD.unpack_from(payload)
    if len(payload) - _CHUNK_HEAD.size != n * dim * 4 or n == 0 or dim != STATE_DIM:
        raise ProtocolViolation("chunk size does not match")
    actions = np.frombuffer(payload, dtype="<f4", count=n * dim, offset=_CHUNK_HEAD.size).reshape(n, dim)
    if not np.all(np.isfinite(actions)):
        raise ProtocolViolation("chunk is not finite")
    return seq, actions.astype(np.float64), float(infer_ms)


# --- 서버 쪽 클라이언트 ---

# launcher(socket_path, log_path) -> subprocess.Popen 같은 객체(poll, terminate, kill, wait, pid). 테스트에서 가짜로 바꿔 끼운다.
Launcher = Callable[[str, str], "subprocess.Popen"]


def default_launcher(socket_path: str, log_path: str) -> subprocess.Popen:
    """train/.venv의 policy_server.py를 낮은 우선순위로 띄운다. 로그는 파일로(파이프가 차서 막히는 일이 없게)."""
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS=str(auto_config.POLICY_THREADS),
        MKL_NUM_THREADS=str(auto_config.POLICY_THREADS),
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        WANDB_MODE="disabled",
    )
    cmd = [
        str(auto_config.POLICY_PYTHON),
        str(auto_config.POLICY_SCRIPT),
        "--socket", socket_path,
        "--checkpoint", str(auto_config.POLICY_CHECKPOINT),
        "--device", auto_config.POLICY_DEVICE,
        "--idle-timeout-s", str(auto_config.POLICY_IDLE_TIMEOUT_S),
        "--nice", str(auto_config.POLICY_NICE),
        "--parent-pid", str(os.getpid()),
        "--cleanup-dir", os.path.dirname(socket_path),
    ]
    with open(log_path, "ab") as log_file:
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file, env=env, cwd=str(config.REPO_ROOT))


class PolicyClient:
    """정책 프로세스 하나를 띄우고, 요청을 보내고, 끈다. 소켓 입출력은 하나씩만(락). request()는 시뮬레이션 스레드가 부른다."""

    def __init__(
        self,
        launcher: Launcher = default_launcher,
        *,
        start_timeout_s: float = auto_config.POLICY_START_TIMEOUT_S,
        request_timeout_s: float = auto_config.POLICY_REQUEST_TIMEOUT_S,
        idle_timeout_s: float = auto_config.POLICY_IDLE_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._launcher = launcher
        self._start_timeout_s = start_timeout_s
        self._request_timeout_s = request_timeout_s
        self._idle_timeout_s = idle_timeout_s
        self._clock = clock
        self._start_lock = threading.Lock()  # 시작/종료를 한 번에 하나씩
        self._io_lock = threading.Lock()  # 소켓 읽기/쓰기를 한 번에 하나씩
        self._state_lock = threading.Lock()
        self._proc = None
        self._sock: socket.socket | None = None
        self._dir: str | None = None
        self._loading = False
        self._seq = 0
        self._last_activity_s: float | None = None
        self.load_s: float | None = None  # 직전 시작에서 잰 프로세스 시작~READY(s)
        self.ready_info: dict = {}
        self.last_error = ""

    # --- 상태 ---

    def _proc_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def is_ready(self) -> bool:
        return self._sock is not None and self._proc_alive()

    def status(self) -> dict:
        """화면용: state는 off / loading / on. on이면 유휴 종료까지 남은 시간(idle_remaining_s)."""
        with self._state_lock:
            if self._loading:
                return {"state": "loading", "idle_remaining_s": None, "load_s": self.load_s, "pid": None}
            if self._sock is not None and not self._proc_alive():
                state = "off"  # 유휴 시간이 지나 스스로 끝났거나 죽었다
            else:
                state = "on" if self._sock is not None else "off"
            remaining = None
            if state == "on" and self._last_activity_s is not None:
                remaining = max(0.0, self._idle_timeout_s - (self._clock() - self._last_activity_s))
            return {
                "state": state,
                "idle_remaining_s": remaining,
                "load_s": self.load_s,
                "pid": self._proc.pid if state == "on" else None,
            }

    # --- 시작/종료 ---

    def ensure_started(self, cancel: threading.Event | None = None) -> None:
        """프로세스가 떠 있고 READY면 바로 돌아온다. 아니면 띄우고 READY까지 기다린다(몇 초 걸린다. 시뮬레이션 스레드에서 부르지 않는다)."""
        with self._start_lock:
            if self.is_ready:
                return
            self._teardown()
            with self._state_lock:
                self._loading = True
            started_s = self._clock()
            try:
                self._start(cancel, started_s)
            except BaseException as err:
                self.last_error = str(err)
                self._teardown()
                raise
            finally:
                with self._state_lock:
                    self._loading = False

    def _start(self, cancel: threading.Event | None, started_s: float) -> None:
        self._dir = tempfile.mkdtemp(prefix=auto_config.SOCKET_DIR_PREFIX, dir=auto_config.SOCKET_DIR_PARENT)  # 0700
        socket_path = os.path.join(self._dir, "policy.sock")
        log_path = os.path.join(self._dir, "policy.log")
        self._proc = self._launcher(socket_path, log_path)
        deadline = time.monotonic() + self._start_timeout_s
        sock: socket.socket | None = None
        while sock is None:
            if cancel is not None and cancel.is_set():
                raise PolicyError("시작이 취소되었습니다")
            if not self._proc_alive():
                raise PolicyError(f"정책 프로세스가 시작 중에 끝났습니다 (코드 {self._proc.poll()}). {self._log_tail(log_path)}")
            if time.monotonic() > deadline:
                raise PolicyError(f"정책 프로세스가 {self._start_timeout_s:.0f}초 안에 준비되지 않았습니다")
            if os.path.exists(socket_path):
                candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    candidate.settimeout(1.0)
                    candidate.connect(socket_path)
                    sock = candidate
                except OSError:
                    candidate.close()
            if sock is None:
                time.sleep(0.05)
        try:
            got = recv_message(sock, max(0.1, deadline - time.monotonic()))
        except (PolicyError, TimeoutError) as err:
            sock.close()
            raise PolicyError(f"READY를 받지 못했습니다: {err}") from err
        if got is None or got[0] != T_READY:
            sock.close()
            raise PolicyError("READY를 받지 못했습니다")
        try:
            self.ready_info = json.loads(got[1].decode("utf-8"))
        except ValueError:
            self.ready_info = {}
        self._sock = sock
        self.load_s = self._clock() - started_s
        self._last_activity_s = self._clock()
        log.info("policy process ready in %.1f s (pid %s)", self.load_s, self._proc.pid)

    @staticmethod
    def _log_tail(log_path: str, n_bytes: int = 600) -> str:
        try:
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - n_bytes))
                return f.read().decode("utf-8", "replace").strip().replace("\n", " | ")
        except OSError:
            return ""

    def _teardown(self) -> None:
        """소켓을 닫고 프로세스가 남아 있으면 끝내고 임시 폴더를 지운다."""
        sock, proc, d = self._sock, self._proc, self._dir
        self._sock = self._proc = self._dir = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(2.0)
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

    def shutdown(self, timeout_s: float = 5.0) -> None:
        """SHUTDOWN을 보내 정상 종료를 기다리고, 안 끝나면 강제로 끝낸다. 꺼져 있으면 아무것도 안 한다."""
        with self._start_lock, self._io_lock:
            sock, proc = self._sock, self._proc
            if sock is not None and proc is not None and proc.poll() is None:
                try:
                    send_message(sock, T_SHUTDOWN)
                    recv_message(sock, timeout_s)
                    proc.wait(timeout_s)
                except (PolicyError, TimeoutError, OSError, subprocess.TimeoutExpired):
                    pass
            self._teardown()

    close = shutdown

    def touch(self) -> None:
        """PING을 보내 프로세스의 유휴 시간을 0으로 되돌린다(AUTO가 끝난 때 호출). 실패하면 꺼진 것으로 본다."""
        if not self._io_lock.acquire(timeout=1.0):
            return
        try:
            if not self.is_ready:
                return
            try:
                send_message(self._sock, T_PING)
                got = recv_message(self._sock, self._request_timeout_s)
                if got is None or got[0] != T_PONG:
                    raise PolicyError("no PONG")
                self._last_activity_s = self._clock()
            except (PolicyError, TimeoutError, OSError) as err:
                self.last_error = f"PING failed: {err}"
                self._mark_dead()
        finally:
            self._io_lock.release()

    def _mark_dead(self) -> None:
        """응답이 없거나 깨졌을 때: 이어 쓸 수 없으니(늦게 온 응답과 섞이면 안 된다) 소켓을 닫고 프로세스를 끝낸다."""
        sock, proc = self._sock, self._proc
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(1.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    # --- 요청 ---

    def request(self, state: np.ndarray, image: np.ndarray | None, reset: bool) -> ChunkResult:
        """관측을 보내고 행동 청크를 받는다. 실패하면 PolicyError(소켓과 프로세스는 정리된다)."""
        with self._io_lock:
            if not self.is_ready:
                raise PolicyError("정책 프로세스가 꺼져 있습니다")
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            seq = self._seq
            t0 = time.perf_counter()
            try:
                send_message(self._sock, T_REQUEST, encode_request(seq, reset, state, image))
                got = recv_message(self._sock, self._request_timeout_s)
            except TimeoutError:
                self._mark_dead()
                raise PolicyError(f"정책 응답이 {self._request_timeout_s:.1f}초 안에 오지 않았습니다") from None
            except (PolicyError, OSError) as err:
                self._mark_dead()
                raise PolicyError(f"정책 프로세스와 연결이 끊겼습니다: {err}") from err
            if got is None:
                self._mark_dead()
                raise PolicyError(f"정책 응답이 {self._request_timeout_s:.1f}초 안에 오지 않았습니다")
            total_ms = (time.perf_counter() - t0) * 1000.0
            msg_type, payload = got
            if msg_type == T_ERROR:
                raise PolicyError(f"정책 프로세스 오류: {payload.decode('utf-8', 'replace')}")
            if msg_type != T_CHUNK:
                self._mark_dead()
                raise PolicyError(f"예상하지 못한 응답 종류 {msg_type}")
            try:
                got_seq, actions, infer_ms = decode_chunk(payload)
            except PolicyError as err:
                self._mark_dead()
                raise PolicyError(f"잘못된 응답: {err}") from err
            if got_seq != seq:
                self._mark_dead()
                raise PolicyError("응답 번호가 요청과 다릅니다")
            self._last_activity_s = self._clock()
            return ChunkResult(actions=actions, infer_ms=infer_ms, total_ms=total_ms)
