"""정책 프로세스 통신(server/policy_ipc.py): 메시지 형식, 클라이언트 수명(시작/READY/요청/유휴/끄기), 프로세스가 죽을 때, 응답이 늦을 때."""

import os
import socket
import stat
import threading
import time

import numpy as np
import pytest
from fake_policy import FakePolicyServer, park_chunk

from server import policy_ipc as ipc


def make_client(fake: FakePolicyServer, **kwargs) -> ipc.PolicyClient:
    kwargs.setdefault("start_timeout_s", 5.0)
    kwargs.setdefault("request_timeout_s", 2.0)
    return ipc.PolicyClient(fake.launcher, **kwargs)


STATE = np.arange(6, dtype=np.float32) * 0.1
IMAGE = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)


# --- 메시지 형식 ---


def test_request_roundtrip_with_and_without_image():
    seq, reset, state, image = ipc.decode_request(ipc.encode_request(5, True, STATE, IMAGE))
    assert (seq, reset) == (5, True) and np.array_equal(state, STATE) and np.array_equal(image, IMAGE)
    seq, reset, state, image = ipc.decode_request(ipc.encode_request(6, False, STATE, None))
    assert (seq, reset, image) == (6, False, None)


@pytest.mark.parametrize(
    "state, image",
    [(np.zeros(5), None), (np.array([np.nan] + [0] * 5), None), (STATE, np.zeros((4, 4), np.uint8)), (STATE, np.zeros((4, 4, 3), np.float32))],
)
def test_encode_request_rejects_bad_input(state, image):
    with pytest.raises(ValueError):
        ipc.encode_request(1, False, state, image)


def test_decode_request_rejects_bad_payloads():
    good = ipc.encode_request(1, False, STATE, IMAGE)
    for bad in (b"", good[:20], good[:-1], good + b"x"):
        with pytest.raises(ipc.ProtocolViolation):
            ipc.decode_request(bad)


def test_chunk_roundtrip_and_validation():
    actions = park_chunk(7)
    seq, got, infer_ms = ipc.decode_chunk(ipc.encode_chunk(9, actions, 12.5))
    assert seq == 9 and infer_ms == 12.5 and got.dtype == np.float64 and np.array_equal(got, actions.astype(np.float64))
    with pytest.raises(ipc.ProtocolViolation):
        ipc.decode_chunk(ipc.encode_chunk(9, actions, 1.0)[:-4])
    with pytest.raises(ipc.ProtocolViolation):
        ipc.decode_chunk(ipc.encode_chunk(9, np.full((3, 6), np.nan), 1.0))


def test_recv_message_rejects_bad_magic_and_huge_length_and_reports_close():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(b"BAD!" + b"\x00" * 8)
        with pytest.raises(ipc.ProtocolViolation):
            ipc.recv_message(b, 1.0)
    a, b = socket.socketpair()
    with a, b:
        a.sendall(ipc._HEADER.pack(ipc.MAGIC, ipc.T_CHUNK, 0, 0, 10**9))
        with pytest.raises(ipc.ProtocolViolation):
            ipc.recv_message(b, 1.0)
    a, b = socket.socketpair()
    with b:
        a.close()
        with pytest.raises(ipc.ConnectionClosed):
            ipc.recv_message(b, 1.0)


def test_recv_message_times_out_quietly_and_on_partial_message():
    a, b = socket.socketpair()
    with a, b:
        assert ipc.recv_message(b, 0.1) is None  # 아무것도 안 옴
        a.sendall(ipc._HEADER.pack(ipc.MAGIC, ipc.T_CHUNK, 0, 0, 100))  # 본문이 안 온다
        with pytest.raises(TimeoutError):
            ipc.recv_message(b, 1.0, io_timeout_s=0.2)


# --- 클라이언트 수명 ---


def test_start_request_and_shutdown():
    fake = FakePolicyServer()
    client = make_client(fake)
    assert client.status()["state"] == "off"
    client.ensure_started()
    status = client.status()
    assert status["state"] == "on" and status["load_s"] is not None and status["idle_remaining_s"] > 590
    # 소켓 폴더 0700, 소켓 파일 0600 (같은 서버의 다른 사용자가 접속하지 못한다)
    sock_dir = client._dir
    assert stat.S_IMODE(os.stat(sock_dir).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(os.path.join(sock_dir, "policy.sock")).st_mode) == 0o600
    result = client.request(STATE, IMAGE, reset=True)
    assert result.actions.shape == (100, 6) and result.total_ms > 0 and result.infer_ms == 1.5
    assert fake.requests[0]["reset"] is True and np.array_equal(fake.requests[0]["image"], IMAGE)
    client.ensure_started()  # 이미 떠 있으면 다시 띄우지 않는다
    assert fake.launch_count == 1
    client.shutdown()
    assert client.status()["state"] == "off" and not os.path.exists(sock_dir) and fake.procs[0].poll() == 0
    client.shutdown()  # 꺼져 있을 때 다시 불러도 안전


def test_status_while_loading_and_cancel():
    fake = FakePolicyServer(ready_delay_s=5.0)
    client = make_client(fake, start_timeout_s=10.0)
    cancel = threading.Event()
    errors = []

    def run():
        try:
            client.ensure_started(cancel)
        except ipc.PolicyError as err:
            errors.append(str(err))

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 2
    while client.status()["state"] != "loading" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.status()["state"] == "loading"
    cancel.set()
    thread.join(5)
    assert errors and "취소" in errors[0]
    assert client.status()["state"] == "off" and fake.procs[0].poll() is not None  # 프로세스도 끝냈다


def test_start_failure_reports_log_and_cleans_up():
    fake = FakePolicyServer(fail_start=True)
    client = make_client(fake)
    with pytest.raises(ipc.PolicyError, match="fake start failure"):
        client.ensure_started()
    assert client.status()["state"] == "off"
    assert client._dir is None


def test_start_timeout_kills_process():
    fake = FakePolicyServer(ready_delay_s=30.0)
    client = make_client(fake, start_timeout_s=0.5)
    with pytest.raises(ipc.PolicyError, match="준비되지 않았습니다"):
        client.ensure_started()
    assert fake.procs[0].poll() is not None


def test_idle_remaining_counts_down_and_resets_on_touch():
    now = [100.0]
    fake = FakePolicyServer()
    client = make_client(fake, idle_timeout_s=600.0, clock=lambda: now[0])
    client.ensure_started()
    now[0] += 250.0
    assert client.status()["idle_remaining_s"] == pytest.approx(350.0)
    client.touch()
    assert client.status()["idle_remaining_s"] == pytest.approx(600.0)
    now[0] += 700.0
    assert client.status()["idle_remaining_s"] == 0.0
    client.shutdown()


def test_status_off_when_process_exits_by_itself():
    fake = FakePolicyServer()
    client = make_client(fake)
    client.ensure_started()
    fake.procs[0].exit(0)  # 유휴 시간이 지나 스스로 끝난 것과 같다
    assert client.status()["state"] == "off"
    client.shutdown()


# --- 오류: 프로세스가 죽을 때, 늦을 때, 오류 응답 ---


def test_process_dies_mid_request_raises_and_next_start_relaunches():
    fake = FakePolicyServer(die_after=1)
    client = make_client(fake)
    client.ensure_started()
    client.request(STATE, IMAGE, True)
    with pytest.raises(ipc.PolicyError, match="끊겼"):
        client.request(STATE, IMAGE, False)
    assert client.status()["state"] == "off"
    with pytest.raises(ipc.PolicyError, match="꺼져"):
        client.request(STATE, IMAGE, False)
    fake.die_after = None
    client.ensure_started()  # 다시 띄운다
    assert fake.launch_count == 2 and client.status()["state"] == "on"
    client.request(STATE, IMAGE, True)
    client.shutdown()


def test_slow_response_times_out_and_process_is_terminated():
    fake = FakePolicyServer(delay_s=1.5)
    client = make_client(fake, request_timeout_s=0.3)
    client.ensure_started()
    t0 = time.monotonic()
    with pytest.raises(ipc.PolicyError, match="안에 오지 않았습니다"):
        client.request(STATE, IMAGE, True)
    assert time.monotonic() - t0 < 1.0
    assert client.status()["state"] == "off" and fake.procs[0].poll() is not None
    client.shutdown()


def test_error_response_raises_but_keeps_the_process():
    fake = FakePolicyServer(error_on=1)
    client = make_client(fake)
    client.ensure_started()
    with pytest.raises(ipc.PolicyError, match="fake error"):
        client.request(STATE, IMAGE, True)
    assert client.status()["state"] == "on"
    assert client.request(STATE, IMAGE, True).actions.shape == (100, 6)
    client.shutdown()


def test_garbage_response_marks_dead():
    def bad_predict(state, image, reset, i):
        return np.full((3, 6), np.nan)

    fake = FakePolicyServer(predict=bad_predict)
    client = make_client(fake)
    client.ensure_started()
    with pytest.raises(ipc.PolicyError, match="잘못된 응답|not finite"):
        client.request(STATE, IMAGE, True)
    assert client.status()["state"] == "off"
    client.shutdown()
