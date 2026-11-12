"""WebSocket으로 AUTO 흐름(4-1): 리셋 → 시작 → 실행 → 시간 초과/중지, 거절 규칙, 상태 필드, 정책 오류, 정책 프로세스 끄기.
정책은 가짜(InProcessPolicy: home 자세 유지)지만 시뮬레이션, 제어, 녹화 서비스, 제어 주체는 모두 진짜다."""

import json
import time

import numpy as np
import pytest
from fake_policy import InProcessPolicy
from fastapi.testclient import TestClient

from server import config
from server.app import create_app

AUTO = lambda action: json.dumps({"type": "auto", "action": action})  # noqa: E731
RESET = json.dumps({"type": "reset"})


def receive_until(ws, predicate, max_messages=600):
    seen = []
    for _ in range(max_messages):
        message = ws.receive()
        item = message["bytes"] if message.get("bytes") is not None else json.loads(message["text"])
        seen.append(item)
        if predicate(item):
            return item, seen
    raise AssertionError(f"not received in {max_messages} messages; last: {seen[-3:]!r}")


def is_state(m):
    return isinstance(m, dict) and m["type"] == "state"


def state_where(ws, predicate, **kwargs):
    item, _ = receive_until(ws, lambda m: is_state(m) and predicate(m), **kwargs)
    return item


def last_state(ws):
    return state_where(ws, lambda m: True)


def error_after(ws, send_text):
    ws.send_text(send_text)
    item, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
    return item["reason"]


def reset_and_wait(ws):
    ws.send_text(RESET)
    return state_where(ws, lambda m: m["record"]["block_ready"])


@pytest.fixture
def policy():
    return InProcessPolicy(load_s=0.3)


@pytest.fixture
def client(tmp_path, policy):
    with TestClient(create_app(record_dir=tmp_path / "raw", policy_client=policy, auto_max_s=2.0)) as c:
        yield c


def test_auto_state_is_idle_by_default(client):
    with client.websocket_connect("/ws") as ws:
        auto = last_state(ws)["auto"]
        assert auto["status"] == "idle" and auto["owner"] == "IDLE" and auto["session_attempts"] == 0
        assert auto["policy"]["state"] == "off" and auto["load_estimate_s"] > 0


def test_start_needs_a_block_reset_first(client):
    with client.websocket_connect("/ws") as ws:
        reason = error_after(ws, AUTO("start"))
        assert "리셋" in reason
        state = last_state(ws)
        assert state["invalid_count"] == 0 and state["auto"]["status"] == "idle" and state["auto"]["owner"] == "IDLE"


def test_invalid_auto_message_is_rejected_and_connection_stays(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "auto", "action": "go"}))
        assert state_where(ws, lambda m: m["invalid_count"] == 1)
        ws.send_text(json.dumps({"type": "ping", "id": 5}))
        receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "pong" and m["id"] == 5)


def test_full_run_times_out_counts_attempt_and_frees_the_robot(client, policy):
    with client.websocket_connect("/ws") as ws:
        ready = reset_and_wait(ws)
        seed = ready["record"]["current_seed"]
        ws.send_text(AUTO("start"))
        loading = state_where(ws, lambda m: m["auto"]["status"] in ("loading", "running"))
        assert loading["auto"]["owner"] == "AUTO"
        running = state_where(ws, lambda m: m["auto"]["status"] == "running" and m["auto"]["sample"] is not None)
        assert running["auto"]["seed"] == seed and running["auto"]["max_s"] == 2.0
        assert running["record"]["block_ready"] is False  # 시작하면 "리셋 직후"를 소비한다
        done = state_where(ws, lambda m: m["auto"]["status"] == "timeout")
        auto = done["auto"]
        assert auto["owner"] == "IDLE" and auto["session_attempts"] == 1 and auto["session_success"] == 0
        assert auto["result"] == {"outcome": "timeout", "seed": seed, "time_s": pytest.approx(2.0, abs=0.05)}
        assert auto["last_latency_ms"] is None or auto["last_latency_ms"] > 0
        assert policy.calls >= 1 and policy.touches == 1  # 끝난 뒤 유휴 타이머를 다시 잰다
        # 끝나면 다시 리셋할 수 있고(주체가 IDLE로 돌아왔다) 시작은 리셋 뒤에만 된다
        assert "리셋" in error_after(ws, AUTO("start"))
        assert reset_and_wait(ws)["record"]["current_seed"] == seed + 1


def test_session_counts_only_success_and_timeout_not_stop(client, policy):
    with client.websocket_connect("/ws") as ws:
        reset_and_wait(ws)
        ws.send_text(AUTO("start"))
        state_where(ws, lambda m: m["auto"]["status"] == "running" and m["auto"]["sample"] is not None)
        ws.send_text(AUTO("stop"))
        stopped = state_where(ws, lambda m: m["auto"]["status"] == "stopped")
        assert stopped["auto"]["session_attempts"] == 0 and stopped["auto"]["owner"] == "IDLE"
        assert stopped["auto"]["result"]["outcome"] == "stopped"


def test_other_requests_are_refused_while_auto_runs_without_invalid_count(client):
    with client.websocket_connect("/ws") as ws:
        reset_and_wait(ws)
        ws.send_text(AUTO("start"))
        state_where(ws, lambda m: m["auto"]["status"] in ("loading", "running"))
        for text in (RESET, json.dumps({"type": "record", "action": "start"}), AUTO("start"), AUTO("shutdown_policy")):
            assert error_after(ws, text)
        state = last_state(ws)
        assert state["invalid_count"] == 0 and state["auto"]["owner"] == "AUTO"
        ws.send_text(AUTO("stop"))
        state_where(ws, lambda m: m["auto"]["status"] in ("stopped", "timeout"))


def test_hand_and_height_input_is_dropped_while_auto_runs(client):
    """AUTO 중에는 손·키보드 입력이 로봇을 움직이지 못한다(ctrl을 쓰는 주체는 하나)."""
    hub = client.app.state.hub
    with client.websocket_connect("/ws") as ws:
        reset_and_wait(ws)
        ws.send_text(AUTO("start"))
        state_where(ws, lambda m: m["auto"]["status"] == "running" and m["auto"]["sample"] is not None)
        before = np.array(hub.worker.snapshot["joints_rad"])
        hand = {
            "type": "hand", "t": 1.0, "detected": True, "wrist": [0.3, 0.3, 0.0], "index_tip": [0.3, 0.3, 0.0], "thumb_tip": [0.3, 0.3, 0.0],
            "pinch": 0.1, "pinch_ratio": 0.9, "palm_open": True, "hand_scale": 0.12,
        }
        for _ in range(30):
            ws.send_text(json.dumps(hand))
            ws.send_text(json.dumps({"type": "height", "dir": -1, "slow": False}))
            time.sleep(0.01)
        state = state_where(ws, lambda m: m["auto"]["status"] == "running")
        assert state["control"]["controlling"] is False
        after = np.array(hub.worker.snapshot["joints_rad"])
        assert np.abs(after - before).max() < 0.05  # 정책은 home을 유지하라고 했다
        ws.send_text(AUTO("stop"))
        state_where(ws, lambda m: m["auto"]["status"] == "stopped")


def test_eval_seed_blocks_cannot_start_auto(client):
    hub = client.app.state.hub
    with client.websocket_connect("/ws") as ws:
        reset_and_wait(ws)
        hub.recording._current_seed = config.EVAL_SEED_RANGE[0]  # 어떤 경로로든 평가 시드가 현재 시드가 되면 막는다
        assert "평가 시드" in error_after(ws, AUTO("start"))
        assert last_state(ws)["auto"]["status"] == "idle"


def test_start_refused_when_arm_was_moved_after_reset(client):
    hub = client.app.state.hub
    with client.websocket_connect("/ws") as ws:
        reset_and_wait(ws)
        hub.worker.set_ctrl(np.array([0.6, -1.0, 0.8, -0.6, 0.0, 0.9]))  # 리셋 뒤 팔이 움직였다
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and np.abs(hub.worker.snapshot["joints_rad"][0]) < 0.3:
            time.sleep(0.05)
        assert "home" in error_after(ws, AUTO("start"))
        assert last_state(ws)["auto"]["owner"] == "IDLE"


def test_policy_start_failure_is_shown_and_server_survives(tmp_path):
    policy = InProcessPolicy(load_s=0.1)
    policy.fail_start = "정책 프로세스가 시작 중에 끝났습니다 (코드 1)"
    with TestClient(create_app(record_dir=tmp_path / "raw", policy_client=policy, auto_max_s=2.0)) as client, client.websocket_connect("/ws") as ws:
        reset_and_wait(ws)
        ws.send_text(AUTO("start"))
        failed = state_where(ws, lambda m: m["auto"]["status"] == "error")
        assert "시작 중에 끝났습니다" in failed["auto"]["error"] and failed["auto"]["owner"] == "IDLE"
        assert failed["record"]["block_ready"] is True  # 블록을 쓰지 않았으니 다시 시도할 수 있다
        policy.fail_start = ""
        ws.send_text(AUTO("start"))
        state_where(ws, lambda m: m["auto"]["status"] == "timeout")


def test_policy_death_mid_run_stops_auto_with_error_and_server_survives(tmp_path):
    policy = InProcessPolicy(load_s=0.1, n_action_steps=20)  # 20샘플(0.67초)마다 요청: 두 번째 요청에서 죽는다
    policy.fail_request_after = 1
    with TestClient(create_app(record_dir=tmp_path / "raw", policy_client=policy, auto_max_s=3.0)) as client, client.websocket_connect("/ws") as ws:
        reset_and_wait(ws)
        ws.send_text(AUTO("start"))
        failed = state_where(ws, lambda m: m["auto"]["status"] == "error")
        assert "끊겼" in failed["auto"]["error"] and failed["auto"]["owner"] == "IDLE"
        assert failed["auto"]["session_attempts"] == 0  # 오류는 시도에 넣지 않는다
        ws.send_text(json.dumps({"type": "ping", "id": 9}))
        receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "pong")  # 서버는 살아 있다
        assert reset_and_wait(ws)  # 다시 리셋도 된다


def test_stop_while_loading_cancels_without_touching_the_block(tmp_path):
    policy = InProcessPolicy(load_s=3.0)
    with TestClient(create_app(record_dir=tmp_path / "raw", policy_client=policy, auto_max_s=2.0)) as client, client.websocket_connect("/ws") as ws:
        reset_and_wait(ws)
        ws.send_text(AUTO("start"))
        loading = state_where(ws, lambda m: m["auto"]["status"] == "loading")
        assert loading["auto"]["load_estimate_s"] > 0
        ws.send_text(AUTO("stop"))
        stopped = state_where(ws, lambda m: m["auto"]["status"] == "stopped")
        assert stopped["auto"]["owner"] == "IDLE" and stopped["record"]["block_ready"] is True


def test_shutdown_policy_button(client, policy):
    with client.websocket_connect("/ws") as ws:
        assert "꺼져" in error_after(ws, AUTO("shutdown_policy"))  # 처음엔 꺼져 있다
        reset_and_wait(ws)
        ws.send_text(AUTO("start"))
        state_where(ws, lambda m: m["auto"]["status"] == "timeout")
        on = last_state(ws)["auto"]["policy"]
        assert on["state"] == "on" and on["idle_remaining_s"] is not None and on["idle_remaining_s"] > 0
        ws.send_text(AUTO("shutdown_policy"))
        off = state_where(ws, lambda m: m["auto"]["policy"]["state"] == "off")
        assert off["auto"]["policy"]["idle_remaining_s"] is None and policy.shutdowns == 1
