"""앱 전체(2-2): hand 메시지가 제어 스레드를 거쳐 로봇을 움직이고 state에 값이 실린다."""

import json
import time

import pytest
from fastapi.testclient import TestClient
from test_app import HAND, receive_until

from server import config
from server.app import create_app

CONTROL_KEYS = {
    "controlling", "consecutive_frames", "target_m", "tip_m", "error_m", "gripper_closed", "gripper_goal_rad",
    "gripper_rad", "ctrl_rad", "ik_fail_count", "ik_last_failure", "loop_error_count", "ik_ms", "latency_ms",
    "jitter_mm",
}


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def state_with(ws, predicate):
    item, _ = receive_until(
        ws, lambda m: isinstance(m, dict) and m["type"] == "state" and predicate(m["control"]), max_messages=600
    )
    return item["control"]


def test_state_has_control_section_with_all_keys(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        control = state_with(ws, lambda c: True)
        assert CONTROL_KEYS <= set(control)
        assert control["controlling"] is False
        assert control["target_m"] is None


def test_state_interval_is_10hz(client):
    assert config.STATE_SEND_INTERVAL_S == pytest.approx(0.1)
    with client.websocket_connect(config.WS_PATH) as ws:
        t0 = time.monotonic()
        count = 0
        while time.monotonic() - t0 < 1.0:
            item, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "state")
            count += 1
        assert 6 <= count <= 12


def test_hand_messages_move_the_robot(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        hand = {**HAND, "wrist": [0.4, 0.4, 0.0], "hand_scale": 0.12}
        for i in range(10):
            ws.send_text(json.dumps({**hand, "t": 100.0 + i}))
            time.sleep(0.03)
        control = state_with(ws, lambda c: c["controlling"] and c["target_m"] is not None)
        assert control["ik_ms"]["n"] >= 1
        assert control["latency_ms"]["n"] >= 1
        # 계속 보내면 실제 집게가 목표 쪽으로 간다(거리가 줄어든다).
        first_error = None
        deadline = time.monotonic() + 8.0
        best = 1e9
        while time.monotonic() < deadline:
            ws.send_text(json.dumps({**hand, "t": 200.0}))
            c = state_with(ws, lambda c: c["error_m"] is not None)
            first_error = c["error_m"] if first_error is None else first_error
            best = min(best, c["error_m"])
            if best < 0.02:
                break
            time.sleep(0.03)
        assert best < 0.02, f"tip did not approach target: first {first_error}, best {best}"


def test_ctrl_not_applied_when_only_two_frames(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        for i in range(2):
            ws.send_text(json.dumps({**HAND, "t": 100.0 + i}))
            time.sleep(0.05)
        time.sleep(0.3)
        control = state_with(ws, lambda c: True)
        assert control["controlling"] is False
        assert control["ctrl_rad"] == pytest.approx(client.app.state.hub.control._home_ctrl().round(3).tolist())
