"""앱 전체(2-5): height 메시지가 목표 높이를 바꾸고 state에 실린다. 잘못된 메시지는 거부하되 연결을 유지. 메시지가 끊기면 멈춘다."""

import json
import time

import pytest
from fastapi.testclient import TestClient
from test_app import HAND, JPEG_START, receive_until

from server import config
from server.app import create_app

SPEED = config.HEIGHT_SPEED_M_S


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def hand_msg(i=0):
    return json.dumps({**HAND, "t": 100.0 + i, "wrist": [0.5, 0.5, 0.0], "hand_scale": 0.13})


def height_msg(direction, slow=False):
    return json.dumps({"type": "height", "dir": direction, "slow": slow})


def control_state(ws, predicate=lambda c: True, max_messages=800):
    item, _ = receive_until(
        ws, lambda m: isinstance(m, dict) and m["type"] == "state" and predicate(m["control"]), max_messages=max_messages
    )
    return item["control"]


def start_control(ws):
    for i in range(6):
        ws.send_text(hand_msg(i))
        time.sleep(0.03)
    return control_state(ws, lambda c: c["controlling"] and c["target_z_m"] is not None)


def test_state_has_the_height_fields(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        state = control_state(ws)
        for key in ("height_dir", "height_slow", "height_active", "height_limit", "target_z_m", "z_from_hand_scale"):
            assert key in state
        assert state["height_dir"] == 0 and state["target_z_m"] is None and state["z_from_hand_scale"] is False


def test_height_before_the_first_clutch_is_ignored(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        for _ in range(8):
            ws.send_text(height_msg(-1))
            time.sleep(0.05)
        state = control_state(ws)
        assert state["target_z_m"] is None and state["height_active"] is False
        assert state["ctrl_rad"] == pytest.approx(client.app.state.hub.control._home_ctrl().round(3).tolist())


def test_holding_a_key_lowers_the_target_and_the_tip_follows(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        started = start_control(ws)
        z0 = started["target_z_m"]
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.5:  # ↓를 0.5초 누른다(20Hz 반복)
            ws.send_text(height_msg(-1))
            time.sleep(0.05)
        ws.send_text(height_msg(0))
        pressed_s = time.monotonic() - t0
        state = control_state(ws, lambda c: c["height_dir"] == 0 and c["target_z_m"] < z0 - 0.02)
        drop = z0 - state["target_z_m"]
        assert drop == pytest.approx(SPEED * pressed_s, abs=0.01)  # 속도 × 누른 시간(타이밍 오차 약 1 cm 이내)
        # 로봇 집게가 실제로 따라 내려온다
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = control_state(ws)
            if state["tip_m"] is not None and abs(state["tip_m"][2] - state["target_z_m"]) < 0.01:
                break
        assert abs(state["tip_m"][2] - state["target_z_m"]) < 0.01


def test_height_stops_by_itself_when_messages_stop(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        z0 = start_control(ws)["target_z_m"]
        ws.send_text(height_msg(-1))  # 한 번만 보내고 브라우저가 죽었다
        time.sleep(0.8)
        first = control_state(ws, lambda c: c["height_dir"] == 0)
        time.sleep(0.6)
        second = control_state(ws)
        assert second["target_z_m"] == first["target_z_m"]  # 더는 안 내려간다
        drop = z0 - second["target_z_m"]
        assert 0 < drop <= SPEED * config.HEIGHT_COMMAND_TIMEOUT_S + 0.005  # 최대 2.4 cm


def test_slow_is_slower(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        z0 = start_control(ws)["target_z_m"]
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.5:
            ws.send_text(height_msg(-1, slow=True))
            time.sleep(0.05)
        ws.send_text(height_msg(0))
        pressed_s = time.monotonic() - t0
        state = control_state(ws, lambda c: c["height_dir"] == 0 and c["target_z_m"] < z0 - 0.003)
        assert z0 - state["target_z_m"] == pytest.approx(config.HEIGHT_SLOW_SPEED_M_S * pressed_s, abs=0.005)


def test_bad_height_messages_are_rejected_but_connection_stays(client):
    bad_texts = [
        json.dumps({"type": "height", "dir": 2, "slow": False}),
        json.dumps({"type": "height", "dir": True, "slow": False}),
        json.dumps({"type": "height", "dir": "1", "slow": False}),
        json.dumps({"type": "height", "dir": 0.5, "slow": False}),
        json.dumps({"type": "height", "dir": 1, "slow": 1}),
        json.dumps({"type": "height", "dir": 1}),
        json.dumps({"type": "height", "dir": 1, "slow": False, "extra": 1}),
        '{"type": "height", "dir": NaN, "slow": false}',
    ]
    with client.websocket_connect(config.WS_PATH) as ws:
        start_control(ws)
        for text in bad_texts:
            ws.send_text(text)
        ws.send_text(json.dumps({"type": "ping", "id": 9}))
        # start_control에서 이미 받은 거부 카운트가 0이었으므로 이번에 정확히 len(bad_texts)개
        state, seen = receive_until(
            ws, lambda m: isinstance(m, dict) and m["type"] == "state" and m["invalid_count"] == len(bad_texts)
        )
        assert any(isinstance(m, dict) and m["type"] == "error" for m in seen)
        assert state["control"]["height_dir"] == 0  # 거부한 메시지는 아무것도 바꾸지 않았다
        # 그 뒤에도 프레임이 오고 올바른 height는 적용된다
        z_before = state["control"]["target_z_m"]
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.3:
            ws.send_text(height_msg(1))
            time.sleep(0.05)
        ws.send_text(height_msg(0))
        control_state(ws, lambda c: c["target_z_m"] > z_before + 0.005)
        frame, _ = receive_until(ws, lambda m: isinstance(m, bytes))
        assert frame.startswith(JPEG_START)
    assert client.app.state.hub.worker.is_alive and client.app.state.hub.control.is_alive
