"""앱 전체(2-3): view 메시지가 서버의 보기 방향을 바꾸고 state에 실린다. 잘못된 view는 거부하되 연결을 유지한다."""

import json
import time

import pytest
from fastapi.testclient import TestClient
from test_app import JPEG_START, receive_until

from server import config
from server.app import create_app
from server.view import CameraView


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def is_state(item, **expected):
    return isinstance(item, dict) and item["type"] == "state" and all(item["view"].get(k) == v for k, v in expected.items())


def test_state_reports_the_default_view(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        state, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "state")
        assert state["view"] == {
            "azimuth_deg": config.VIEW_DEFAULT["azimuth_deg"],
            "elevation_deg": config.VIEW_DEFAULT["elevation_deg"],
            "distance_m": config.VIEW_DEFAULT["distance_m"],
        }


def test_view_message_changes_server_view_and_state(client):
    worker = client.app.state.hub.worker
    with client.websocket_connect(config.WS_PATH) as ws:
        ws.send_text(json.dumps({"type": "view", "azimuth": 30, "elevation": -50, "distance": 0.6}))
        state, _ = receive_until(ws, lambda m: is_state(m, azimuth_deg=30.0, elevation_deg=-50.0, distance_m=0.6))
        assert worker.view == CameraView(30.0, -50.0, 0.6)


def test_rendering_follows_the_view_message(client):
    worker = client.app.state.hub.worker
    with client.websocket_connect(config.WS_PATH) as ws:
        receive_until(ws, lambda m: isinstance(m, bytes))
        ws.send_text(json.dumps({"type": "view", "azimuth": -120, "elevation": -25, "distance": 0.7}))
        deadline = time.monotonic() + 5
        while worker.rendered_view != CameraView(-120.0, -25.0, 0.7) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert worker.rendered_view == CameraView(-120.0, -25.0, 0.7)
        frame, _ = receive_until(ws, lambda m: isinstance(m, bytes))
        assert frame.startswith(JPEG_START)


def test_out_of_range_view_is_clamped(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        ws.send_text(json.dumps({"type": "view", "azimuth": 270, "elevation": 40, "distance": 1000}))
        receive_until(
            ws,
            lambda m: is_state(
                m,
                azimuth_deg=-90.0,
                elevation_deg=config.VIEW_ELEVATION_RANGE_DEG[1],
                distance_m=config.VIEW_DISTANCE_RANGE_M[1],
            ),
        )


def test_last_view_wins_across_connections(client):
    with client.websocket_connect(config.WS_PATH) as ws1, client.websocket_connect(config.WS_PATH) as ws2:
        ws1.send_text(json.dumps({"type": "view", "azimuth": 10, "elevation": -30, "distance": 0.8}))
        time.sleep(0.2)
        ws2.send_text(json.dumps({"type": "view", "azimuth": 20, "elevation": -40, "distance": 0.9}))
        for ws in (ws1, ws2):  # 둘 다 마지막에 보낸 값을 본다
            receive_until(ws, lambda m: is_state(m, azimuth_deg=20.0, elevation_deg=-40.0, distance_m=0.9))


def test_bad_view_messages_are_rejected_but_connection_stays(client):
    worker = client.app.state.hub.worker
    before = worker.view
    bad_texts = [
        json.dumps({"type": "view", "azimuth": "x", "elevation": -30, "distance": 0.8}),
        json.dumps({"type": "view", "azimuth": True, "elevation": -30, "distance": 0.8}),
        json.dumps({"type": "view", "azimuth": 10, "elevation": -30}),
        json.dumps({"type": "view", "azimuth": 10, "elevation": -30, "distance": 0.8, "extra": 1}),
        '{"type": "view", "azimuth": NaN, "elevation": -30, "distance": 0.8}',
        '{"type": "view", "azimuth": 10, "elevation": Infinity, "distance": 0.8}',
    ]
    with client.websocket_connect(config.WS_PATH) as ws:
        for text in bad_texts:
            ws.send_text(text)
        ws.send_text(json.dumps({"type": "ping", "id": 5}))
        state, seen = receive_until(
            ws, lambda m: isinstance(m, dict) and m["type"] == "state" and m["invalid_count"] == len(bad_texts)
        )
        assert {"type": "pong", "id": 5} in seen or receive_until(ws, lambda m: m == {"type": "pong", "id": 5})
        assert any(isinstance(m, dict) and m["type"] == "error" for m in seen)
        assert worker.view == before  # 거부한 메시지는 각도를 바꾸지 않았다
        # 그 뒤에도 프레임이 계속 오고, 올바른 view는 적용된다.
        ws.send_text(json.dumps({"type": "view", "azimuth": 5, "elevation": -45, "distance": 0.7}))
        receive_until(ws, lambda m: is_state(m, azimuth_deg=5.0, elevation_deg=-45.0, distance_m=0.7))
        frame, _ = receive_until(ws, lambda m: isinstance(m, bytes))
        assert frame.startswith(JPEG_START)
    assert client.app.state.hub.worker.is_alive


def test_view_message_without_viewers_does_not_render(client):
    """연결이 끊긴 뒤에는 렌더링이 멈춘다는 2-1 동작은 view가 있어도 그대로다."""
    hub = client.app.state.hub
    with client.websocket_connect(config.WS_PATH) as ws:
        receive_until(ws, lambda m: isinstance(m, bytes))
        ws.send_text(json.dumps({"type": "view", "azimuth": 0, "elevation": -60, "distance": 0.7}))
        receive_until(ws, lambda m: is_state(m, azimuth_deg=0.0, elevation_deg=-60.0, distance_m=0.7))
    deadline = time.monotonic() + 2
    while hub.connections and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.3)
    stopped_at = hub.worker.frames_rendered
    time.sleep(0.6)
    assert hub.worker.frames_rendered == stopped_at
