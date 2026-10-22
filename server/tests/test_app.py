"""WebSocket 엔드포인트: 잘못된 메시지가 와도 연결과 서버가 살아 있는지, pong/state/프레임."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from server import config
from server.app import create_app

JPEG_START = b"\xff\xd8"

HAND = {
    "type": "hand",
    "t": 100.0,
    "detected": True,
    "wrist": [0.52, 0.61, -0.03],
    "index_tip": [0.55, 0.48, -0.05],
    "thumb_tip": [0.50, 0.50, -0.04],
    "pinch": 0.04,
    "pinch_ratio": 0.21,
    "palm_open": False,
    "hand_scale": 0.18,
}


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def hub_of(client):
    return client.app.state.hub


def receive_until(ws, predicate, max_messages=200):
    """서버가 보내는 메시지를 predicate가 참이 될 때까지 받는다. 프레임(15fps)과 state(2Hz)가 계속 오므로 멈추지 않는다."""
    seen = []
    for _ in range(max_messages):
        message = ws.receive()
        item = message["bytes"] if message.get("bytes") is not None else json.loads(message["text"])
        seen.append(item)
        if predicate(item):
            return item, seen
    raise AssertionError(f"not received in {max_messages} messages; last: {seen[-3:]!r}")


def is_state(item, **expected):
    return isinstance(item, dict) and item["type"] == "state" and all(item[k] == v for k, v in expected.items())


def test_no_connection_means_no_frames(client):
    hub = hub_of(client)
    start_sim_s = hub.worker.sim_time_s
    time.sleep(0.5)
    assert hub.worker.sim_time_s > start_sim_s  # 시뮬레이션은 돈다
    assert hub.worker.frames_rendered == 0  # 렌더링은 안 한다


def test_frames_only_while_connected(client):
    hub = hub_of(client)
    with client.websocket_connect(config.WS_PATH) as ws:
        frame, _ = receive_until(ws, lambda item: isinstance(item, bytes))
        assert frame.startswith(JPEG_START)
    # 연결이 끝나면 렌더링이 멈춘다.
    deadline = time.monotonic() + 2.0
    while hub.connections and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not hub.connections
    time.sleep(0.2)
    stopped_at = hub.worker.frames_rendered
    time.sleep(0.5)
    assert hub.worker.frames_rendered == stopped_at


def test_ping_pong(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        ws.send_text(json.dumps({"type": "ping", "id": 42}))
        pong, _ = receive_until(ws, lambda item: isinstance(item, dict) and item["type"] == "pong")
        assert pong == {"type": "pong", "id": 42}


def test_hand_counted_in_state(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        for i in range(3):
            ws.send_text(json.dumps({**HAND, "t": 100.0 + i}))
        state, _ = receive_until(ws, lambda item: is_state(item, hand_count=3))
        assert state["invalid_count"] == 0
        assert state["last_hand"] == {
            "t": 102.0,
            "detected": True,
            "wrist": [0.52, 0.61, -0.03],
            "pinch_ratio": 0.21,
            "palm_open": False,
            "hand_scale": 0.18,
        }
        assert state["sim_time_s"] > 0


def test_bad_messages_do_not_kill_connection(client):
    bad_texts = [
        "not json",
        "{",
        "[]",
        json.dumps({"type": "video", "data": "AAAA"}),
        json.dumps({**HAND, "wrist": [1, 2]}),
        json.dumps({**HAND, "pinch": "x"}),
        json.dumps(HAND).replace('"pinch": 0.04', '"pinch": NaN'),
        "x" * (config.MAX_MESSAGE_BYTES + 1),
    ]
    with client.websocket_connect(config.WS_PATH) as ws:
        for text in bad_texts:
            ws.send_text(text)
        ws.send_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
        ws.send_text(json.dumps(HAND))
        ws.send_text(json.dumps({"type": "ping", "id": 1}))

        n_bad = len(bad_texts) + 1
        state, seen = receive_until(ws, lambda item: is_state(item, hand_count=1, invalid_count=n_bad))
        assert any(isinstance(item, dict) and item["type"] == "error" for item in seen)
        assert {"type": "pong", "id": 1} in seen
        # 그 뒤에도 프레임이 계속 온다.
        frame, _ = receive_until(ws, lambda item: isinstance(item, bytes))
        assert frame.startswith(JPEG_START)

    # 같은 서버에 다시 붙어도 정상 (서버가 죽지 않았다). 카운트는 연결마다 새로 센다.
    with client.websocket_connect(config.WS_PATH) as ws:
        ws.send_text(json.dumps({"type": "ping", "id": 2}))
        receive_until(ws, lambda item: item == {"type": "pong", "id": 2})
        receive_until(ws, lambda item: is_state(item, hand_count=0, invalid_count=0))
    assert hub_of(client).worker.is_alive


def test_two_connections_both_get_frames(client):
    with client.websocket_connect(config.WS_PATH) as ws1, client.websocket_connect(config.WS_PATH) as ws2:
        assert len(hub_of(client).connections) == 2
        receive_until(ws1, lambda item: isinstance(item, bytes))
        receive_until(ws2, lambda item: isinstance(item, bytes))


def test_worker_stopped_on_shutdown():
    with TestClient(create_app()) as c:
        worker = c.app.state.hub.worker
        assert worker.is_alive
    assert not worker.is_alive
