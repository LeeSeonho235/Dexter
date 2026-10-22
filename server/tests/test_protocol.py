"""브라우저 → 서버 메시지 검사."""

import json

import pytest

from server.protocol import ProtocolError, hand_summary, parse_client_message

MAX_BYTES = 4096


def hand(**overrides):
    msg = {
        "type": "hand",
        "t": 1729570000.123,
        "detected": True,
        "wrist": [0.52, 0.61, -0.03],
        "index_tip": [0.55, 0.48, -0.05],
        "thumb_tip": [0.50, 0.50, -0.04],
        "pinch": 0.04,
        "pinch_ratio": 0.21,
        "palm_open": False,
        "hand_scale": 0.18,
    }
    msg.update(overrides)
    return msg


NO_HAND = {
    "type": "hand",
    "t": 12.5,
    "detected": False,
    "wrist": None,
    "index_tip": None,
    "thumb_tip": None,
    "pinch": None,
    "pinch_ratio": None,
    "palm_open": None,
    "hand_scale": None,
}


def parse(msg):
    return parse_client_message(json.dumps(msg), MAX_BYTES)


def test_valid_hand():
    assert parse(hand()) == hand()


def test_valid_no_hand():
    assert parse(NO_HAND) == NO_HAND


def test_pinch_ratio_may_be_null():
    # hand_scale이 0이면 브라우저가 pinch_ratio를 null로 보낸다.
    assert parse(hand(hand_scale=0, pinch_ratio=None))["pinch_ratio"] is None


def test_coordinates_outside_0_1_are_allowed():
    assert parse(hand(wrist=[-0.05, 1.02, 0]))["wrist"] == [-0.05, 1.02, 0]


def test_valid_ping():
    assert parse({"type": "ping", "id": 7}) == {"type": "ping", "id": 7}


@pytest.mark.parametrize(
    "msg",
    [
        pytest.param({k: v for k, v in hand().items() if k != "pinch"}, id="missing-field"),
        pytest.param(hand(extra=1), id="unknown-field"),
        pytest.param(hand(t="now"), id="t-string"),
        pytest.param(hand(t=True), id="t-bool"),
        pytest.param(hand(detected=1), id="detected-int"),
        pytest.param(hand(wrist=[0.5, 0.5]), id="wrist-2-values"),
        pytest.param(hand(wrist=[0.5, 0.5, 0.0, 0.1]), id="wrist-4-values"),
        pytest.param(hand(wrist=[0.5, "a", 0.0]), id="wrist-string-value"),
        pytest.param(hand(wrist=[0.5, None, 0.0]), id="wrist-null-value"),
        pytest.param(hand(wrist={"x": 0.5}), id="wrist-object"),
        pytest.param(hand(wrist=None), id="wrist-null-when-detected"),
        pytest.param(hand(pinch=-0.01), id="pinch-negative"),
        pytest.param(hand(hand_scale="big"), id="hand-scale-string"),
        pytest.param(hand(palm_open="yes"), id="palm-open-string"),
        pytest.param({**NO_HAND, "pinch": 0.1}, id="value-when-not-detected"),
        pytest.param({"type": "ping"}, id="ping-without-id"),
        pytest.param({"type": "ping", "id": "1"}, id="ping-id-string"),
        pytest.param({"type": "ping", "id": True}, id="ping-id-bool"),
        pytest.param({"type": "video", "data": "..."}, id="unknown-type"),
        pytest.param({"no_type": 1}, id="no-type"),
        pytest.param([1, 2, 3], id="json-array"),
        pytest.param("hand", id="json-string"),
    ],
)
def test_invalid_messages(msg):
    with pytest.raises(ProtocolError):
        parse(msg)


@pytest.mark.parametrize("raw", ["", "not json", "{", '{"type": "hand",}'])
def test_not_json(raw):
    with pytest.raises(ProtocolError):
        parse_client_message(raw, MAX_BYTES)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_rejected(constant):
    raw = json.dumps(hand()).replace('"pinch": 0.04', f'"pinch": {constant}')
    assert constant in raw
    with pytest.raises(ProtocolError):
        parse_client_message(raw, MAX_BYTES)


def test_binary_rejected():
    with pytest.raises(ProtocolError, match="binary"):
        parse_client_message(b"\xff\xd8\xff\xe0", MAX_BYTES)


def test_too_large_rejected():
    raw = json.dumps(hand()) + " " * MAX_BYTES
    with pytest.raises(ProtocolError, match="larger"):
        parse_client_message(raw, MAX_BYTES)


def test_hand_summary():
    assert hand_summary(hand()) == {
        "t": 1729570000.123,
        "detected": True,
        "wrist": [0.52, 0.61, -0.03],
        "pinch_ratio": 0.21,
        "palm_open": False,
        "hand_scale": 0.18,
    }
