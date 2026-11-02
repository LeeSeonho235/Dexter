"""height(키보드 높이) 메시지 검사: dir은 정수 -1/0/1만, slow는 bool만. 그 외는 모두 거부."""

import json

import pytest

from server.protocol import ProtocolError, parse_client_message

MAX_BYTES = 4096


def height(**overrides):
    msg = {"type": "height", "dir": -1, "slow": False}
    msg.update(overrides)
    return msg


def parse(msg):
    return parse_client_message(json.dumps(msg), MAX_BYTES)


@pytest.mark.parametrize("direction", [-1, 0, 1])
@pytest.mark.parametrize("slow", [True, False])
def test_valid_height_is_returned_as_is(direction, slow):
    msg = height(dir=direction, slow=slow)
    assert parse(msg) == msg


@pytest.mark.parametrize("bad", [2, -2, 100, 0.5, -1.0, 1.0, 0.0, "1", "-1", "up", None, True, False, [1], {"d": 1}])
def test_invalid_dir_is_rejected(bad):
    with pytest.raises(ProtocolError):
        parse(height(dir=bad))


@pytest.mark.parametrize("bad", [0, 1, "true", "false", None, [True], {"s": True}])
def test_invalid_slow_is_rejected(bad):
    with pytest.raises(ProtocolError):
        parse(height(slow=bad))


@pytest.mark.parametrize("missing", ["dir", "slow"])
def test_missing_field_is_rejected(missing):
    msg = height()
    del msg[missing]
    with pytest.raises(ProtocolError, match="missing"):
        parse(msg)


def test_unknown_field_is_rejected():
    with pytest.raises(ProtocolError, match="unknown"):
        parse(height(speed=1.0))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_are_rejected(constant):
    raw = json.dumps(height()).replace('"dir": -1', f'"dir": {constant}')
    assert constant in raw
    with pytest.raises(ProtocolError):
        parse_client_message(raw, MAX_BYTES)


def test_out_of_range_is_not_clamped_but_rejected():
    """view와 달리 height는 범위를 자르지 않는다: 잘못된 값으로 로봇이 움직이면 안 되므로 거부한다."""
    with pytest.raises(ProtocolError):
        parse(height(dir=5))
