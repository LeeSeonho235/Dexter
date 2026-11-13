"""보조 카메라 메시지 검사: {"type":"aux","enabled":true|false}만 받고 나머지는 거부."""

import json

import pytest

from server.protocol import ProtocolError, parse_client_message

MAX = 4096


@pytest.mark.parametrize("enabled", [True, False])
def test_valid_messages(enabled):
    assert parse_client_message(json.dumps({"type": "aux", "enabled": enabled}), MAX) == {"type": "aux", "enabled": enabled}


@pytest.mark.parametrize(
    "msg",
    [
        {"type": "aux"},
        {"type": "aux", "enabled": 1},  # bool만
        {"type": "aux", "enabled": 0},
        {"type": "aux", "enabled": "true"},
        {"type": "aux", "enabled": None},
        {"type": "aux", "enabled": [True]},
        {"type": "aux", "enabled": True, "size": "small"},  # 크기는 서버 설정
        {"type": "aux", "enabled": True, "extra": 1},
    ],
)
def test_invalid_messages_are_rejected(msg):
    with pytest.raises(ProtocolError):
        parse_client_message(json.dumps(msg), MAX)


def test_binary_and_unknown_types_are_still_rejected():
    with pytest.raises(ProtocolError):
        parse_client_message(b'{"type":"aux","enabled":true}', MAX)
    with pytest.raises(ProtocolError):
        parse_client_message(json.dumps({"type": "aux_frame", "enabled": True}), MAX)  # aux_frame은 서버 → 브라우저 전용
