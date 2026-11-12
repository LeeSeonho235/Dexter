"""AUTO 메시지 검사: {"type":"auto","action":"start"|"stop"|"shutdown_policy"}만 받고 나머지는 거부."""

import json

import pytest

from server.protocol import ProtocolError, parse_client_message

MAX = 4096


@pytest.mark.parametrize("action", ["start", "stop", "shutdown_policy"])
def test_valid_actions(action):
    assert parse_client_message(json.dumps({"type": "auto", "action": action}), MAX) == {"type": "auto", "action": action}


@pytest.mark.parametrize(
    "msg",
    [
        {"type": "auto"},
        {"type": "auto", "action": "go"},
        {"type": "auto", "action": "START"},
        {"type": "auto", "action": None},
        {"type": "auto", "action": 1},
        {"type": "auto", "action": ["start"]},
        {"type": "auto", "action": "start", "seed": 1000},  # 시드는 서버가 정한다
        {"type": "auto", "action": "start", "extra": True},
    ],
)
def test_invalid_messages_are_rejected(msg):
    with pytest.raises(ProtocolError):
        parse_client_message(json.dumps(msg), MAX)
