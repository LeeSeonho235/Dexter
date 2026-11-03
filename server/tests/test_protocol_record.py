"""record, reset 메시지 검사: 올바른 것은 통과, 잘못된 값은 ProtocolError."""

import json

import pytest

from server import config
from server.protocol import ProtocolError, parse_client_message


def parse(msg):
    return parse_client_message(json.dumps(msg), config.MAX_MESSAGE_BYTES)


@pytest.mark.parametrize("action", ["start", "stop_success", "stop_fail", "discard"])
def test_valid_record_actions(action):
    assert parse({"type": "record", "action": action}) == {"type": "record", "action": action}


def test_delete_last_needs_an_episode_number():
    assert parse({"type": "record", "action": "delete_last", "episode": 0})["episode"] == 0
    assert parse({"type": "record", "action": "delete_last", "episode": 17})["episode"] == 17
    with pytest.raises(ProtocolError):
        parse({"type": "record", "action": "delete_last"})  # 번호 없음


@pytest.mark.parametrize(
    "msg",
    [
        {"type": "record"},  # action 없음
        {"type": "record", "action": "explode"},
        {"type": "record", "action": ""},
        {"type": "record", "action": 1},
        {"type": "record", "action": None},
        {"type": "record", "action": ["start"]},
        {"type": "record", "action": "START"},  # 대소문자 구분
        {"type": "record", "action": "start", "episode": 1},  # delete_last 아닌데 episode
        {"type": "record", "action": "discard", "extra": True},
        {"type": "record", "action": "delete_last", "episode": -1},
        {"type": "record", "action": "delete_last", "episode": 1.0},
        {"type": "record", "action": "delete_last", "episode": "3"},
        {"type": "record", "action": "delete_last", "episode": True},
        {"type": "record", "action": "delete_last", "episode": None},
        {"type": "record", "action": "delete_last", "episode": 1, "extra": 0},
    ],
)
def test_invalid_record_messages_are_rejected(msg):
    with pytest.raises(ProtocolError):
        parse(msg)


def test_nan_episode_is_rejected():
    raw = '{"type": "record", "action": "delete_last", "episode": NaN}'
    with pytest.raises(ProtocolError):
        parse_client_message(raw, config.MAX_MESSAGE_BYTES)


def test_reset_takes_no_fields():
    assert parse({"type": "reset"}) == {"type": "reset"}
    for extra in ({"seed": 3}, {"seed": 1000}, {"x": 1}):
        with pytest.raises(ProtocolError):
            parse({"type": "reset", **extra})  # 시드는 서버가 정한다(평가 시드를 화면에서 못 고른다)


def test_existing_message_types_still_work():
    assert parse({"type": "ping", "id": 1})["id"] == 1
    assert parse({"type": "height", "dir": 1, "slow": False})["dir"] == 1
