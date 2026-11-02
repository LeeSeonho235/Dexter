"""view(보기 방향) 메시지 검사: 정상, 범위 밖은 자름, NaN·무한대·타입 오류·필드 오류는 거부."""

import json

import pytest

from server import config
from server.protocol import ProtocolError, parse_client_message

MAX_BYTES = 4096


def view(**overrides):
    msg = {"type": "view", "azimuth": 110.0, "elevation": -35.0, "distance": 0.8}
    msg.update(overrides)
    return msg


def parse(msg):
    return parse_client_message(json.dumps(msg), MAX_BYTES)


def test_valid_view_is_returned_as_is():
    assert parse(view()) == view()


def test_integers_are_accepted():
    assert parse(view(azimuth=90, elevation=-20, distance=1)) == view(azimuth=90, elevation=-20, distance=1)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"elevation": 10.0}, {"elevation": config.VIEW_ELEVATION_RANGE_DEG[1]}),
        ({"elevation": -500.0}, {"elevation": config.VIEW_ELEVATION_RANGE_DEG[0]}),
        ({"distance": 0.0}, {"distance": config.VIEW_DISTANCE_RANGE_M[0]}),
        ({"distance": 50.0}, {"distance": config.VIEW_DISTANCE_RANGE_M[1]}),
        ({"distance": 1e308}, {"distance": config.VIEW_DISTANCE_RANGE_M[1]}),
        ({"azimuth": 270.0}, {"azimuth": -90.0}),
        ({"azimuth": -190.0}, {"azimuth": 170.0}),
    ],
)
def test_out_of_range_is_clamped_not_rejected(overrides, expected):
    assert parse(view(**overrides)) == view(**expected)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("field", ["azimuth", "elevation", "distance"])
def test_nan_and_infinity_are_rejected(field, constant):
    raw = json.dumps(view()).replace(f'"{field}": {json.dumps(view()[field])}', f'"{field}": {constant}')
    assert constant in raw
    with pytest.raises(ProtocolError):
        parse_client_message(raw, MAX_BYTES)


@pytest.mark.parametrize("bad", ["90", None, True, False, [90], {"v": 90}])
@pytest.mark.parametrize("field", ["azimuth", "elevation", "distance"])
def test_wrong_types_are_rejected(field, bad):
    with pytest.raises(ProtocolError):
        parse(view(**{field: bad}))


@pytest.mark.parametrize("missing", ["azimuth", "elevation", "distance"])
def test_missing_field_is_rejected(missing):
    msg = view()
    del msg[missing]
    with pytest.raises(ProtocolError, match="missing"):
        parse(msg)


def test_unknown_field_is_rejected():
    with pytest.raises(ProtocolError, match="unknown"):
        parse(view(lookat=[0, 0, 0]))
