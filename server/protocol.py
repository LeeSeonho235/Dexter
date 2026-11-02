"""브라우저 → 서버 메시지 검사 (docs/04-architecture.md의 hand 형식 + pinch_ratio, ping, view).

순수 함수만 둔다. 잘못된 메시지는 ProtocolError를 던지고, 서버는 이를 받아 연결을 유지한 채 무시한다.
좌표는 MediaPipe 정규화 좌표(화면 0~1)지만 손이 화면 가장자리에 걸치면 0~1 밖도 나오므로 범위는 검사하지 않고
유한한 숫자인지만 본다.

view(보기 방향): {"type": "view", "azimuth": 도, "elevation": 도, "distance": m}. 유한한 숫자가 아니거나 타입이 틀리거나
필드가 빠지면 거부하고, 범위 밖 숫자는 거부하지 않고 허용 범위로 자른다(server/view.py clamp_view).

height(키보드 높이): {"type": "height", "dir": -1|0|1, "slow": true|false}. dir은 정수 -1(내리기)/0(정지)/1(올리기)만,
slow는 bool만 받는다. 범위를 자르지 않고 그 외는 모두 거부한다(안전: 잘못된 값으로 로봇이 움직이면 안 된다).
"""

import json
import math
from typing import Any

from server.view import clamp_view

POINT_FIELDS = ("wrist", "index_tip", "thumb_tip")
HAND_FIELDS = frozenset(
    {"type", "t", "detected", *POINT_FIELDS, "pinch", "pinch_ratio", "palm_open", "hand_scale"}
)
PING_FIELDS = frozenset({"type", "id"})
VIEW_FIELDS = frozenset({"type", "azimuth", "elevation", "distance"})
HEIGHT_FIELDS = frozenset({"type", "dir", "slow"})
HEIGHT_DIRECTIONS = (-1, 0, 1)


class ProtocolError(ValueError):
    """잘못된 메시지. str(err)는 브라우저에 돌려보내는 짧은 이유."""


def parse_client_message(raw: str | bytes, max_bytes: int) -> dict[str, Any]:
    """텍스트 한 개를 검사해서 dict로 돌려준다. hand 또는 ping만 받는다."""
    if isinstance(raw, bytes):
        raise ProtocolError("binary messages are not accepted")
    if len(raw.encode("utf-8")) > max_bytes:
        raise ProtocolError(f"message larger than {max_bytes} bytes")
    try:
        # json.loads는 기본으로 NaN, Infinity를 받아들이므로 막는다.
        msg = json.loads(raw, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ProtocolError) as err:
        raise ProtocolError(f"invalid JSON: {err}") from None
    if not isinstance(msg, dict):
        raise ProtocolError("message must be a JSON object")

    msg_type = msg.get("type")
    if msg_type == "hand":
        return validate_hand(msg)
    if msg_type == "ping":
        return validate_ping(msg)
    if msg_type == "view":
        return validate_view(msg)
    if msg_type == "height":
        return validate_height(msg)
    raise ProtocolError(f"unknown type: {msg_type!r}")


def validate_hand(msg: dict[str, Any]) -> dict[str, Any]:
    _check_fields(msg, HAND_FIELDS)
    _check_number(msg, "t")
    if not isinstance(msg["detected"], bool):
        raise ProtocolError("detected must be true/false")

    if not msg["detected"]:
        for name in HAND_FIELDS - {"type", "t", "detected"}:
            if msg[name] is not None:
                raise ProtocolError(f"{name} must be null when detected is false")
        return msg

    for name in POINT_FIELDS:
        point = msg[name]
        if not isinstance(point, list) or len(point) != 3:
            raise ProtocolError(f"{name} must be a list of 3 numbers")
        for value in point:
            if not _is_finite_number(value):
                raise ProtocolError(f"{name} must be a list of 3 numbers")
    _check_number(msg, "pinch", min_value=0.0)
    _check_number(msg, "hand_scale", min_value=0.0)
    # hand_scale이 0이면 브라우저가 pinch_ratio를 null로 보낸다 (web/src/hand/features.js pinchRatio)
    if msg["pinch_ratio"] is not None:
        _check_number(msg, "pinch_ratio", min_value=0.0)
    if not isinstance(msg["palm_open"], bool):
        raise ProtocolError("palm_open must be true/false")
    return msg


def validate_ping(msg: dict[str, Any]) -> dict[str, Any]:
    _check_fields(msg, PING_FIELDS)
    ping_id = msg["id"]
    if isinstance(ping_id, bool) or not isinstance(ping_id, int):
        raise ProtocolError("ping id must be an integer")
    return msg


def validate_view(msg: dict[str, Any]) -> dict[str, Any]:
    """돌려주는 값은 범위를 맞춘 view 메시지(방위각은 -180~180으로 돌림)."""
    _check_fields(msg, VIEW_FIELDS)
    for name in ("azimuth", "elevation", "distance"):
        _check_number(msg, name)
    view = clamp_view(msg["azimuth"], msg["elevation"], msg["distance"])
    return {
        "type": "view",
        "azimuth": view.azimuth_deg,
        "elevation": view.elevation_deg,
        "distance": view.distance_m,
    }


def validate_height(msg: dict[str, Any]) -> dict[str, Any]:
    """키보드 높이: dir은 정수 -1(내리기), 0(정지), 1(올리기)만, slow는 true/false만 받는다. 그 외는 거부."""
    _check_fields(msg, HEIGHT_FIELDS)
    direction = msg["dir"]
    # bool은 int의 하위 타입이라 따로 뺀다. 1.0 같은 실수도 받지 않는다(브라우저는 정수로 보낸다).
    if isinstance(direction, bool) or not isinstance(direction, int) or direction not in HEIGHT_DIRECTIONS:
        raise ProtocolError("dir must be -1, 0 or 1")
    if not isinstance(msg["slow"], bool):
        raise ProtocolError("slow must be true/false")
    return msg


def hand_summary(msg: dict[str, Any]) -> dict[str, Any]:
    """상태 메시지에 넣을 마지막 hand 값 요약."""
    return {
        "t": msg["t"],
        "detected": msg["detected"],
        "wrist": msg["wrist"],
        "pinch_ratio": msg["pinch_ratio"],
        "palm_open": msg["palm_open"],
        "hand_scale": msg["hand_scale"],
    }


def _reject_constant(name: str) -> None:
    raise ProtocolError(f"{name} is not allowed")


def _check_fields(msg: dict[str, Any], expected: frozenset[str]) -> None:
    keys = set(msg)
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise ProtocolError(f"missing fields: {sorted(missing)}")
    if extra:
        raise ProtocolError(f"unknown fields: {sorted(extra)}")


def _is_finite_number(value: Any) -> bool:
    # bool은 int의 하위 타입이라 따로 뺀다.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_number(msg: dict[str, Any], name: str, min_value: float | None = None) -> None:
    value = msg[name]
    if not _is_finite_number(value):
        raise ProtocolError(f"{name} must be a finite number")
    if min_value is not None and value < min_value:
        raise ProtocolError(f"{name} must be >= {min_value}")
