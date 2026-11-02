"""집게 열고 닫기 기준(두 기준, 사이는 유지)."""

import pytest

from server import config
from server.gripper import GripperLatch


def latch(closed=False):
    return GripperLatch(config.PINCH_CLOSE_BELOW, config.PINCH_OPEN_ABOVE, closed=closed)


def test_config_thresholds_are_the_requested_ones():
    assert config.PINCH_CLOSE_BELOW == 0.3
    assert config.PINCH_OPEN_ABOVE == 0.5


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.29])
def test_below_close_threshold_closes(ratio):
    assert latch().update(ratio) is True


@pytest.mark.parametrize("ratio", [0.51, 0.8, 1.0])
def test_above_open_threshold_opens(ratio):
    assert latch(closed=True).update(ratio) is False


@pytest.mark.parametrize("ratio", [0.3, 0.4, 0.5])
def test_between_thresholds_keeps_state(ratio):
    assert latch(closed=True).update(ratio) is True
    assert latch(closed=False).update(ratio) is False


def test_no_flicker_around_a_single_value():
    g = latch()
    g.update(0.1)  # 닫힘
    states = [g.update(r) for r in (0.28, 0.32, 0.29, 0.35, 0.31)]  # 0.3 근처에서 흔들려도
    assert states == [True] * 5
    assert g.update(0.55) is False  # 0.5를 넘어야 열림
    assert [g.update(r) for r in (0.45, 0.49, 0.32)] == [False] * 3  # 사이 값은 열림 유지


def test_none_keeps_state():
    g = latch(closed=True)
    assert g.update(None) is True
    g.update(0.9)
    assert g.update(None) is False


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        GripperLatch(0.5, 0.3)
    with pytest.raises(ValueError):
        GripperLatch(0.4, 0.4)
