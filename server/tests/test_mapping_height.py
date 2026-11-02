"""매퍼의 키보드 높이(2-5): 오프셋, 상자 끝(죽은 구간 없음), 손 없을 때, 재클러치, Z_FROM_HAND_SCALE 옵션."""

import numpy as np
import pytest

from server import config
from server.mapping import HandMapper

Z_LO, Z_HI = config.WORKSPACE_Z_M
SCALE_MID = 0.13
NO_FILTER = {"min_cutoff_hz": 1e6, "beta": 0.0, "d_cutoff_hz": 1e6}
CENTER_Z = config.RELATIVE_START_POSITION_M[2]


def hand(x=0.5, y=0.5, scale=SCALE_MID, detected=True):
    if not detected:
        return {"type": "hand", "t": 0.0, "detected": False, "wrist": None, "index_tip": None, "thumb_tip": None,
                "pinch": None, "pinch_ratio": None, "palm_open": None, "hand_scale": None}
    return {"type": "hand", "t": 0.0, "detected": True, "wrist": [x, y, 0.0], "index_tip": [x, y, 0.0],
            "thumb_tip": [x, y, 0.0], "pinch": 0.1, "pinch_ratio": 0.8, "palm_open": True, "hand_scale": scale}


class Feeder:
    def __init__(self, mapper, az=0.0):
        self.mapper, self.az, self.t = mapper, az, 0.0

    def feed(self, x=0.5, y=0.5, scale=SCALE_MID, detected=True):
        self.t += 1 / 30
        return self.mapper.update(hand(x, y, scale, detected), self.t, self.az)

    def clutch(self, x=0.5, y=0.5, scale=SCALE_MID):
        result = None
        for _ in range(self.mapper.start_frames):
            result = self.feed(x, y, scale)
        return result


def make(mode, z_from_hand_scale=False, filtered=False, **kw):
    extra = {} if filtered else NO_FILTER
    return HandMapper(mode=mode, start_frames=3, z_from_hand_scale=z_from_hand_scale, **extra, **kw)


MODES = ["relative", "absolute"]
# NO_FILTER도 30Hz에서는 완전한 통과가 아니다(계수가 1에서 약 5e-6 모자란다). 손 크기가 갑자기 바뀌는 테스트의 허용 오차.
FILTER_RESIDUE_M = 1e-5


@pytest.mark.parametrize("mode", MODES)
def test_nudge_before_any_target_is_ignored(mode):
    m = make(mode)
    assert m.nudge_height(0.01) is None
    f = Feeder(m)
    f.feed()
    f.feed()  # 아직 시작 조건 전
    assert m.nudge_height(0.01) is None
    assert m.last_target_m is None


@pytest.mark.parametrize("mode", MODES)
def test_nudge_moves_only_z(mode):
    m = make(mode)
    f = Feeder(m)
    r = f.clutch(0.4, 0.45)
    before = r.target_m.copy()
    after = m.nudge_height(-0.02)
    assert after[2] == pytest.approx(before[2] - 0.02, abs=1e-12)
    assert after[0] == before[0] and after[1] == before[1]
    np.testing.assert_array_equal(m.last_target_m, after)
    assert m.last_target_m is not after  # 돌려준 것은 복사본이다


@pytest.mark.parametrize("mode", MODES)
def test_nudge_is_not_filtered_so_increments_are_exact(mode):
    """키보드 높이는 One Euro 필터를 거치지 않는다: 기본 필터 설정에서도 10번 0.001씩 올리면 정확히 0.01."""
    m = make(mode, filtered=True)
    f = Feeder(m)
    start = f.clutch().target_m[2]
    for _ in range(10):
        m.nudge_height(0.001)
    assert m.last_target_m[2] == pytest.approx(start + 0.01, abs=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_offset_survives_later_hand_messages(mode):
    m = make(mode)
    f = Feeder(m)
    f.clutch()
    m.nudge_height(-0.03)
    z_after_nudge = m.last_target_m[2]
    for k in range(10):  # 손이 계속 움직여도(좌우) 키보드로 바꾼 높이는 그대로
        r = f.feed(0.5 - 0.005 * k, 0.5)
        assert r.target_m[2] == pytest.approx(z_after_nudge, abs=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_top_and_bottom_are_clamped(mode):
    m = make(mode)
    f = Feeder(m)
    f.clutch()
    assert m.nudge_height(+10.0)[2] == pytest.approx(Z_HI)
    assert m.height_limit == "top"
    assert m.nudge_height(-10.0)[2] == pytest.approx(Z_LO)
    assert m.height_limit == "bottom"
    assert m.nudge_height(+0.01)[2] == pytest.approx(Z_LO + 0.01)
    assert m.height_limit is None


@pytest.mark.parametrize("mode", MODES)
def test_no_dead_zone_after_pushing_past_the_limit(mode):
    """맨 아래에서 ↓를 오래 눌렀다 ↑로 바꾸면 바로 올라간다(눌렀던 만큼 쌓여 있지 않다)."""
    m = make(mode)
    f = Feeder(m)
    f.clutch()
    for _ in range(100):
        m.nudge_height(-0.01)  # 1 m나 내리려 한다
    assert m.last_target_m[2] == pytest.approx(Z_LO)
    assert m.nudge_height(+0.005)[2] == pytest.approx(Z_LO + 0.005)  # 곧바로 반응
    for _ in range(100):
        m.nudge_height(+0.01)
    assert m.nudge_height(-0.005)[2] == pytest.approx(Z_HI - 0.005)


def test_height_limit_is_none_without_a_target():
    assert make("relative").height_limit is None


@pytest.mark.parametrize("mode", MODES)
def test_nudge_works_while_the_hand_is_not_visible(mode):
    m = make(mode)
    f = Feeder(m)
    f.clutch(0.4, 0.45)
    xy = m.last_target_m[:2].copy()
    for _ in range(5):
        f.feed(detected=False)  # 손이 화면 밖
    z0 = m.last_target_m[2]
    after = m.nudge_height(-0.02)
    assert after[2] == pytest.approx(z0 - 0.02, abs=1e-12)
    np.testing.assert_array_equal(after[:2], xy)  # x, y는 마지막 목표 그대로


def test_relative_reclutch_does_not_count_the_keyboard_height_twice():
    m = make("relative")
    f = Feeder(m)
    f.clutch()
    m.nudge_height(-0.03)
    held = m.last_target_m.copy()
    f.feed(detected=False)
    r = f.clutch(0.7, 0.6)  # 다른 곳에서 손을 다시 잡는다
    np.testing.assert_allclose(r.target_m, held, atol=1e-12)  # 높이가 튀지 않는다(이중으로 더해지지 않는다)
    again = m.nudge_height(-0.01)
    assert again[2] == pytest.approx(held[2] - 0.01, abs=1e-12)


def test_relative_keyboard_height_persists_through_reclutch():
    m = make("relative")
    f = Feeder(m)
    f.clutch()
    m.nudge_height(+0.03)
    f.feed(detected=False)
    f.clutch()
    for k in range(5):
        r = f.feed(0.5 - 0.01 * k, 0.5)
    assert r.target_m[2] == pytest.approx(CENTER_Z + 0.03, abs=1e-12)


# --- Z_FROM_HAND_SCALE ---


def test_default_code_value_keeps_the_2_4_behavior():
    assert HandMapper().z_from_hand_scale is True  # 코드상 기본은 2-4 동작(기존 테스트가 기대한다)
    assert config.Z_FROM_HAND_SCALE is False  # 설정값 기본은 꺼짐: 높이는 키보드


def test_relative_off_hand_size_does_not_change_the_height():
    f = Feeder(make("relative", z_from_hand_scale=False))
    f.clutch(0.5, 0.5, SCALE_MID)
    for scale in (0.07, 0.19, 0.30, 0.01):
        r = f.feed(0.5, 0.5, scale)
        assert r.target_m[2] == pytest.approx(CENTER_Z, abs=1e-12)
        assert r.raw_target_m[2] == pytest.approx(CENTER_Z, abs=1e-12)


def test_absolute_off_height_is_constant_at_the_start_height_and_keyboard_moves_it():
    m = make("absolute", z_from_hand_scale=False)
    f = Feeder(m)
    for scale in (0.07, 0.13, 0.19):
        r = f.clutch(0.5, 0.5, scale)
        assert r.target_m[2] == pytest.approx(CENTER_Z, abs=1e-12)
    m.nudge_height(-0.02)
    assert f.feed(0.5, 0.5, 0.19).target_m[2] == pytest.approx(CENTER_Z - 0.02, abs=1e-12)


def test_relative_on_follows_hand_size_like_2_4():
    f = Feeder(make("relative", z_from_hand_scale=True))
    f.clutch(0.5, 0.5, SCALE_MID)
    r = f.feed(0.5, 0.5, SCALE_MID + 0.05)
    assert r.target_m[2] == pytest.approx(CENTER_Z + config.RELATIVE_GAIN_Z_M_PER_SCALE * 0.05, abs=FILTER_RESIDUE_M)
    assert r.raw_target_m[2] == pytest.approx(CENTER_Z + config.RELATIVE_GAIN_Z_M_PER_SCALE * 0.05, abs=1e-12)  # 필터 전


def test_relative_on_keyboard_adds_to_hand_height():
    m = make("relative", z_from_hand_scale=True)
    f = Feeder(m)
    f.clutch(0.5, 0.5, SCALE_MID)
    m.nudge_height(+0.02)
    r = f.feed(0.5, 0.5, SCALE_MID + 0.02)  # 손이 가까워져서 -0.0175
    assert r.target_m[2] == pytest.approx(
        CENTER_Z + 0.02 + config.RELATIVE_GAIN_Z_M_PER_SCALE * 0.02, abs=FILTER_RESIDUE_M
    )


def test_absolute_on_is_the_2_2_height_plus_keyboard():
    m = make("absolute", z_from_hand_scale=True)
    f = Feeder(m)
    r0 = f.clutch(0.5, 0.5, 0.10)
    hand_z = r0.target_m[2]
    m.nudge_height(-0.01)
    r1 = f.feed(0.5, 0.5, 0.15)
    expected_hand_z = np.interp(0.15, config.HAND_SCALE_RANGE, config.ROBOT_Z_FOR_SCALE_M)
    assert hand_z == pytest.approx(np.interp(0.10, config.HAND_SCALE_RANGE, config.ROBOT_Z_FOR_SCALE_M), abs=1e-12)
    assert r1.target_m[2] == pytest.approx(expected_hand_z - 0.01, abs=FILTER_RESIDUE_M)


def test_clamp_with_hand_height_and_keyboard_together_has_no_dead_zone():
    m = make("relative", z_from_hand_scale=True)
    f = Feeder(m)
    f.clutch(0.5, 0.5, SCALE_MID)
    for _ in range(50):
        m.nudge_height(+0.01)
        f.feed(0.5, 0.5, SCALE_MID)
    assert m.last_target_m[2] == pytest.approx(Z_HI)
    assert m.nudge_height(-0.004)[2] == pytest.approx(Z_HI - 0.004)
