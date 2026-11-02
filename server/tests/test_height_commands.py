"""키보드 높이 명령 적분(server/height.py): 방향 × 속도 × 흐른 시간. 시계 기준, 틱 횟수와 무관."""

import pytest

from server import config
from server.height import HeightCommands

SPEED = 0.08
SLOW = 0.02
TIMEOUT = 0.3
TICK_S = 1 / 30


def make(**kw):
    return HeightCommands(kw.get("speed", SPEED), kw.get("slow", SLOW), kw.get("timeout", TIMEOUT), kw.get("max_dt", 0.1))


def run(h, until_s, start_s=0.0, tick_s=TICK_S):
    """start_s부터 until_s까지 tick_s 간격으로 advance를 부르고 이동량(m)의 합을 돌려준다. 마지막은 until_s에 맞춘다."""
    total = 0.0
    t = start_s
    while t + tick_s < until_s - 1e-12:
        t += tick_s
        total += h.advance(t)
    return total + h.advance(until_s)


def press(h, direction, slow, t0, seconds, repeat_hz=20):
    """t0에 누르고, repeat_hz로 반복 메시지를 보내고, t0 + seconds에 뗀다. 그 사이 틱마다 advance한 합(m)."""
    h.command(direction, slow, t0)
    h.advance(t0)  # 누르는 순간 틱
    sent = t0
    total = 0.0
    t = t0
    end = t0 + seconds
    while t + TICK_S < end - 1e-12:
        t += TICK_S
        while sent + 1 / repeat_hz <= t + 1e-12:
            sent += 1 / repeat_hz
            h.command(direction, slow, sent)
        total += h.advance(t)
    h.command(0, False, end)
    total += h.advance(end)
    return total


def test_config_defaults():
    assert config.HEIGHT_SPEED_M_S == 0.08
    assert config.HEIGHT_SLOW_SPEED_M_S == 0.02
    assert config.HEIGHT_COMMAND_TIMEOUT_S == 0.3
    assert config.Z_FROM_HAND_SCALE is False
    # 맨 위 → 맨 아래 시간은 속도만으로 정해진다
    height_range = config.WORKSPACE_Z_M[1] - config.WORKSPACE_Z_M[0]
    assert height_range / config.HEIGHT_SPEED_M_S == pytest.approx(1.3125)
    assert height_range / config.HEIGHT_SLOW_SPEED_M_S == pytest.approx(5.25)


def test_no_command_means_no_motion():
    h = make()
    assert run(h, 2.0) == 0.0
    assert h.effective(2.0) == (0, False)


@pytest.mark.parametrize("direction,sign", [(1, +1), (-1, -1)])
def test_motion_is_direction_times_speed_times_time(direction, sign):
    h = make()
    dz = press(h, direction, False, t0=10.0, seconds=1.0)
    assert dz == pytest.approx(sign * SPEED * 1.0, abs=1e-12)


def test_slow_speed():
    h = make()
    assert press(h, -1, True, t0=3.0, seconds=1.0) == pytest.approx(-SLOW, abs=1e-12)


@pytest.mark.parametrize("seconds", [0.05, 0.1, 0.5, 1.0, 2.5])
def test_result_does_not_depend_on_tick_rate(seconds):
    for tick_s in (1 / 15, 1 / 30, 1 / 60, 0.007):
        h = make()
        h.command(-1, False, 5.0)
        total = 0.0
        t = 5.0
        sent = 5.0
        end = 5.0 + seconds
        while t + tick_s < end - 1e-12:
            t += tick_s
            while sent + 0.05 <= t + 1e-12:
                sent += 0.05
                h.command(-1, False, sent)
            total += h.advance(t)
        h.command(0, False, end)
        total += h.advance(end)
        assert total == pytest.approx(-SPEED * seconds, abs=1e-12), tick_s


def test_counting_starts_at_the_press_time_not_at_the_next_tick():
    h = make()
    h.advance(1.0)
    h.command(1, False, 1.01)
    assert h.advance(1.03) == pytest.approx(SPEED * 0.02, abs=1e-12)  # 1.01부터 1.03까지


def test_counting_ends_at_the_release_time():
    h = make()
    h.command(1, False, 1.0)
    h.advance(1.0)
    h.command(0, False, 1.04)
    assert h.advance(1.07) == pytest.approx(SPEED * 0.04, abs=1e-12)  # 1.04에 뗐으니 1.07까지 가지 않는다
    assert h.advance(1.5) == 0.0


def test_timeout_stops_motion_when_messages_stop():
    h = make()
    h.command(-1, False, 1.0)  # 한 번 보내고 브라우저가 죽었다
    total = run(h, 3.0, start_s=1.0)
    assert total == pytest.approx(-SPEED * TIMEOUT, abs=1e-12)  # 0.3초까지만 움직인다
    assert h.effective(1.2) == (-1, False)
    assert h.effective(1.31) == (0, False)
    assert run(h, 5.0, start_s=3.0) == 0.0


def test_timeout_is_measured_from_the_last_message():
    h = make()
    h.command(-1, False, 1.0)
    h.command(-1, False, 1.2)  # 반복 메시지가 오면 유효 시간이 늘어난다
    total = run(h, 3.0, start_s=1.0)
    assert total == pytest.approx(-SPEED * (1.2 + TIMEOUT - 1.0), abs=1e-12)


def test_worst_case_drift_after_crash_is_speed_times_timeout():
    for slow, speed in ((False, SPEED), (True, SLOW)):
        h = make()
        h.command(1, slow, 0.0)
        assert abs(run(h, 10.0)) == pytest.approx(speed * TIMEOUT, abs=1e-12)
        assert speed * TIMEOUT <= 0.025


def test_direction_change_while_holding():
    # advance를 한 번만 불러 명령 여러 개를 재생하므로, 멈칫 방어 상한(max_dt)은 넉넉하게 준다(상한 동작은 따로 검사한다).
    h = make(max_dt=10.0)
    h.command(-1, False, 1.0)
    h.advance(1.0)
    h.command(1, False, 1.2)  # 내리다가 올리기로 바꿈
    h.command(0, False, 1.5)
    total = h.advance(1.6)
    assert total == pytest.approx(-SPEED * 0.2 + SPEED * 0.3, abs=1e-12)


def test_slow_change_applies_from_the_message_time():
    h = make(max_dt=10.0)  # 위와 같은 이유로 상한을 넉넉하게
    for k in range(10):  # 실제 브라우저처럼 누르는 동안 20Hz로 반복 메시지(1.0 ~ 1.45)
        h.command(-1, False, 1.0 + 0.05 * k)
    h.advance(1.0)
    for k in range(10):  # 1.5에 Shift를 눌러 천천히로 바뀜(1.5 ~ 1.95)
        h.command(-1, True, 1.5 + 0.05 * k)
    h.command(0, False, 2.0)
    assert h.advance(2.1) == pytest.approx(-SPEED * 0.5 - SLOW * 0.5, abs=1e-12)


def test_a_stalled_control_thread_does_not_cause_a_jump():
    h = make(max_dt=0.1)
    h.command(-1, False, 1.0)
    h.advance(1.0)
    for k in range(1, 80):  # 브라우저는 계속 반복 메시지를 보낸다(4초)
        h.command(-1, False, 1.0 + 0.05 * k)
    dz = h.advance(5.0)  # 제어 스레드가 4초 만에 돌아왔다
    assert dz == pytest.approx(-SPEED * 0.1, abs=1e-12)  # 상한(0.1초)만큼만 센다 — 0.32 m가 한꺼번에 가지 않는다


def test_first_advance_after_press_counts_from_the_press():
    h = make()
    h.command(-1, False, 7.0)
    assert h.advance(7.05) == pytest.approx(-SPEED * 0.05, abs=1e-12)


def test_future_messages_wait_for_their_time():
    h = make(max_dt=10.0)
    h.command(-1, False, 10.0)
    assert h.advance(5.0) == 0.0  # 아직 오지 않은 시각의 명령은 다음에
    assert h.advance(10.5) == pytest.approx(-SPEED * TIMEOUT, abs=1e-12)  # 10.0부터 타임아웃 0.3초까지만


def test_reset_clears_everything():
    h = make()
    h.command(-1, False, 1.0)
    h.reset()
    assert h.advance(1.1) == 0.0
    assert h.effective(1.1) == (0, False)


def test_effective_reports_pending_commands_before_advance():
    h = make()
    h.command(1, True, 1.0)
    assert h.effective(1.1) == (1, True)
    assert h.effective(0.5) == (0, False)  # 아직 오지 않은 시각
