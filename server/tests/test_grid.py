"""시간 격자: 30 Hz 샘플이 5 ms 물리 스텝에서 어떤 스텝에 찍히는지(녹화와 평가가 공유하는 순수 계산)."""

import pytest

from server import config
from server.grid import grid_deviation_s, grid_step_index, grid_step_indices, sample_due, steps_done

TS = 0.005
FPS = 30


def test_first_steps_match_documented_pattern():
    assert grid_step_indices(7, FPS, TS) == [0, 7, 13, 20, 27, 33, 40]


def test_spacing_alternates_30_and_35_ms():
    steps = grid_step_indices(300, FPS, TS)
    spacing_ms = {round((b - a) * TS * 1000) for a, b in zip(steps, steps[1:], strict=False)}
    assert spacing_ms == {30, 35}


def test_no_drift_over_a_long_episode():
    n = 1800  # 60 초
    last_s = grid_step_index(n, FPS, TS) * TS
    assert last_s == pytest.approx(n / FPS, abs=TS / 2)


def test_deviation_is_at_most_half_a_physics_step():
    assert max(abs(grid_deviation_s(k, FPS, TS)) for k in range(2000)) <= TS / 2 + 1e-12


def test_exact_multiple_keeps_exact_step():
    # fps 25, timestep 0.005 → 정확히 8 스텝 간격
    assert grid_step_indices(5, 25, TS) == [0, 8, 16, 24, 32]


def test_sample_due_matches_index_formula_when_stepping():
    """물리 스텝을 하나씩 진행하며 sample_due로 찍은 스텝이 grid_step_indices와 같다(워커의 실제 사용 방식)."""
    t0 = 1.235  # 시작 시각은 0이 아니어도 된다
    k = 0
    taken = []
    for step in range(0, 400):
        t = t0 + step * TS
        if sample_due(t, t0, k, FPS, TS):
            taken.append(step)
            k += 1
    assert taken == grid_step_indices(len(taken), FPS, TS)
    assert len(taken) == 60  # 400 스텝 = 2.0 초 → 30 Hz 60개


def test_steps_done_rounds_float_error_away():
    t = 0.0
    for _ in range(1000):
        t += TS  # 더하기를 반복해 생긴 오차
    assert steps_done(t, 0.0, TS) == 1000


def test_config_fps_and_timestep_give_documented_spacing():
    assert config.RECORD_FPS == 30
    assert grid_step_indices(3, config.RECORD_FPS, TS) == [0, 7, 13]
