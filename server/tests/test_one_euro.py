"""One Euro Filter: 순수 계산."""

import numpy as np
import pytest

from server.one_euro import OneEuroFilter

DT_S = 1 / 30


def run(filt, values, dt_s=DT_S):
    return np.array([filt(v, i * dt_s) for i, v in enumerate(values)])


def test_first_value_passes_through():
    assert OneEuroFilter(1.0, 0.0, 1.0)(0.123, 5.0) == 0.123


def test_constant_input_stays_constant():
    out = run(OneEuroFilter(1.0, 10.0, 1.0), [0.2] * 50)
    np.testing.assert_allclose(out, 0.2)


def test_noise_is_reduced_when_still():
    rng = np.random.default_rng(0)
    noisy = 0.2 + rng.normal(0, 0.005, 300)  # 5 mm 떨림(m)
    out = run(OneEuroFilter(1.0, 5.0, 1.0), noisy)
    assert out[60:].std() < 0.5 * noisy[60:].std()


def test_larger_beta_follows_fast_motion_with_less_lag():
    ramp = np.linspace(0.0, 0.3, 30)  # 1초 동안 0.3 m 이동
    slow = run(OneEuroFilter(1.0, 0.0, 1.0), ramp)
    fast = run(OneEuroFilter(1.0, 50.0, 1.0), ramp)
    assert abs(ramp[-1] - fast[-1]) < abs(ramp[-1] - slow[-1])


def test_reset_accepts_next_value_as_is():
    filt = OneEuroFilter(1.0, 0.0, 1.0)
    run(filt, [0.0] * 10)
    filt.reset()
    assert filt(0.9, 100.0) == 0.9


def test_non_increasing_time_does_not_crash():
    filt = OneEuroFilter(1.0, 1.0, 1.0)
    filt(0.0, 1.0)
    assert np.isfinite(filt(0.1, 1.0))  # 같은 시각
    assert np.isfinite(filt(0.2, 0.5))  # 거꾸로 간 시각


@pytest.mark.parametrize("min_cutoff,d_cutoff", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_invalid_cutoff_rejected(min_cutoff, d_cutoff):
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff, 0.0, d_cutoff)
