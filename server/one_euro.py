"""One Euro Filter: 떨림 제거. 멈춰 있을 때는 많이 거르고, 빠르게 움직일 때는 덜 거른다(지연이 적다).

원리(Casiez et al. 2012): 속도를 구해 거르고, 그 속도로 값 거르기 세기(cutoff)를 정한다.
    cutoff = min_cutoff_hz + beta * |속도|
시각은 호출하는 쪽이 초(s) 단위로 넣는다. 순수 계산이라 시계에 의존하지 않는다.
"""

import math

# 시각이 거꾸로 가거나 같을 때 쓰는 간격(s). 30Hz 기준.
FALLBACK_DT_S = 1.0 / 30.0


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    """저역통과 필터 계수(0~1). 클수록 새 값을 많이 반영한다."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


class OneEuroFilter:
    """숫자 하나를 거른다. 단위는 입력 값의 단위를 따른다(예: m)."""

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float) -> None:
        if min_cutoff_hz <= 0 or d_cutoff_hz <= 0:
            raise ValueError("cutoff must be > 0")
        self.min_cutoff_hz = min_cutoff_hz
        self.beta = beta
        self.d_cutoff_hz = d_cutoff_hz
        self.reset()

    def reset(self) -> None:
        """다음 값을 그대로 받아들인다(손이 다시 나타났을 때 옛 값이 끌려오지 않게)."""
        self._t_s: float | None = None
        self._x: float | None = None
        self._dx = 0.0

    def __call__(self, x: float, t_s: float) -> float:
        if self._x is None or self._t_s is None:
            self._t_s, self._x, self._dx = t_s, x, 0.0
            return x
        dt_s = t_s - self._t_s
        if dt_s <= 0:
            dt_s = FALLBACK_DT_S
        raw_dx = (x - self._x) / dt_s
        a_d = _alpha(self.d_cutoff_hz, dt_s)
        self._dx = a_d * raw_dx + (1.0 - a_d) * self._dx
        cutoff_hz = self.min_cutoff_hz + self.beta * abs(self._dx)
        a = _alpha(cutoff_hz, dt_s)
        self._x = a * x + (1.0 - a) * self._x
        self._t_s = t_s
        return self._x
