"""시뮬레이션 시간 격자: 녹화(3-2)와 평가(3-4)가 같은 규칙으로 샘플을 찍기 위한 순수 계산.

문제: 모델 timestep이 0.005 s라서 1/30 s(33.3 ms)가 물리 스텝의 정수 배가 아니다. 그래서 샘플 k의 목표 시각을 t0 + k/fps로 두고,
"그 시각에 가장 가까운 물리 스텝"에서 찍는다(반올림, 딱 중간이면 늦은 쪽). 시작 시각 t0은 첫 샘플을 찍은 시뮬레이션 시각(s)이다.

    스텝 번호 s_k = ceil(k / (fps * timestep) - 0.5)       (t0에서 몇 스텝 뒤인가)
    예: fps 30, timestep 0.005 → s = 0, 7, 13, 20, 27, 33, 40, ... (간격 35, 30, 35, 35, 30, 35 ms)

격자와의 차이는 최대 timestep / 2 = 2.5 ms다(실제로는 1.67 ms). 각 샘플의 정확한 시각은 sim_time_s에 그대로 저장한다.

평가도 같은 격자로 돌려야 한다: 정책을 부른 샘플 k와 k+1 사이에는 grid_step_indices()가 알려주는 만큼(6 또는 7개) mj_step을 진행한다.
틱 간격(벽시계)과는 상관이 없다.
"""

import math

# 부동소수점 오차로 정확히 중간 값이 한 칸 어긋나는 것을 막는 작은 값
_EPS = 1e-9


def grid_step_index(k: int, fps: float, timestep_s: float) -> int:
    """샘플 k가 t0에서 몇 번째 물리 스텝에서 찍히는가(k=0이면 0)."""
    return math.ceil(k / (fps * timestep_s) - 0.5 - _EPS)


def grid_step_indices(n: int, fps: float, timestep_s: float) -> list[int]:
    """샘플 0..n-1의 스텝 번호 목록. 평가 루프가 정책 호출 사이에 진행할 스텝 수를 정할 때 쓴다."""
    return [grid_step_index(k, fps, timestep_s) for k in range(n)]


def steps_done(sim_time_s: float, t0_s: float, timestep_s: float) -> int:
    """t0 이후 진행한 물리 스텝 수(시각은 timestep의 정수 배라서 반올림으로 오차를 지운다)."""
    return round((sim_time_s - t0_s) / timestep_s)


def sample_due(sim_time_s: float, t0_s: float, k: int, fps: float, timestep_s: float) -> bool:
    """지금(물리 스텝 직전)이 샘플 k를 찍을 때인가."""
    return steps_done(sim_time_s, t0_s, timestep_s) >= grid_step_index(k, fps, timestep_s)


def grid_deviation_s(k: int, fps: float, timestep_s: float) -> float:
    """샘플 k를 찍는 실제 시각과 이상적인 시각 t0 + k/fps의 차이(s). 양수면 늦게 찍는다."""
    return grid_step_index(k, fps, timestep_s) * timestep_s - k / fps
