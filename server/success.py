"""성공 자동 판정. 녹화(3-2)와 평가(3-4)가 같은 함수를 쓴다.

한 순간의 조건 is_success()는 순수 함수이고, "0.5초 이상 유지"는 시뮬레이션 시간으로 세는 SuccessHold가 맡는다.
저장된 에피소드(sim_time_s, qpos)에서 다시 판정하려면 success_over_time()을 쓴다.

조건(모두 만족): 블록 중심 xy가 목표 중심에서 (목표 반변 + 여유) 안쪽(경계 포함) / 블록 중심 z < z_max(경계 제외) /
집게 실제 각도 >= 열림 기준(경계 포함). 단위: 길이 m, 각도 rad, 시간 s. 기준값은 server/config.py.
"""

from collections.abc import Sequence

import numpy as np

from server import config

QPOS_BLOCK_XYZ = slice(6, 9)  # 전체 qpos 13개 중 블록 자유 관절의 위치(월드 좌표 m)
QPOS_GRIPPER = 5  # 전체 qpos 중 집게 관절


def is_success(
    block_m: Sequence[float],
    gripper_rad: float,
    *,
    target_center_m: Sequence[float] = config.TARGET_CENTER_M,
    target_half_m: float = config.TARGET_HALF_M,
    xy_margin_m: float = config.SUCCESS_XY_MARGIN_M,
    z_max_m: float = config.SUCCESS_BLOCK_Z_MAX_M,
    gripper_open_min_rad: float = config.SUCCESS_GRIPPER_OPEN_MIN_RAD,
) -> bool:
    """한 순간의 성공 조건. block_m은 블록 중심 (x, y, z)(m), gripper_rad는 집게 실제 각도(rad)."""
    limit_m = target_half_m + xy_margin_m
    inside = abs(block_m[0] - target_center_m[0]) <= limit_m and abs(block_m[1] - target_center_m[1]) <= limit_m
    return bool(inside and block_m[2] < z_max_m and gripper_rad >= gripper_open_min_rad)


def success_from_qpos(qpos: Sequence[float]) -> bool:
    """전체 qpos(팔 6개 + 블록 7개)에서 한 순간의 조건을 판정한다."""
    return is_success(qpos[QPOS_BLOCK_XYZ], float(qpos[QPOS_GRIPPER]))


class SuccessHold:
    """조건이 hold_s(시뮬레이션 시간) 이상 계속 맞았는지 센다. 한 번이라도 끊기면 처음부터 다시 센다."""

    def __init__(self, hold_s: float = config.SUCCESS_HOLD_S) -> None:
        self.hold_s = hold_s
        self._since_s: float | None = None
        self.satisfied = False

    def update(self, sim_time_s: float, ok: bool) -> bool:
        """이 시각의 조건 결과를 넣고, 지금까지 hold_s 이상 유지됐으면 True."""
        if not ok:
            self._since_s = None
            self.satisfied = False
            return False
        if self._since_s is None:
            self._since_s = sim_time_s
        self.satisfied = sim_time_s - self._since_s >= self.hold_s - 1e-9
        return self.satisfied

    def reset(self) -> None:
        self._since_s = None
        self.satisfied = False


def success_over_time(sim_time_s: np.ndarray, qpos: np.ndarray, hold_s: float = config.SUCCESS_HOLD_S):
    """저장된 에피소드에서 판정: (한 번이라도 유지 달성했나, 처음 달성한 시각 s 또는 None, 끝 시점에 달성 상태인가)."""
    hold = SuccessHold(hold_s)
    first_s: float | None = None
    for t, q in zip(sim_time_s, qpos, strict=True):
        if hold.update(float(t), success_from_qpos(q)) and first_s is None:
            first_s = float(t)
    return first_s is not None, first_s, hold.satisfied
