"""손 → 로봇 목표 위치(m). 좌표 변환은 순수 함수, 상태가 있는 부분(필터, 시작 조건, 마지막 목표 유지)은 HandMapper.

입력 hand 메시지의 좌표는 MediaPipe 정규화 좌표(0~1, 좌우 뒤집지 않은 원본). 출력은 로봇 베이스 기준 x, y, z(m):
x는 앞쪽(top 카메라 화면 위), y는 왼쪽(화면 오른쪽이 -y), z는 위.
"""

from dataclasses import dataclass, field

import numpy as np

from server import config
from server.one_euro import OneEuroFilter

Range = tuple[float, float]


def lerp(value: float, src: Range, dst: Range) -> float:
    """src 구간의 value를 dst 구간으로 직선 대응한다. 구간 밖은 늘려서 계산한다(자르지 않는다)."""
    a, b = src
    c, d = dst
    if a == b:
        return (c + d) / 2.0
    return c + (value - a) / (b - a) * (d - c)


@dataclass(frozen=True)
class MappingParams:
    img_x_range: Range = config.HAND_IMG_X_RANGE
    img_y_range: Range = config.HAND_IMG_Y_RANGE
    robot_y_for_img_x_m: Range = config.ROBOT_Y_FOR_IMG_X_M
    robot_x_for_img_y_m: Range = config.ROBOT_X_FOR_IMG_Y_M
    hand_scale_range: Range = config.HAND_SCALE_RANGE
    robot_z_for_scale_m: Range = config.ROBOT_Z_FOR_SCALE_M
    workspace_x_m: Range = config.WORKSPACE_X_M
    workspace_y_m: Range = config.WORKSPACE_Y_M
    workspace_z_m: Range = config.WORKSPACE_Z_M


def clamp_to_workspace(target_m: np.ndarray, params: MappingParams = MappingParams()) -> np.ndarray:
    lows = np.array([params.workspace_x_m[0], params.workspace_y_m[0], params.workspace_z_m[0]])
    highs = np.array([params.workspace_x_m[1], params.workspace_y_m[1], params.workspace_z_m[1]])
    return np.clip(target_m, lows, highs)


def hand_to_raw_target(
    img_x: float, img_y: float, hand_scale: float, params: MappingParams = MappingParams()
) -> np.ndarray:
    """손목 위치와 손 크기 → 목표 위치(m). 상자로 자르지 않은 값."""
    y_m = lerp(img_x, params.img_x_range, params.robot_y_for_img_x_m)
    x_m = lerp(img_y, params.img_y_range, params.robot_x_for_img_y_m)
    z_m = lerp(hand_scale, params.hand_scale_range, params.robot_z_for_scale_m)
    return np.array([x_m, y_m, z_m])


def hand_to_target(
    img_x: float, img_y: float, hand_scale: float, params: MappingParams = MappingParams()
) -> np.ndarray:
    """필터 없이 변환하고 상자 안으로 자른 목표 위치(m)."""
    return clamp_to_workspace(hand_to_raw_target(img_x, img_y, hand_scale, params), params)


@dataclass
class MapResult:
    target_m: np.ndarray | None  # 지금 따라갈 목표. 아직 조종을 시작하지 않았으면 None
    controlling: bool  # 조종 손이 지금 보이고 시작 조건을 채웠는가
    consecutive_frames: int  # 연속으로 보인 프레임 수
    raw_target_m: np.ndarray | None = None  # 이번 메시지의 필터 전 값(상자로 자른 것). 흔들림 확인용
    new_sample: bool = False  # 이번 호출이 목표를 새로 계산했는가


@dataclass
class HandMapper:
    params: MappingParams = field(default_factory=MappingParams)
    start_frames: int = config.START_CONSECUTIVE_FRAMES
    min_cutoff_hz: float = config.ONE_EURO_MIN_CUTOFF_HZ
    beta: float = config.ONE_EURO_BETA
    d_cutoff_hz: float = config.ONE_EURO_D_CUTOFF_HZ

    def __post_init__(self) -> None:
        self._filters = [OneEuroFilter(self.min_cutoff_hz, self.beta, self.d_cutoff_hz) for _ in range(3)]
        self._last_target_m: np.ndarray | None = None
        self._consecutive = 0

    @property
    def last_target_m(self) -> np.ndarray | None:
        return self._last_target_m

    def update(self, hand: dict | None, t_s: float) -> MapResult:
        """hand 메시지(검사를 통과한 dict)와 받은 시각(s)을 넣는다.

        - 손이 없거나(detected false) hand가 None이면 마지막 목표를 그대로 돌려주고, 연속 프레임 수와 필터를 초기화한다.
        - 손이 start_frames 연속으로 보이기 전에는 목표를 바꾸지 않는다.
        """
        if hand is None or not hand["detected"]:
            self._consecutive = 0
            for f in self._filters:
                f.reset()
            return MapResult(self._last_target_m, False, 0)

        self._consecutive += 1
        if self._consecutive < max(1, self.start_frames):
            return MapResult(self._last_target_m, False, self._consecutive)

        wrist = hand["wrist"]
        raw_m = hand_to_target(wrist[0], wrist[1], hand["hand_scale"], self.params)
        filtered = np.array([f(float(v), t_s) for f, v in zip(self._filters, raw_m)])
        self._last_target_m = clamp_to_workspace(filtered, self.params)
        return MapResult(self._last_target_m, True, self._consecutive, raw_target_m=raw_m, new_sample=True)
