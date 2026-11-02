"""손 → 로봇 목표 위치(m). 좌표 변환은 순수 함수, 상태가 있는 부분(필터, 시작 조건, 마지막 목표 유지)은 HandMapper.

입력 hand 메시지의 좌표는 MediaPipe 정규화 좌표(0~1, 좌우 뒤집지 않은 원본). 출력은 로봇 베이스 기준 x, y, z(m):
x는 앞쪽(top 카메라 화면 위), y는 왼쪽(화면 오른쪽이 -y), z는 위.

매핑 방식(config.MAPPING_MODE):
- "absolute": 손 위치 → 로봇 위치. 월드 방향 고정(보기 방위각을 쓰지 않는다).
- "relative": 조종 시작 시점의 손 위치를 기준으로 "손이 움직인 양 × 이득"만큼 목표를 옮긴다(클러치).
  방향은 현재 보기 방위각 기준이라 손을 오른쪽으로 움직이면 화면에서도 오른쪽, 손을 위로 올리면 화면에서 안쪽(멀어지는 방향).
  움직임은 걸음마다 목표에 더하므로, 보기가 중간에 바뀌어도 이미 간 위치는 그대로이고 이후 움직임에만 새 방향이 적용된다.
"""

import math
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


def view_basis(azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """보기 방위각(도)에서 바닥 위의 화면 기준 방향 (forward_xy, right_xy)을 돌려준다(단위 벡터, 월드 x, y).

    forward: 화면에서 안쪽(위)으로 멀어지는 방향, right: 화면 오른쪽. MuJoCo 자유 카메라 규칙이다.
    고도각에는 의존하지 않으므로 거의 수직(-89°)으로 내려다볼 때도 안정적이다.
    예: 방위각 0°는 forward = +x, right = -y(위에서 보기), 90°는 right = +x.
    server/tests/test_screen_mapping.py가 MuJoCo 실제 카메라 축과 같은지 확인한다.
    """
    az = math.radians(azimuth_deg)
    forward = np.array([math.cos(az), math.sin(az)])
    right = np.array([math.sin(az), -math.cos(az)])
    return forward, right


def screen_step_to_world(
    dx_img: float, dy_img: float, azimuth_deg: float, gain_m_per_width: float, aspect: float = config.IMAGE_ASPECT
) -> np.ndarray:
    """영상에서 손이 (dx_img, dy_img)(정규화 좌표, 좌우 뒤집지 않은 원본) 움직였을 때의 월드 xy 이동(m).

    - 손이 오른쪽으로 가면 영상 x는 줄어든다 → 화면 오른쪽(right) 방향.
    - 손이 위로 가면 영상 y는 줄어든다 → 화면 안쪽(forward) 방향.
    - y는 영상 높이 기준이라 가로(너비) 길이 단위로 바꾼다: dy × (높이 / 너비) = dy / aspect.
    gain_m_per_width: 손이 영상 너비만큼 움직일 때 로봇이 가는 거리(m).
    """
    forward, right = view_basis(azimuth_deg)
    along_right_m = -dx_img * gain_m_per_width
    along_forward_m = -(dy_img / aspect) * gain_m_per_width
    return along_right_m * right + along_forward_m * forward


@dataclass
class MapResult:
    target_m: np.ndarray | None  # 지금 따라갈 목표. 아직 조종을 시작하지 않았으면 None
    controlling: bool  # 조종 손이 지금 보이고 시작 조건을 채웠는가
    consecutive_frames: int  # 연속으로 보인 프레임 수
    raw_target_m: np.ndarray | None = None  # 이번 메시지의 필터 전 값(상자로 자른 것). 흔들림 확인용
    new_sample: bool = False  # 이번 호출이 목표를 새로 계산했는가
    # 상대 모드에서 클러치가 잡혀 있는 동안의 기준 손 위치(영상 정규화 x, y). 그 밖에는 None. 브라우저가 십자를 그린다.
    reference_hand: tuple[float, float] | None = None


@dataclass
class HandMapper:
    params: MappingParams = field(default_factory=MappingParams)
    start_frames: int = config.START_CONSECUTIVE_FRAMES
    min_cutoff_hz: float = config.ONE_EURO_MIN_CUTOFF_HZ
    beta: float = config.ONE_EURO_BETA
    d_cutoff_hz: float = config.ONE_EURO_D_CUTOFF_HZ
    # 코드상 기본은 "absolute"(2-2 동작 그대로). 설정값 config.MAPPING_MODE는 ControlWorker가 만들 때 넘긴다.
    mode: str = "absolute"
    gain_xy_m_per_width: float = config.RELATIVE_GAIN_XY_M_PER_WIDTH
    gain_z_m_per_scale: float = config.RELATIVE_GAIN_Z_M_PER_SCALE
    image_aspect: float = config.IMAGE_ASPECT
    start_position_m: tuple[float, float, float] = config.RELATIVE_START_POSITION_M
    # 손 크기(hand_scale)로 높이를 정할지. 코드상 기본은 True(2-4 동작 그대로). 설정값 config.Z_FROM_HAND_SCALE(기본 False,
    # 높이는 키보드)은 ControlWorker가 만들 때 넘긴다. False면 손 크기는 높이에 영향이 없다: 상대 모드는 클러치 기준 높이,
    # 절대 모드는 start_position_m의 높이에서 시작해 키보드(nudge_height)로만 바뀐다.
    z_from_hand_scale: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("relative", "absolute"):
            raise ValueError(f"mapping mode must be 'relative' or 'absolute', got {self.mode!r}")
        self._filters = [OneEuroFilter(self.min_cutoff_hz, self.beta, self.d_cutoff_hz) for _ in range(3)]
        self._last_target_m: np.ndarray | None = None
        self._consecutive = 0
        # 키보드 높이: 최종 z = 작업 상자로 자른 (손에서 나온 z(필터 뒤) + 오프셋). 필터를 거치지 않아 지연이 없고
        # "변화량 = 속도 × 시간"이 정확하다. 오프셋은 상자 끝에서 잘린 만큼 되돌려 둬서 죽은 구간이 없다.
        self._base_z: float | None = None  # 손에서 나온 z(오프셋 전)
        self._z_offset = 0.0
        # 상대 모드 상태
        self._start_samples: list[tuple[float, float, float]] = []  # 시작 조건 동안 모은 (x, y, hand_scale)
        self._reference: tuple[float, float, float] | None = None  # 기준 손(3프레임 평균)
        self._previous: tuple[float, float, float] | None = None  # 직전 프레임의 손
        self._accumulated_m: np.ndarray | None = None  # 필터 전, 월드 상자로 자른 누적 목표

    @property
    def last_target_m(self) -> np.ndarray | None:
        return self._last_target_m

    @property
    def reference_hand(self) -> tuple[float, float] | None:
        return None if self._reference is None else (self._reference[0], self._reference[1])

    @property
    def height_limit(self) -> str | None:
        """지금 목표 높이가 작업 상자의 맨 위("top")나 맨 아래("bottom")에 닿아 있는가."""
        if self._last_target_m is None:
            return None
        z = float(self._last_target_m[2])
        lo, hi = self.params.workspace_z_m
        if z >= hi - 1e-9:
            return "top"
        if z <= lo + 1e-9:
            return "bottom"
        return None

    def nudge_height(self, dz_m: float) -> np.ndarray | None:
        """키보드로 목표 높이를 dz_m(m, 위가 +)만큼 바꾼다. 손이 안 보이는 동안에도 마지막 목표의 z만 바꾼다(x, y는 그대로).

        아직 한 번도 조종이 시작되지 않아 목표가 없으면 None(무시). 작업 상자 높이 범위로 자른다.
        돌려주는 값: 새 목표(복사본).
        """
        if self._last_target_m is None or self._base_z is None:
            return None
        self._z_offset += dz_m
        final_z = self._clamp_z(self._base_z + self._z_offset)
        self._z_offset = final_z - self._base_z  # 상자 끝에서 잘린 만큼은 버린다(반대 키가 바로 반응하게)
        target = self._last_target_m.copy()
        target[2] = final_z
        self._last_target_m = target
        return target.copy()

    def _clamp_z(self, z_m: float) -> float:
        lo, hi = self.params.workspace_z_m
        return min(hi, max(lo, z_m))

    def _compose_target(self, filtered: np.ndarray) -> np.ndarray:
        """필터를 거친 값(손에서 나온 x, y, z)에 키보드 높이 오프셋을 더하고 작업 상자로 잘라 마지막 목표로 삼는다."""
        target = clamp_to_workspace(filtered, self.params)
        base_z = float(target[2])
        final_z = self._clamp_z(base_z + self._z_offset)
        self._z_offset = final_z - base_z
        self._base_z = base_z
        target[2] = final_z
        self._last_target_m = target
        return target

    def update(self, hand: dict | None, t_s: float, azimuth_deg: float = 0.0) -> MapResult:
        """hand 메시지(검사를 통과한 dict)와 받은 시각(s), 현재 보기 방위각(도)을 넣는다.

        - 손이 없거나(detected false) hand가 None이면 마지막 목표를 그대로 돌려주고, 연속 프레임 수와 필터를 초기화한다.
        - 손이 start_frames 연속으로 보이기 전에는 목표를 바꾸지 않는다.
        - azimuth_deg는 상대 모드에서만 쓴다(absolute는 월드 방향 고정).
        """
        if hand is None or not hand["detected"]:
            self._consecutive = 0
            self._start_samples = []
            self._reference = None
            self._previous = None
            self._accumulated_m = None
            for f in self._filters:
                f.reset()
            return MapResult(self._last_target_m, False, 0)

        self._consecutive += 1
        wrist = hand["wrist"]
        sample = (float(wrist[0]), float(wrist[1]), float(hand["hand_scale"]))
        if self.mode == "relative" and self._reference is None:
            self._start_samples.append(sample)  # 기준 손을 잡을 때까지만 모은다(그 뒤에는 쌓지 않는다)
        if self._consecutive < max(1, self.start_frames):
            return MapResult(self._last_target_m, False, self._consecutive)

        if self.mode == "relative":
            return self._update_relative(sample, t_s, azimuth_deg)

        raw_m = hand_to_target(wrist[0], wrist[1], hand["hand_scale"], self.params)
        if not self.z_from_hand_scale:
            raw_m[2] = self._clamp_z(self.start_position_m[2])  # 손 크기는 높이에 영향이 없다
        filtered = np.array([f(float(v), t_s) for f, v in zip(self._filters, raw_m)])
        target = self._compose_target(filtered)
        return MapResult(target, True, self._consecutive, raw_target_m=raw_m, new_sample=True)

    def _update_relative(self, sample: tuple[float, float, float], t_s: float, azimuth_deg: float) -> MapResult:
        if self._reference is None:
            # 조종 시작(클러치): 손이 연속으로 보인 프레임들의 평균을 기준 손으로 잡는다.
            n = len(self._start_samples)
            self._reference = (
                sum(s[0] for s in self._start_samples) / n,
                sum(s[1] for s in self._start_samples) / n,
                sum(s[2] for s in self._start_samples) / n,
            )
            self._previous = self._reference
            # 로봇 쪽 기준: 마지막 목표(손을 놓았다 다시 잡은 경우) 또는 처음이면 작업 상자 중심.
            base = self._last_target_m if self._last_target_m is not None else np.asarray(self.start_position_m, float)
            self._accumulated_m = clamp_to_workspace(np.array(base, dtype=float), self.params)
            # 마지막 목표는 키보드 높이를 이미 포함하므로 새 기준에서는 오프셋을 0으로 되돌린다(높이가 두 번 더해지지 않게).
            self._z_offset = 0.0

        assert self._previous is not None and self._accumulated_m is not None
        dx = sample[0] - self._previous[0]
        dy = sample[1] - self._previous[1]
        dscale = sample[2] - self._previous[2]
        step_xy_m = screen_step_to_world(dx, dy, azimuth_deg, self.gain_xy_m_per_width, self.image_aspect)
        step_z_m = self.gain_z_m_per_scale * dscale if self.z_from_hand_scale else 0.0
        step_m = np.array([step_xy_m[0], step_xy_m[1], step_z_m])
        # 걸음마다 월드 상자로 자른다: 상자 밖으로 더 뻗은 만큼은 버려져서 손을 되돌리면 로봇이 바로 되돌아온다.
        self._accumulated_m = clamp_to_workspace(self._accumulated_m + step_m, self.params)
        self._previous = sample

        filtered = np.array([f(float(v), t_s) for f, v in zip(self._filters, self._accumulated_m)])
        target = self._compose_target(filtered)
        return MapResult(
            target,
            True,
            self._consecutive,
            raw_target_m=self._accumulated_m.copy(),
            new_sample=True,
            reference_hand=self.reference_hand,
        )
