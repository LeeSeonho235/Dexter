"""자유 카메라 보기 방향(2-3): 값 범위 자르기와 MuJoCo 카메라 만들기.

보기 방향은 서버에 하나뿐이다(시뮬레이션이 하나라서). 여러 브라우저가 붙어 있으면 마지막에 보낸 값이 적용된다.
단위: 각도는 도(deg), 거리는 m. (좌표 규칙은 server/config.py의 "자유 카메라 보기 방향" 주석)
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from server import config

if TYPE_CHECKING:  # protocol.py가 이 파일의 clamp_view를 쓰므로, mujoco는 카메라를 만들 때만 불러온다.
    import mujoco


@dataclass(frozen=True)
class CameraView:
    azimuth_deg: float
    elevation_deg: float
    distance_m: float

    def as_dict(self) -> dict[str, float]:
        return {
            "azimuth_deg": round(self.azimuth_deg, 2),
            "elevation_deg": round(self.elevation_deg, 2),
            "distance_m": round(self.distance_m, 3),
        }


def wrap_azimuth_deg(azimuth_deg: float) -> float:
    """방위각은 한 바퀴를 돌아 같은 방향이 되므로 자르지 않고 -180 이상 180 미만으로 돌려 쓴다."""
    return (azimuth_deg + 180.0) % 360.0 - 180.0


def clamp_view(azimuth_deg: float, elevation_deg: float, distance_m: float) -> CameraView:
    """유한한 숫자 세 개를 허용 범위 안으로 맞춘다. NaN이나 무한대는 호출하는 쪽(protocol)이 이미 걸러야 한다."""
    for value in (azimuth_deg, elevation_deg, distance_m):
        if not math.isfinite(value):
            raise ValueError("view values must be finite")
    lo_el, hi_el = config.VIEW_ELEVATION_RANGE_DEG
    lo_d, hi_d = config.VIEW_DISTANCE_RANGE_M
    return CameraView(
        wrap_azimuth_deg(azimuth_deg),
        min(hi_el, max(lo_el, elevation_deg)),
        min(hi_d, max(lo_d, distance_m)),
    )


def default_view() -> CameraView:
    return clamp_view(**config.VIEW_DEFAULT)


def preset_view(name: str) -> CameraView:
    return clamp_view(**config.VIEW_PRESETS[name])


def make_free_camera(
    view: CameraView, lookat_m: tuple[float, float, float] = config.VIEW_LOOKAT_M
) -> "mujoco.MjvCamera":
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    apply_view(camera, view, lookat_m)
    return camera


def apply_view(
    camera: "mujoco.MjvCamera", view: CameraView, lookat_m: tuple[float, float, float] = config.VIEW_LOOKAT_M
) -> None:
    camera.azimuth = view.azimuth_deg
    camera.elevation = view.elevation_deg
    camera.distance = view.distance_m
    camera.lookat[:] = lookat_m
