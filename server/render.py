"""상태(qpos) → 이미지 렌더링. 3-2의 카메라 확인과 3-3의 데이터셋 이미지 생성이 같이 쓴다.

- 카메라는 server/config.py의 RECORD_CAMERA_*(마우스 보기 VIEW_*와 별개). 해상도는 RECORD_IMAGE_*_PX.
- 그림자·반사 설정은 실시간 화면과 같은 config(RENDER_SHADOW, RENDER_REFLECTION)를 쓴다. set_render_flags()를 SimWorker도 부른다.
- Renderer(EGL)는 만든 스레드에서만 쓰고, 끝나면 close()를 직접 불러야 한다(with 문을 쓰면 자동).
  MUJOCO_GL은 mujoco를 import하기 전에 정해져 있어야 한다(MUJOCO_GL=egl).
"""

from pathlib import Path

import mujoco
import numpy as np

from server import config
from server.view import CameraView, make_free_camera


def set_render_flags(renderer: mujoco.Renderer, shadow: bool, reflection: bool) -> None:
    """그림자·반사 켜기/끄기. update_scene 뒤에 매번 불러야 적용된다."""
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflection


def record_view() -> CameraView:
    return CameraView(
        config.RECORD_CAMERA_AZIMUTH_DEG, config.RECORD_CAMERA_ELEVATION_DEG, config.RECORD_CAMERA_DISTANCE_M
    )


def make_record_camera() -> mujoco.MjvCamera:
    """학습·재생용 카메라(고정). 마우스로 돌리는 화면과 상관없다."""
    return make_free_camera(record_view(), config.RECORD_CAMERA_LOOKAT_M)


class StateRenderer:
    """전체 qpos(13개)로 장면을 복원해 그린다. 팔과 블록 위치만 필요하다(속도는 그림에 영향이 없다)."""

    def __init__(
        self,
        scene_path: Path = config.SCENE_PATH,
        width_px: int = config.RECORD_IMAGE_WIDTH_PX,
        height_px: int = config.RECORD_IMAGE_HEIGHT_PX,
        shadow: bool = config.RENDER_SHADOW,
        reflection: bool = config.RENDER_REFLECTION,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self._shadow = shadow
        self._reflection = reflection
        self._camera = make_record_camera()
        self._renderer = mujoco.Renderer(self.model, height_px, width_px)

    def _set_state(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.model.nq,):
            raise ValueError(f"qpos must have {self.model.nq} values, got shape {qpos.shape}")
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def render(self, qpos: np.ndarray) -> np.ndarray:
        """(높이, 너비, 3) uint8 RGB 이미지."""
        self._set_state(qpos)
        self._renderer.update_scene(self.data, camera=self._camera)
        set_render_flags(self._renderer, self._shadow, self._reflection)
        return self._renderer.render().copy()

    def render_segmentation(self, qpos: np.ndarray) -> np.ndarray:
        """(높이, 너비, 2) int 이미지: [물체 번호, 물체 종류]. 블록·목표가 몇 픽셀인지 세는 확인용."""
        self._set_state(qpos)
        self._renderer.enable_segmentation_rendering()
        try:
            self._renderer.update_scene(self.data, camera=self._camera)
            return self._renderer.render().copy()
        finally:
            self._renderer.disable_segmentation_rendering()

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "StateRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
