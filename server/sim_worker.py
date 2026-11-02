"""시뮬레이션 스레드: MuJoCo 진행 + top 카메라 렌더링 → JPEG.

- MjModel, MjData, Renderer는 모두 이 스레드 안에서 만들고 이 스레드에서만 쓴다
  (EGL 렌더링 상태는 만든 스레드에 묶인다). 웹 서버의 메시지 받기는 asyncio 쪽에서 따로 돈다.
- 시뮬레이션(mj_step)은 실제 시간에 맞춰 항상 진행한다.
- 렌더링과 JPEG 변환은 화면을 보는 브라우저가 1개 이상일 때만 한다(공용 서버 CPU 절약).
- 끝날 때 renderer.close()를 직접 부른다. 안 부르면 종료 시 EGL 에러가 난다.

MUJOCO_GL은 mujoco를 import하기 전에 정해져 있어야 한다(server/app.py, server/tests/conftest.py에서 정함).
"""

import io
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from server.view import CameraView, apply_view, default_view, make_free_camera

log = logging.getLogger(__name__)

# camera 이름이 이 값이면 마우스로 돌릴 수 있는 자유 카메라(MjvCamera)로 그린다. 다른 이름은 scene.xml의 고정 카메라.
FREE_CAMERA = "free"

# 집게 끝 위치를 읽을 site 이름(server/config.py의 IK_SITE와 같다). sim_worker는 config에 의존하지 않게 상수로 둔다.
TIP_SITE = "gripperframe"

# (프레임 번호, JPEG bytes)를 받는 함수. 시뮬레이션 스레드에서 불리므로 빨리 끝나야 한다.
FrameCallback = Callable[[int, bytes], None]


class SimWorker:
    def __init__(
        self,
        scene_path: Path,
        *,
        keyframe: str,
        camera: str,
        width_px: int,
        height_px: int,
        frame_fps: float,
        jpeg_quality: int,
        max_catch_up_s: float,
        shadow: bool = True,
        reflection: bool = True,
        on_frame: FrameCallback | None = None,
        tick_hz: float | None = None,
        view: CameraView | None = None,
    ) -> None:
        """tick_hz: 물리 진행과 ctrl 반영 주기(Hz). 안 주면 frame_fps와 같다(2-1 동작).
        화면 렌더링은 틱 N번에 한 번(N = tick_hz / frame_fps, 반올림), 화면 보는 브라우저가 있을 때만 한다.
        view: camera가 "free"일 때 처음 보기 방향. 안 주면 server/config.py의 기본값."""
        self._scene_path = scene_path
        self._keyframe = keyframe
        self._camera = camera
        self._width_px = width_px
        self._height_px = height_px
        tick_hz = tick_hz or frame_fps
        self._tick_s = 1.0 / tick_hz
        self._render_every = max(1, round(tick_hz / frame_fps))
        self._jpeg_quality = jpeg_quality
        self._max_catch_up_s = max_catch_up_s
        self._shadow = shadow
        self._reflection = reflection
        self._on_frame = on_frame

        self._lock = threading.Lock()
        self._viewer_count = 0
        self._frames_rendered = 0
        self._sim_time_s = 0.0
        self._latest_frame: tuple[int, bytes] | None = None
        self._ctrl: np.ndarray | None = None  # 제어 스레드가 정한 모터 목표(rad). None이면 keyframe의 ctrl 그대로
        self._snapshot: dict | None = None  # 틱마다 갱신하는 읽기 전용 값(집게 끝 위치 등)
        self._view = view or default_view()  # 보기 방향은 하나뿐: 마지막에 set_view로 받은 값이 다음 렌더링부터 적용된다
        self._rendered_view: CameraView | None = None  # 가장 최근 프레임을 그린 보기 방향(고정 카메라면 None)
        self._free_camera: mujoco.MjvCamera | None = None  # 시뮬레이션 스레드에서만 만들고 쓴다

        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    # --- 다른 스레드에서 부르는 함수 ---

    def start(self, timeout_s: float = 10.0) -> None:
        """스레드를 시작하고 장면 로드와 Renderer 생성이 끝날 때까지 기다린다."""
        self._thread = threading.Thread(target=self._run, name="sim-worker", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise TimeoutError("sim worker did not start in time")
        if self._error is not None:
            raise RuntimeError("sim worker failed to start") from self._error

    def stop(self, timeout_s: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                log.warning("sim worker did not stop within %.1f s", timeout_s)

    def set_viewer_count(self, count: int) -> None:
        """화면을 받는 브라우저 수. 0이면 렌더링을 멈춘다."""
        with self._lock:
            self._viewer_count = max(0, count)

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """모터 목표(관절 6개, rad)를 정한다. 다음 틱부터 적용된다. 크기가 다르거나 NaN이 있으면 ValueError."""
        ctrl = np.asarray(ctrl, dtype=float).copy()
        if ctrl.shape != (6,) or not np.all(np.isfinite(ctrl)):
            raise ValueError("ctrl must be 6 finite numbers")
        with self._lock:
            self._ctrl = ctrl

    def set_view(self, view: CameraView) -> None:
        """자유 카메라의 보기 방향을 정한다(다음 렌더링부터). 화면 보는 브라우저가 없으면 렌더링은 여전히 안 한다."""
        with self._lock:
            self._view = view

    @property
    def view(self) -> CameraView:
        """지금 정해진 보기 방향(가장 최근 set_view 값)."""
        with self._lock:
            return self._view

    @property
    def uses_free_camera(self) -> bool:
        """자유 카메라로 그리는가(False면 scene.xml의 고정 카메라). 손 → 로봇 방향 계산이 이 값을 본다."""
        return self._camera == FREE_CAMERA

    @property
    def rendered_view(self) -> CameraView | None:
        """가장 최근에 그린 프레임에 쓴 보기 방향. 아직 안 그렸거나 고정 카메라면 None."""
        with self._lock:
            return self._rendered_view

    @property
    def snapshot(self) -> dict | None:
        """가장 최근 틱의 값: tip_m(집게 끝 x,y,z m), joints_rad(관절 6개), block_m(블록 중심 x,y,z m). 시작 전이면 None."""
        with self._lock:
            return self._snapshot

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def frames_rendered(self) -> int:
        with self._lock:
            return self._frames_rendered

    @property
    def sim_time_s(self) -> float:
        with self._lock:
            return self._sim_time_s

    @property
    def latest_frame(self) -> tuple[int, bytes] | None:
        with self._lock:
            return self._latest_frame

    # --- 시뮬레이션 스레드 ---

    def _run(self) -> None:
        renderer = None
        try:
            model = mujoco.MjModel.from_xml_path(str(self._scene_path))
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key(self._keyframe).id)
            mujoco.mj_forward(model, data)
            renderer = mujoco.Renderer(model, self._height_px, self._width_px)
        except BaseException as err:  # noqa: BLE001 - start()에서 다시 던진다
            self._error = err
            self._ready.set()
            if renderer is not None:
                renderer.close()
            return
        self._ready.set()

        try:
            self._loop(model, data, renderer)
        except Exception:
            log.exception("sim worker crashed")
        finally:
            renderer.close()
            log.info("sim worker stopped, renderer closed")

    def _loop(self, model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> None:
        timestep_s = model.opt.timestep
        # 시뮬레이션 시간 = 실제 경과 시간 + offset_s. 너무 늦으면 offset_s를 당겨 따라잡기를 포기한다.
        wall_start_s = time.monotonic()
        offset_s = data.time
        next_tick_s = wall_start_s
        tip_site_id = model.site(TIP_SITE).id
        block_body_id = model.body("block").id
        tick_count = 0

        while not self._stop.is_set():
            tick_count += 1
            with self._lock:
                ctrl = self._ctrl
            if ctrl is not None:
                data.ctrl[:6] = ctrl
            target_sim_s = time.monotonic() - wall_start_s + offset_s
            if target_sim_s - data.time > self._max_catch_up_s:
                offset_s -= target_sim_s - data.time - self._max_catch_up_s
                target_sim_s = data.time + self._max_catch_up_s
            while data.time + timestep_s <= target_sim_s:
                mujoco.mj_step(model, data)

            # mj_step 뒤의 site, body 위치는 한 스텝 전 값이라 위치만 다시 계산한다(상태는 안 바뀐다).
            mujoco.mj_kinematics(model, data)
            snapshot = {
                "tip_m": data.site_xpos[tip_site_id].copy(),
                "joints_rad": data.qpos[:6].copy(),
                "block_m": data.xpos[block_body_id].copy(),
            }
            with self._lock:
                self._sim_time_s = data.time
                self._snapshot = snapshot
                should_render = self._viewer_count > 0

            if should_render and tick_count % self._render_every == 0:
                self._render_and_publish(renderer, data)

            next_tick_s += self._tick_s
            now_s = time.monotonic()
            if next_tick_s < now_s - self._tick_s:
                # 한 틱보다 더 밀렸으면 밀린 틱을 몰아서 하지 않는다. 한 틱 안쪽으로 밀린 것은 그대로 둬서
                # 렌더링 틱(약 55ms) 뒤에 빠른 틱이 바로 따라오게 한다. 리셋하면 틱 30Hz에서 화면이 15fps에 못 미친다.
                next_tick_s = now_s
            self._stop.wait(max(0.0, next_tick_s - now_s))

    def _render_and_publish(self, renderer: mujoco.Renderer, data: mujoco.MjData) -> None:
        view: CameraView | None = None
        camera: str | mujoco.MjvCamera = self._camera
        if self._camera == FREE_CAMERA:
            with self._lock:
                view = self._view
            if self._free_camera is None:
                self._free_camera = make_free_camera(view)
            else:
                apply_view(self._free_camera, view)
            camera = self._free_camera
        renderer.update_scene(data, camera=camera)
        # 그리기 옵션은 매번 정한다(server/config.py RENDER_SHADOW, RENDER_REFLECTION).
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self._shadow
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self._reflection
        pixels = renderer.render()
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="JPEG", quality=self._jpeg_quality)
        jpeg = buf.getvalue()

        with self._lock:
            self._frames_rendered += 1
            seq = self._frames_rendered
            self._latest_frame = (seq, jpeg)
            self._rendered_view = view
        if self._on_frame is not None:
            try:
                self._on_frame(seq, jpeg)
            except Exception:
                log.exception("on_frame callback failed")
