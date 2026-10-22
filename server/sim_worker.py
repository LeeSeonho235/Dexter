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
from PIL import Image

log = logging.getLogger(__name__)

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
    ) -> None:
        self._scene_path = scene_path
        self._keyframe = keyframe
        self._camera = camera
        self._width_px = width_px
        self._height_px = height_px
        self._tick_s = 1.0 / frame_fps
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

        while not self._stop.is_set():
            target_sim_s = time.monotonic() - wall_start_s + offset_s
            if target_sim_s - data.time > self._max_catch_up_s:
                offset_s -= target_sim_s - data.time - self._max_catch_up_s
                target_sim_s = data.time + self._max_catch_up_s
            while data.time + timestep_s <= target_sim_s:
                mujoco.mj_step(model, data)

            with self._lock:
                self._sim_time_s = data.time
                should_render = self._viewer_count > 0

            if should_render:
                self._render_and_publish(renderer, data)

            next_tick_s += self._tick_s
            now_s = time.monotonic()
            if next_tick_s < now_s:
                # 렌더링이 틱보다 오래 걸렸으면 밀린 틱을 몰아서 하지 않는다.
                next_tick_s = now_s
            self._stop.wait(next_tick_s - now_s)

    def _render_and_publish(self, renderer: mujoco.Renderer, data: mujoco.MjData) -> None:
        renderer.update_scene(data, camera=self._camera)
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
        if self._on_frame is not None:
            try:
                self._on_frame(seq, jpeg)
            except Exception:
                log.exception("on_frame callback failed")
