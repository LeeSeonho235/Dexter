"""보조 카메라 렌더 스레드(2-6): 보기 전용 작은 화면 두 개(옆에서, 비스듬히). 이 파일만 aux를 안다.

한 번에 두 시점을 그린다: GL 컨텍스트 하나, 오프스크린 버퍼 하나에 viewport 둘을 가로로 나란히 그리고, 읽기와 JPEG 인코딩은 한 번만 한다
(프로브 측정으로 컨텍스트 둘을 따로 쓰는 것보다 약 15~20% 빠르다). 이미지 한 장 = 시점 수 × 시점 너비 (보통 320x120).
로봇 메시는 시작할 때 한 번 삼각형을 줄인 모델로 그린다(server/aux_mesh.py, AUX_MESH_FRACTION). 줄이지 않으면 렌더가 약 5배 느리다.

격리 설계 (server/tests/test_aux_isolation.py, test_aux_view.py가 확인한다):
- 자기만의 MjModel(줄인 메시) / MjData / GL 컨텍스트를 쓴다. 시뮬레이션 스레드의 MjData나 학습용 StateRenderer와 아무것도 공유하지 않는다.
- 시뮬레이션 스레드는 offer_qpos()로 qpos 13개(rad, m)의 사본만 넘긴다. 락 안에서 복사만 하므로 렌더 시간이 물리 틱을 늘리지 않는다.
- GL 컨텍스트(EGL)는 이 파일의 스레드 안에서 만들고 쓰고 닫는다.
- 렌더·녹화·AUTO·평가 코드(render.py, recorder.py, auto.py, eval_env.py, correction*.py)는 이 파일과 aux_mesh.py를 import하지 않는다.

끄는 장치:
1. AUX_ENABLED가 False면 start()는 스레드도 렌더러도 만들지 않고, offer_qpos()는 아무 일도 안 한다.
2. 보는 연결이 0개(set_viewer_count)이면 렌더하지 않는다.
3. 크기 두 단계(set_size_level), fps는 설정.
4. 렌더가 AUX_MAX_RENDER_MS를 넘으면 fps를 반으로 줄인다. AUTO 중에는 AUX_PAUSE_DURING_AUTO가 True일 때만 멈춘다(기본 False).
5. 스레드가 죽으면 state()["error"]에 이유가 남는다(웹이 2D 보조 그림으로 돌아가는 데 쓴다).
"""

import io
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
from mujoco import gl_context
from PIL import Image

from server import aux_view_config as ac
from server import config
from server.aux_mesh import build_model
from server.render import set_render_flags
from server.view import clamp_view, make_free_camera

log = logging.getLogger(__name__)

# (프레임 번호, JPEG bytes)를 받는 함수. 보조 스레드에서 불리므로 빨리 끝나야 한다.
AuxFrameCallback = Callable[[int, bytes], None]

QPOS_DIM = 13  # 팔 6 + 블록 자유 관절 7


def is_due(now_s: float, next_s: float) -> bool:
    """지금(s)이 다음 렌더 시각(s) 이후인가. 시계를 읽지 않는 순수 함수라 테스트에서 가짜 시각을 넣을 수 있다."""
    return now_s >= next_s


def next_render_time(next_s: float, period_s: float, after_render_s: float) -> float:
    """한 장 그린 뒤 다음 렌더 시각(s). 밀렸으면(렌더가 주기보다 오래 걸렸으면) 따라잡으려 연달아 그리지 않고 지금으로 맞춘다."""
    return max(next_s + period_s, after_render_s)


class _MultiViewRenderer:
    """GL 컨텍스트 하나에 같은 크기 viewport n개를 가로로 나란히 그린다. 만든 스레드에서만 쓰고 close()까지 부른다.

    mujoco.Renderer와 같은 순서로 컨텍스트를 만들지만 viewport 여러 개를 한 버퍼에 그리려고 저수준 함수(mjr_render)를 직접 쓴다.
    scene 속성은 server.render.set_render_flags가 읽는 이름이다.
    """

    def __init__(self, model: mujoco.MjModel, view_w_px: int, view_h_px: int, n_views: int) -> None:
        self._model = model
        self.view_w_px, self.view_h_px, self.n_views = view_w_px, view_h_px, n_views
        self.width_px = view_w_px * n_views
        self._gl = gl_context.GLContext(self.width_px, view_h_px)
        self._ctx = None
        try:
            self._gl.make_current()
            self._ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._ctx)
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self._option = mujoco.MjvOption()
            self._out = np.empty((view_h_px, self.width_px, 3), dtype=np.uint8)
        except BaseException:
            self.close()
            raise

    def render(self, data: mujoco.MjData, cameras: list) -> np.ndarray:
        """cameras[i]가 왼쪽에서 i번째 viewport. 반환: (높이, 너비 × 시점 수, 3) uint8(위가 위인 방향, 새 배열)."""
        self._gl.make_current()
        for i, camera in enumerate(cameras):
            mujoco.mjv_updateScene(self._model, data, self._option, None, camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
            set_render_flags(self, config.RENDER_SHADOW, config.RENDER_REFLECTION)
            mujoco.mjr_render(mujoco.MjrRect(i * self.view_w_px, 0, self.view_w_px, self.view_h_px), self.scene, self._ctx)
        mujoco.mjr_readPixels(self._out, None, mujoco.MjrRect(0, 0, self.width_px, self.view_h_px), self._ctx)
        return np.ascontiguousarray(np.flipud(self._out))

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.free()
            self._ctx = None
        if self._gl is not None:
            self._gl.free()
            self._gl = None


class AuxViewWorker:
    def __init__(
        self,
        scene_path: Path = config.SCENE_PATH,
        *,
        enabled: bool | None = None,
        fps: float | None = None,
        size_level: str | None = None,
        mesh_fraction: float | None = None,
        on_frame: AuxFrameCallback | None = None,
    ) -> None:
        """enabled를 안 주면 aux_view_config.AUX_ENABLED. 꺼져 있으면 아무것도 만들지 않는다.
        mesh_fraction을 안 주면 AUX_MESH_FRACTION(로봇 메시 삼각형을 이 비율로 줄인 모델로 그린다)."""
        self._scene_path = scene_path
        self.enabled = ac.AUX_ENABLED if enabled is None else enabled
        self._mesh_fraction = float(ac.AUX_MESH_FRACTION if mesh_fraction is None else mesh_fraction)
        self._base_fps = float(ac.AUX_FPS if fps is None else fps)
        self._on_frame = on_frame
        level = size_level or ac.AUX_SIZE_DEFAULT
        if level not in ac.AUX_SIZE_LEVELS:
            raise ValueError(f"unknown aux size level {level!r}")

        self._lock = threading.Lock()
        self._size_level = level
        self._viewer_count = 0
        self._qpos = np.zeros(QPOS_DIM)
        self._have_qpos = False
        self._paused = False
        self._fps = self._base_fps  # 지금 쓰는 fps(자동 감속으로 줄어들 수 있다)
        self._slowed = False
        self._frames = 0
        self._latest: tuple[int, bytes] | None = None
        self._last_render_ms: float | None = None
        self._renderer: _MultiViewRenderer | None = None  # 보조 스레드에서만 만들고 닫는다

        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None  # 시작 실패(start()가 다시 던진다)
        self._error_text: str | None = None  # 스레드가 죽은 이유(state()["error"]로 보여 준다). 정상이면 None
        self._thread: threading.Thread | None = None

    # --- 다른 스레드에서 부르는 함수 ---

    def start(self, timeout_s: float = 10.0) -> None:
        """꺼져 있으면 아무것도 안 한다. 켜져 있으면 스레드를 시작하고 모델 로드와 Renderer 생성이 끝날 때까지 기다린다."""
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, name="aux-view", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise TimeoutError("aux view worker did not start in time")
        if self._error is not None:
            raise RuntimeError("aux view worker failed to start") from self._error

    def stop(self, timeout_s: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                log.warning("aux view worker did not stop within %.1f s", timeout_s)

    def offer_qpos(self, qpos: np.ndarray, auto_running: bool = False) -> None:
        """시뮬레이션 스레드가 틱마다 부른다. 락 안에서 13개를 복사만 한다. 꺼져 있으면 바로 돌아온다."""
        if not self.enabled:
            return
        with self._lock:
            self._qpos[:] = qpos[:QPOS_DIM]
            self._have_qpos = True
            self._paused = bool(auto_running and ac.AUX_PAUSE_DURING_AUTO)

    def set_viewer_count(self, count: int) -> None:
        """보조 화면을 켠 연결 수. 0이면 렌더하지 않는다. 0에서 다시 생기면 자동 감속을 풀고 설정 fps로 돌아간다."""
        with self._lock:
            count = max(0, int(count))
            if self._viewer_count == 0 and count > 0:
                self._fps = self._base_fps
                self._slowed = False
            self._viewer_count = count

    def set_size_level(self, level: str) -> None:
        """"small" 또는 "normal". 다음 렌더 전에 Renderer를 다시 만든다. 모르는 값이면 ValueError."""
        if level not in ac.AUX_SIZE_LEVELS:
            raise ValueError(f"unknown aux size level {level!r}")
        with self._lock:
            self._size_level = level

    @property
    def latest_frame(self) -> tuple[int, bytes] | None:
        with self._lock:
            return self._latest

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def state(self) -> dict:
        with self._lock:
            view_w_px, height_px = ac.AUX_SIZE_LEVELS[self._size_level]
            views = len(ac.AUX_CAMERA_VIEWS)
            return {
                "enabled": self.enabled,
                "running": self.is_alive,
                "viewers": self._viewer_count,
                "paused": self._paused,
                "size_level": self._size_level,
                "views": views,
                "view_width_px": view_w_px,  # 시점 하나
                "width_px": view_w_px * views,  # 프레임 한 장(시점을 가로로 이어 붙인 것)
                "height_px": height_px,
                "fps": self._fps,
                "slowed": self._slowed,
                "frames": self._frames,
                "last_render_ms": None if self._last_render_ms is None else round(self._last_render_ms, 2),
                "error": self._error_text,
            }

    # --- 보조 스레드 ---

    def _run(self) -> None:
        renderer = None
        try:
            model = build_model(self._scene_path, self._mesh_fraction)  # 시작할 때 한 번(약 0.6 s). 시뮬레이션의 모델과는 별개다
            data = mujoco.MjData(model)
            cameras = [make_free_camera(clamp_view(**view), ac.AUX_CAMERA_LOOKAT_M) for view in ac.AUX_CAMERA_VIEWS]
            level = self._size_level
            renderer = self._make_renderer(model, level)
        except BaseException as err:  # noqa: BLE001 - start()에서 다시 던진다
            self._error = err
            self._record_error(err)
            self._ready.set()
            self._close_renderer_quietly()
            return
        self._ready.set()

        try:
            level = self._loop(model, data, cameras, renderer, level)
        except Exception as err:
            log.exception("aux view worker crashed")
            self._record_error(err)
        finally:
            self._close_renderer_quietly()
            log.info("aux view worker stopped")

    def _record_error(self, err: BaseException) -> None:
        with self._lock:
            self._error_text = f"{type(err).__name__}: {err}"

    def _make_renderer(self, model: mujoco.MjModel, level: str) -> _MultiViewRenderer:
        view_w_px, view_h_px = ac.AUX_SIZE_LEVELS[level]
        self._renderer = _MultiViewRenderer(model, view_w_px, view_h_px, len(ac.AUX_CAMERA_VIEWS))
        return self._renderer

    def _close_renderer_quietly(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _loop(self, model, data, cameras, renderer, level: str) -> str:
        next_s = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                should_render = self._viewer_count > 0 and self._have_qpos and not self._paused
                wanted_level = self._size_level
                period_s = 1.0 / self._fps
            now_s = time.monotonic()
            if not should_render:
                next_s = now_s  # 다시 켜지면 바로 한 장 그린다
                self._stop.wait(0.05)
                continue
            if not is_due(now_s, next_s):
                self._stop.wait(min(next_s - now_s, 0.05))
                continue
            if wanted_level != level:
                renderer.close()
                self._renderer = None  # 다시 만들다 실패해도 _run의 정리가 닫힌 것을 또 닫지 않게
                renderer = self._make_renderer(model, wanted_level)
                level = wanted_level
            self._render_once(model, data, cameras, renderer)
            next_s = next_render_time(next_s, period_s, time.monotonic())
        return level

    def _note_render_ms(self, render_ms: float) -> None:
        """렌더+JPEG 시간(ms)을 기록하고, AUX_MAX_RENDER_MS를 넘었으면 fps를 반으로 줄인다(AUX_MIN_FPS가 바닥)."""
        with self._lock:
            self._last_render_ms = render_ms
            if render_ms > ac.AUX_MAX_RENDER_MS and self._fps > ac.AUX_MIN_FPS:
                self._fps = max(ac.AUX_MIN_FPS, self._fps / 2.0)
                self._slowed = True

    def _render_once(self, model, data, cameras, renderer) -> None:
        t0 = time.perf_counter()
        with self._lock:
            data.qpos[:QPOS_DIM] = self._qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        pixels = renderer.render(data, cameras)  # 두 시점을 한 버퍼에 그리고 한 번에 읽는다
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="JPEG", quality=ac.AUX_JPEG_QUALITY)
        jpeg = buf.getvalue()
        render_ms = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            self._frames += 1
            seq = self._frames
            self._latest = (seq, jpeg)
        self._note_render_ms(render_ms)
        if self._on_frame is not None:
            try:
                self._on_frame(seq, jpeg)
            except Exception:
                log.exception("aux on_frame callback failed")
