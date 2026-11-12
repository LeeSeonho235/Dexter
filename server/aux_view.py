"""보조 카메라 렌더 스레드(2-6): 보기 전용 작은 화면. 이 파일만 aux를 안다.

격리 설계 (server/tests/test_aux_isolation.py가 확인한다):
- 자기만의 MjModel / MjData / Renderer를 쓴다. 시뮬레이션 스레드의 MjData나 학습용 StateRenderer와 아무것도 공유하지 않는다.
- 시뮬레이션 스레드는 offer_qpos()로 qpos 13개(rad, m)의 사본만 넘긴다. 락 안에서 복사만 하므로 렌더 시간이 물리 틱을 늘리지 않는다.
- Renderer(EGL)는 이 파일의 스레드 안에서 만들고 쓰고 닫는다.
- 렌더·녹화·AUTO·평가 코드(render.py, recorder.py, auto.py, eval_env.py, correction*.py)는 이 파일을 import하지 않는다.

끄는 장치:
1. AUX_ENABLED가 False면 start()는 스레드도 렌더러도 만들지 않고, offer_qpos()는 아무 일도 안 한다.
2. 보는 연결이 0개(set_viewer_count)이면 렌더하지 않는다.
3. 크기 두 단계(set_size_level), fps는 설정.
4. 렌더가 AUX_MAX_RENDER_MS를 넘으면 fps를 반으로 줄인다. AUTO 중에는(AUX_PAUSE_DURING_AUTO) 멈춘다.
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

from server import aux_view_config as ac
from server import config
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


class AuxViewWorker:
    def __init__(
        self,
        scene_path: Path = config.SCENE_PATH,
        *,
        enabled: bool | None = None,
        fps: float | None = None,
        size_level: str | None = None,
        on_frame: AuxFrameCallback | None = None,
    ) -> None:
        """enabled를 안 주면 aux_view_config.AUX_ENABLED. 꺼져 있으면 아무것도 만들지 않는다."""
        self._scene_path = scene_path
        self.enabled = ac.AUX_ENABLED if enabled is None else enabled
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
        self._renderer: mujoco.Renderer | None = None  # 보조 스레드에서만 만들고 닫는다

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
            width_px, height_px = ac.AUX_SIZE_LEVELS[self._size_level]
            return {
                "enabled": self.enabled,
                "running": self.is_alive,
                "viewers": self._viewer_count,
                "paused": self._paused,
                "size_level": self._size_level,
                "width_px": width_px,
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
            model = mujoco.MjModel.from_xml_path(str(self._scene_path))
            data = mujoco.MjData(model)
            camera = make_free_camera(clamp_view(**ac.AUX_CAMERA_VIEW), ac.AUX_CAMERA_LOOKAT_M)
            level = self._size_level
            renderer = self._make_renderer(model, level)
        except BaseException as err:  # noqa: BLE001 - start()에서 다시 던진다
            self._error = err
            self._record_error(err)
            self._ready.set()
            if renderer is not None:
                renderer.close()
            return
        self._ready.set()

        try:
            level = self._loop(model, data, camera, renderer, level)
        except Exception as err:
            log.exception("aux view worker crashed")
            self._record_error(err)
        finally:
            self._close_renderer_quietly()
            log.info("aux view worker stopped")

    def _record_error(self, err: BaseException) -> None:
        with self._lock:
            self._error_text = f"{type(err).__name__}: {err}"

    def _make_renderer(self, model: mujoco.MjModel, level: str) -> mujoco.Renderer:
        width_px, height_px = ac.AUX_SIZE_LEVELS[level]
        self._renderer = mujoco.Renderer(model, height_px, width_px)
        return self._renderer

    def _close_renderer_quietly(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _loop(self, model, data, camera, renderer, level: str) -> str:
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
            self._render_once(model, data, camera, renderer)
            next_s = next_render_time(next_s, period_s, time.monotonic())
        return level

    def _note_render_ms(self, render_ms: float) -> None:
        """렌더+JPEG 시간(ms)을 기록하고, AUX_MAX_RENDER_MS를 넘었으면 fps를 반으로 줄인다(AUX_MIN_FPS가 바닥)."""
        with self._lock:
            self._last_render_ms = render_ms
            if render_ms > ac.AUX_MAX_RENDER_MS and self._fps > ac.AUX_MIN_FPS:
                self._fps = max(ac.AUX_MIN_FPS, self._fps / 2.0)
                self._slowed = True

    def _render_once(self, model, data, camera, renderer) -> None:
        t0 = time.perf_counter()
        with self._lock:
            data.qpos[:QPOS_DIM] = self._qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        set_render_flags(renderer, config.RENDER_SHADOW, config.RENDER_REFLECTION)
        pixels = renderer.render()
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
