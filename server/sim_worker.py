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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from server import auto_config, config
from server.owner import ControlOwner
from server.render import StateRenderer, set_render_flags
from server.view import CameraView, apply_view, default_view, make_free_camera

log = logging.getLogger(__name__)

# camera 이름이 이 값이면 마우스로 돌릴 수 있는 자유 카메라(MjvCamera)로 그린다. 다른 이름은 scene.xml의 고정 카메라.
FREE_CAMERA = "free"

# 집게 끝 위치를 읽을 site 이름(server/config.py의 IK_SITE와 같다). sim_worker는 config에 의존하지 않게 상수로 둔다.
TIP_SITE = "gripperframe"

# (프레임 번호, JPEG bytes)를 받는 함수. 시뮬레이션 스레드에서 불리므로 빨리 끝나야 한다.
FrameCallback = Callable[[int, bytes], None]


@dataclass
class PlaceResult:
    """place_block 결과. block_m은 놓은 뒤 블록 중심(m)."""

    ok: bool
    reason: str = ""
    block_m: tuple[float, float, float] | None = None


@dataclass(eq=False)
class _PlaceCommand:
    x_m: float
    y_m: float
    done: threading.Event = field(default_factory=threading.Event)
    result: PlaceResult = field(default_factory=lambda: PlaceResult(False, "not executed"))


class TickStats:
    """틱 시작 사이의 벽시계 간격(ms): 시작 이후 전체와 최근 N개. 스레드 안전.

    공칭 주기는 1/SIM_TICK_HZ(33.3 ms)다. 화면 렌더링이 있는 틱은 약 55 ms 걸려서 그 뒤 틱이 짧게 따라오는 것은 정상이다.
    """

    def __init__(self, window: int = config.TICK_STATS_WINDOW, over_ms: tuple[float, ...] = config.TICK_OVER_MS) -> None:
        self._lock = threading.Lock()
        self._over_ms = over_ms
        self._window: deque[float] = deque(maxlen=window)
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._window.clear()
            self._n = 0
            self._sum_ms = 0.0
            self._max_ms = 0.0
            self._over = [0] * len(self._over_ms)

    def add(self, interval_s: float) -> None:
        ms = interval_s * 1000.0
        with self._lock:
            self._window.append(ms)
            self._n += 1
            self._sum_ms += ms
            self._max_ms = max(self._max_ms, ms)
            for i, limit in enumerate(self._over_ms):
                if ms > limit:
                    self._over[i] += 1

    def summary(self) -> dict:
        """전체(시작 또는 reset 이후)와 최근 구간의 평균, 최대, 기준 초과 횟수."""
        with self._lock:
            window = list(self._window)
            return {
                "n": self._n,
                "mean_ms": round(self._sum_ms / self._n, 2) if self._n else None,
                "max_ms": round(self._max_ms, 2) if self._n else None,
                "over": {f"{limit:g}": count for limit, count in zip(self._over_ms, self._over, strict=True)},
                "recent_mean_ms": round(sum(window) / len(window), 2) if window else None,
                "recent_max_ms": round(max(window), 2) if window else None,
            }


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
        self._recorder = None  # 녹화기(server/recorder.py). 있으면 물리 스텝 직전마다 on_step을 부른다
        self._aux = None  # 보조 카메라(2-6, server/aux_view.py). 있으면 틱마다 qpos 사본만 넘긴다(읽기 전용, 이 파일은 aux_view를 import하지 않는다)
        self._place_cmd: _PlaceCommand | None = None  # 다른 스레드가 맡긴 "블록 놓기" 명령 한 개
        # 제어 주체(server/owner.py). AUTO이면 ControlWorker가 정한 ctrl을 적용하지 않는다(ctrl을 쓰는 주체는 하나).
        self.owner = ControlOwner()
        self._auto_cmd = None  # 다른 스레드가 맡긴 AUTO 에피소드(server/auto.py AutoEpisode) 한 개. 시뮬레이션 스레드가 집어간다
        self._auto = None  # 지금 실행 중인 AUTO 에피소드(시뮬레이션 스레드에서만 만지고 바꾼다)
        self._state_renderer: StateRenderer | None = None  # 학습과 같은 카메라로 그리는 렌더러. 첫 AUTO 때 시뮬레이션 스레드에서 만든다
        self._timestep_s: float | None = None  # 모델 로드 뒤에 정해진다
        self.tick_stats = TickStats()

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

    def attach_recorder(self, recorder) -> None:
        """녹화기를 연결한다(None이면 떼어낸다). 시뮬레이션 스레드가 물리 스텝 직전마다 recorder.on_step(시각, qpos, ctrl)을 부른다."""
        self._recorder = recorder

    def attach_aux(self, aux) -> None:
        """보조 카메라(offer_qpos(qpos, auto_running)를 가진 객체)를 연결한다(None이면 뗀다).

        시뮬레이션 스레드가 매 틱 물리 진행이 끝난 뒤 qpos 사본 하나만 넘긴다. aux는 받은 값을 읽기만 하고 시뮬레이션 상태에 쓸 길이 없다.
        offer_qpos가 예외를 던지면 aux를 떼고 시뮬레이션은 계속 돈다(보기 전용 기능이 시뮬레이션을 죽이면 안 된다)."""
        self._aux = aux

    def start_auto(self, episode) -> str | None:
        """AUTO 에피소드를 시뮬레이션 스레드에 맡긴다(다음 틱에 시작). 받아들였으면 None, 아니면 이유.
        호출하는 쪽이 먼저 owner를 AUTO로 잡아야 한다(server/auto.py AutoRunner)."""
        with self._lock:
            if self._auto_cmd is not None or self._auto is not None:
                return "AUTO가 이미 실행 중입니다"
            if not self.is_alive:
                return "시뮬레이션 스레드가 꺼져 있습니다"
            self._auto_cmd = episode
        return None

    @property
    def timestep_s(self) -> float | None:
        """모델의 물리 스텝(s). start()가 끝난 뒤에 값이 있다."""
        return self._timestep_s

    def place_block(self, x_m: float, y_m: float, timeout_s: float = 1.0) -> PlaceResult:
        """블록을 (x, y)(m) 바닥 위에 회전 없이 놓고 속도를 0으로 만든다. 실제 일은 시뮬레이션 스레드가 다음 틱에 하고 여기서는 기다린다.

        놓은 자리에서 블록이 바닥 말고 다른 것(팔, 집게)과 닿으면 놓지 않고(원래 자리로 되돌림) 실패로 돌려준다.
        """
        cmd = _PlaceCommand(float(x_m), float(y_m))
        with self._lock:
            if self._place_cmd is not None:
                return PlaceResult(False, "another place_block is pending")
            self._place_cmd = cmd
        if not cmd.done.wait(timeout_s):
            with self._lock:
                if self._place_cmd is cmd:  # 아직 시뮬레이션 스레드가 집어가지 않았다: 거둔다
                    self._place_cmd = None
                    return PlaceResult(False, "sim thread did not run the command in time")
            cmd.done.wait(1.0)  # 막 집어간 경우: 끝나기를 조금 더 기다린다
            if not cmd.done.is_set():
                return PlaceResult(False, "place_block timed out")
        return cmd.result

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
            self._timestep_s = float(model.opt.timestep)
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
            if self._state_renderer is not None:
                self._state_renderer.close()
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
        last_tick_start_s: float | None = None

        while not self._stop.is_set():
            tick_start_s = time.perf_counter()
            if last_tick_start_s is not None:
                self.tick_stats.add(tick_start_s - last_tick_start_s)
            last_tick_start_s = tick_start_s
            tick_count += 1
            with self._lock:
                ctrl = self._ctrl
                place_cmd, self._place_cmd = self._place_cmd, None
                auto_cmd, self._auto_cmd = self._auto_cmd, None
            if place_cmd is not None:
                self._do_place_block(model, data, place_cmd)
            if auto_cmd is not None:
                self._begin_auto(model, data, auto_cmd)
                # 시작(렌더러 만들기, exact_start의 시간 되돌리기)에 걸린 시간과 시각 변화는 따라잡지 않는다: 시뮬레이션 시각 기준을 다시 잡는다.
                offset_s = data.time - (time.monotonic() - wall_start_s)
            # ControlWorker가 정한 ctrl은 owner가 IDLE이거나 HUMAN일 때만 적용한다. AUTO(정책이 씀)와 HUMAN_WAIT(직전 ctrl 유지)에서는 적용하지 않는다.
            if ctrl is not None and self.owner.control_writes_ctrl:
                data.ctrl[:6] = ctrl
            target_sim_s = time.monotonic() - wall_start_s + offset_s
            # AUTO 중에는 따라잡기 상한을 줄인다: 렌더링 틱 정도의 지연만 따라잡고, 몰아서 돌지 않는다.
            catch_up_s = auto_config.AUTO_CATCH_UP_S if self._auto is not None else self._max_catch_up_s
            if target_sim_s - data.time > catch_up_s:
                offset_s -= target_sim_s - data.time - catch_up_s
                target_sim_s = data.time + catch_up_s
            recorder = self._recorder
            while data.time + timestep_s <= target_sim_s:
                auto = self._auto
                if auto is not None:
                    blocked = auto.before_step(model, data)  # 샘플 시각이면 관측 → (필요하면 정책을 기다림) → ctrl
                    if not auto.active:
                        self._auto = None
                    if blocked:
                        # 정책을 기다리는 동안 물리 시간은 멈춰 있었다. 벽시계가 흐른 만큼을 따라잡지 않도록 시각 기준을 다시 잡고 이 틱의 물리를 끝낸다.
                        offset_s = data.time - (time.monotonic() - wall_start_s)
                        break
                if recorder is not None:
                    # 물리 스텝 직전: qpos와 지금 적용 중인 ctrl을 한 번에 복사한다(server/recorder.py의 짝 맞추기 규칙).
                    recorder.on_step(data.time, data.qpos, data.ctrl)
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

            # 보조 카메라(2-6): 물리가 끝난 뒤 qpos 사본(13개)만 넘긴다. 사본이라 aux가 값을 바꿔도 시뮬레이션은 영향이 없다.
            aux = self._aux
            if aux is not None:
                try:
                    aux.offer_qpos(data.qpos.copy(), auto_running=self._auto is not None)
                except Exception:
                    log.exception("aux offer_qpos failed; detaching the aux view (simulation continues)")
                    self._aux = None

            if should_render and tick_count % self._render_every == 0:
                self._render_and_publish(renderer, data)

            next_tick_s += self._tick_s
            now_s = time.monotonic()
            if next_tick_s < now_s - self._tick_s:
                # 한 틱보다 더 밀렸으면 밀린 틱을 몰아서 하지 않는다. 한 틱 안쪽으로 밀린 것은 그대로 둬서
                # 렌더링 틱(약 55ms) 뒤에 빠른 틱이 바로 따라오게 한다. 리셋하면 틱 30Hz에서 화면이 15fps에 못 미친다.
                next_tick_s = now_s
            self._stop.wait(max(0.0, next_tick_s - now_s))

    def _begin_auto(self, model: mujoco.MjModel, data: mujoco.MjData, episode) -> None:
        """시뮬레이션 스레드에서만 부른다. 학습용 렌더러(EGL)를 이 스레드에서 처음 만들고 에피소드를 시작한다. 실패하면 에피소드를 오류로 끝낸다."""
        try:
            if self._state_renderer is None:
                renderer = StateRenderer()
                # 렌더러 인스턴스의 첫 렌더링만 몇 픽셀이 다르다(docs/07-lerobot-notes.md): 평가(EvalEnv)처럼 버리는 렌더링을 한 번 한다.
                renderer.render(model.key(config.START_KEYFRAME).qpos)
                self._state_renderer = renderer
            episode.begin(model, data, self._state_renderer)
            self.tick_stats.reset()  # AUTO 중 틱 간격만 재려고(docs/12-auto.md)
            self._auto = episode
        except Exception as err:  # noqa: BLE001 - 시작하지 못하면 AUTO를 오류로 끝내 owner를 돌려준다
            log.exception("auto begin failed")
            episode.active = True  # _finish가 한 번만 동작하도록
            episode._finish(data, "error", f"{type(err).__name__}: {err}")

    def _do_place_block(self, model: mujoco.MjModel, data: mujoco.MjData, cmd: _PlaceCommand) -> None:
        """시뮬레이션 스레드에서만 부른다. 블록 qpos를 옮기고 qvel을 0으로 만든 뒤, 바닥 외의 접촉이 생기면 되돌린다."""
        try:
            joint = model.joint("block_free")
            q_adr, v_adr = int(joint.qposadr[0]), int(joint.dofadr[0])
            block_geom, floor_geom = model.geom("block").id, model.geom("floor").id
            half_m = float(model.geom("block").size[2])
            old_q = data.qpos[q_adr : q_adr + 7].copy()
            old_v = data.qvel[v_adr : v_adr + 6].copy()
            data.qpos[q_adr : q_adr + 7] = [cmd.x_m, cmd.y_m, half_m, 1.0, 0.0, 0.0, 0.0]
            data.qvel[v_adr : v_adr + 6] = 0.0
            mujoco.mj_forward(model, data)
            blocked = False
            for i in range(data.ncon):
                pair = {data.contact[i].geom1, data.contact[i].geom2}
                if block_geom in pair and floor_geom not in pair:
                    blocked = True
                    break
            if blocked:
                data.qpos[q_adr : q_adr + 7] = old_q
                data.qvel[v_adr : v_adr + 6] = old_v
                mujoco.mj_forward(model, data)
                cmd.result = PlaceResult(False, "block would touch the robot at that position")
            else:
                cmd.result = PlaceResult(True, "", (cmd.x_m, cmd.y_m, half_m))
        except Exception as err:  # noqa: BLE001 - 대기 중인 호출이 영원히 기다리지 않게 항상 알린다
            log.exception("place_block failed")
            cmd.result = PlaceResult(False, f"{type(err).__name__}: {err}")
        finally:
            cmd.done.set()

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
        # 그리기 옵션은 매번 정한다(server/config.py RENDER_SHADOW, RENDER_REFLECTION). 학습 이미지(server/render.py)와 같은 함수다.
        set_render_flags(renderer, self._shadow, self._reflection)
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
