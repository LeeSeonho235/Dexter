"""AUTO(4-1): 학습된 정책이 혼자 로봇을 움직인다.

- AutoEpisode: 한 에피소드의 루프. 시뮬레이션 스레드가 물리 스텝 직전마다 before_step()을 부른다(녹화기 on_step과 같은 자리).
  샘플은 3-2의 시간 격자(server/grid.py)에서 찍고, 평가(server/eval_env.py rollout)와 같은 순서로 돈다:
    샘플 k: 관측(state = qpos[:6] float32, 이미지 = StateRenderer) → 성공 판정(is_success + SuccessHold, 시각은 시작 기준) →
            큐가 비었으면 정책에 청크를 요청(그동안 물리 시간을 멈춘다) → 큐에서 행동 하나 → data.ctrl[:6]에 그대로(변화량 제한 없음)
  종료: 성공 / 시간 초과(AUTO_MAX_S) / 사용자 중지 / 오류(정책 프로세스 문제). 끝나면 on_finish(AutoResult)를 시뮬레이션 스레드에서 부른다.
- AutoRunner: 브라우저 명령(start/stop/shutdown_policy)을 받아들이거나 거절하고, 정책 프로세스를 띄우고(스레드), 에피소드를 시뮬레이션 스레드에 맡기고,
  끝난 결과와 세션 성공/시도 횟수를 세고, 화면용 상태를 만든다. 제어 주체(server/owner.py)를 AUTO로 잡아 손·키보드가 ctrl을 못 쓰게 한다.

단위: 길이 m, 각도 rad, 시간 s(시뮬레이션 시간은 sim, 벽시계는 wall로 표시).
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mujoco
import numpy as np

from server import auto_config, config
from server import eval_config as ec
from server.grid import grid_step_index, steps_done
from server.owner import ControlOwner
from server.policy_ipc import PolicyClient, PolicyError
from server.seeds import DEMO, block_qpos_for_seed, seed_kind
from server.success import QPOS_BLOCK_XYZ, QPOS_GRIPPER, SuccessHold, is_success

log = logging.getLogger(__name__)

ARM_DOF = 6


class AutoStatus(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    RUNNING = "running"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    STOPPED = "stopped"
    ERROR = "error"


TERMINAL = (AutoStatus.SUCCESS, AutoStatus.TIMEOUT, AutoStatus.STOPPED, AutoStatus.ERROR)


@dataclass
class AutoResult:
    outcome: str  # success | timeout | stopped | error
    seed: int
    time_s: float  # 끝난 시각(시뮬레이션 시간, 시작 기준)
    time_to_success_s: float | None
    n_samples: int  # 판정한 샘플 수(평가의 n_samples와 같은 규칙: 마지막으로 판정한 샘플 번호 + 1)
    policy_calls: int
    final_qpos: np.ndarray  # (13,)
    final_ctrl: np.ndarray  # (6,)
    error: str = ""
    requests: list[dict[str, float]] = field(default_factory=list)  # 요청마다 render_ms, request_ms, infer_ms, stall_ms
    tick_stats: dict | None = None
    qpos_trace: np.ndarray | None = None  # (n_samples, 13) 샘플마다 qpos(record_trace일 때. 평가 비교용)
    ctrl_trace: np.ndarray | None = None  # (n_samples - 1, 6) 샘플마다 낸 ctrl

    @property
    def final_block_m(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.final_qpos[QPOS_BLOCK_XYZ])


def apply_eval_start(model: mujoco.MjModel, data: mujoco.MjData, seed: int) -> None:
    """평가(server/eval_env.py EvalEnv.reset)와 같은 시작 상태: 데이터 초기화, home 키프레임, park ctrl, 블록을 시드 위치에, EVAL_SETTLE_S 동안 가라앉히기.
    검증용(exact_start). 시간이 0으로 돌아가고 팔이 순간 이동하므로 화면 버튼에는 연결하지 않는다."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_resetDataKeyframe(model, data, model.key(config.START_KEYFRAME).id)
    park = np.array(model.key(config.START_KEYFRAME).ctrl[:ARM_DOF], dtype=float)
    park[QPOS_GRIPPER] = config.GRIPPER_OPEN_RAD
    data.ctrl[:ARM_DOF] = park
    data.qpos[ARM_DOF:13] = block_qpos_for_seed(seed)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(round(ec.EVAL_SETTLE_S / float(model.opt.timestep))):
        mujoco.mj_step(model, data)


class AutoEpisode:
    """한 에피소드. begin()과 before_step()은 시뮬레이션 스레드에서만 부른다. request_stop()과 progress()는 아무 스레드에서나."""

    def __init__(
        self,
        seed: int,
        policy: Any,
        *,
        max_s: float = auto_config.AUTO_MAX_S,
        fps: float = ec.EVAL_FPS,
        exact_start: bool = False,
        record_trace: bool = False,
        on_finish: Callable[[AutoResult], None] | None = None,
    ) -> None:
        """policy: request(state float32[6], image uint8 HxWx3, reset: bool) -> ChunkResult 를 가진 객체(PolicyClient 또는 가짜)."""
        self.seed = int(seed)
        self._policy = policy
        self._fps = fps
        self._n_max = int(round(max_s * fps))
        self._max_s = max_s
        self._exact_start = exact_start
        self._record_trace = record_trace
        self._on_finish = on_finish
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._progress: dict[str, Any] = {"sample": 0, "elapsed_s": 0.0, "chunk_pos": None, "chunk_len": None, "last_latency_ms": None}
        self.active = False
        self.result: AutoResult | None = None
        self.done = threading.Event()
        self._renderer = None
        self._timestep_s = 0.005
        self._t0 = 0.0
        self._k = 0
        self._next_step = 0
        self._hold = SuccessHold()
        self._time_to_success: float | None = None
        self._queue: deque[np.ndarray] = deque()
        self._chunk_len = 0
        self._calls = 0
        self._requests: list[dict[str, float]] = []
        self._trace: list[np.ndarray] = []
        self._ctrls: list[np.ndarray] = []
        self._last_qpos = np.zeros(13)
        self._last_sim_s = 0.0

    # --- 다른 스레드 ---

    def request_stop(self) -> None:
        self._stop.set()

    def progress(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._progress)

    # --- 시뮬레이션 스레드 ---

    def begin(self, model: mujoco.MjModel, data: mujoco.MjData, renderer) -> None:
        self._renderer = renderer
        self._timestep_s = float(model.opt.timestep)
        if self._exact_start:
            apply_eval_start(model, data, self.seed)
        self._t0 = float(data.time)
        self._last_qpos = np.array(data.qpos[:13], dtype=np.float64)
        self._k = 0
        self._next_step = grid_step_index(0, self._fps, self._timestep_s)
        self._hold = SuccessHold()
        self.active = True

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """물리 스텝 직전에 부른다. 이번 호출에서 정책을 기다렸다면(그동안 시뮬레이션 시간은 멈춰 있다) True."""
        if not self.active:
            return False
        if self._stop.is_set():
            self._finish(data, "stopped")
            return False
        if steps_done(float(data.time), self._t0, self._timestep_s) < self._next_step:
            return False
        try:
            return self._sample(data)
        except PolicyError as err:
            self._finish(data, "error", str(err))
        except Exception as err:  # noqa: BLE001 - 에피소드 오류는 AUTO 오류로 멈추고 시뮬레이션 스레드는 살린다
            log.exception("auto sample failed")
            self._finish(data, "error", f"{type(err).__name__}: {err}")
        return False

    def _sample(self, data: mujoco.MjData) -> bool:
        k = self._k
        qpos = np.array(data.qpos[:13], dtype=np.float64)  # 복사
        sim_s = float(data.time) - self._t0
        self._last_qpos, self._last_sim_s = qpos, sim_s
        if self._record_trace:
            self._trace.append(qpos)
        if self._hold.update(sim_s, is_success(qpos[QPOS_BLOCK_XYZ], float(qpos[QPOS_GRIPPER]))) and self._time_to_success is None:
            self._time_to_success = sim_s
            self._finish(data, "success")
            return False
        if k == self._n_max:
            self._finish(data, "timeout")
            return False

        blocked = False
        latency_ms = None
        if not self._queue:
            blocked = True
            wall0 = time.perf_counter()
            image = self._renderer.render(qpos)
            render_ms = (time.perf_counter() - wall0) * 1000.0
            result = self._policy.request(qpos[:ARM_DOF].astype(np.float32), image, reset=(self._calls == 0))
            stall_ms = (time.perf_counter() - wall0) * 1000.0
            self._calls += 1
            actions = np.asarray(result.actions, dtype=np.float64)
            self._queue.extend(actions)
            self._chunk_len = len(actions)
            latency_ms = render_ms + result.total_ms
            self._requests.append(
                {"render_ms": render_ms, "request_ms": result.total_ms, "infer_ms": result.infer_ms, "stall_ms": stall_ms}
            )
        action = self._queue.popleft()
        if action.shape[0] < ARM_DOF or not np.all(np.isfinite(action[:ARM_DOF])):
            raise PolicyError("정책이 잘못된 행동을 냈습니다")
        data.ctrl[:ARM_DOF] = action[:ARM_DOF]
        if self._record_trace:
            self._ctrls.append(action[:ARM_DOF].copy())
        self._k = k + 1
        self._next_step = grid_step_index(self._k, self._fps, self._timestep_s)
        with self._lock:
            self._progress.update(
                sample=k, elapsed_s=sim_s, chunk_pos=self._chunk_len - len(self._queue) - 1, chunk_len=self._chunk_len
            )
            if latency_ms is not None:
                self._progress["last_latency_ms"] = latency_ms
        return blocked

    def _finish(self, data: mujoco.MjData, outcome: str, error: str = "") -> None:
        if not self.active:
            return
        self.active = False
        qpos = self._last_qpos
        result = AutoResult(
            outcome=outcome,
            seed=self.seed,
            time_s=self._last_sim_s,
            time_to_success_s=self._time_to_success,
            n_samples=self._k + 1 if outcome in ("success", "timeout") else self._k,
            policy_calls=self._calls,
            final_qpos=qpos.copy(),
            final_ctrl=np.array(data.ctrl[:ARM_DOF], dtype=np.float64),
            error=error,
            requests=list(self._requests),
            qpos_trace=np.array(self._trace) if self._record_trace else None,
            ctrl_trace=np.array(self._ctrls) if self._record_trace else None,
        )
        self.result = result
        try:
            if self._on_finish is not None:
                self._on_finish(result)
        except Exception:  # noqa: BLE001
            log.exception("auto on_finish failed")
        finally:
            self.done.set()


class AutoRunner:
    """브라우저 명령 처리 + 상태. start/stop/shutdown_policy는 asyncio 스레드에서 불리고, 느린 일은 스레드에서 한다."""

    def __init__(
        self,
        worker,
        owner: ControlOwner,
        policy: PolicyClient,
        recording,
        *,
        control=None,
        arm_check: Callable[[], str | None] | None = None,
        max_s: float = auto_config.AUTO_MAX_S,
        load_estimate_s: float = auto_config.POLICY_LOAD_ESTIMATE_S,
    ) -> None:
        self._worker = worker
        self._owner = owner
        self._policy = policy
        self._recording = recording
        self._control = control
        self._arm_check = arm_check
        self._max_s = max_s
        self._load_estimate_s = load_estimate_s

        self._lock = threading.RLock()
        self._status = AutoStatus.IDLE
        self._cancel = threading.Event()
        self._episode: AutoEpisode | None = None
        self._seed: int | None = None
        self._error = ""
        self._loading_since_s: float | None = None
        self._result: dict[str, Any] | None = None
        self._success = 0
        self._attempts = 0
        self._launcher_thread: threading.Thread | None = None
        self._last_event: dict[str, Any] | None = None
        self._event_id = 0

    # --- 브라우저 명령. 돌려주는 값: None이면 받아들임, 문자열이면 거절 이유 ---

    def handle(self, action: str) -> str | None:
        if action == "start":
            reason = self.handle_start()
        elif action == "stop":
            reason = self.handle_stop()
        elif action == "shutdown_policy":
            reason = self.handle_shutdown_policy()
        else:
            reason = f"unknown action: {action}"
        if reason is not None:
            self._event("warn", f"거절됨: {reason}")  # 화면의 AUTO 상자에 이유가 보이게(녹화 상자와 같은 방식)
        return reason

    def handle_start(self) -> str | None:
        with self._lock:
            if self._status in (AutoStatus.LOADING, AutoStatus.RUNNING):
                return "AUTO가 이미 실행 중입니다"
            reason = self._recording.auto_precheck()
            if reason is None and self._arm_check is not None:
                reason = self._arm_check()
            if reason is not None:
                return reason
            if not self._owner.try_acquire_auto():
                return "지금은 다른 주체가 로봇을 제어 중입니다"
            self._status = AutoStatus.LOADING
            self._cancel.clear()
            self._error = ""
            self._result = None
            self._episode = None
            self._seed = None
            self._loading_since_s = time.monotonic()
        self._event("info", "AUTO 시작: 정책 프로세스를 준비합니다")
        self._launcher_thread = threading.Thread(target=self._launch, name="auto-launch", daemon=True)
        self._launcher_thread.start()
        return None

    def handle_stop(self) -> str | None:
        with self._lock:
            if self._status is AutoStatus.LOADING:
                self._cancel.set()
                return None
            if self._status is AutoStatus.RUNNING and self._episode is not None:
                self._episode.request_stop()
                return None
        return "AUTO가 실행 중이 아닙니다"

    def handle_shutdown_policy(self) -> str | None:
        with self._lock:
            if self._status in (AutoStatus.LOADING, AutoStatus.RUNNING):
                return "AUTO 실행 중에는 정책 프로세스를 끌 수 없습니다. 먼저 중지하세요"
            if self._policy.status()["state"] == "off":
                return "정책 프로세스가 이미 꺼져 있습니다"
        threading.Thread(target=self._shutdown_policy, name="auto-shutdown", daemon=True).start()
        return None

    def _shutdown_policy(self) -> None:
        self._policy.shutdown()
        self._event("info", "정책 프로세스를 껐습니다")

    def close(self) -> None:
        """서버 종료: 실행 중이면 멈추고 정책 프로세스를 끈다."""
        self._cancel.set()
        episode = self._episode
        if episode is not None and episode.active:
            episode.request_stop()
            episode.done.wait(2.0)
        thread = self._launcher_thread
        if thread is not None:
            thread.join(5.0)
        self._policy.shutdown()

    # --- 정책 프로세스 시작과 에피소드 맡기기 (스레드) ---

    def _launch(self) -> None:
        try:
            self._policy.ensure_started(self._cancel)
        except PolicyError as err:
            if self._cancel.is_set():
                self._abort(AutoStatus.STOPPED, "", "AUTO를 중지했습니다(정책을 불러오는 중)")
            else:
                self._abort(AutoStatus.ERROR, str(err), f"AUTO 오류: {err}")
            return
        if self._cancel.is_set():
            self._abort(AutoStatus.STOPPED, "", "AUTO를 중지했습니다(정책을 불러오는 중)")
            return
        seed, reason = self._recording.begin_auto()
        if reason is not None or seed is None:
            self._abort(AutoStatus.ERROR, reason or "시드를 정하지 못했습니다", f"AUTO를 시작하지 못했습니다: {reason}")
            return
        episode = AutoEpisode(seed, self._policy, max_s=self._max_s, on_finish=self._on_finished)
        with self._lock:
            self._episode = episode
            self._seed = seed
            self._status = AutoStatus.RUNNING
        reason = self._worker.start_auto(episode)
        if reason is not None:
            self._abort(AutoStatus.ERROR, reason, f"AUTO를 시작하지 못했습니다: {reason}")
            return
        self._event("info", f"AUTO 실행 중: 시드 {seed}")

    def _abort(self, status: AutoStatus, error: str, message: str) -> None:
        """에피소드가 시작되기 전에 끝난다(정책 프로세스 문제, 취소). ctrl은 건드린 적이 없다."""
        with self._lock:
            self._status = status
            self._error = error
        self._owner.release_auto()
        self._event("error" if status is AutoStatus.ERROR else "info", message)

    # --- 시뮬레이션 스레드에서 불린다 ---

    def _on_finished(self, result: AutoResult) -> None:
        # 마지막 ctrl을 SimWorker(손 입력이 다시 쓸 값)와 ControlWorker 양쪽에 이어준다: 주체가 바뀌는 순간 팔이 튀지 않게.
        self._worker.set_ctrl(result.final_ctrl)
        if self._control is not None and hasattr(self._control, "request_adopt"):
            self._control.request_adopt(result.final_ctrl)
        self._owner.release_auto()
        status = AutoStatus(result.outcome)
        with self._lock:
            self._status = status
            self._error = result.error
            if status in (AutoStatus.SUCCESS, AutoStatus.TIMEOUT):
                self._attempts += 1
                self._success += int(status is AutoStatus.SUCCESS)
            self._result = {"outcome": result.outcome, "seed": result.seed, "time_s": round(result.time_s, 2)}
        text = {
            "success": f"성공! 시드 {result.seed}, {result.time_s:.1f}초",
            "timeout": f"시간 초과: 시드 {result.seed}, {result.time_s:.0f}초 안에 성공하지 못했습니다",
            "stopped": f"중지했습니다: 시드 {result.seed}, {result.time_s:.1f}초",
            "error": f"오류로 멈췄습니다: {result.error}",
        }[result.outcome]
        self._event("error" if result.outcome == "error" else "info", text)
        if result.outcome != "error":
            # 유휴 종료 타이머를 지금부터 다시 잰다(느린 일이라 시뮬레이션 스레드 밖에서).
            threading.Thread(target=self._policy.touch, name="auto-touch", daemon=True).start()

    # --- 상태 ---

    def _event(self, level: str, text: str) -> None:
        with self._lock:
            self._event_id += 1
            self._last_event = {"id": self._event_id, "level": level, "text": text}
        log.info("auto: %s", text)

    def state(self) -> dict[str, Any]:
        policy = self._policy.status()
        with self._lock:
            status = self._status
            episode = self._episode
            progress = episode.progress() if episode is not None and status is AutoStatus.RUNNING else {}
            loading_s = None
            if status is AutoStatus.LOADING and self._loading_since_s is not None:
                loading_s = time.monotonic() - self._loading_since_s
            load_s = policy.get("load_s")
            return {
                "status": status.value,
                "owner": self._owner.current.value,
                "seed": self._seed,
                "elapsed_s": progress.get("elapsed_s") if status is AutoStatus.RUNNING else (self._result or {}).get("time_s"),
                "max_s": self._max_s,
                "sample": progress.get("sample"),
                "chunk_pos": progress.get("chunk_pos"),
                "chunk_len": progress.get("chunk_len"),
                "last_latency_ms": None if progress.get("last_latency_ms") is None else round(progress["last_latency_ms"], 1),
                "session_success": self._success,
                "session_attempts": self._attempts,
                "error": self._error,
                "result": self._result,
                "loading_s": None if loading_s is None else round(loading_s, 1),
                "load_estimate_s": max(1, round(load_s if load_s else self._load_estimate_s)),
                "policy": {
                    "state": policy["state"],
                    "idle_remaining_s": None if policy["idle_remaining_s"] is None or status is AutoStatus.RUNNING else round(policy["idle_remaining_s"]),
                },
                "last_event": self._last_event,
            }
