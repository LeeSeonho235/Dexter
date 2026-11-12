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
from server import correction_config as cc
from server import eval_config as ec
from server.correction import CorrectionRecording, owner_time_s
from server.correction_store import KIND_AUTO_ONLY, KIND_DISCARDED, CorrectionStore, CorrectionWriter
from server.grid import grid_step_index, steps_done
from server.owner import ControlOwner, Owner
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
    # 교정(4-2): 샘플마다 (시각, qpos, 실제 적용된 ctrl, 주체)를 기록했을 때만 있다(record_samples=True). 저장 후보
    recording: CorrectionRecording | None = None
    end_reason: str = ""  # success | auto_limit | total_limit | user_stop | error
    latencies: dict[str, list[float]] = field(default_factory=dict)  # 넘겨받기/돌려주기 요청 → 전환 지연(ms)

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
        owner: ControlOwner | None = None,
        control=None,
        hold_ctrl: Callable[[np.ndarray], None] | None = None,
        record_samples: bool = False,
        total_max_s: float = cc.TOTAL_MAX_S,
    ) -> None:
        """policy: request(state float32[6], image uint8 HxWx3, reset: bool) -> ChunkResult 를 가진 객체(PolicyClient 또는 가짜).

        교정(4-2): owner를 주면 request_takeover()/request_handback()으로 AUTO → HUMAN_WAIT → HUMAN → AUTO 전환을 할 수 있다(전환은 이 에피소드가
        시뮬레이션 스레드에서만 한다). control은 request_hold(ctrl)를 가진 ControlWorker, hold_ctrl은 SimWorker.set_ctrl(넘겨받는 순간 ctrl을 맞춘다).
        record_samples=True면 샘플마다 (시각, qpos, 실제 적용 ctrl, 주체)를 모아 결과에 CorrectionRecording으로 담는다."""
        self._owner = owner
        self._control = control
        self._hold_ctrl = hold_ctrl
        self._total_n_max = int(round(total_max_s * fps))
        self._rows: dict[str, list] | None = {"t": [], "qpos": [], "ctrl": [], "owner": []} if record_samples else None
        self._takeover_req_s: float | None = None
        self._handback_req_s: float | None = None
        self._need_reset = False  # 돌려준 뒤 첫 정책 요청은 policy.reset()을 부르게 한다
        self._auto_n = 0  # AUTO 주체로 행동을 낸 샘플 수(AUTO 시간 제한의 기준)
        self._counts = {cc.OWNER_AUTO: 0, cc.OWNER_HUMAN: 0, cc.OWNER_HUMAN_WAIT: 0}
        self._prev_code: int | None = None
        self._interventions = 0
        self._latencies: dict[str, list[float]] = {"takeover_request_to_wait_ms": [], "handback_request_to_auto_ms": []}
        self._block_start_m = (0.0, 0.0, 0.0)
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

    def request_takeover(self) -> str | None:
        """사람이 넘겨받기를 요청한다(Space, 버튼). 받아들였으면 None, 아니면 이유. 실제 전환은 시뮬레이션 스레드가 다음 물리 스텝 직전에 한다."""
        with self._lock:
            if self._owner is None or self._control is None:
                return "이 에피소드는 넘겨받기를 지원하지 않습니다"
            if not self.active:
                return "AUTO가 실행 중이 아닙니다"
            current = self._owner.current
            if current is Owner.HUMAN_WAIT:
                return "이미 손을 기다리는 중입니다"
            if current is Owner.HUMAN:
                return "이미 사람이 조종 중입니다"
            if current is not Owner.AUTO:
                return "AUTO가 실행 중이 아닙니다"
            if self._takeover_req_s is not None:
                return "이미 넘겨받기를 요청했습니다"
            self._takeover_req_s = time.monotonic()
            return None

    def request_handback(self) -> str | None:
        """사람이 정책에게 돌려준다(Space 한 번 더). HUMAN일 때만 받아들인다."""
        with self._lock:
            if self._owner is None:
                return "이 에피소드는 돌려주기를 지원하지 않습니다"
            if not self.active:
                return "AUTO가 실행 중이 아닙니다"
            current = self._owner.current
            if current is Owner.HUMAN_WAIT:
                return "아직 손을 기다리는 중입니다. 손을 보여 주어 조종을 시작한 뒤에 돌려줄 수 있습니다"
            if current is not Owner.HUMAN:
                return "사람이 조종 중이 아닙니다"
            if self._handback_req_s is not None:
                return "이미 돌려주기를 요청했습니다"
            self._handback_req_s = time.monotonic()
            return None

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
        self._block_start_m = tuple(float(v) for v in data.qpos[6:9])
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
        self._process_transition_requests(data)
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

    # --- 전환(시뮬레이션 스레드). 소유자는 이 에피소드가 정하고, HUMAN_WAIT → HUMAN만 ControlWorker가 손 3프레임을 보고 정한다 ---

    def _owner_code(self) -> int:
        current = self._owner.current if self._owner is not None else Owner.AUTO
        if current is Owner.HUMAN:
            return cc.OWNER_HUMAN
        if current is Owner.HUMAN_WAIT:
            return cc.OWNER_HUMAN_WAIT
        return cc.OWNER_AUTO

    def _process_transition_requests(self, data: mujoco.MjData) -> None:
        if self._owner is None:
            return
        current = self._owner.current
        if self._takeover_req_s is not None:
            if current is Owner.AUTO:
                self._do_takeover(data)
            else:
                self._takeover_req_s = None  # 그 사이 상황이 바뀌었다
        if self._handback_req_s is not None:
            if current is Owner.HUMAN:
                self._do_handback(data)
            else:
                self._handback_req_s = None  # HUMAN이 아니게 된 사이에 온 요청은 버린다

    def _do_takeover(self, data: mujoco.MjData) -> None:
        """AUTO → HUMAN_WAIT. 로봇은 직전 ctrl 그대로 멈춘다. 정책 호출은 이미 끝나 있고(같은 스레드), 대기 중인 청크 큐는 버린다.
        ControlWorker와 SimWorker가 같은 ctrl을 갖게 한 뒤에 소유자를 바꾼다(낡은 ctrl이 덮어쓰지 않게)."""
        assert self._owner is not None
        self._queue.clear()
        ctrl = np.array(data.ctrl[:6], dtype=np.float64)
        if self._control is not None:
            self._control.request_hold(ctrl)
        if self._hold_ctrl is not None:
            self._hold_ctrl(ctrl)
        requested = self._takeover_req_s
        if self._owner.transition(Owner.AUTO, Owner.HUMAN_WAIT):
            self._takeover_req_s = None
            if requested is not None:
                self._latencies["takeover_request_to_wait_ms"].append((time.monotonic() - requested) * 1000.0)

    def _do_handback(self, data: mujoco.MjData) -> None:
        """HUMAN → AUTO. 청크 큐를 비우고, 다음 샘플에서 새 관측으로 정책에 요청한다(reset=True: policy.reset())."""
        assert self._owner is not None
        self._queue.clear()
        self._need_reset = True
        requested = self._handback_req_s
        if self._owner.transition(Owner.HUMAN, Owner.AUTO):
            self._handback_req_s = None
            if requested is not None:
                self._latencies["handback_request_to_auto_ms"].append((time.monotonic() - requested) * 1000.0)

    def _append_row(self, data: mujoco.MjData, qpos: np.ndarray, code: int) -> None:
        """샘플 한 개를 기록한다: qpos(행동 적용 전)와 지금 data.ctrl(실제로 적용 중인 값)을 같은 순간에 복사한다."""
        self._counts[code] += 1
        if code == cc.OWNER_HUMAN and self._prev_code != cc.OWNER_HUMAN:
            self._interventions += 1
        self._prev_code = code
        if self._rows is None:
            return
        self._rows["t"].append(float(data.time))
        self._rows["qpos"].append(qpos)
        self._rows["ctrl"].append(np.array(data.ctrl[:ARM_DOF], dtype=np.float64))
        self._rows["owner"].append(code)

    def _sample(self, data: mujoco.MjData) -> bool:
        k = self._k
        qpos = np.array(data.qpos[:13], dtype=np.float64)  # 복사
        sim_s = float(data.time) - self._t0
        self._last_qpos, self._last_sim_s = qpos, sim_s
        if self._record_trace:
            self._trace.append(qpos)
        code = self._owner_code()
        # 성공 판정은 AUTO, HUMAN_WAIT, HUMAN 모두에서 돈다
        if self._hold.update(sim_s, is_success(qpos[QPOS_BLOCK_XYZ], float(qpos[QPOS_GRIPPER]))) and self._time_to_success is None:
            self._time_to_success = sim_s
            self._append_row(data, qpos, code)
            self._finish(data, "success", end_reason="success")
            return False
        # AUTO 시간 제한은 AUTO 주체로 보낸 샘플만 센다(손을 기다린 시간과 사람 시간은 뺀다). 순수 AUTO에서는 예전과 같다.
        if code == cc.OWNER_AUTO and self._auto_n >= self._n_max:
            self._append_row(data, qpos, code)
            self._finish(data, "timeout", end_reason="auto_limit")
            return False
        if k >= self._total_n_max:
            self._append_row(data, qpos, code)
            self._finish(data, "timeout", end_reason="total_limit")
            return False

        blocked = False
        latency_ms = None
        if code == cc.OWNER_AUTO:
            if not self._queue:
                blocked = True
                wall0 = time.perf_counter()
                image = self._renderer.render(qpos)
                render_ms = (time.perf_counter() - wall0) * 1000.0
                result = self._policy.request(qpos[:ARM_DOF].astype(np.float32), image, reset=(self._calls == 0 or self._need_reset))
                self._need_reset = False
                stall_ms = (time.perf_counter() - wall0) * 1000.0
                self._calls += 1
                actions = np.asarray(result.actions, dtype=np.float64)
                latency_ms = render_ms + result.total_ms
                self._requests.append(
                    {"render_ms": render_ms, "request_ms": result.total_ms, "infer_ms": result.infer_ms, "stall_ms": stall_ms}
                )
                if self._takeover_req_s is not None and self._owner is not None:
                    # 정책을 기다리는 동안 넘겨받기를 눌렀다: 받아 온 청크는 통째로 버리고 직전 ctrl을 유지한 채 손을 기다린다
                    self._process_transition_requests(data)
                    code = self._owner_code()
                else:
                    self._queue.extend(actions)
                    self._chunk_len = len(actions)
            if code == cc.OWNER_AUTO:
                action = self._queue.popleft()
                if action.shape[0] < ARM_DOF or not np.all(np.isfinite(action[:ARM_DOF])):
                    raise PolicyError("정책이 잘못된 행동을 냈습니다")
                data.ctrl[:ARM_DOF] = action[:ARM_DOF]
                if self._record_trace:
                    self._ctrls.append(action[:ARM_DOF].copy())
                self._auto_n += 1
        # HUMAN, HUMAN_WAIT 샘플: ctrl을 쓰지 않는다. data.ctrl은 사람의 ctrl(HUMAN) 또는 직전 값(HUMAN_WAIT) 그대로다.
        self._append_row(data, qpos, code)
        self._k = k + 1
        self._next_step = grid_step_index(self._k, self._fps, self._timestep_s)
        with self._lock:
            self._progress.update(
                sample=k, elapsed_s=sim_s,
                chunk_pos=(self._chunk_len - len(self._queue) - 1) if code == cc.OWNER_AUTO else None,
                chunk_len=self._chunk_len if code == cc.OWNER_AUTO else None,
                owner=("HUMAN" if code == cc.OWNER_HUMAN else "HUMAN_WAIT" if code == cc.OWNER_HUMAN_WAIT else "AUTO"),
                auto_s=self._counts[cc.OWNER_AUTO] / self._fps, human_s=self._counts[cc.OWNER_HUMAN] / self._fps,
                wait_s=self._counts[cc.OWNER_HUMAN_WAIT] / self._fps, total_s=(k + 1) / self._fps, interventions=self._interventions,
            )
            if latency_ms is not None:
                self._progress["last_latency_ms"] = latency_ms
        return blocked

    def _build_recording(self, outcome: str, end_reason: str, error: str) -> CorrectionRecording | None:
        if self._rows is None or not self._rows["t"]:
            return None
        return CorrectionRecording(
            seed=self.seed,
            block_start_m=self._block_start_m,
            sim_time_s=np.array(self._rows["t"], dtype=np.float64),
            qpos=np.array(self._rows["qpos"], dtype=np.float64),
            ctrl=np.array(self._rows["ctrl"], dtype=np.float64),
            owner=np.array(self._rows["owner"], dtype=np.int8),
            outcome=outcome,
            end_reason=end_reason,
            time_to_success_s=self._time_to_success,
            error=error,
            fps=float(self._fps),
        )

    def _finish(self, data: mujoco.MjData, outcome: str, error: str = "", end_reason: str = "") -> None:
        if not self.active:
            return
        self.active = False
        qpos = self._last_qpos
        end_reason = end_reason or {"stopped": "user_stop", "error": "error"}.get(outcome, outcome)
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
            recording=self._build_recording(outcome, end_reason, error),
            end_reason=end_reason,
            latencies={key: list(values) for key, values in self._latencies.items()},
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
        store: CorrectionStore | None = None,
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
        # 교정(4-2): 끝난 에피소드에 사람 구간이 있으면 사용자가 저장/버리기를 고를 때까지 메모리에 들고 있는다(고르기 전에는 저장하지 않는다)
        self._store = store
        self._pending: CorrectionRecording | None = None
        self._last_counts: dict[str, Any] | None = None  # 끝난 에피소드의 시간·개입 횟수(화면 표시용)
        self._corr_saved = 0
        self._corr_saved_human_s = 0.0
        self._corr_save_errors = 0
        self._last_saved: dict[str, Any] | None = None
        self._writer: CorrectionWriter | None = None
        if store is not None:
            for meta in store.scan():
                self._corr_saved += 1
                self._corr_saved_human_s += float(meta.get("human_time_s", 0.0))
            self._writer = CorrectionWriter(
                store, on_saved=self._on_correction_saved, on_failed=self._on_correction_failed, on_csv_failed=self._on_csv_failed
            )
            self._writer.start()

    # --- 브라우저 명령. 돌려주는 값: None이면 받아들임, 문자열이면 거절 이유 ---

    def handle(self, action: str) -> str | None:
        if action == "start":
            reason = self.handle_start()
        elif action == "stop":
            reason = self.handle_stop()
        elif action == "shutdown_policy":
            reason = self.handle_shutdown_policy()
        elif action == "takeover":
            reason = self.handle_takeover()
        elif action == "handback":
            reason = self.handle_handback()
        elif action == "save_correction":
            reason = self.handle_decision(save=True)
        elif action == "discard_correction":
            reason = self.handle_decision(save=False)
        else:
            reason = f"unknown action: {action}"
        if reason is not None:
            self._event("warn", f"거절됨: {reason}")  # 화면의 AUTO 상자에 이유가 보이게(녹화 상자와 같은 방식)
        return reason

    def _running_episode(self) -> tuple[AutoEpisode | None, str | None]:
        with self._lock:
            if self._status is not AutoStatus.RUNNING or self._episode is None:
                return None, "AUTO가 실행 중이 아닙니다"
            return self._episode, None

    def handle_takeover(self) -> str | None:
        episode, reason = self._running_episode()
        return reason if episode is None else episode.request_takeover()

    def handle_handback(self) -> str | None:
        episode, reason = self._running_episode()
        return reason if episode is None else episode.request_handback()

    def handle_decision(self, *, save: bool) -> str | None:
        """끝난 에피소드의 교정을 저장하거나 버린다. 고르기 전에는 아무것도 저장하지 않는다."""
        with self._lock:
            if self._pending is None or self._writer is None:
                return "저장하거나 버릴 교정이 없습니다"
            rec, self._pending = self._pending, None
        if save:
            self._event("info", f"교정 저장 중: 시드 {rec.seed}")
            self._writer.submit_save(rec)
        else:
            self._writer.submit_csv(rec, KIND_DISCARDED)
            self._event("info", f"교정을 버렸습니다 (저장 안 됨): 시드 {rec.seed}")
        return None

    def _on_correction_saved(self, rec: CorrectionRecording, meta: dict[str, Any]) -> None:
        with self._lock:
            self._corr_saved += 1
            self._corr_saved_human_s += float(meta["human_time_s"])
            self._last_saved = {"index": meta["episode_index"], "seed": meta["seed"], "human_time_s": meta["human_time_s"], "interventions": meta["intervention_count"]}
        self._event(
            "info", f"교정 {meta['episode_index']}번 저장됨 (시드 {meta['seed']}, 사람 {meta['human_time_s']:.1f}초, 개입 {meta['intervention_count']}번)"
        )

    def _on_correction_failed(self, rec: CorrectionRecording, error: str) -> None:
        with self._lock:
            self._corr_save_errors += 1
        self._event("error", f"교정 저장 실패(시드 {rec.seed}): {error}. 저장된 것으로 세지 않았습니다. 디스크 공간과 폴더 권한을 확인하세요")

    def _on_csv_failed(self, rec: CorrectionRecording, error: str) -> None:
        self._event("error", f"auto_runs.csv에 기록하지 못했습니다(시드 {rec.seed}): {error}")

    def handle_start(self) -> str | None:
        with self._lock:
            if self._status in (AutoStatus.LOADING, AutoStatus.RUNNING):
                return "AUTO가 이미 실행 중입니다"
            if self._pending is not None:
                return "먼저 지난 교정을 저장할지 버릴지 고르세요"
            reason = self._recording.auto_precheck()
            if reason is None and self._arm_check is not None:
                reason = self._arm_check()
            if reason is not None:
                return reason
            if not self._owner.try_acquire_auto():
                return "지금은 다른 주체가 로봇을 제어 중입니다"
            self._status = AutoStatus.LOADING
            self._cancel.clear()
            self._last_counts = None
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
        with self._lock:
            self._pending = None  # 고르지 않은 교정은 저장하지 않고 버린다(서버를 끄면 사라진다)
        if self._writer is not None:
            self._writer.close()

    def has_pending_decision(self) -> bool:
        with self._lock:
            return self._pending is not None

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
        episode = AutoEpisode(
            seed, self._policy, max_s=self._max_s, on_finish=self._on_finished,
            owner=self._owner, control=self._control if hasattr(self._control, "request_hold") else None,
            hold_ctrl=self._worker.set_ctrl, record_samples=self._store is not None,
        )
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
        self._owner.release_to_idle()  # 끝난 순간 주체가 AUTO, HUMAN_WAIT, HUMAN 무엇이든 IDLE로
        status = AutoStatus(result.outcome)
        rec = result.recording
        with self._lock:
            if rec is not None and self._writer is not None:
                self._last_counts = {**owner_time_s(rec.owner, rec.fps), "interventions": rec.intervention_count}
                if rec.human_samples > 0 and rec.n_ticks >= 2:
                    self._pending = rec  # 사용자가 저장/버리기를 고르기 전에는 저장하지 않는다
                else:
                    self._writer.submit_csv(rec, KIND_AUTO_ONLY)  # 순수 AUTO: 메타 한 줄만
            elif rec is not None:
                self._last_counts = {**owner_time_s(rec.owner, rec.fps), "interventions": rec.intervention_count}
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
        if self.has_pending_decision():
            text += " — 사람이 고친 구간이 있습니다. \"교정 저장\" 또는 \"버리기\"를 고르세요"
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
            running = status is AutoStatus.RUNNING
            counts = self._last_counts or {}
            seed_info = self._recording.seed_info() if hasattr(self._recording, "seed_info") else {}
            pending = self._pending
            return {
                "status": status.value,
                "owner": self._owner.current.value,
                # 교정(4-2): 지금 제어 주체와 시간(AUTO 시간은 45초 제한의 기준, 사람 시간은 HUMAN 구간만)
                "auto_s": progress.get("auto_s") if running else counts.get("auto_time_s"),
                "human_s": progress.get("human_s") if running else counts.get("human_time_s"),
                "wait_s": progress.get("wait_s") if running else counts.get("human_wait_time_s"),
                "total_s": progress.get("total_s") if running else counts.get("total_s"),
                "total_max_s": cc.TOTAL_MAX_S,
                "interventions": progress.get("interventions") if running else counts.get("interventions"),
                "correction": {
                    "pending": None
                    if pending is None
                    else {
                        "seed": pending.seed,
                        "outcome": pending.outcome,
                        "end_reason": pending.end_reason,
                        "human_time_s": round(owner_time_s(pending.owner, pending.fps)["human_time_s"], 2),
                        "interventions": pending.intervention_count,
                    },
                    "saved_count": self._corr_saved,
                    "saved_human_s": round(self._corr_saved_human_s, 1),
                    "save_error_count": self._corr_save_errors,
                    "last_saved": self._last_saved,
                },
                "ready_seed": seed_info.get("ready_seed"),
                "ready_seed_purpose": cc.purpose_for_seed(seed_info.get("ready_seed")),
                "next_seed": seed_info.get("next_seed"),
                "next_seed_purpose": cc.purpose_for_seed(seed_info.get("next_seed")),
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
