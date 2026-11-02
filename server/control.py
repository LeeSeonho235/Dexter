"""제어 스레드: 손 메시지 → 목표 위치 → 역기구학 → 변화량 제한 → 시뮬레이션 ctrl.

- asyncio의 메시지 받기는 submit_hand()로 "최신 hand 메시지 한 개"를 덮어쓰기만 한다. 계산이 늦어도 받기와 렌더링은
  막히지 않고, 처리 못 한 옛 메시지는 쌓이지 않고 버려진다.
- 새 메시지가 오면 바로 깨어나(다음 주기 틱을 기다리지 않아 지연이 짧다) 매핑과 IK로 새 목표 관절 각도(goal)를 정하고,
  메시지가 없어도 CONTROL_HZ 주기로 한 번씩 돈다. 틱마다 ctrl이 goal 쪽으로 움직이되, 한 틱(1/CONTROL_HZ초)당 최대
  변화량 × (지난 걸음 이후 흐른 시간 / 한 틱)만큼만 움직인다. 그래서 메시지가 언제 오든 속도 상한(rad/s)이 같고,
  처음 손을 인식했을 때도 로봇이 튀지 않는다.
- IK가 실패하거나 NaN이 나오면 goal을 바꾸지 않는다(ctrl은 직전 goal 쪽으로 계속 가던 대로 간다). 실패 횟수를 센다.
  어떤 예외가 나도 스레드는 죽지 않고 loop_error_count만 올린다.
- 시각은 모두 서버의 time.monotonic()(s). 브라우저 시계와 섞지 않는다.
"""

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from server import config
from server.gripper import GripperLatch
from server.ik import ArmIK, limit_joint_step
from server.mapping import HandMapper
from server.sim_worker import SimWorker

log = logging.getLogger(__name__)

ARM_DOF = 5  # ctrl 앞 5개가 팔 관절, 6번째가 집게


class RollingStats:
    """최근 N개 값의 마지막, 평균, 최대와 시작 이후 최대."""

    def __init__(self, window: int = config.STATS_WINDOW_SAMPLES) -> None:
        self._values: deque[float] = deque(maxlen=window)
        self._lifetime_max = float("-inf")
        self.count = 0

    def add(self, value: float) -> None:
        self._values.append(value)
        self._lifetime_max = max(self._lifetime_max, value)
        self.count += 1

    def summary(self) -> dict[str, float | int | None]:
        if not self._values:
            return {"last": None, "mean": None, "max": None, "max_since_start": None, "n": 0}
        return {
            "last": round(self._values[-1], 3),
            "mean": round(sum(self._values) / len(self._values), 3),
            "max": round(max(self._values), 3),
            "max_since_start": round(self._lifetime_max, 3),
            "n": self.count,
        }


def std_mm(samples_m: list[np.ndarray]) -> float | None:
    """목표 위치 샘플들의 흔들림(mm). x, y, z 분산을 더해 제곱근(= 평균 위치에서 벗어난 거리의 제곱평균제곱근)."""
    if len(samples_m) < config.JITTER_MIN_SAMPLES:
        return None
    arr = np.asarray(samples_m)
    return float(math.sqrt(float(arr.var(axis=0).sum())) * 1000.0)


class ControlWorker:
    def __init__(
        self,
        sim: SimWorker,
        *,
        ik_factory: Callable[[], ArmIK] = ArmIK,
        mapper: HandMapper | None = None,
        gripper: GripperLatch | None = None,
        control_hz: float = config.CONTROL_HZ,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sim = sim
        self._ik_factory = ik_factory
        self._mapper = mapper or HandMapper()
        self._gripper = gripper or GripperLatch(config.PINCH_CLOSE_BELOW, config.PINCH_OPEN_ABOVE)
        self._tick_s = 1.0 / control_hz
        self._clock = clock

        self._lock = threading.Lock()
        self._pending: tuple[dict | None, float] | None = None
        self._wake = threading.Event()  # 새 메시지가 오거나 멈추라고 할 때 켠다
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # 아래는 제어 스레드가 쓰고 state()가 락 안에서 읽는다.
        self._ik: ArmIK | None = None
        self._goal_arm_rad: np.ndarray | None = None
        self._ctrl = np.zeros(6)
        self._ctrl_published = False
        self._last_step_s: float | None = None  # ctrl을 마지막으로 움직인 시각
        self._controlling = False
        self._consecutive = 0
        self._target_m: np.ndarray | None = None
        self._ik_fail_count = 0
        self._ik_last_failure = ""
        self._loop_error_count = 0
        self._ik_ms = RollingStats()
        self._latency_ms = RollingStats()
        self._samples: deque[tuple[float, np.ndarray, np.ndarray]] = deque()  # (시각, 필터 전, 필터 후) 목표 위치(m)

    # --- 다른 스레드에서 부르는 함수 ---

    def prepare(self) -> None:
        """IK 모델을 읽고 ctrl을 home 값으로 맞춘다. start()가 부르며, 테스트에서는 스레드 없이 tick()을 직접 돌릴 때 쓴다."""
        self._ik = self._ik_factory()
        self._ctrl = self._home_ctrl()

    def start(self) -> None:
        self.prepare()
        self._thread = threading.Thread(target=self._run, name="control-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                log.warning("control worker did not stop within %.1f s", timeout_s)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit_hand(self, hand: dict | None, recv_s: float | None = None) -> None:
        """검사를 통과한 hand 메시지(또는 None=손 없음)를 넣는다. 아직 처리 안 된 이전 메시지는 버려진다."""
        with self._lock:
            self._pending = (hand, self._clock() if recv_s is None else recv_s)
        self._wake.set()

    def state(self) -> dict:
        """상태 메시지에 넣을 값(JSON으로 보낼 수 있는 것만)."""
        snapshot = self._sim.snapshot
        tip_m = snapshot["tip_m"] if snapshot else None
        joints_rad = snapshot["joints_rad"] if snapshot else None
        now_s = self._clock()
        with self._lock:
            target_m = self._target_m
            error_m = None
            if target_m is not None and tip_m is not None:
                error_m = float(np.linalg.norm(tip_m - target_m))
            window = [s for s in self._samples if s[0] >= now_s - config.JITTER_WINDOW_S]
            filtered_mm = std_mm([s[2] for s in window])
            raw_mm = std_mm([s[1] for s in window])
            return {
                "controlling": self._controlling,
                "consecutive_frames": self._consecutive,
                "start_frames": self._mapper.start_frames,
                "target_m": _round_list(target_m),
                "tip_m": _round_list(tip_m),
                "error_m": None if error_m is None else round(error_m, 4),
                "gripper_closed": self._gripper.closed,
                "gripper_goal_rad": round(_gripper_goal_rad(self._gripper.closed), 3),
                "gripper_rad": None if joints_rad is None else round(float(joints_rad[5]), 3),
                "ctrl_rad": _round_list(self._ctrl, 3),
                "ik_fail_count": self._ik_fail_count,
                "ik_last_failure": self._ik_last_failure,
                "loop_error_count": self._loop_error_count,
                "ik_ms": self._ik_ms.summary(),
                "latency_ms": self._latency_ms.summary(),
                "jitter_mm": {
                    "filtered": None if filtered_mm is None else round(filtered_mm, 2),
                    "raw": None if raw_mm is None else round(raw_mm, 2),
                },
            }

    # --- 제어 스레드 ---

    def _home_ctrl(self) -> np.ndarray:
        assert self._ik is not None
        return np.array(self._ik.model.key(config.START_KEYFRAME).ctrl[:6], dtype=float)

    def _run(self) -> None:
        while not self._stop.is_set():
            # wake를 먼저 끄고 틱을 돈다: 틱 도중에 새 메시지가 오면 wake가 다시 켜져서 바로 다음 틱이 돈다.
            self._wake.clear()
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - 제어 스레드는 죽지 않는다
                log.exception("control tick failed")
                with self._lock:
                    self._loop_error_count += 1
            # 새 메시지가 오면 바로, 안 오면 한 틱 뒤에 다시 돈다(goal 쪽으로 계속 가야 하므로).
            self._wake.wait(self._tick_s)

    def tick(self) -> None:
        """한 틱: 새 hand 메시지 처리 → ctrl 한 걸음. 스레드가 메시지가 올 때와 CONTROL_HZ 주기로 부른다."""
        with self._lock:
            pending, self._pending = self._pending, None
        processed_recv_s: float | None = None
        if pending is not None:
            hand, recv_s = pending
            if self._on_new_hand(hand, recv_s):
                processed_recv_s = recv_s
        self._step_ctrl(processed_recv_s)

    def _on_new_hand(self, hand: dict | None, recv_s: float) -> bool:
        """새 메시지로 목표와 goal을 갱신한다. 이번 메시지의 IK 결과가 goal에 반영되면 True."""
        assert self._ik is not None
        result = self._mapper.update(hand, recv_s)
        with self._lock:
            self._controlling = result.controlling
            self._consecutive = result.consecutive_frames
            self._target_m = result.target_m
        if not result.new_sample or result.target_m is None:
            return False

        self._gripper.update(hand["pinch_ratio"])
        with self._lock:
            self._samples.append((recv_s, result.raw_target_m.copy(), result.target_m.copy()))
            while self._samples and self._samples[0][0] < recv_s - config.JITTER_WINDOW_S:
                self._samples.popleft()

        ik_result = self._ik.solve(result.target_m)
        with self._lock:
            self._ik_ms.add(ik_result.elapsed_ms)
            if not ik_result.ok:
                self._ik_fail_count += 1
                self._ik_last_failure = ik_result.reason
        if not ik_result.ok:
            return False
        self._goal_arm_rad = ik_result.q_rad
        return True

    def _step_ctrl(self, processed_recv_s: float | None) -> None:
        """ctrl을 goal 쪽으로 옮기고 시뮬레이션에 넘긴다. 최대 변화량은 지난 걸음 이후 흐른 시간에 비례한다."""
        if self._goal_arm_rad is None:
            return  # 아직 조종 시작 전: ctrl을 바꾸지 않는다(home 그대로)
        now_s = self._clock()
        if self._last_step_s is None:
            scale = 1.0
        else:
            scale = min(1.0, max(0.0, (now_s - self._last_step_s) / self._tick_s))
        self._last_step_s = now_s
        new_ctrl = self._ctrl.copy()
        new_ctrl[:ARM_DOF] = limit_joint_step(
            self._ctrl[:ARM_DOF], self._goal_arm_rad, config.MAX_JOINT_STEP_RAD * scale
        )
        new_ctrl[ARM_DOF] = limit_joint_step(
            self._ctrl[ARM_DOF], _gripper_goal_rad(self._gripper.closed), config.MAX_GRIPPER_STEP_RAD * scale
        )
        if not np.all(np.isfinite(new_ctrl)):
            raise ValueError("ctrl is not finite")
        if self._ctrl_published and np.array_equal(new_ctrl, self._ctrl):
            if processed_recv_s is not None:
                self._record_latency(processed_recv_s)  # 이미 목표에 도착해 있어도 이 메시지는 반영된 것이다
            return
        self._sim.set_ctrl(new_ctrl)
        with self._lock:
            self._ctrl = new_ctrl
        self._ctrl_published = True
        if processed_recv_s is not None:
            self._record_latency(processed_recv_s)

    def _record_latency(self, recv_s: float) -> None:
        with self._lock:
            self._latency_ms.add((self._clock() - recv_s) * 1000.0)


def _gripper_goal_rad(closed: bool) -> float:
    return config.GRIPPER_CLOSED_RAD if closed else config.GRIPPER_OPEN_RAD


def _round_list(values: np.ndarray | None, digits: int = 4) -> list[float] | None:
    if values is None:
        return None
    return [round(float(v), digits) for v in values]
