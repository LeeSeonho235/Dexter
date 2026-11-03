"""제어 스레드: 손 메시지 → 목표 위치 → 역기구학 → 변화량 제한 → 시뮬레이션 ctrl.

- asyncio의 메시지 받기는 submit_hand()로 "최신 hand 메시지 한 개"를 덮어쓰기만 한다. 계산이 늦어도 받기와 렌더링은
  막히지 않고, 처리 못 한 옛 메시지는 쌓이지 않고 버려진다.
- 새 메시지가 오면 바로 깨어나(다음 주기 틱을 기다리지 않아 지연이 짧다) 매핑과 IK로 새 목표 관절 각도(goal)를 정하고,
  메시지가 없어도 CONTROL_HZ 주기로 한 번씩 돈다. 틱마다 ctrl이 goal 쪽으로 움직이되, 한 틱(1/CONTROL_HZ초)당 최대
  변화량 × (지난 걸음 이후 흐른 시간 / 한 틱)만큼만 움직인다. 그래서 메시지가 언제 오든 속도 상한(rad/s)이 같고,
  처음 손을 인식했을 때도 로봇이 튀지 않는다.
- 키보드 높이(2-5): submit_height()로 받은 명령(dir, slow)을 HeightCommands가 "방향 × 속도 × 흐른 시간"으로 바꾸고,
  매퍼의 목표 z에 더한다(손이 안 보여도 동작, 조종이 한 번도 시작되지 않았으면 무시). 손 메시지와 같은 틱에 겹쳐도
  IK는 한 번만 풀고, 결과는 똑같이 변화량 제한을 거쳐 ctrl로 간다.
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
from server.height import HeightCommands
from server.ik import ArmIK, limit_joint_step
from server.mapping import HandMapper
from server.reset import BlockReset, ResetAction
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
        # 설정값 MAPPING_MODE(기본 "relative"), Z_FROM_HAND_SCALE(기본 False: 높이는 키보드)을 여기서 넘긴다.
        # HandMapper 자체의 기본값은 "absolute"와 z_from_hand_scale=True(2-2, 2-4 동작)다.
        self._mapper = mapper or self._new_mapper()
        self._height = HeightCommands(
            config.HEIGHT_SPEED_M_S,
            config.HEIGHT_SLOW_SPEED_M_S,
            config.HEIGHT_COMMAND_TIMEOUT_S,
            config.HEIGHT_MAX_DT_S,
        )
        self._gripper = gripper or GripperLatch(config.PINCH_CLOSE_BELOW, config.PINCH_OPEN_ABOVE)
        self._tick_s = 1.0 / control_hz
        self._clock = clock

        # 블록 리셋(3-2): request_reset()이 요청을 넣고, 제어 스레드가 tick마다 한 걸음씩 진행한다.
        self._reset = BlockReset(
            tolerance_rad=config.RESET_PARK_TOLERANCE_RAD,
            hold_s=config.RESET_PARK_HOLD_S,
            timeout_s=config.RESET_TIMEOUT_S,
            settle_s=config.RESET_SETTLE_S,
            block_tolerance_m=config.RESET_BLOCK_TOLERANCE_M,
            block_half_m=config.BLOCK_HALF_M,
        )
        self._reset_request: tuple[int, tuple[float, float], Callable[[int, bool, str], None]] | None = None
        self._reset_callback: Callable[[int, bool, str], None] | None = None

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
        self._reference_hand: tuple[float, float] | None = None  # 상대 모드 기준 손 위치(영상 정규화 좌표)
        self._azimuth_deg = 0.0  # 마지막으로 매핑에 쓴 보기 방위각
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

    def submit_height(self, direction: int, slow: bool, recv_s: float | None = None) -> None:
        """검사를 통과한 키보드 높이 명령(dir -1/0/1, slow)을 넣는다. 받은 순서대로 기록되어 시각 기준으로 계산된다.
        조종이 한 번도 시작되지 않았다면 무시된다(목표가 없다)."""
        with self._lock:
            self._height.command(direction, slow, self._clock() if recv_s is None else recv_s)
        self._wake.set()

    def request_reset(
        self, seed: int, xy_m: tuple[float, float], on_done: Callable[[int, bool, str], None]
    ) -> str | None:
        """블록 리셋을 요청한다(팔을 올리고 → 블록을 (x, y)(m)에 놓는다). 받아들였으면 None, 아니면 이유.
        끝나면 제어 스레드가 on_done(seed, 성공 여부, 이유)를 부른다. 리셋 중에는 손과 키보드 높이 입력을 무시한다."""
        with self._lock:
            if self._reset.active or self._reset_request is not None:
                return "a reset is already running"
            if self._ik is None:
                return "control is not started"
            self._reset_request = (seed, xy_m, on_done)
        self._wake.set()
        return None

    @property
    def resetting(self) -> bool:
        with self._lock:
            return self._reset.active or self._reset_request is not None

    @property
    def reset_phase(self) -> str:
        with self._lock:
            return "requested" if self._reset_request is not None else self._reset.phase.value

    def state(self) -> dict:
        """상태 메시지에 넣을 값(JSON으로 보낼 수 있는 것만)."""
        snapshot = self._sim.snapshot
        tip_m = snapshot["tip_m"] if snapshot else None
        joints_rad = snapshot["joints_rad"] if snapshot else None
        now_s = self._clock()
        with self._lock:
            target_m = self._target_m
            height_dir, height_slow = self._height.effective(now_s)
            height_limit = self._mapper.height_limit
            # 키보드 높이가 지금 실제로 움직이는 중인가: 방향이 있고, 목표가 있고(조종이 시작됐고), 그 방향의 상자 끝이 아니다.
            height_active = (
                height_dir != 0
                and self._mapper.last_target_m is not None
                and not (height_dir > 0 and height_limit == "top")
                and not (height_dir < 0 and height_limit == "bottom")
            )
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
                "mapping_mode": self._mapper.mode,
                # 상대 모드에서 클러치가 잡혀 있는 동안의 기준 손 위치(영상 정규화 x, y). 브라우저가 십자를 그린다.
                "reference_hand": _round_list(self._reference_hand, 4),
                "view_azimuth_deg": round(self._azimuth_deg, 2),
                # 키보드 높이(2-5): 유효 방향(-1 내림, 0 정지, 1 올림, 타임아웃 반영), 천천히 여부, 지금 움직이는지,
                # 목표 z(m), 상자 끝에 닿았는지("top"/"bottom"/None), 손 크기로도 높이를 정하는지
                "height_dir": height_dir,
                "height_slow": height_slow,
                "height_active": height_active,
                "height_limit": height_limit,
                "target_z_m": None if target_m is None else round(float(target_m[2]), 4),
                "z_from_hand_scale": self._mapper.z_from_hand_scale,
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

    def _view_azimuth_deg(self) -> float:
        """매핑에 쓸 보기 방위각(도). 자유 카메라가 아니면(고정 top 카메라) 위에서 보기와 같은 0°.
        view를 모르는 시뮬레이션(테스트용 가짜)도 0°다."""
        view = getattr(self._sim, "view", None)
        if view is None or not getattr(self._sim, "uses_free_camera", True):
            return 0.0
        return float(view.azimuth_deg)

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

    @staticmethod
    def _new_mapper() -> HandMapper:
        return HandMapper(mode=config.MAPPING_MODE, z_from_hand_scale=config.Z_FROM_HAND_SCALE)

    def _park_ctrl(self) -> np.ndarray:
        """리셋 때 팔이 가는 자세: home(위로 세움) + 집게 열림."""
        park = self._home_ctrl()
        park[ARM_DOF] = config.GRIPPER_OPEN_RAD
        return park

    def _tick_reset(self) -> None:
        """리셋 한 걸음: 팔을 park 자세로 한 틱 옮기고, 도착하면 블록을 놓고, 가라앉으면 끝낸다. 손·높이 입력은 버린다."""
        with self._lock:
            self._pending = None
            self._height.reset()
        now_s = self._clock()
        park = self._park_ctrl()
        scale = 1.0 if self._last_step_s is None else min(1.0, max(0.0, (now_s - self._last_step_s) / self._tick_s))
        self._last_step_s = now_s
        new_ctrl = self._ctrl.copy()
        new_ctrl[:ARM_DOF] = limit_joint_step(self._ctrl[:ARM_DOF], park[:ARM_DOF], config.MAX_JOINT_STEP_RAD * scale)
        new_ctrl[ARM_DOF] = limit_joint_step(
            self._ctrl[ARM_DOF], park[ARM_DOF], config.MAX_GRIPPER_STEP_RAD * scale
        )
        if not np.array_equal(new_ctrl, self._ctrl) or not self._ctrl_published:
            self._sim.set_ctrl(new_ctrl)
            with self._lock:
                self._ctrl = new_ctrl
            self._ctrl_published = True

        snapshot = self._sim.snapshot
        joints = None if snapshot is None else snapshot["joints_rad"]
        block = None if snapshot is None else snapshot["block_m"]
        step = self._reset.update(now_s, joints, park, block)
        if step.action == ResetAction.PLACE:
            assert self._reset.xy_m is not None
            placed = self._sim.place_block(*self._reset.xy_m)
            if placed.ok:
                self._reset.placed(self._clock())
            else:
                step = self._reset.fail(placed.reason)
        if step.action in (ResetAction.DONE, ResetAction.FAILED):
            self._finish_reset(step.action == ResetAction.DONE, step.reason)

    def _finish_reset(self, ok: bool, reason: str) -> None:
        """리셋이 끝났다(성공이든 실패든): 이전 목표·필터·집게 상태를 모두 지워 손이 다시 보일 때 처음부터 시작하게 한다.
        ctrl은 지금 값 그대로 둔다(팔이 튀지 않는다)."""
        assert self._ik is not None
        self._ik.reset()
        self._mapper = self._new_mapper()
        self._gripper = GripperLatch(config.PINCH_CLOSE_BELOW, config.PINCH_OPEN_ABOVE)
        with self._lock:
            self._height.reset()
            self._pending = None
            self._goal_arm_rad = None
            self._controlling = False
            self._consecutive = 0
            self._target_m = None
            self._reference_hand = None
            self._samples.clear()
        self._last_step_s = None
        callback, seed = self._reset_callback, self._reset.seed
        self._reset_callback = None
        if callback is not None and seed is not None:
            try:
                callback(seed, ok, reason)
            except Exception:  # noqa: BLE001 - 콜백이 실패해도 제어 스레드는 죽지 않는다
                log.exception("reset callback failed")

    def tick(self) -> None:
        """한 틱: 새 hand 메시지 처리 → ctrl 한 걸음. 스레드가 메시지가 올 때와 CONTROL_HZ 주기로 부른다."""
        with self._lock:
            request, self._reset_request = self._reset_request, None
        if request is not None:
            seed, xy_m, callback = request
            self._reset.start(self._clock(), seed, xy_m)
            self._reset_callback = callback
            self._last_step_s = None
        if self._reset.active:
            self._tick_reset()
            return
        with self._lock:
            pending, self._pending = self._pending, None
        hand_changed_target = False
        recv_s = 0.0
        if pending is not None:
            hand, recv_s = pending
            hand_changed_target = self._on_new_hand(hand, recv_s)
        height_changed_target = self._apply_keyboard_height()

        # 손 메시지와 키보드 높이가 같은 틱에 겹쳐도 IK는 한 번만 푼다. 결과는 변화량 제한을 거쳐 ctrl로 간다.
        processed_recv_s: float | None = None
        if hand_changed_target or height_changed_target:
            solved = self._solve_goal(self._mapper.last_target_m)
            if solved and hand_changed_target:
                processed_recv_s = recv_s
        self._step_ctrl(processed_recv_s)

    def _apply_keyboard_height(self) -> bool:
        """키보드 높이 명령을 흐른 시간만큼 목표 z에 반영한다. 목표 z가 바뀌었으면 True.

        조종이 한 번도 시작되지 않아 목표가 없으면 무시한다(키보드로 로봇이 갑자기 움직이는 사고를 막는다).
        손이 안 보이는 동안에도 마지막 목표의 z는 바꾼다.
        """
        now_s = self._clock()
        with self._lock:
            dz_m = self._height.advance(now_s)
            if self._mapper.last_target_m is None:
                self._height.reset()  # 조종 시작 전 명령은 모두 버린다
                return False
        if dz_m == 0.0:
            return False
        before = self._mapper.last_target_m
        target = self._mapper.nudge_height(dz_m)
        if target is None or before is None or target[2] == before[2]:
            return False  # 상자 끝에 막혀 높이가 그대로
        with self._lock:
            self._target_m = target
        return True

    def _solve_goal(self, target_m: np.ndarray | None) -> bool:
        """목표 위치를 IK로 풀어 goal(관절 목표)을 갱신한다. 실패하면 goal을 바꾸지 않고 실패 횟수를 센다."""
        assert self._ik is not None
        if target_m is None:
            return False
        ik_result = self._ik.solve(target_m)
        with self._lock:
            self._ik_ms.add(ik_result.elapsed_ms)
            if not ik_result.ok:
                self._ik_fail_count += 1
                self._ik_last_failure = ik_result.reason
        if not ik_result.ok:
            return False
        self._goal_arm_rad = ik_result.q_rad
        return True

    def _on_new_hand(self, hand: dict | None, recv_s: float) -> bool:
        """새 손 메시지로 목표를 갱신한다(IK는 tick에서 푼다). 이번 메시지가 새 목표를 만들었으면 True."""
        azimuth_deg = self._view_azimuth_deg()
        result = self._mapper.update(hand, recv_s, azimuth_deg)
        with self._lock:
            self._controlling = result.controlling
            self._consecutive = result.consecutive_frames
            self._target_m = result.target_m
            self._reference_hand = result.reference_hand if result.controlling else None
            self._azimuth_deg = azimuth_deg
        if not result.new_sample or result.target_m is None:
            return False

        self._gripper.update(hand["pinch_ratio"])
        with self._lock:
            self._samples.append((recv_s, result.raw_target_m.copy(), result.target_m.copy()))
            while self._samples and self._samples[0][0] < recv_s - config.JITTER_WINDOW_S:
                self._samples.popleft()
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
