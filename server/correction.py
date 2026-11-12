"""교정(4-2)의 순수 계산: 집게 넘겨받기, 한 에피소드의 기록(CorrectionRecording), 개입 구간과 점프 계산. MuJoCo나 스레드에 의존하지 않는다.

기록의 짝 맞추기 규칙은 3-2와 같다(server/recorder.py 맨 위): 샘플 k = 시간 격자(server/grid.py)의 한 순간에
  qpos[k] = 그 순간의 전체 qpos 13개(행동을 적용하기 전)
  ctrl[k] = 그 순간 data.ctrl[:6](실제로 로봇에 적용 중인 값: AUTO 샘플이면 정책 행동, HUMAN 샘플이면 사람의 ctrl, HUMAN_WAIT이면 직전 값)
  owner[k] = 그 샘플의 제어 주체(0 AUTO, 1 HUMAN, 2 HUMAN_WAIT)
을 같은 호출에서 한꺼번에 복사한 것이다. 소유자가 바뀌는 샘플도 같은 규칙이라 짝이 어긋나지 않는다.
is_intervention[k] = (owner[k] == HUMAN). HUMAN_WAIT는 로봇이 멈춰 있는 구간이라 개입으로 세지 않는다(학습에 "멈춰라"를 가르치지 않게).
"""

from dataclasses import dataclass, field

import numpy as np

from server import config, correction_config as cc
from server.gripper import GripperLatch


class HandoffGripper:
    """넘겨받는 순간 집게가 갑자기 바뀌지 않게 하는 집게 판정.

    - 시작: 집게 ctrl은 AUTO가 마지막으로 낸 값을 그대로 둔다(hold_rad). 로봇 집게의 닫힘/열림(robot_closed)은 그 값이 가운데(mid_rad)보다 작은가로 정한다.
    - 사용자의 pinch 판정이 로봇 상태와 처음으로 같아지면(synced) 그때부터 일반 GripperLatch처럼 손을 따라간다.
      예: 블록을 쥔 채(닫힘) 넘겨받았는데 손이 펴져 있으면, 사용자가 손을 쥘 때까지 집게는 닫힌 채로 있다(블록을 떨어뜨리지 않는다).
    - pinch_ratio가 닫힘 기준과 열림 기준 사이(애매한 구간)이거나 없으면(None) 사용자 상태를 아직 모른다고 본다.
    """

    def __init__(
        self,
        robot_ctrl_rad: float,
        close_below: float,
        open_above: float,
        mid_rad: float = cc.GRIPPER_SYNC_MID_RAD,
        require_match: bool = cc.GRIPPER_SYNC_REQUIRE_MATCH,
    ) -> None:
        self.robot_closed = bool(robot_ctrl_rad < mid_rad)
        self._hold_rad = float(robot_ctrl_rad)
        self._close_below = close_below
        self._open_above = open_above
        self._latch = GripperLatch(close_below, open_above, closed=self.robot_closed)
        self.synced = not require_match
        self.user_closed: bool | None = None

    @property
    def closed(self) -> bool:
        return self._latch.closed

    @property
    def hold_rad(self) -> float | None:
        """아직 일치하지 않았으면 집게 ctrl로 유지할 값(rad), 일치했으면 None(손을 따라간다)."""
        return None if self.synced else self._hold_rad

    @property
    def waiting(self) -> bool:
        """화면에 "집게를 로봇에 맞추세요"를 띄울 조건: 아직 일치하지 않았다."""
        return not self.synced

    def update(self, pinch_ratio: float | None) -> bool:
        if pinch_ratio is not None:
            if pinch_ratio < self._close_below:
                self.user_closed = True
            elif pinch_ratio > self._open_above:
                self.user_closed = False
        if self.synced:
            self._latch.update(pinch_ratio)
        elif self.user_closed is not None and self.user_closed == self.robot_closed:
            self.synced = True
            self._latch.closed = self.user_closed
        return self.closed


@dataclass
class CorrectionRecording:
    """끝난 에피소드의 기록(저장 후보). 배열 길이는 모두 T."""

    seed: int
    block_start_m: tuple[float, float, float]
    sim_time_s: np.ndarray  # (T,) float64
    qpos: np.ndarray  # (T, 13) float64
    ctrl: np.ndarray  # (T, 6) float64
    owner: np.ndarray  # (T,) int8: 0 AUTO, 1 HUMAN, 2 HUMAN_WAIT
    outcome: str  # success | timeout | stopped | error
    end_reason: str  # success | auto_limit | total_limit | user_stop | error
    time_to_success_s: float | None = None
    error: str = ""
    fps: float = float(cc.CORRECTION_FPS)
    extras: dict = field(default_factory=dict)

    @property
    def n_ticks(self) -> int:
        return int(len(self.sim_time_s))

    @property
    def is_intervention(self) -> np.ndarray:
        return self.owner == cc.OWNER_HUMAN

    @property
    def human_samples(self) -> int:
        return int(np.count_nonzero(self.is_intervention))

    @property
    def intervention_count(self) -> int:
        return len(intervention_segments(self.owner))


def intervention_segments(owner: np.ndarray) -> list[tuple[int, int]]:
    """owner가 HUMAN인 연속 구간 [(시작 샘플, 끝 샘플)] (끝 샘플 포함)."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for k, value in enumerate(np.asarray(owner)):
        if value == cc.OWNER_HUMAN:
            if start is None:
                start = k
        elif start is not None:
            segments.append((start, k - 1))
            start = None
    if start is not None:
        segments.append((start, len(owner) - 1))
    return segments


def owner_time_s(owner: np.ndarray, fps: float = float(cc.CORRECTION_FPS)) -> dict[str, float]:
    """주체별 시간(s) = 샘플 수 / fps. 사람 시간은 HUMAN 구간만(06 문서의 "같은 시간 비교"에 쓰는 값)."""
    owner = np.asarray(owner)
    return {
        "auto_time_s": float(np.count_nonzero(owner == cc.OWNER_AUTO)) / fps,
        "human_time_s": float(np.count_nonzero(owner == cc.OWNER_HUMAN)) / fps,
        "human_wait_time_s": float(np.count_nonzero(owner == cc.OWNER_HUMAN_WAIT)) / fps,
        "total_s": float(len(owner)) / fps,
    }


def max_joint_delta(a: np.ndarray, b: np.ndarray) -> float:
    """두 ctrl(6개)의 관절별 차이 중 가장 큰 값(rad)."""
    return float(np.max(np.abs(np.asarray(a, dtype=float)[:6] - np.asarray(b, dtype=float)[:6])))


def handback_jumps_rad(owner: np.ndarray, ctrl: np.ndarray) -> list[float]:
    """돌려준 순간마다 (정책 첫 행동 - 직전 사람 ctrl)의 최대 관절 차이(rad). HUMAN 다음에 처음 나오는 AUTO 샘플이 대상이다."""
    owner = np.asarray(owner)
    jumps: list[float] = []
    for k in range(1, len(owner)):
        if owner[k] == cc.OWNER_AUTO and owner[k - 1] == cc.OWNER_HUMAN:
            jumps.append(max_joint_delta(ctrl[k], ctrl[k - 1]))
    return jumps


def takeover_first_second_steps_rad(owner: np.ndarray, ctrl: np.ndarray, fps: float = float(cc.CORRECTION_FPS)) -> list[float]:
    """넘겨받은 직후(HUMAN 구간이 시작되는 샘플부터 1초 동안) 샘플 사이 ctrl의 최대 관절 변화(rad/샘플)를 구간마다 하나씩."""
    owner = np.asarray(owner)
    result: list[float] = []
    horizon = int(round(fps))
    for start, _end in intervention_segments(owner):
        stop = min(len(owner), start + horizon + 1)
        lo = max(0, start - 1)  # 직전(HUMAN_WAIT) 샘플과의 차이도 포함한다
        diffs = np.abs(np.diff(np.asarray(ctrl)[lo:stop, :6], axis=0))
        result.append(float(diffs.max()) if diffs.size else 0.0)
    return result


def typical_step_rad(ctrl: np.ndarray) -> dict[str, float]:
    """샘플 사이 ctrl 변화(관절 중 최대, rad/샘플)의 평균·95번째 백분위·최대. 평상시 기준값."""
    ctrl = np.asarray(ctrl)
    if len(ctrl) < 2:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    steps = np.abs(np.diff(ctrl[:, :6], axis=0)).max(axis=1)
    return {"mean": float(steps.mean()), "p95": float(np.percentile(steps, 95)), "max": float(steps.max())}


def check_alignment(rec: CorrectionRecording) -> list[str]:
    """저장 전에 기록 자체가 정합한지 본다. 문제 목록(비어 있으면 통과)."""
    problems: list[str] = []
    n = rec.n_ticks
    if rec.qpos.shape != (n, 13) or rec.ctrl.shape != (n, 6) or rec.owner.shape != (n,):
        problems.append(f"shape mismatch: t{rec.sim_time_s.shape} qpos{rec.qpos.shape} ctrl{rec.ctrl.shape} owner{rec.owner.shape}")
    if n and not np.all(np.diff(rec.sim_time_s) > 0):
        problems.append("sim_time_s is not strictly increasing")
    if not np.all(np.isin(rec.owner, (cc.OWNER_AUTO, cc.OWNER_HUMAN, cc.OWNER_HUMAN_WAIT))):
        problems.append("owner has unknown values")
    # 손을 기다리는 동안과 그 직전 AUTO 마지막 샘플은 ctrl이 같아야 한다(로봇이 그 자리에 멈춘다)
    for k in range(1, n):
        if rec.owner[k] == cc.OWNER_HUMAN_WAIT and rec.owner[k - 1] in (cc.OWNER_AUTO, cc.OWNER_HUMAN_WAIT) and not np.array_equal(rec.ctrl[k], rec.ctrl[k - 1]):
            problems.append(f"ctrl changed while waiting for the hand at sample {k}")
            break
    return problems


# 설정값 모음(보고용)
def describe_settings() -> dict:
    return {
        "total_max_s": cc.TOTAL_MAX_S,
        "gripper_sync_mid_rad": cc.GRIPPER_SYNC_MID_RAD,
        "gripper_sync_require_match": cc.GRIPPER_SYNC_REQUIRE_MATCH,
        "human_ik_warm_start": cc.HUMAN_IK_WARM_START,
        "start_consecutive_frames": config.START_CONSECUTIVE_FRAMES,
    }
