"""에피소드 녹화기: 시뮬레이션 스레드가 물리 스텝 직전에 on_step()을 불러 샘플을 쌓는다. 파일 쓰기는 여기서 하지 않는다.

★ 샘플 한 개의 짝(틱 정렬 규칙) — 학습 데이터셋(3-3)이 이 규칙으로 만들어진다:
    샘플 k = 시각 t_k(시뮬레이션 시간 격자, server/grid.py)의 "물리 스텝 직전" 한 순간에 같이 복사한 값
      qpos[k] : 그 순간의 전체 qpos 13개(팔 6개 + 블록 자유 관절 7개)   → observation.state[k] = qpos[k][:6]
      ctrl[k] : 그 순간 data.ctrl[:6](모터 목표, 이 순간부터 적용 중인 값)  → action[k] = ctrl[k]
      이미지[k]: 3-3에서 qpos[k] 전체로 다시 그린다
  state와 action을 같은 호출에서 한 번에 복사하므로 한 샘플 어긋날 수 없다. 이 순간의 ctrl이 만드는 효과는 qpos[k+1]부터 보인다.
  mj_step 뒤의 값(SimWorker.snapshot)은 쓰지 않는다.

- on_step()은 락 안에서 미리 만든 배열에 복사만 한다(수 µs). 에피소드가 끝나면(stop, 최대 길이) 복사본을 만들어 sink에 넘기고,
  sink(쓰기 스레드 큐)는 기다리지 않고 바로 돌아와야 한다. 그래서 디스크가 느려도 틱 루프는 늦어지지 않는다.
- 정지 구간: 이웃한 샘플 사이 ctrl 변화(6개 중 최대 절대값, rad)가 STILL_CTRL_DELTA_RAD보다 작으면 "정지" 샘플.
- is_intervention은 샘플마다 False(4단계 교정용 자리).
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from server import config
from server.grid import sample_due
from server.success import QPOS_BLOCK_XYZ, QPOS_GRIPPER, SuccessHold, is_success

QPOS_DIM = 13  # 팔 6 + 블록 자유 관절 7
CTRL_DIM = 6
MIN_SAMPLES = 2  # 이보다 짧으면 저장하지 않고 버린다

END_STOP_SUCCESS = "stop_success"
END_STOP_FAIL = "stop_fail"
END_MAX_LENGTH = "max_length"


@dataclass
class FinishedEpisode:
    episode_index: int
    seed: int
    block_start_m: tuple[float, float, float]
    target_m: tuple[float, float]
    sim_time_s: np.ndarray  # (T,) float64
    qpos: np.ndarray  # (T, 13) float64
    ctrl: np.ndarray  # (T, 6) float64
    is_intervention: np.ndarray  # (T,) bool
    success: bool  # 내가 버튼으로 정한 최종 결과(최대 길이로 끝났으면 False)
    end_reason: str
    auto_success_ever: bool
    auto_success_first_s: float | None
    auto_success_at_end: bool
    still_ratio: float
    still_longest_s: float
    created_at: str
    source: str = config.RECORD_SOURCE

    @property
    def n_ticks(self) -> int:
        return len(self.sim_time_s)


def still_stats(ctrl: np.ndarray, threshold_rad: float, fps: float) -> tuple[float, float]:
    """(정지 샘플 비율, 가장 긴 연속 정지 시간 s). 첫 샘플은 비교할 앞이 없어서 세지 않는다(분모는 샘플 수 - 1)."""
    if len(ctrl) < 2:
        return 0.0, 0.0
    delta = np.abs(np.diff(ctrl, axis=0)).max(axis=1)
    still = delta < threshold_rad
    longest = run = 0
    for flag in still:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return float(still.mean()), longest / fps


class EpisodeRecorder:
    def __init__(
        self,
        *,
        sink: Callable[[FinishedEpisode], None],
        timestep_s: float,
        fps: float = config.RECORD_FPS,
        max_s: float = config.RECORD_MAX_S,
        still_delta_rad: float = config.STILL_CTRL_DELTA_RAD,
        hold_s: float = config.SUCCESS_HOLD_S,
    ) -> None:
        self._sink = sink
        self._timestep_s = timestep_s
        self._fps = fps
        self._max_samples = int(round(max_s * fps))
        self._still_delta_rad = still_delta_rad
        self._hold_s = hold_s

        self._lock = threading.Lock()
        self._active = False  # on_step()가 락 없이 먼저 보는 값
        self._t = np.zeros(self._max_samples)
        self._qpos = np.zeros((self._max_samples, QPOS_DIM))
        self._ctrl = np.zeros((self._max_samples, CTRL_DIM))
        self._n = 0
        self._t0_s = 0.0
        self._meta: dict = {}
        self._hold = SuccessHold(hold_s)
        self._auto_first_s: float | None = None
        self._still_n = 0
        self._still_run = 0
        self._still_longest = 0

    # --- 시뮬레이션 스레드 ---

    def on_step(self, sim_time_s: float, qpos: np.ndarray, ctrl: np.ndarray) -> None:
        """물리 스텝 직전에 부른다. 녹화 중이 아니면 바로 돌아온다. 격자 시각이 아니면 아무것도 안 한다."""
        if not self._active:
            return
        finished: FinishedEpisode | None = None
        with self._lock:
            if not self._active:
                return
            k = self._n
            if k == 0:
                self._t0_s = sim_time_s
            elif not sample_due(sim_time_s, self._t0_s, k, self._fps, self._timestep_s):
                return
            self._t[k] = sim_time_s
            self._qpos[k] = qpos[:QPOS_DIM]
            self._ctrl[k] = ctrl[:CTRL_DIM]
            if k > 0:
                if np.abs(self._ctrl[k] - self._ctrl[k - 1]).max() < self._still_delta_rad:
                    self._still_n += 1
                    self._still_run += 1
                    self._still_longest = max(self._still_longest, self._still_run)
                else:
                    self._still_run = 0
            ok = is_success(self._qpos[k][QPOS_BLOCK_XYZ], float(self._qpos[k][QPOS_GRIPPER]))
            if self._hold.update(sim_time_s, ok) and self._auto_first_s is None:
                self._auto_first_s = sim_time_s
            self._n = k + 1
            if self._n >= self._max_samples:
                finished = self._finish_locked(False, END_MAX_LENGTH)
        if finished is not None:
            self._sink(finished)

    # --- 다른 스레드 ---

    @property
    def active(self) -> bool:
        return self._active

    def begin(
        self,
        episode_index: int,
        seed: int,
        block_start_m: tuple[float, float, float],
        target_m: tuple[float, float],
    ) -> None:
        """녹화를 시작한다. 첫 샘플은 다음 물리 스텝 직전에 찍힌다."""
        with self._lock:
            self._n = 0
            self._hold = SuccessHold(self._hold_s)
            self._auto_first_s = None
            self._still_n = self._still_run = self._still_longest = 0
            self._meta = {
                "episode_index": episode_index,
                "seed": seed,
                "block_start_m": tuple(float(v) for v in block_start_m),
                "target_m": tuple(float(v) for v in target_m),
            }
            self._active = True

    def stop(self, success: bool) -> FinishedEpisode | None:
        """내가 정한 결과로 끝낸다. 녹화 중이 아니거나 샘플이 너무 적어 버렸으면 None."""
        with self._lock:
            if not self._active:
                return None
            reason = END_STOP_SUCCESS if success else END_STOP_FAIL
            finished = self._finish_locked(success, reason)
        if finished is not None:
            self._sink(finished)
        return finished

    def discard(self) -> bool:
        """저장하지 않고 버린다. 녹화 중이었으면 True."""
        with self._lock:
            was_active = self._active
            self._active = False
            self._n = 0
        return was_active

    def state(self) -> dict:
        with self._lock:
            n = self._n
            elapsed_s = float(self._t[n - 1] - self._t0_s) if n > 0 else 0.0
            pairs = max(0, n - 1)
            return {
                "recording": self._active,
                "ticks": n,
                "elapsed_s": round(elapsed_s, 2),
                "max_s": self._max_samples / self._fps,
                "success_condition_met": bool(self._active and self._hold.satisfied),
                "success_condition_ever": bool(self._active and self._auto_first_s is not None),
                "still_ratio": round(self._still_n / pairs, 3) if pairs > 0 else None,
                "still_longest_s": round(self._still_longest / self._fps, 2) if pairs > 0 else None,
            }

    # --- 내부 ---

    def _finish_locked(self, success: bool, reason: str) -> FinishedEpisode | None:
        """락 안에서 복사본을 만들고 녹화를 끝낸다. 샘플이 MIN_SAMPLES보다 적으면 None."""
        n = self._n
        self._active = False
        self._n = 0
        if n < MIN_SAMPLES:
            return None
        ctrl = self._ctrl[:n].copy()
        still_ratio, still_longest_s = still_stats(ctrl, self._still_delta_rad, self._fps)
        return FinishedEpisode(
            episode_index=self._meta["episode_index"],
            seed=self._meta["seed"],
            block_start_m=self._meta["block_start_m"],
            target_m=self._meta["target_m"],
            sim_time_s=self._t[:n].copy(),
            qpos=self._qpos[:n].copy(),
            ctrl=ctrl,
            is_intervention=np.zeros(n, dtype=bool),
            success=success,
            end_reason=reason,
            auto_success_ever=self._auto_first_s is not None,
            auto_success_first_s=self._auto_first_s,
            auto_success_at_end=self._hold.satisfied,
            still_ratio=still_ratio,
            still_longest_s=still_longest_s,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
