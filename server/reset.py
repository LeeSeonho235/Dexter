"""블록 리셋 순서 기계(시간과 관측값만 받는 순수 계산). 실제로 로봇을 움직이고 블록을 놓는 일은 ControlWorker가 한다.

순서(블록이 집게에 끼는 일이 없게, 팔이 먼저 비켜난 다음에 블록을 놓는다):
  1. PARKING : 팔을 home 자세(위로 세움, 집게 열림)로 보낸다. 모든 관절이 목표에서 tolerance_rad 안으로 hold_s 동안 있으면 도착.
               timeout_s 안에 도착하지 못하면 블록을 건드리지 않고 FAILED.
  2. (place) : 도착하면 "블록을 놓아라" 신호를 낸다(update가 PLACE를 돌려준다). 시뮬레이션 스레드가 블록 위치를 시드 위치로
               옮기고 속도를 0으로 만든다. 놓은 자리에서 블록이 바닥 외의 것과 닿으면 놓지 않고 FAILED.
  3. SETTLING: settle_s 기다린 뒤 블록이 시드 위치(xy)와 바닥 높이에 있으면 DONE, 아니면 FAILED.
단위: 각도 rad, 길이 m, 시간 s(호출하는 쪽의 시계).
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ResetPhase(str, Enum):
    IDLE = "idle"
    PARKING = "parking"
    SETTLING = "settling"
    DONE = "done"
    FAILED = "failed"


class ResetAction(str, Enum):
    NONE = "none"
    PLACE = "place"  # 지금 블록을 시드 위치에 놓아라
    DONE = "done"
    FAILED = "failed"


@dataclass
class ResetStep:
    action: ResetAction
    reason: str = ""


class BlockReset:
    def __init__(
        self, *, tolerance_rad: float, hold_s: float, timeout_s: float, settle_s: float, block_tolerance_m: float,
        block_half_m: float,
    ) -> None:
        self.tolerance_rad = tolerance_rad
        self.hold_s = hold_s
        self.timeout_s = timeout_s
        self.settle_s = settle_s
        self.block_tolerance_m = block_tolerance_m
        self.block_half_m = block_half_m
        self.phase = ResetPhase.IDLE
        self.seed: int | None = None
        self.xy_m: tuple[float, float] | None = None
        self._start_s = 0.0
        self._arrived_since_s: float | None = None
        self._placed_s = 0.0

    @property
    def active(self) -> bool:
        return self.phase in (ResetPhase.PARKING, ResetPhase.SETTLING)

    def start(self, now_s: float, seed: int, xy_m: tuple[float, float]) -> None:
        self.phase = ResetPhase.PARKING
        self.seed = seed
        self.xy_m = xy_m
        self._start_s = now_s
        self._arrived_since_s = None

    def update(
        self, now_s: float, joints_rad: np.ndarray | None, park_rad: np.ndarray, block_m: np.ndarray | None
    ) -> ResetStep:
        """한 틱의 관측(팔 관절 6개 rad, 블록 중심 m)을 넣는다. 다음에 할 일을 돌려준다."""
        if self.phase == ResetPhase.PARKING:
            if now_s - self._start_s > self.timeout_s:
                return self._fail("arm did not reach the park pose in time")
            arrived = joints_rad is not None and bool(
                np.all(np.abs(np.asarray(joints_rad) - np.asarray(park_rad)) < self.tolerance_rad)
            )
            if not arrived:
                self._arrived_since_s = None
                return ResetStep(ResetAction.NONE)
            if self._arrived_since_s is None:
                self._arrived_since_s = now_s
            if now_s - self._arrived_since_s >= self.hold_s - 1e-9:
                return ResetStep(ResetAction.PLACE)
            return ResetStep(ResetAction.NONE)

        if self.phase == ResetPhase.SETTLING:
            if now_s - self._placed_s < self.settle_s - 1e-9:
                return ResetStep(ResetAction.NONE)
            assert self.xy_m is not None
            if block_m is None:
                return self._fail("block position unknown")
            off_xy_m = float(np.hypot(block_m[0] - self.xy_m[0], block_m[1] - self.xy_m[1]))
            off_z_m = abs(float(block_m[2]) - self.block_half_m)
            if off_xy_m > self.block_tolerance_m or off_z_m > self.block_tolerance_m:
                return self._fail(f"block not settled at the seed position (off {off_xy_m * 1000:.1f} mm)")
            self.phase = ResetPhase.DONE
            return ResetStep(ResetAction.DONE)
        return ResetStep(ResetAction.NONE)

    def placed(self, now_s: float) -> None:
        """블록을 놓는 데 성공했을 때 부른다."""
        self.phase = ResetPhase.SETTLING
        self._placed_s = now_s

    def fail(self, reason: str) -> ResetStep:
        """호출하는 쪽이 실패를 알릴 때(블록 놓기 실패 등)."""
        return self._fail(reason)

    def _fail(self, reason: str) -> ResetStep:
        self.phase = ResetPhase.FAILED
        return ResetStep(ResetAction.FAILED, reason)
