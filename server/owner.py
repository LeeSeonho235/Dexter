"""제어 주체(4-1, 4-2): 지금 로봇의 ctrl을 쓰는 것은 누구인가. 한 순간에 하나뿐이다.

IDLE      : 손·키보드 입력(ControlWorker)이 ctrl을 쓴다(기존 동작).
AUTO      : 학습된 정책(server/auto.py, 시뮬레이션 스레드)이 ctrl을 쓴다. ControlWorker의 입력은 버려지고 SimWorker는 ControlWorker의 ctrl을 적용하지 않는다.
HUMAN_WAIT: 사람이 넘겨받기를 요청했고 조종 손이 3프레임 연속 보이기를 기다리는 중. 아무도 ctrl을 쓰지 않는다(로봇은 직전 ctrl 그대로 멈춤).
HUMAN     : 사람이 손으로 조종한다(ControlWorker가 ctrl을 쓴다). 에피소드(AutoEpisode)는 기록과 성공 판정만 한다.

허용 전환: IDLE→AUTO, AUTO→HUMAN_WAIT→HUMAN→AUTO(돌려주기). 에피소드가 끝나면 AUTO/HUMAN_WAIT/HUMAN → IDLE.
전환은 락 안에서 "지금 값이 기대한 값일 때만" 바꾼다. 전환 순간에 로봇이 튀지 않게 하는 일(ctrl 이어받기)은 호출하는 쪽의 몫이다.
"""

import threading
from enum import Enum


class Owner(str, Enum):
    IDLE = "IDLE"
    AUTO = "AUTO"
    HUMAN_WAIT = "HUMAN_WAIT"
    HUMAN = "HUMAN"


ALLOWED_TRANSITIONS = frozenset(
    {
        (Owner.IDLE, Owner.AUTO),
        (Owner.AUTO, Owner.IDLE),
        (Owner.AUTO, Owner.HUMAN_WAIT),
        (Owner.HUMAN_WAIT, Owner.HUMAN),
        (Owner.HUMAN, Owner.AUTO),
        (Owner.HUMAN_WAIT, Owner.IDLE),
        (Owner.HUMAN, Owner.IDLE),
    }
)
# 허용하지 않는 전환(건너뛰기와 거꾸로 가기). 테스트가 모두 거부되는지 확인한다.
REFUSED_TRANSITIONS = frozenset(
    {
        (Owner.IDLE, Owner.HUMAN_WAIT),
        (Owner.IDLE, Owner.HUMAN),
        (Owner.AUTO, Owner.HUMAN),
        (Owner.HUMAN_WAIT, Owner.AUTO),
        (Owner.HUMAN, Owner.HUMAN_WAIT),
    }
)


class ControlOwner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = Owner.IDLE

    @property
    def current(self) -> Owner:
        return self._current

    @property
    def is_auto(self) -> bool:
        return self._current is Owner.AUTO

    @property
    def is_busy(self) -> bool:
        """IDLE이 아니면 True(AUTO 에피소드가 로봇을 쓰는 중). 녹화와 블록 리셋은 busy일 때 받지 않는다."""
        return self._current is not Owner.IDLE

    @property
    def control_writes_ctrl(self) -> bool:
        """ControlWorker가 정한 ctrl을 SimWorker가 적용해도 되는가(IDLE 또는 HUMAN)."""
        return self._current in (Owner.IDLE, Owner.HUMAN)

    def transition(self, expected: Owner, new: Owner) -> bool:
        """지금 값이 expected일 때만 new로 바꾼다. 허용되지 않은 전환이거나 값이 다르면 False(바꾸지 않는다)."""
        if (expected, new) not in ALLOWED_TRANSITIONS:
            return False
        with self._lock:
            if self._current is not expected:
                return False
            self._current = new
            return True

    def try_acquire_auto(self) -> bool:
        return self.transition(Owner.IDLE, Owner.AUTO)

    def release_auto(self) -> bool:
        return self.transition(Owner.AUTO, Owner.IDLE)

    def release_to_idle(self) -> bool:
        """에피소드가 끝났다: 지금 주체가 AUTO, HUMAN_WAIT, HUMAN 중 무엇이든 IDLE로. 이미 IDLE이면 False."""
        with self._lock:
            if self._current is Owner.IDLE:
                return False
            self._current = Owner.IDLE
            return True
