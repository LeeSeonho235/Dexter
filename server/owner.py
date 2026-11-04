"""제어 주체(4-1): 지금 로봇의 ctrl을 쓰는 것은 누구인가. 한 순간에 하나뿐이다.

IDLE : 손·키보드 입력(ControlWorker)이 ctrl을 쓴다(기존 동작).
AUTO : 학습된 정책(server/auto.py, 시뮬레이션 스레드)이 ctrl을 쓴다. ControlWorker의 손·키보드 입력은 버려지고 SimWorker는 ControlWorker의 ctrl을 적용하지 않는다.
HUMAN: 4-2(손으로 넘겨받기)에서 쓴다. 이번에는 전환 자리만 있고 켜지 않았다(FUTURE_TRANSITIONS).

전환은 락 안에서 "지금 값이 기대한 값일 때만" 바꾼다. 전환 순간에 로봇이 튀지 않게 하는 일(ctrl 이어받기)은 호출하는 쪽의 몫이다
(AUTO가 끝나면 AutoRunner가 마지막 ctrl을 SimWorker와 ControlWorker에 같이 넘긴다).
"""

import threading
from enum import Enum


class Owner(str, Enum):
    IDLE = "IDLE"
    AUTO = "AUTO"
    HUMAN = "HUMAN"


ALLOWED_TRANSITIONS = frozenset({(Owner.IDLE, Owner.AUTO), (Owner.AUTO, Owner.IDLE)})
# 4-2에서 켤 전환(지금은 거부): AUTO → HUMAN(손을 올려 넘겨받기), HUMAN → AUTO(손을 내려 돌려주기), HUMAN → IDLE
FUTURE_TRANSITIONS = frozenset({(Owner.AUTO, Owner.HUMAN), (Owner.HUMAN, Owner.AUTO), (Owner.HUMAN, Owner.IDLE), (Owner.IDLE, Owner.HUMAN)})


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
