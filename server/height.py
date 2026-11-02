"""키보드 높이 명령(↑/↓)을 "흐른 시간 × 속도"로 바꾸는 순수 계산(2-5). 시계에 의존하지 않고 시각(s)을 받는다.

브라우저는 키를 누르고 있는 동안 20Hz로 {dir, slow}를 보내고, 키를 놓으면 dir 0을 한 번 보낸다. 이 클래스는 받은 명령을 시각과
함께 기록해 두었다가 advance(now)가 불릴 때 그 사이 높이가 얼마나 변해야 하는지(m)를 돌려준다.

- 이동량 = 방향(+1 위, -1 아래) × 속도(m/s) × 흐른 시간(s). 틱 횟수가 아니라 시계 기준이다.
- 누른 시각부터(첫 메시지), 놓은 시각까지(dir 0 메시지) 정확히 센다. 제어 틱 간격에 따라 달라지지 않는다.
- 마지막 명령을 받은 지 timeout_s가 지나면 dir 0으로 본다. 그 시점까지만 센다.
- 한 번에 세는 경과 시간은 max_dt_s로 자른다(제어 스레드가 멈칫해도 높이가 뛰지 않게).
스레드 안전하지 않다: 부르는 쪽(ControlWorker)이 락 안에서 쓴다.
"""

from dataclasses import dataclass


@dataclass
class _Command:
    direction: int  # -1 아래, 0 정지, 1 위
    slow: bool
    recv_s: float


class HeightCommands:
    def __init__(self, speed_m_s: float, slow_speed_m_s: float, timeout_s: float, max_dt_s: float) -> None:
        self.speed_m_s = speed_m_s
        self.slow_speed_m_s = slow_speed_m_s
        self.timeout_s = timeout_s
        self.max_dt_s = max_dt_s
        self._events: list[_Command] = []  # 아직 적용 안 한 명령(받은 순서)
        self._state = _Command(0, False, float("-inf"))  # 지금 적용 중인 명령
        self._cursor_s: float | None = None  # 여기까지 셌다

    def command(self, direction: int, slow: bool, recv_s: float) -> None:
        """받은 명령을 기록한다(받은 순서대로 넣는다). direction은 -1, 0, 1."""
        self._events.append(_Command(direction, slow, recv_s))

    def reset(self) -> None:
        """기록과 상태를 비운다(조종 시작 전에는 키보드 높이를 무시하려고 쓴다)."""
        self._events.clear()
        self._state = _Command(0, False, float("-inf"))
        self._cursor_s = None

    def advance(self, now_s: float) -> float:
        """지난번 advance 이후 now_s까지 높이가 변해야 할 양(m). 위가 +."""
        if self._cursor_s is None:
            # 처음 부르는 때: 그 사이에 누른 명령이 있어도 누른 시각부터 셀 수 있게 상한만큼 앞에서 시작한다.
            # (상태가 정지라서 명령이 없으면 이 구간은 0이다.)
            self._cursor_s = now_s - self.max_dt_s
        # 제어 스레드가 오래 멈췄다면 그 시간은 세지 않는다(뛰지 않게). 명령 기록은 계속 반영한다.
        cursor = max(self._cursor_s, now_s - self.max_dt_s)
        total_m = 0.0
        events, self._events = self._events, []
        for event in events:
            if event.recv_s > now_s:  # 시계가 다른 곳에서 온 미래 시각은 다음에
                self._events.append(event)
                continue
            total_m += self._integrate(cursor, event.recv_s)
            cursor = max(cursor, min(event.recv_s, now_s))
            self._state = event
        total_m += self._integrate(cursor, now_s)
        self._cursor_s = now_s
        return total_m

    def effective(self, now_s: float) -> tuple[int, bool]:
        """지금 유효한 (방향, slow). 타임아웃이 지났거나 명령이 없으면 (0, False). 아직 반영 안 한 명령도 본다."""
        state = self._state
        for event in self._events:
            if event.recv_s <= now_s:
                state = event
        if state.direction != 0 and now_s - state.recv_s <= self.timeout_s:
            return state.direction, state.slow
        return 0, False

    def _integrate(self, start_s: float, end_s: float) -> float:
        """현재 상태(self._state)로 [start_s, end_s]에서 변하는 높이. 타임아웃 시각에서 끊는다."""
        state = self._state
        if state.direction == 0 or end_s <= start_s:
            return 0.0
        end_s = min(end_s, state.recv_s + self.timeout_s)
        begin_s = max(start_s, state.recv_s)
        if end_s <= begin_s:
            return 0.0
        speed = self.slow_speed_m_s if state.slow else self.speed_m_s
        return state.direction * speed * (end_s - begin_s)
