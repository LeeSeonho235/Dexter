"""집게 열고 닫기 판단: pinch_ratio에 두 기준을 둔다(히스테리시스).

  pinch_ratio < close_below  → 닫기
  pinch_ratio > open_above   → 열기
  그 사이, 값이 없음(null), 손이 안 보임 → 직전 상태 유지
기준 근처에서 값이 조금 흔들려도 집게가 깜빡이지 않게 하려는 것이다.
"""


class GripperLatch:
    def __init__(self, close_below: float, open_above: float, closed: bool = False) -> None:
        if close_below >= open_above:
            raise ValueError("close_below must be < open_above")
        self.close_below = close_below
        self.open_above = open_above
        self.closed = closed

    def update(self, pinch_ratio: float | None) -> bool:
        """pinch_ratio(손 크기 대비 엄지-검지 거리, 단위 없음)를 넣고 닫힘 여부를 돌려준다."""
        if pinch_ratio is not None:
            if pinch_ratio < self.close_below:
                self.closed = True
            elif pinch_ratio > self.open_above:
                self.closed = False
        return self.closed
