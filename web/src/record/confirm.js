// 2단계 확인(삭제처럼 되돌릴 수 없는 버튼용): 첫 번째 누름은 "확인 상태"로 바꾸기만 하고, timeoutMs 안의 두 번째 누름이 실제 실행이다.
// 시간이 지나면 확인 상태가 저절로 풀린다. React에 의존하지 않는다.

import { CONFIRM_TIMEOUT_MS } from './config.js'

export function createConfirm({
  timeoutMs = CONFIRM_TIMEOUT_MS,
  setTimer = (fn, ms) => setTimeout(fn, ms),
  clearTimer = (id) => clearTimeout(id),
  onChange = () => {},
} = {}) {
  let armed = false
  let timer = null

  function set(value) {
    if (armed === value) return
    armed = value
    onChange(armed)
  }

  function stopTimer() {
    if (timer !== null) clearTimer(timer)
    timer = null
  }

  return {
    // 버튼을 눌렀다. 두 번째 누름이면 true(실행해도 된다), 첫 번째면 false(확인 상태로 바뀜).
    press() {
      if (armed) {
        stopTimer()
        set(false)
        return true
      }
      set(true)
      stopTimer()
      timer = setTimer(() => {
        timer = null
        set(false)
      }, timeoutMs)
      return false
    },
    // 다른 버튼을 누르거나 상태가 바뀌어 확인을 취소한다.
    cancel() {
      stopTimer()
      set(false)
    },
    isArmed() {
      return armed
    },
  }
}
