// 빈도 제한 + "마지막 값은 반드시 보낸다". 시각은 ms.
//
// net/rate.js의 createRateLimiter는 제한을 넘는 값을 그냥 버린다. 손 좌표는 계속 새 값이 오니 괜찮지만,
// 마우스 드래그의 마지막 값을 버리면 화면이 마지막 위치에서 멈추지 않으므로 여기서는 최신 값 하나를 기억해 두었다가
// 허용 시각이 되면 보낸다. 보내는 간격은 항상 1000/maxHz ms 이상이다.

export function createTrailingThrottle(
  send,
  { maxHz, now = () => performance.now(), setTimer = setTimeout, clearTimer = clearTimeout },
) {
  const intervalMs = 1000 / maxHz
  let nextAllowedMs = -Infinity
  let pending = null // { value } 아직 안 보낸 최신 값
  let timer = null

  function fire(nowMs) {
    const { value } = pending
    pending = null
    nextAllowedMs = nowMs + intervalMs
    send(value)
  }

  function tryFlush() {
    if (pending === null) return
    const nowMs = now()
    if (nowMs >= nextAllowedMs) {
      fire(nowMs)
    } else if (timer === null) {
      timer = setTimer(() => {
        timer = null
        tryFlush() // 타이머가 조금 일찍 깨도 아직 이르면 다시 기다린다
      }, nextAllowedMs - nowMs)
    }
  }

  return {
    // 최신 값을 넣는다. 아직 못 보낸 이전 값은 이 값으로 대체된다.
    push(value) {
      pending = { value }
      tryFlush()
    },
    // 기다리는 값과 타이머를 버린다(연결이 끝날 때).
    cancel() {
      pending = null
      if (timer !== null) clearTimer(timer)
      timer = null
    },
  }
}
