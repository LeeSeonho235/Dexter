// 빈도 제한과 초당 개수 세기. 시각은 모두 ms (performance.now() 같은 단조 증가 시계).

// 최대 maxHz로 제한한다. 호출할 때마다 now를 넣고, true면 보낸다.
// 다음 허용 시각을 "이전 허용 시각 + 간격"으로 잡아서, 60FPS 입력이 조금씩 흔들려도 평균이 30Hz 아래로 떨어지지 않게 한다.
// 오래 쉬었다가 다시 오면 밀린 만큼 몰아서 보내지 않는다.
export function createRateLimiter(maxHz) {
  const intervalMs = 1000 / maxHz
  let nextAllowedMs = -Infinity
  return function shouldSend(nowMs) {
    if (nowMs < nextAllowedMs) return false
    // 한 간격 이상 늦었으면(처음 호출, 오래 쉼) 지금부터 다시 센다.
    nextAllowedMs = nowMs - nextAllowedMs >= intervalMs ? nowMs + intervalMs : nextAllowedMs + intervalMs
    return true
  }
}

// 최근 windowMs 동안 일어난 횟수를 초당 개수로 돌려준다.
export function createRateCounter(windowMs = 1000) {
  let times = []
  function prune(nowMs) {
    const from = nowMs - windowMs
    let i = 0
    while (i < times.length && times[i] <= from) i++
    if (i > 0) times = times.slice(i)
  }
  return {
    add(nowMs) {
      times.push(nowMs)
      prune(nowMs)
    },
    perSecond(nowMs) {
      prune(nowMs)
      return (times.length * 1000) / windowMs
    },
    reset() {
      times = []
    },
  }
}
