import { describe, expect, it } from 'vitest'
import { createTrailingThrottle } from './throttle.js'

// 시계와 타이머를 직접 움직이는 테스트 도구
function makeClock() {
  let nowMs = 0
  let timers = [] // { id, at, fn }
  let nextId = 1
  return {
    now: () => nowMs,
    setTimer: (fn, ms) => {
      const id = nextId++
      timers.push({ id, at: nowMs + ms, fn })
      return id
    },
    clearTimer: (id) => {
      timers = timers.filter((t) => t.id !== id)
    },
    pendingTimers: () => timers.length,
    // 시각을 toMs까지 진행하며 그 사이 타이머를 순서대로 실행
    advanceTo(toMs) {
      for (;;) {
        const due = timers.filter((t) => t.at <= toMs).sort((a, b) => a.at - b.at)[0]
        if (!due) break
        timers = timers.filter((t) => t !== due)
        nowMs = Math.max(nowMs, due.at)
        due.fn()
      }
      nowMs = toMs
    },
  }
}

function setup(maxHz = 15) {
  const clock = makeClock()
  const sent = [] // { value, at }
  const throttle = createTrailingThrottle((value) => sent.push({ value, at: clock.now() }), {
    maxHz,
    now: clock.now,
    setTimer: clock.setTimer,
    clearTimer: clock.clearTimer,
  })
  return { clock, sent, throttle }
}

describe('createTrailingThrottle', () => {
  it('첫 값은 바로 보낸다', () => {
    const { sent, throttle } = setup()
    throttle.push('a')
    expect(sent.map((s) => s.value)).toEqual(['a'])
  })

  it('제한 안에 온 값은 기다렸다가, 허용 시각에 최신 값 하나만 보낸다', () => {
    const { clock, sent, throttle } = setup(15)
    throttle.push(1)
    clock.advanceTo(10)
    throttle.push(2)
    clock.advanceTo(20)
    throttle.push(3)
    expect(sent.map((s) => s.value)).toEqual([1])
    clock.advanceTo(1000 / 15 + 1)
    expect(sent.map((s) => s.value)).toEqual([1, 3]) // 2는 3으로 대체됐다
  })

  it('마지막 값은 반드시 보낸다(드래그가 끝난 자리)', () => {
    const { clock, sent, throttle } = setup(15)
    for (let i = 0; i < 20; i++) {
      clock.advanceTo(i * 5) // 200Hz로 계속 들어옴
      throttle.push(i)
    }
    clock.advanceTo(2000)
    expect(sent[sent.length - 1].value).toBe(19)
  })

  it('보내는 간격이 항상 1000/15 ms 이상이고 어느 1초 구간에서도 16개를 넘지 않는다', () => {
    const { clock, sent, throttle } = setup(15)
    for (let t = 0; t <= 10000; t += 4) {
      clock.advanceTo(t)
      throttle.push(t) // 250Hz 입력
    }
    clock.advanceTo(11000)
    const times = sent.map((s) => s.at)
    for (let i = 1; i < times.length; i++) {
      expect(times[i] - times[i - 1]).toBeGreaterThanOrEqual(1000 / 15 - 1e-9)
    }
    // 간격이 정확히 1000/15 ms면 1초 구간 양 끝을 포함해 16개가 들어간다(15개 간격 = 1000 ms). net/rate.test.js도 한계 + 1을 허용한다.
    for (let i = 0; i < times.length; i++) {
      const inWindow = times.filter((t) => t >= times[i] && t < times[i] + 1000 + 1e-6).length
      expect(inWindow).toBeLessThanOrEqual(16)
    }
    expect(times.length).toBeGreaterThanOrEqual(140) // 10초 동안 약 150개
  })

  it('오래 쉬었다 오면 바로 보내고, 밀린 것을 몰아 보내지 않는다', () => {
    const { clock, sent, throttle } = setup(15)
    throttle.push('a')
    clock.advanceTo(5000)
    throttle.push('b')
    expect(sent.map((s) => s.value)).toEqual(['a', 'b'])
    expect(clock.pendingTimers()).toBe(0)
  })

  it('cancel하면 기다리던 값과 타이머가 없어진다', () => {
    const { clock, sent, throttle } = setup(15)
    throttle.push(1)
    throttle.push(2)
    expect(clock.pendingTimers()).toBe(1)
    throttle.cancel()
    expect(clock.pendingTimers()).toBe(0)
    clock.advanceTo(1000)
    expect(sent.map((s) => s.value)).toEqual([1])
  })

  it('타이머가 조금 일찍 깨도 허용 시각 전에는 보내지 않는다', () => {
    const clock = makeClock()
    const sent = []
    const early = []
    const throttle = createTrailingThrottle((v) => sent.push({ v, at: clock.now() }), {
      maxHz: 10,
      now: clock.now,
      setTimer: (fn, ms) => {
        early.push(fn)
        return clock.setTimer(() => {}, ms)
      },
      clearTimer: clock.clearTimer,
    })
    throttle.push('a')
    throttle.push('b')
    clock.advanceTo(50)
    early[0]() // 100ms 전에 타이머가 깨어났다
    expect(sent.map((s) => s.v)).toEqual(['a'])
    clock.advanceTo(100)
    early[early.length - 1]()
    expect(sent.map((s) => s.v)).toEqual(['a', 'b'])
  })
})
