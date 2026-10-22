import { describe, expect, it } from 'vitest'
import { createRateCounter, createRateLimiter } from './rate.js'

// 60FPS 입력 시각 (ms). jitterMs만큼 앞뒤로 흔든다.
function frameTimes(seconds, fps = 60, jitterMs = 0) {
  const n = Math.round(seconds * fps)
  return Array.from({ length: n }, (_, i) => i * (1000 / fps) + (i % 2 ? jitterMs : -jitterMs))
}

describe('createRateLimiter', () => {
  it('60FPS 입력을 30Hz로 줄인다', () => {
    const shouldSend = createRateLimiter(30)
    const sent = frameTimes(10).filter((t) => shouldSend(t)).length
    expect(sent).toBe(300)
  })

  it('입력 시각이 흔들려도 30Hz를 넘지 않고 크게 떨어지지 않는다', () => {
    const shouldSend = createRateLimiter(30)
    const sent = frameTimes(10, 60, 3).filter((t) => shouldSend(t)).length
    expect(sent).toBeLessThanOrEqual(301)
    expect(sent).toBeGreaterThanOrEqual(290)
  })

  it('어느 1초 구간에서도 31개를 넘지 않는다', () => {
    const shouldSend = createRateLimiter(30)
    const sentTimes = frameTimes(5, 120, 2).filter((t) => shouldSend(t))
    for (const start of sentTimes) {
      const inWindow = sentTimes.filter((t) => t >= start && t < start + 1000).length
      expect(inWindow).toBeLessThanOrEqual(31)
    }
  })

  it('오래 쉬었다 와도 몰아서 보내지 않는다', () => {
    const shouldSend = createRateLimiter(30)
    expect(shouldSend(0)).toBe(true)
    expect(shouldSend(5000)).toBe(true)
    expect(shouldSend(5001)).toBe(false)
    expect(shouldSend(5020)).toBe(false)
    expect(shouldSend(5034)).toBe(true)
  })

  it('입력이 30Hz보다 느리면 전부 보낸다', () => {
    const shouldSend = createRateLimiter(30)
    const sent = frameTimes(2, 20).filter((t) => shouldSend(t)).length
    expect(sent).toBe(40)
  })
})

describe('createRateCounter', () => {
  it('최근 1초 동안의 개수', () => {
    const c = createRateCounter(1000)
    for (let t = 0; t < 2000; t += 100) c.add(t) // 10Hz로 2초
    expect(c.perSecond(1950)).toBe(10)
    expect(c.perSecond(2850)).toBe(1) // 구간 (1850, 2850]: 1900만 남음
    expect(c.perSecond(2900)).toBe(0) // 구간 (1900, 2900]: 정확히 1초 전인 1900은 빠짐
    expect(c.perSecond(5000)).toBe(0)
  })

  it('reset', () => {
    const c = createRateCounter(1000)
    c.add(0)
    c.reset()
    expect(c.perSecond(10)).toBe(0)
  })
})
