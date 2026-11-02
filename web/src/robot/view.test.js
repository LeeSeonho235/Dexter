import { describe, expect, it } from 'vitest'
import { barFraction, controlStatusText, fmtStats, fmtVec } from './view.js'

describe('barFraction', () => {
  it('0~max 사이는 비율, 밖은 자른다', () => {
    expect(barFraction(0.075, 0.15)).toBeCloseTo(0.5)
    expect(barFraction(-0.1, 0.15)).toBe(0)
    expect(barFraction(0.5, 0.15)).toBe(1)
  })
  it('값이 없거나 max가 0 이하이면 null', () => {
    expect(barFraction(null, 0.15)).toBeNull()
    expect(barFraction(0.1, 0)).toBeNull()
  })
})

describe('format', () => {
  it('fmtVec', () => {
    expect(fmtVec([0.23, 0, 0.08])).toBe('(0.230, 0.000, 0.080)')
    expect(fmtVec(null)).toBe('—')
  })
  it('fmtStats: 마지막 / 평균 / 최대', () => {
    expect(fmtStats({ last: 0.5, mean: 0.4, max: 1.25, n: 10 })).toBe('0.50 / 0.40 / 1.25')
    expect(fmtStats({ last: null, mean: null, max: null, n: 0 })).toBe('—')
    expect(fmtStats(undefined)).toBe('—')
  })
})

describe('controlStatusText', () => {
  const base = { controlling: false, consecutive_frames: 0, start_frames: 3, target_m: null }
  it('연결이 없으면 그렇게 알려준다', () => {
    expect(controlStatusText(base, false)).toBe('서버에 연결되지 않음')
    expect(controlStatusText(undefined, true)).toBe('서버 상태 기다리는 중')
  })
  it('시작 조건 진행 상황', () => {
    expect(controlStatusText({ ...base, consecutive_frames: 2 }, true)).toContain('2/3')
  })
  it('따라가는 중 / 손이 사라짐 / 시작 전', () => {
    expect(controlStatusText({ ...base, controlling: true, target_m: [0.2, 0, 0.1] }, true)).toBe('손을 따라가는 중')
    expect(controlStatusText({ ...base, target_m: [0.2, 0, 0.1] }, true)).toContain('마지막 목표')
    expect(controlStatusText(base, true)).toContain('시작 전')
  })
})
