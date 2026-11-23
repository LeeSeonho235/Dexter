import { describe, expect, it } from 'vitest'
import { estimateMinutes, estimateText, parseSteps, STEPS_DEFAULT, STEPS_MAX, STEPS_MIN, stepsError } from './trainTime.js'

describe('parseSteps', () => {
  it.each([
    ['20000', 20000],
    [' 1000 ', 1000],
    [5000, 5000],
  ])('%s → %s', (input, expected) => expect(parseSteps(input)).toBe(expected))

  it.each([[''], ['abc'], ['1e4'], ['2000.5'], ['-1000'], [null], [undefined]])('정수가 아니면 null: %s', (input) => {
    expect(parseSteps(input)).toBeNull()
  })
})

describe('stepsError', () => {
  it('기본값과 범위 양 끝은 통과한다', () => {
    for (const v of [STEPS_DEFAULT, STEPS_MIN, STEPS_MAX]) expect(stepsError(v)).toBeNull()
  })

  it.each([['999'], ['100001'], ['0'], [''], ['abc'], ['1500.5']])('범위 밖이거나 정수가 아니면 이유를 돌려준다: %s', (v) => {
    expect(stepsError(v)).toBeTruthy()
  })
})

describe('예상 시간', () => {
  it('20000스텝 × 0.14초 = 2800초 = 46.7분 → 올려서 47분', () => {
    expect(estimateMinutes('20000')).toBe(47)
    expect(estimateText('20000')).toBe('약 47분 (설치·확인 포함 약 +10분)')
  })

  it('1000스텝은 140초 → 3분, 100000스텝은 14000초 → 234분', () => {
    expect(estimateMinutes(1000)).toBe(3)
    expect(estimateMinutes(100000)).toBe(234)
  })

  it('입력이 안전하지 않으면 null', () => {
    expect(estimateMinutes('500')).toBeNull()
    expect(estimateText('abc')).toBeNull()
  })
})
