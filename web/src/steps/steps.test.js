import { describe, expect, it } from 'vitest'
import { busyReason, isBusy, STEP_IDS } from './steps.js'

describe('단계 목록', () => {
  it('녹화 → 학습 → 테스트·교정 순서', () => {
    expect(STEP_IDS).toEqual(['record', 'train', 'test'])
  })
})

describe('isBusy / busyReason', () => {
  it.each([
    ['서버 상태 없음(연결 전)', undefined, undefined, false],
    ['녹화 중', { recording: true }, { status: 'idle' }, true],
    ['AUTO 불러오는 중', { recording: false }, { status: 'loading' }, true],
    ['AUTO 실행(교정 포함) 중', { recording: false }, { status: 'running', owner: 'HUMAN' }, true],
    ['AUTO 끝남', { recording: false }, { status: 'success' }, false],
    ['아무것도 안 함', { recording: false }, { status: 'idle' }, false],
  ])('%s', (_label, record, auto, expected) => {
    expect(isBusy(record, auto)).toBe(expected)
    expect(busyReason(record, auto) !== '').toBe(expected)
  })

  it('녹화 이유가 AUTO 이유보다 먼저 나온다', () => {
    expect(busyReason({ recording: true }, { status: 'running' })).toContain('녹화')
  })
})
