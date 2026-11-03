// 녹화 상자 표시 규칙: 버튼 활성 조건, 버튼 → 메시지, 상태 문구, 정지 구간 표시. 서버 state.record 형식은 docs/08-recording.md.
import { describe, expect, it } from 'vitest'
import { BUTTON_IDS, buttonStates, commandFor, eventLevel, formatElapsed, statusText, stillIsHigh, stillText, summaryRows } from './view.js'

const IDLE = {
  recording: false,
  resetting: false,
  block_ready: false,
  current_seed: null,
  next_seed: 0,
  seeds_left: 100,
  saved_success: 0,
  saved_fail: 0,
  saved_total: 0,
  pending_writes: 0,
  save_error_count: 0,
  last_episode: null,
  ticks: 0,
  elapsed_s: 0,
  max_s: 60,
  still_ratio: null,
  still_longest_s: null,
  success_condition_met: false,
  last_event: null,
  tick_ms: null,
}
const READY = { ...IDLE, block_ready: true, current_seed: 0, next_seed: 1 }
const RECORDING = { ...READY, recording: true, block_ready: false, ticks: 45, elapsed_s: 1.5, still_ratio: 0.2, still_longest_s: 0.4 }
const WITH_SAVED = { ...IDLE, saved_success: 3, saved_fail: 1, saved_total: 4, last_episode: { index: 3, seed: 3, success: true, ticks: 900 }, next_seed: 4 }

function enabled(record, connected = true) {
  const s = buttonStates(record, connected)
  return Object.fromEntries(BUTTON_IDS.map((id) => [id, s[id].enabled]))
}

describe('buttonStates', () => {
  it('연결이 없거나 record가 없으면 모두 꺼진다(이유 문구 포함)', () => {
    for (const record of [IDLE, READY, undefined, null]) {
      expect(Object.values(enabled(record, false)).some(Boolean)).toBe(false)
    }
    expect(Object.values(enabled(undefined)).some(Boolean)).toBe(false)
    expect(buttonStates(IDLE, false).start.reason).toContain('연결')
    expect(buttonStates(undefined, true).start.reason).toContain('녹화 상태')
  })

  it('처음에는 리셋만 가능하다(리셋 전에는 시작할 수 없다)', () => {
    expect(enabled(IDLE)).toEqual({ start: false, stop_success: false, stop_fail: false, discard: false, reset: true, delete_last: false })
    expect(buttonStates(IDLE, true).start.reason).toContain('리셋')
  })

  it('리셋이 끝나면 시작할 수 있다', () => {
    expect(enabled(READY)).toMatchObject({ start: true, reset: true, stop_success: false, discard: false })
  })

  it('녹화 중에는 끝/취소만 가능하고 리셋·시작·삭제는 막힌다', () => {
    expect(enabled({ ...RECORDING, last_episode: WITH_SAVED.last_episode })).toEqual({
      start: false, stop_success: true, stop_fail: true, discard: true, reset: false, delete_last: false,
    })
  })

  it('리셋 중에는 시작·리셋·삭제가 막힌다', () => {
    expect(enabled({ ...WITH_SAVED, resetting: true })).toMatchObject({ start: false, reset: false, delete_last: false })
  })

  it('저장된 에피소드가 있고 쓰는 중이 아닐 때만 삭제할 수 있다', () => {
    expect(enabled(WITH_SAVED).delete_last).toBe(true)
    expect(enabled({ ...WITH_SAVED, pending_writes: 1 }).delete_last).toBe(false)
    expect(enabled(IDLE).delete_last).toBe(false)
  })

  it('시연 시드를 모두 쓰면 리셋이 막힌다', () => {
    expect(enabled({ ...IDLE, next_seed: null, seeds_left: 0 }).reset).toBe(false)
  })
})

describe('commandFor (버튼 → 서버 메시지)', () => {
  it('녹화 버튼은 record 메시지가 된다', () => {
    for (const action of ['start', 'stop_success', 'stop_fail', 'discard']) {
      expect(commandFor(action, IDLE)).toEqual({ type: 'record', action })
    }
  })

  it('블록 리셋은 reset 메시지다(필드 없음 — 시드는 서버가 정한다)', () => {
    expect(commandFor('reset', IDLE)).toEqual({ type: 'reset' })
  })

  it('삭제는 화면이 아는 가장 최근 에피소드 번호를 같이 보낸다', () => {
    expect(commandFor('delete_last', WITH_SAVED)).toEqual({ type: 'record', action: 'delete_last', episode: 3 })
    expect(commandFor('delete_last', { ...IDLE, last_episode: { index: 0 } })).toEqual({ type: 'record', action: 'delete_last', episode: 0 })
  })

  it('번호를 모르면 삭제 메시지를 만들지 않는다', () => {
    expect(commandFor('delete_last', IDLE)).toBeNull()
    expect(commandFor('delete_last', null)).toBeNull()
    expect(commandFor('delete_last', { last_episode: { index: '3' } })).toBeNull()
    expect(commandFor('delete_last', { last_episode: { index: -1 } })).toBeNull()
  })

  it('모르는 버튼 id는 null', () => {
    expect(commandFor('explode', IDLE)).toBeNull()
    expect(commandFor(undefined, IDLE)).toBeNull()
  })
})

describe('상태 문구와 표시', () => {
  it('상태 한 줄', () => {
    expect(statusText(IDLE, false)).toContain('연결되지 않음')
    expect(statusText(undefined, true)).toContain('모름')
    expect(statusText(IDLE, true)).toContain('블록 리셋')
    expect(statusText({ ...IDLE, resetting: true }, true)).toContain('리셋 중')
    expect(statusText(READY, true)).toContain('시드 0')
    expect(statusText(RECORDING, true)).toContain('녹화 중')
  })

  it('경과 시간', () => {
    expect(formatElapsed(12.34, 60)).toBe('12.3 / 60 s')
    expect(formatElapsed(0, 60)).toBe('0.0 / 60 s')
    expect(formatElapsed(null, 60)).toBe('—')
  })

  it('정지 구간 문구와 경고 기준', () => {
    expect(stillText(0.314, 1.23)).toBe('31% (가장 긴 정지 1.2 s)')
    expect(stillText(null, null)).toBe('—')
    expect(stillIsHigh(0.49, 2.9)).toBe(false)
    expect(stillIsHigh(0.5, 0)).toBe(true)
    expect(stillIsHigh(0.1, 3.0)).toBe(true)
    expect(stillIsHigh(null, null)).toBe(false)
  })

  it('이벤트 단계는 info/warn/error만(그 밖은 info)', () => {
    expect(eventLevel({ level: 'error' })).toBe('error')
    expect(eventLevel({ level: 'warn' })).toBe('warn')
    expect(eventLevel({ level: 'x' })).toBe('info')
    expect(eventLevel(null)).toBe('info')
  })

  it('표: 저장 수, 녹화 중 표시값, record가 없으면 빈 표', () => {
    const rows = Object.fromEntries(summaryRows(RECORDING))
    expect(rows['경과 시간']).toBe('1.5 / 60 s')
    expect(rows['기록한 틱 수']).toBe('45')
    expect(rows['정지 구간 비율']).toBe('20% (가장 긴 정지 0.4 s)')
    const saved = Object.fromEntries(summaryRows(WITH_SAVED))
    expect(saved['저장된 에피소드 (성공 / 실패)']).toBe('3 / 1')
    expect(saved['마지막 저장']).toContain('에피소드 3')
    expect(saved['경과 시간']).toBe('—')
    expect(summaryRows(undefined)).toEqual([])
    expect(summaryRows({})).not.toHaveLength(0) // 일부 필드가 없어도 깨지지 않는다
  })

  it('틱 간격 표시', () => {
    const rows = Object.fromEntries(summaryRows({ ...IDLE, tick_ms: { recent_mean_ms: 33.34, recent_max_ms: 47.2 } }))
    expect(rows['틱 간격 ms (최근 평균 / 최대)']).toBe('33.3 / 47.2')
  })
})
