// AUTO 상자 표시 규칙: 버튼 활성 조건, 버튼 → 메시지, 상태 문구(로딩 문구의 N초), 정책 프로세스 표시, 표 줄. 서버 state.auto 형식은 docs/12-auto.md.
import { describe, expect, it } from 'vitest'
import {
  AUTO_BUTTON_IDS,
  buttonStates,
  commandFor,
  eventLevel,
  formatElapsed,
  formatSeconds,
  policyText,
  statusClass,
  statusText,
  summaryRows,
} from './view.js'
import { LABELED_BUTTONS } from '../components/AutoPanel.jsx'

const POLICY_OFF = { state: 'off', idle_remaining_s: null }
const IDLE = {
  status: 'idle', owner: 'IDLE', seed: null, elapsed_s: null, max_s: 45, sample: null, chunk_pos: null, chunk_len: null, last_latency_ms: null,
  session_success: 0, session_attempts: 0, error: '', result: null, loading_s: null, load_estimate_s: 3, policy: POLICY_OFF, last_event: null,
}
const LOADING = { ...IDLE, status: 'loading', owner: 'AUTO', loading_s: 1.2 }
const RUNNING = {
  ...IDLE, status: 'running', owner: 'AUTO', seed: 28, elapsed_s: 12.3, sample: 369, chunk_pos: 68, chunk_len: 100, last_latency_ms: 141.4,
  policy: { state: 'on', idle_remaining_s: null },
}
const DONE = { ...IDLE, status: 'success', seed: 28, result: { outcome: 'success', seed: 28, time_s: 13.5 }, session_success: 1, session_attempts: 1, policy: { state: 'on', idle_remaining_s: 598 } }
const READY = { recording: false, resetting: false, block_ready: true, current_seed: 28 }
const NOT_READY = { recording: false, resetting: false, block_ready: false, current_seed: null }

function enabled(auto, record, connected = true) {
  const s = buttonStates(auto, record, connected)
  return Object.fromEntries(AUTO_BUTTON_IDS.map((id) => [id, s[id].enabled]))
}

describe('commandFor', () => {
  it('버튼 id를 auto 메시지로 바꾼다', () => {
    expect(commandFor('start')).toEqual({ type: 'auto', action: 'start' })
    expect(commandFor('stop')).toEqual({ type: 'auto', action: 'stop' })
    expect(commandFor('shutdown_policy')).toEqual({ type: 'auto', action: 'shutdown_policy' })
    expect(commandFor('go')).toBeNull()
    expect(commandFor(undefined)).toBeNull()
  })

  it('AutoPanel의 버튼 이름표가 버튼 종류와 같다', () => {
    expect([...LABELED_BUTTONS]).toEqual(AUTO_BUTTON_IDS)
  })
})

describe('buttonStates', () => {
  it('연결이 없거나 auto가 없으면 모두 꺼진다(이유 문구 포함)', () => {
    expect(Object.values(enabled(IDLE, READY, false)).some(Boolean)).toBe(false)
    expect(Object.values(enabled(undefined, READY)).some(Boolean)).toBe(false)
    expect(buttonStates(IDLE, READY, false).start.reason).toContain('연결')
    expect(buttonStates(undefined, READY, true).start.reason).toContain('AUTO 상태')
  })

  it('시작은 블록 리셋 직후에만 켜진다', () => {
    expect(enabled(IDLE, NOT_READY)).toEqual({ start: false, stop: false, shutdown_policy: false })
    expect(buttonStates(IDLE, NOT_READY, true).start.reason).toContain('리셋')
    expect(enabled(IDLE, READY).start).toBe(true)
    expect(enabled(IDLE, undefined).start).toBe(false)
  })

  it('녹화 중이거나 리셋 중에는 시작할 수 없다', () => {
    expect(enabled(IDLE, { ...READY, recording: true }).start).toBe(false)
    expect(enabled(IDLE, { ...READY, resetting: true }).start).toBe(false)
  })

  it('로딩과 실행 중에는 중지만 켜지고 시작과 프로세스 끄기는 꺼진다', () => {
    for (const auto of [LOADING, RUNNING]) {
      expect(enabled(auto, READY)).toEqual({ start: false, stop: true, shutdown_policy: false })
    }
  })

  it('끝난 뒤에는 정책 프로세스가 켜져 있을 때만 끄기 버튼이 켜진다', () => {
    expect(enabled(DONE, NOT_READY)).toEqual({ start: false, stop: false, shutdown_policy: true })
    expect(enabled({ ...DONE, policy: POLICY_OFF }, READY)).toEqual({ start: true, stop: false, shutdown_policy: false })
    expect(buttonStates({ ...DONE, policy: POLICY_OFF }, READY, true).shutdown_policy.reason).toContain('꺼져')
  })

  it('중지·시간 초과·오류 뒤에는 다시 리셋해야 시작할 수 있다(서버가 리셋 직후 표시를 소비한다)', () => {
    for (const status of ['stopped', 'timeout', 'error', 'success']) {
      expect(enabled({ ...IDLE, status }, NOT_READY).start).toBe(false)
    }
  })
})

describe('statusText', () => {
  it('로딩 문구에 측정한 N초가 들어간다', () => {
    expect(statusText({ ...LOADING, load_estimate_s: 3, loading_s: null }, READY, true)).toBe('정책 불러오는 중, 약 3초')
    expect(statusText({ ...LOADING, load_estimate_s: 5, loading_s: 2.4 }, READY, true)).toBe('정책 불러오는 중, 약 5초 (2초 지남)')
  })

  it('각 상태 문구', () => {
    expect(statusText(undefined, READY, true)).toContain('AUTO 상태를 모름')
    expect(statusText(IDLE, READY, false)).toContain('연결되지 않음')
    expect(statusText(IDLE, NOT_READY, true)).toContain('블록 리셋')
    expect(statusText(IDLE, READY, true)).toContain('시드 28')
    expect(statusText(RUNNING, READY, true)).toBe('● AUTO 실행 중: 시드 28')
    expect(statusText(DONE, READY, true)).toBe('성공! 시드 28, 13.5초')
    expect(statusText({ ...IDLE, status: 'timeout', seed: 29, result: { outcome: 'timeout', seed: 29, time_s: 45 } }, READY, true)).toContain('시간 초과: 시드 29, 45초')
    expect(statusText({ ...IDLE, status: 'stopped', seed: 30 }, READY, true)).toContain('중지함')
    expect(statusText({ ...IDLE, status: 'error', error: '정책 응답이 3.0초 안에 오지 않았습니다' }, READY, true)).toContain('오류: 정책 응답이')
  })

  it('상태별 CSS 클래스', () => {
    expect([LOADING, RUNNING, DONE, { ...IDLE, status: 'timeout' }, { ...IDLE, status: 'error' }, IDLE, undefined].map(statusClass)).toEqual([
      'running', 'running', 'success', 'fail', 'error', 'idle', 'idle',
    ])
  })
})

describe('policyText', () => {
  it('꺼짐 / 켜짐(남은 시간) / 불러오는 중 / 실행 중', () => {
    expect(policyText(IDLE, true)).toBe('정책 프로세스: 꺼짐')
    expect(policyText(DONE, true)).toBe('정책 프로세스: 켜짐 (9분 58초 후 종료)')
    expect(policyText({ ...DONE, policy: { state: 'on', idle_remaining_s: 42 } }, true)).toBe('정책 프로세스: 켜짐 (42초 후 종료)')
    expect(policyText(RUNNING, true)).toBe('정책 프로세스: 켜짐 (AUTO 실행 중)')
    expect(policyText({ ...IDLE, policy: { state: 'loading', idle_remaining_s: null } }, true)).toBe('정책 프로세스: 불러오는 중')
    expect(policyText(IDLE, false)).toBe('정책 프로세스: —')
    expect(policyText(undefined, true)).toBe('정책 프로세스: —')
  })

  it('formatSeconds', () => {
    expect(formatSeconds(0)).toBe('0초')
    expect(formatSeconds(59.4)).toBe('59초')
    expect(formatSeconds(600)).toBe('10분 0초')
    expect(formatSeconds(null)).toBe('—')
  })
})

describe('summaryRows / 기타', () => {
  it('실행 중에는 샘플 번호, 청크 안 위치, 지연을 보여 준다', () => {
    const rows = Object.fromEntries(summaryRows(RUNNING))
    expect(rows['시드']).toBe(28)
    expect(rows['경과 시간 (시뮬레이션)']).toBe('12.3 / 45 s')
    expect(rows['샘플 번호']).toBe('369')
    expect(rows['청크 안 위치']).toBe('69 / 100')
    expect(rows['마지막 추론 총 지연 (렌더+추론, ms)']).toBe('141')
    expect(rows['이번 세션 성공 / 시도']).toBe('0 / 0')
  })

  it('실행 중이 아니면 샘플 숫자는 "—", 세션 횟수는 그대로', () => {
    const rows = Object.fromEntries(summaryRows(DONE))
    expect(rows['샘플 번호']).toBe('—')
    expect(rows['청크 안 위치']).toBe('—')
    expect(rows['이번 세션 성공 / 시도']).toBe('1 / 1')
    expect(summaryRows(undefined)).toEqual([])
  })

  it('formatElapsed, eventLevel', () => {
    expect(formatElapsed(null, 45)).toBe('—')
    expect(formatElapsed(3.14159, 45)).toBe('3.1 / 45 s')
    expect(eventLevel({ level: 'warn' })).toBe('warn')
    expect(eventLevel({ level: 'x' })).toBe('info')
    expect(eventLevel(null)).toBe('info')
  })
})
