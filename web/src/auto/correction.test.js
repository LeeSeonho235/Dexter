// 교정(4-2) 표시 규칙: 제어 주체 큰 표시(문구+기호), 토글 버튼과 Space 메시지, 집게 안내가 뜨고 사라지는 조건, 저장/버리기, 시드 안내.
import { describe, expect, it } from 'vitest'
import {
  correctionRows,
  decisionPending,
  decisionText,
  gripperHint,
  ownerBanner,
  ownerKind,
  seedGuideText,
  seedPurposeWarning,
  spaceCommand,
  toggleButton,
} from './correction.js'
import { CORRECTION_BUTTON_IDS, commandFor } from './view.js'

const BASE = {
  status: 'running', owner: 'AUTO', seed: 41, max_s: 45, auto_s: 12.34, human_s: 0, wait_s: 0, total_s: 12.34, total_max_s: 120, interventions: 0,
  correction: { pending: null, saved_count: 0, saved_human_s: 0, save_error_count: 0, last_saved: null },
  ready_seed: null, ready_seed_purpose: null, next_seed: 42, next_seed_purpose: 'extra_demo',
}
const AUTO = BASE
const WAIT = { ...BASE, owner: 'HUMAN_WAIT', wait_s: 0.7 }
const HUMAN = { ...BASE, owner: 'HUMAN', human_s: 3.5, interventions: 1 }
const IDLE = { ...BASE, status: 'idle', owner: 'IDLE' }
const SYNC_WAITING_CLOSED = { gripper_sync: { active: true, waiting: true, robot_closed: true } }
const SYNC_WAITING_OPEN = { gripper_sync: { active: true, waiting: true, robot_closed: false } }
const SYNC_DONE = { gripper_sync: { active: true, waiting: false, robot_closed: true } }
const SYNC_OFF = { gripper_sync: { active: false, waiting: false, robot_closed: null } }

describe('제어 주체 큰 표시', () => {
  it('주체마다 다른 문구와 기호가 있다(색만으로 구분하지 않는다)', () => {
    const banners = [AUTO, WAIT, HUMAN].map((a) => ownerBanner(a, true))
    expect(banners.map((b) => b.title)).toEqual(['로봇이 혼자 하는 중(AUTO)', '손을 보여주세요(대기)', '내가 조종 중'])
    expect(banners.map((b) => b.kind)).toEqual(['auto', 'wait', 'human'])
    expect(new Set(banners.map((b) => b.mark)).size).toBe(3)
    expect(banners[1].detail).toContain('손을 화면 중앙에 보여주세요')
    expect(banners[0].detail).toContain('Space')
    expect(banners[2].detail).toContain('Space')
  })

  it('AUTO가 실행 중이 아니거나 연결이 없으면 표시하지 않는다(로딩은 따로)', () => {
    expect(ownerBanner(IDLE, true)).toBeNull()
    expect(ownerBanner(AUTO, false)).toBeNull()
    expect(ownerBanner(undefined, true)).toBeNull()
    expect(ownerBanner({ ...BASE, status: 'success', owner: 'IDLE' }, true)).toBeNull()
    expect(ownerBanner({ ...BASE, status: 'loading' }, true).title).toBe('정책 불러오는 중')
    expect(ownerKind({ ...BASE, owner: 'IDLE' }, true)).toBe('idle')
  })
})

describe('토글 버튼과 Space 메시지', () => {
  it('AUTO면 넘겨받기, HUMAN이면 돌려주기, 대기와 그 외에는 눌리지 않는다', () => {
    expect(toggleButton(AUTO, true)).toEqual({ id: 'takeover', label: '넘겨받기 (Space)', enabled: true, reason: '' })
    expect(toggleButton(HUMAN, true)).toMatchObject({ id: 'handback', enabled: true })
    expect(toggleButton(HUMAN, true).label).toContain('Space')
    const wait = toggleButton(WAIT, true)
    expect(wait.enabled).toBe(false)
    expect(wait.reason).toContain('손을 화면 중앙에')
    expect(toggleButton(IDLE, true).enabled).toBe(false)
    expect(toggleButton({ ...BASE, status: 'loading' }, true).enabled).toBe(false)
    expect(toggleButton(AUTO, false).reason).toContain('연결')
    expect(toggleButton(undefined, true).enabled).toBe(false)
  })

  it('Space는 버튼과 같은 규칙으로 메시지를 정한다', () => {
    expect(spaceCommand(AUTO, true)).toEqual({ type: 'auto', action: 'takeover' })
    expect(spaceCommand(HUMAN, true)).toEqual({ type: 'auto', action: 'handback' })
    expect(spaceCommand(WAIT, true)).toBeNull()
    expect(spaceCommand(IDLE, true)).toBeNull()
    expect(spaceCommand(AUTO, false)).toBeNull()
    expect(spaceCommand(null, true)).toBeNull()
  })

  it('새 버튼 id도 auto 메시지가 된다', () => {
    for (const id of CORRECTION_BUTTON_IDS) expect(commandFor(id)).toEqual({ type: 'auto', action: id })
    expect(commandFor('take_over')).toBeNull()
  })
})

describe('집게 안내("집게를 로봇(닫힘)에 맞추세요")가 뜨는 조건과 사라지는 조건', () => {
  it('블록을 쥔 채(닫힘) 넘겨받고 손이 안 맞으면 뜬다', () => {
    const hint = gripperHint(HUMAN, SYNC_WAITING_CLOSED, true)
    expect(hint.text).toBe('집게를 로봇(닫힘)에 맞추세요')
    expect(hint.detail).toContain('손을 쥐세요')
    expect(gripperHint(HUMAN, SYNC_WAITING_OPEN, true).text).toBe('집게를 로봇(열림)에 맞추세요')
  })

  it('사용자가 로봇 상태와 일치시키면(waiting false) 사라진다', () => {
    expect(gripperHint(HUMAN, SYNC_DONE, true)).toBeNull()
  })

  it('HUMAN이 아니거나 세션이 없거나 연결이 없으면 뜨지 않는다', () => {
    expect(gripperHint(WAIT, SYNC_WAITING_CLOSED, true)).toBeNull() // 대기 중에는 "손을 보여주세요"가 먼저
    expect(gripperHint(AUTO, SYNC_WAITING_CLOSED, true)).toBeNull()
    expect(gripperHint(HUMAN, SYNC_OFF, true)).toBeNull()
    expect(gripperHint(HUMAN, SYNC_WAITING_CLOSED, false)).toBeNull()
    expect(gripperHint(HUMAN, undefined, true)).toBeNull()
    expect(gripperHint(HUMAN, {}, true)).toBeNull()
  })
})

describe('끝난 뒤 저장/버리기', () => {
  const PENDING = { ...BASE, status: 'success', owner: 'IDLE', correction: { ...BASE.correction, pending: { seed: 41, outcome: 'success', end_reason: 'success', human_time_s: 4.25, interventions: 2 } } }

  it('사람 구간이 있으면 선택을 기다린다', () => {
    expect(decisionPending(PENDING)).toBe(true)
    expect(decisionPending({ ...PENDING, correction: { ...PENDING.correction, pending: null } })).toBe(false)
    expect(decisionPending(undefined)).toBe(false)
    expect(decisionText(PENDING)).toContain('성공')
    expect(decisionText(PENDING)).toContain('2번')
    expect(decisionText(PENDING)).toContain('4.3초')
    expect(decisionText(AUTO)).toBe('')
  })
})

describe('시드 안내와 표', () => {
  it('다음 시드와 용도를 안내한다(평가 시드는 서버가 리셋에서 내지 않아 화면에 나오지 않는다)', () => {
    expect(seedGuideText({ ...BASE, next_seed: 40, next_seed_purpose: 'extra_demo' })).toBe('다음 시드: 40 → 추가 시연용')
    expect(seedGuideText({ ...BASE, next_seed: 52, next_seed_purpose: 'correction' })).toBe('다음 시드: 52 → 교정용')
    expect(seedGuideText({ ...BASE, next_seed: 5, next_seed_purpose: null })).toBe('다음 시드: 5 → 배분표에 없는 시드')
    expect(seedGuideText({ ...BASE, ready_seed: 28, ready_seed_purpose: 'correction' })).toBe('준비된 블록: 시드 28 → 교정용')
    expect(seedGuideText({ ...BASE, next_seed: null })).toBe('시연 시드를 모두 썼습니다')
    expect(seedGuideText(null)).toBe('—')
  })

  it('준비된 시드가 추가 시연용이면 경고만 한다(막지 않는다)', () => {
    expect(seedPurposeWarning({ ...BASE, ready_seed: 40, ready_seed_purpose: 'extra_demo' })).toContain('추가 시연용')
    expect(seedPurposeWarning({ ...BASE, ready_seed: 28, ready_seed_purpose: 'correction' })).toBeNull()
    expect(seedPurposeWarning({ ...BASE, ready_seed: null })).toBeNull()
  })

  it('표: 사람 시간(HUMAN만), 개입 횟수, AUTO 시간 / 제한', () => {
    const rows = Object.fromEntries(correctionRows(HUMAN))
    expect(rows['사람 시간 (HUMAN 구간만)']).toBe('3.5초')
    expect(rows['개입 횟수']).toBe('1번')
    expect(rows['AUTO 시간 / 제한']).toBe('12.3 / 45 초')
    expect(rows['전체 시간 / 상한']).toBe('12.3 / 120 초')
    expect(rows['저장한 교정']).toBe('0개 (사람 시간 합계 0.0초)')
    expect(correctionRows(undefined)).toEqual([])
    expect(Object.fromEntries(correctionRows({ ...BASE, human_s: null, interventions: null, auto_s: null }))['개입 횟수']).toBe('—')
  })
})
