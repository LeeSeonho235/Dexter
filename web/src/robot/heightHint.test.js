import { describe, expect, it } from 'vitest'
import { heightHint } from './view.js'

const started = { target_z_m: 0.07, height_active: false, height_dir: 0, height_slow: false, height_limit: null }

describe('heightHint (키보드 높이 안내 한 줄)', () => {
  it('서버 상태를 아직 못 받았으면 빈 문자열', () => {
    expect(heightHint(undefined)).toBe('')
    expect(heightHint(null)).toBe('')
  })

  it('조종 시작 전에는 키보드가 아직 안 먹는다고 알려 준다', () => {
    expect(heightHint({ ...started, target_z_m: null })).toBe('손으로 조종을 시작한 뒤부터 동작합니다')
  })

  it('내려가는 중 / 올라가는 중, 천천히', () => {
    expect(heightHint({ ...started, height_active: true, height_dir: -1 })).toBe('내려가는 중')
    expect(heightHint({ ...started, height_active: true, height_dir: 1 })).toBe('올라가는 중')
    expect(heightHint({ ...started, height_active: true, height_dir: -1, height_slow: true })).toBe('내려가는 중 (천천히)')
  })

  it('상자 끝에 닿으면 맨 위 / 맨 아래', () => {
    expect(heightHint({ ...started, height_limit: 'top' })).toBe('상자 맨 위')
    expect(heightHint({ ...started, height_limit: 'bottom' })).toBe('상자 맨 아래')
  })

  it('아무것도 안 누르는 평소에는 빈 문자열', () => {
    expect(heightHint(started)).toBe('')
  })
})
