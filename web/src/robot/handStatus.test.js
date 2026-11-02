import { describe, expect, it } from 'vitest'
import { handStatusText } from './view.js'

const base = { controlling: false, consecutive_frames: 0, start_frames: 3, target_m: null }

describe('handStatusText (웹캠 화면 아래 조종 상태 문구)', () => {
  it('기준점 잡는 중 (n/3)', () => {
    expect(handStatusText({ ...base, consecutive_frames: 1 }, true)).toBe('손 기준점 잡는 중 (1/3)')
    expect(handStatusText({ ...base, consecutive_frames: 2 }, true)).toBe('손 기준점 잡는 중 (2/3)')
  })

  it('손을 따라가는 중', () => {
    expect(handStatusText({ ...base, controlling: true, consecutive_frames: 9, target_m: [0.2, 0, 0.1] }, true)).toBe(
      '손을 따라가는 중',
    )
  })

  it('손이 사라지면: 손 안 보임 → 정지 (중앙에서 다시 넣으세요)', () => {
    expect(handStatusText({ ...base, target_m: [0.2, 0, 0.1] }, true)).toBe('손 안 보임 → 정지 (중앙에서 다시 넣으세요)')
  })

  it('처음 조종 시작 전', () => {
    expect(handStatusText(base, true)).toContain('시작 전')
  })

  it('서버 연결이 없거나 상태를 아직 못 받았을 때', () => {
    expect(handStatusText(base, false)).toBe('서버에 연결되지 않음')
    expect(handStatusText(undefined, true)).toBe('서버 상태 기다리는 중')
  })
})
