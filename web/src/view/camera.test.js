import { describe, expect, it } from 'vitest'
import {
  applyDrag,
  applyWheel,
  clampView,
  formatView,
  pickDisplayedView,
  presetView,
  viewMessage,
  viewsEqual,
  wrapAzimuthDeg,
} from './camera.js'
import {
  AZIMUTH_DEG_PER_PX,
  DISTANCE_RANGE_M,
  ELEVATION_DEG_PER_PX,
  ELEVATION_RANGE_DEG,
  PRESET_LABELS,
  PRESETS,
} from './config.js'

const VIEW = { azimuth_deg: 110, elevation_deg: -35, distance_m: 0.8 }

describe('applyDrag', () => {
  it('좌우로 끌면 방위각만, 상하로 끌면 고도각만 바뀐다', () => {
    const horizontal = applyDrag(VIEW, 100, 0)
    expect(horizontal.azimuth_deg).toBeCloseTo(110 + 100 * AZIMUTH_DEG_PER_PX)
    expect(horizontal.elevation_deg).toBe(-35)
    expect(horizontal.distance_m).toBe(0.8)

    const vertical = applyDrag(VIEW, 0, 20)
    expect(vertical.azimuth_deg).toBe(110)
    expect(vertical.elevation_deg).toBeCloseTo(-35 + 20 * ELEVATION_DEG_PER_PX)
  })

  it('반대로 끌면 반대로 바뀌고, 끌었다 되돌리면 원래대로', () => {
    const there = applyDrag(VIEW, 40, 15)
    const back = applyDrag(there, -40, -15)
    expect(back.azimuth_deg).toBeCloseTo(VIEW.azimuth_deg)
    expect(back.elevation_deg).toBeCloseTo(VIEW.elevation_deg)
  })

  it('방위각은 한 바퀴 돌아도 -180~180 안에 있다', () => {
    const spun = applyDrag(VIEW, 5000, 0)
    expect(spun.azimuth_deg).toBeGreaterThanOrEqual(-180)
    expect(spun.azimuth_deg).toBeLessThan(180)
  })

  it('고도각은 범위 밖으로 안 나간다(위에서 거의 수직, 수평선 아래 금지)', () => {
    expect(applyDrag(VIEW, 0, 100000).elevation_deg).toBe(ELEVATION_RANGE_DEG[0])
    expect(applyDrag(VIEW, 0, -100000).elevation_deg).toBe(ELEVATION_RANGE_DEG[1])
  })

  it('원래 값을 바꾸지 않는다', () => {
    const copy = { ...VIEW }
    applyDrag(VIEW, 30, 30)
    expect(VIEW).toEqual(copy)
  })
})

describe('applyWheel', () => {
  it('아래로 굴리면 멀어지고 위로 굴리면 가까워진다', () => {
    expect(applyWheel(VIEW, 100).distance_m).toBeGreaterThan(0.8)
    expect(applyWheel(VIEW, -100).distance_m).toBeLessThan(0.8)
  })

  it('같은 만큼 굴렸다 되돌리면 원래 거리', () => {
    const back = applyWheel(applyWheel(VIEW, 240), -240)
    expect(back.distance_m).toBeCloseTo(0.8)
  })

  it('거리는 범위 안으로 자른다', () => {
    expect(applyWheel(VIEW, 1e6).distance_m).toBe(DISTANCE_RANGE_M[1])
    expect(applyWheel(VIEW, -1e6).distance_m).toBe(DISTANCE_RANGE_M[0])
  })

  it('각도는 건드리지 않는다', () => {
    const next = applyWheel(VIEW, 100)
    expect(next.azimuth_deg).toBe(110)
    expect(next.elevation_deg).toBe(-35)
  })
})

describe('프리셋', () => {
  it('버튼 3개: 위에서, 옆에서, 비스듬히(기본)', () => {
    expect(PRESET_LABELS.map(([, label]) => label)).toEqual(['위에서', '옆에서', '비스듬히 (기본)'])
    expect(PRESET_LABELS.map(([name]) => name)).toEqual(['top', 'side', 'default'])
  })

  it('프리셋 값은 범위 안이고 그대로 적용된다', () => {
    for (const name of Object.keys(PRESETS)) {
      expect(presetView(name)).toEqual(PRESETS[name])
    }
  })

  it('위에서는 거의 수직으로 내려다보고, 옆에서는 낮은 각도이며, 기본은 비스듬하다', () => {
    expect(presetView('top').elevation_deg).toBeLessThan(-80)
    expect(presetView('side').elevation_deg).toBeGreaterThan(-20)
    expect(presetView('default').elevation_deg).toBeCloseTo(-35)
  })

  it('옆에서 프리셋의 방위각은 +x가 화면 오른쪽인 90° 부근', () => {
    expect(Math.abs(presetView('side').azimuth_deg - 90)).toBeLessThanOrEqual(30)
  })

  it('없는 프리셋은 오류', () => {
    expect(() => presetView('nope')).toThrow()
  })
})

describe('clampView / wrapAzimuthDeg', () => {
  it('방위각 돌리기', () => {
    expect(wrapAzimuthDeg(270)).toBe(-90)
    expect(wrapAzimuthDeg(-190)).toBe(170)
    expect(wrapAzimuthDeg(180)).toBe(-180)
    expect(wrapAzimuthDeg(0)).toBe(0)
  })
  it('범위 밖 값 자르기', () => {
    expect(clampView({ azimuth_deg: 0, elevation_deg: 10, distance_m: 99 })).toEqual({
      azimuth_deg: 0,
      elevation_deg: ELEVATION_RANGE_DEG[1],
      distance_m: DISTANCE_RANGE_M[1],
    })
  })
})

describe('viewMessage', () => {
  it('서버 view 메시지 형식', () => {
    expect(viewMessage(VIEW)).toEqual({ type: 'view', azimuth: 110, elevation: -35, distance: 0.8 })
  })
})

describe('pickDisplayedView (서버 값 따라가기)', () => {
  const server = { azimuth_deg: 10, elevation_deg: -20, distance_m: 1 }
  it('직접 조작한 직후에는 내 값', () => {
    expect(pickDisplayedView({ local: VIEW, server, lastInteractionMs: 1000, nowMs: 1200, holdMs: 500 })).toBe(VIEW)
  })
  it('시간이 지나면 서버 값(다른 브라우저가 바꾼 값도 반영)', () => {
    expect(pickDisplayedView({ local: VIEW, server, lastInteractionMs: 1000, nowMs: 1600, holdMs: 500 })).toBe(server)
  })
  it('서버 값이 아직 없으면 내 값', () => {
    expect(pickDisplayedView({ local: VIEW, server: null, lastInteractionMs: -Infinity, nowMs: 5000 })).toBe(VIEW)
  })
})

describe('viewsEqual', () => {
  it('거의 같은 값은 같다고 본다', () => {
    expect(viewsEqual(VIEW, { ...VIEW, azimuth_deg: 110.004 })).toBe(true)
    expect(viewsEqual(VIEW, { ...VIEW, azimuth_deg: 111 })).toBe(false)
    expect(viewsEqual(VIEW, { ...VIEW, distance_m: 0.81 })).toBe(false)
  })
})

describe('formatView', () => {
  it('숫자로 표시', () => {
    expect(formatView(VIEW)).toBe('방위각 110° · 고도각 -35° · 거리 0.80 m')
    expect(formatView(null)).toBe('—')
  })
})
