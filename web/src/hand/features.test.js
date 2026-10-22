import { describe, expect, it } from 'vitest'
import { distance2d, LM, makeHandMessage, pinchRatio } from './features.js'

// 위를 향한 가짜 손. 손목 (0.5, 0.7), 중지 뿌리 (0.5, 0.6) → hand_scale 0.1 (aspect 1)
// curled: 접을 손가락 번호 목록 (0=검지 ... 3=새끼)
function makeHand({ curled = [] } = {}) {
  const lm = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.7, z: 0 }))
  lm[LM.WRIST] = { x: 0.5, y: 0.7, z: 0 }
  // 엄지 1~4: 왼쪽 옆으로
  lm[1] = { x: 0.47, y: 0.67, z: 0 }
  lm[2] = { x: 0.45, y: 0.65, z: 0 }
  lm[3] = { x: 0.44, y: 0.63, z: 0 }
  lm[4] = { x: 0.43, y: 0.61, z: -0.01 }
  const fingerX = [0.48, 0.5, 0.52, 0.54]
  for (let f = 0; f < 4; f++) {
    const mcp = 5 + f * 4
    const x = fingerX[f]
    lm[mcp] = { x, y: 0.6, z: 0 }
    lm[mcp + 1] = { x, y: 0.56, z: 0 } // PIP
    if (curled.includes(f)) {
      // 접힘: DIP, TIP이 손바닥 쪽으로 돌아옴 (손목에 더 가까움)
      lm[mcp + 2] = { x, y: 0.58, z: 0 }
      lm[mcp + 3] = { x, y: 0.61, z: 0 }
    } else {
      lm[mcp + 2] = { x, y: 0.53, z: 0 }
      lm[mcp + 3] = { x, y: 0.5, z: -0.02 }
    }
  }
  return lm
}

// 손목을 중심으로 크기만 factor배
function scaleHand(lm, factor) {
  const w = lm[LM.WRIST]
  return lm.map((p) => ({ x: w.x + (p.x - w.x) * factor, y: w.y + (p.y - w.y) * factor, z: p.z * factor }))
}

describe('makeHandMessage', () => {
  it('손이 없으면 detected=false, 나머지 null', () => {
    for (const input of [null, undefined, []]) {
      const msg = makeHandMessage(input, { t: 1.5 })
      expect(msg).toEqual({
        type: 'hand',
        t: 1.5,
        detected: false,
        wrist: null,
        index_tip: null,
        thumb_tip: null,
        pinch: null,
        pinch_ratio: null,
        palm_open: null,
        hand_scale: null,
      })
    }
  })

  it('메시지 필드와 좌표 배열', () => {
    const lm = makeHand()
    const msg = makeHandMessage(lm, { t: 10 })
    expect(Object.keys(msg).sort()).toEqual(
      ['detected', 'hand_scale', 'index_tip', 'palm_open', 'pinch', 'pinch_ratio', 't', 'thumb_tip', 'type', 'wrist'].sort(),
    )
    expect(msg.detected).toBe(true)
    expect(msg.wrist).toEqual([0.5, 0.7, 0])
    expect(msg.index_tip).toEqual([0.48, 0.5, -0.02])
    expect(msg.thumb_tip).toEqual([0.43, 0.61, -0.01])
  })

  it('hand_scale = 손목~중지 뿌리 거리', () => {
    expect(makeHandMessage(makeHand(), { t: 0 }).hand_scale).toBeCloseTo(0.1, 9)
  })

  it('pinch = 엄지 끝~검지 끝 거리', () => {
    // 엄지 끝 (0.43, 0.61), 검지 끝 (0.48, 0.5)
    const expected = Math.hypot(0.05, 0.11)
    expect(makeHandMessage(makeHand(), { t: 0 }).pinch).toBeCloseTo(expected, 9)
  })

  it('화면 비율 보정: x 차이에만 aspect를 곱한다', () => {
    const a = { x: 0.1, y: 0.2 }
    const b = { x: 0.4, y: 0.6 }
    expect(distance2d(a, b, 4 / 3)).toBeCloseTo(Math.hypot(0.4, 0.4), 9)
    expect(distance2d(a, b)).toBeCloseTo(Math.hypot(0.3, 0.4), 9)
  })

  it('palm_open: 4개 다 펴면 true, 주먹이면 false, 하나라도 접으면 false', () => {
    expect(makeHandMessage(makeHand(), { t: 0 }).palm_open).toBe(true)
    expect(makeHandMessage(makeHand({ curled: [0, 1, 2, 3] }), { t: 0 }).palm_open).toBe(false)
    expect(makeHandMessage(makeHand({ curled: [3] }), { t: 0 }).palm_open).toBe(false)
  })

  it('같은 손 모양을 크기만 2배로 키워도 pinch_ratio가 같다', () => {
    const small = makeHandMessage(makeHand(), { t: 0, aspect: 4 / 3 })
    const big = makeHandMessage(scaleHand(makeHand(), 2), { t: 0, aspect: 4 / 3 })
    expect(big.pinch).toBeCloseTo(small.pinch * 2, 9)
    expect(big.hand_scale).toBeCloseTo(small.hand_scale * 2, 9)
    expect(big.pinch_ratio).toBeCloseTo(small.pinch_ratio, 9)
  })
})

describe('pinchRatio', () => {
  it('pinch / hand_scale', () => {
    expect(pinchRatio(0.04, 0.2)).toBeCloseTo(0.2, 9)
  })

  it('hand_scale이 0이거나 null이면 null', () => {
    expect(pinchRatio(0.04, 0)).toBeNull()
    expect(pinchRatio(0.04, null)).toBeNull()
    expect(pinchRatio(0.04, undefined)).toBeNull()
    expect(pinchRatio(null, 0.2)).toBeNull()
  })
})
