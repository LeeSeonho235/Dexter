import { describe, expect, it } from 'vitest'
import { EDGE_MARGIN } from './config.js'
import { edgeMessage, handEdgeStatus, overlayGeometry } from './edge.js'

// 21개 점이 (minX..maxX, minY..maxY) 상자 안에 고르게 퍼진 가짜 손. 가장자리 점이 정확히 상자 끝에 오게 한다.
function handBox(minX, maxX, minY, maxY) {
  return Array.from({ length: 21 }, (_, i) => ({
    x: minX + ((maxX - minX) * (i % 5)) / 4,
    y: minY + ((maxY - minY) * Math.floor(i / 5)) / 4,
    z: 0,
  }))
}

describe('handEdgeStatus', () => {
  it('기본 여유는 10%', () => {
    expect(EDGE_MARGIN).toBe(0.1)
  })

  it('중앙에 있으면 가장자리가 아니다', () => {
    expect(handEdgeStatus(handBox(0.3, 0.7, 0.3, 0.7))).toEqual({
      near: false, left: false, right: false, top: false, bottom: false,
    })
  })

  it('여유 바로 안쪽(0.11~0.89)까지는 괜찮다', () => {
    expect(handEdgeStatus(handBox(0.11, 0.89, 0.11, 0.89)).near).toBe(false)
  })

  it('각 방향으로 여유 안에 들어오면 그 방향이 켜진다', () => {
    expect(handEdgeStatus(handBox(0.05, 0.5, 0.3, 0.7))).toMatchObject({ near: true, left: true, right: false })
    expect(handEdgeStatus(handBox(0.5, 0.95, 0.3, 0.7))).toMatchObject({ near: true, right: true, left: false })
    expect(handEdgeStatus(handBox(0.3, 0.7, 0.04, 0.5))).toMatchObject({ near: true, top: true, bottom: false })
    expect(handEdgeStatus(handBox(0.3, 0.7, 0.5, 0.96))).toMatchObject({ near: true, bottom: true, top: false })
  })

  it('경계(정확히 10%)도 가장자리로 본다', () => {
    expect(handEdgeStatus(handBox(0.1, 0.5, 0.3, 0.7)).near).toBe(true)
    expect(handEdgeStatus(handBox(0.3, 0.9, 0.3, 0.7)).near).toBe(true)
    expect(handEdgeStatus(handBox(0.3, 0.7, 0.1, 0.5)).near).toBe(true)
    expect(handEdgeStatus(handBox(0.3, 0.7, 0.3, 0.9)).near).toBe(true)
  })

  it('손목은 중앙이어도 손가락 끝 하나만 가장자리에 닿으면 가장자리다(손 21개 점 전체를 본다)', () => {
    const lm = handBox(0.4, 0.6, 0.4, 0.6)
    lm[8] = { x: 0.5, y: 0.03, z: 0 } // 검지 끝만 위쪽으로
    expect(handEdgeStatus(lm)).toMatchObject({ near: true, top: true })
  })

  it('영상 밖(0 미만, 1 초과)도 가장자리', () => {
    expect(handEdgeStatus(handBox(-0.1, 0.3, 0.3, 0.7)).near).toBe(true)
    expect(handEdgeStatus(handBox(0.3, 1.2, 0.3, 0.7)).near).toBe(true)
  })

  it('여유를 바꿀 수 있다', () => {
    const lm = handBox(0.2, 0.8, 0.2, 0.8)
    expect(handEdgeStatus(lm, 0.1).near).toBe(false)
    expect(handEdgeStatus(lm, 0.25).near).toBe(true)
  })

  it('손이 없으면(null, 빈 배열) 가장자리가 아니다', () => {
    expect(handEdgeStatus(null).near).toBe(false)
    expect(handEdgeStatus(undefined).near).toBe(false)
    expect(handEdgeStatus([]).near).toBe(false)
  })
})

describe('overlayGeometry', () => {
  it('기준점 십자, 손목 점, 둘 사이 선의 캔버스 좌표', () => {
    const g = overlayGeometry({ reference: [0.3, 0.4], wrist: [0.6, 0.7] }, 640, 480)
    expect(g.cross).toEqual({ x: 192, y: 192, half: 10 })
    expect(g.dot).toEqual({ x: 384, y: 336 })
    expect(g.line).toEqual({ x1: 192, y1: 192, x2: 384, y2: 336 })
  })

  it('기준점이나 손목이 없으면 그리지 않는다', () => {
    expect(overlayGeometry({ reference: null, wrist: [0.5, 0.5] }, 640, 480)).toBeNull()
    expect(overlayGeometry({ reference: [0.5, 0.5], wrist: null }, 640, 480)).toBeNull()
    expect(overlayGeometry({}, 640, 480)).toBeNull()
  })

  it('손이 기준점에서 안 움직였으면 점이 십자 위에 있고 선 길이는 0', () => {
    const g = overlayGeometry({ reference: [0.5, 0.5], wrist: [0.5, 0.5] }, 640, 480)
    expect(g.line.x1).toBe(g.line.x2)
    expect(g.line.y1).toBe(g.line.y2)
  })
})

describe('edgeMessage', () => {
  it('상대 모드: 손을 빼서 중앙에서 다시 넣으라고 안내', () => {
    expect(edgeMessage('relative')).toBe('화면 가장자리: 손을 화면 밖으로 뺐다가 중앙에서 다시 넣으면 이어서 움직여요')
  })
  it('절대 모드는 다른 문구', () => {
    expect(edgeMessage('absolute')).not.toBe(edgeMessage('relative'))
    expect(edgeMessage('absolute')).toContain('가장자리')
  })
})
