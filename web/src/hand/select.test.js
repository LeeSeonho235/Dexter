import { describe, expect, it } from 'vitest'
import { selectHand } from './select.js'

const points = (tag) => Array.from({ length: 21 }, () => ({ x: tag, y: tag, z: 0 }))
const cat = (categoryName, score) => [{ categoryName, score, index: 0, displayName: categoryName }]

describe('selectHand', () => {
  it('고른 라벨의 손만 조종 손이 된다', () => {
    const result = { landmarks: [points(0.1), points(0.9)], handedness: [cat('Left', 0.99), cat('Right', 0.97)] }
    const { chosen, hands } = selectHand(result, 'Right')
    expect(chosen.index).toBe(1)
    expect(chosen.landmarks[0].x).toBe(0.9)
    expect(hands).toEqual([
      { index: 0, label: 'Left', score: 0.99, controlling: false },
      { index: 1, label: 'Right', score: 0.97, controlling: true },
    ])
  })

  it('손 순서가 바뀌어도 라벨로 고른다', () => {
    const result = { landmarks: [points(0.9), points(0.1)], handedness: [cat('Right', 0.9), cat('Left', 0.9)] }
    expect(selectHand(result, 'Right').chosen.landmarks[0].x).toBe(0.9)
    expect(selectHand(result, 'Left').chosen.landmarks[0].x).toBe(0.1)
  })

  it('조종 손이 안 보이면 다른 손이 있어도 null (조종이 넘어가지 않는다)', () => {
    const result = { landmarks: [points(0.1)], handedness: [cat('Left', 0.99)] }
    const { chosen, hands } = selectHand(result, 'Right')
    expect(chosen).toBeNull()
    expect(hands).toEqual([{ index: 0, label: 'Left', score: 0.99, controlling: false }])
  })

  it('같은 라벨이 둘이면 score가 높은 쪽', () => {
    const result = { landmarks: [points(0.2), points(0.8)], handedness: [cat('Right', 0.6), cat('Right', 0.95)] }
    const { chosen, hands } = selectHand(result, 'Right')
    expect(chosen.index).toBe(1)
    expect(hands.map((h) => h.controlling)).toEqual([false, true])
  })

  it('손이 없거나 결과가 비어 있으면 null', () => {
    expect(selectHand({ landmarks: [], handedness: [] }, 'Right')).toEqual({ chosen: null, hands: [] })
    expect(selectHand(undefined, 'Right')).toEqual({ chosen: null, hands: [] })
    expect(selectHand({}, 'Left')).toEqual({ chosen: null, hands: [] })
  })

  it('라벨 정보가 없는 손은 조종 손이 되지 않는다', () => {
    const result = { landmarks: [points(0.5)], handedness: [] }
    const { chosen, hands } = selectHand(result, 'Right')
    expect(chosen).toBeNull()
    expect(hands[0].label).toBeNull()
  })
})
