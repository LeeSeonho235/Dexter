import { describe, expect, it } from 'vitest'
import { compassArrows, MIN_VISIBLE_LENGTH, visibleLength, worldDirectionOnScreen } from './compass.js'

const sinEl = (el) => Math.sin((Math.abs(el) * Math.PI) / 180)

describe('worldDirectionOnScreen: 로봇 앞쪽(+x)이 화면 어느 쪽인가', () => {
  it('방위각 0°(위에서/뒤에서 보기): +x는 화면 위, 비스듬하면 짧아 보인다', () => {
    const a = worldDirectionOnScreen([1, 0], 0, -35)
    expect(a.angle_deg).toBeCloseTo(0)
    expect(a.length).toBeCloseTo(sinEl(-35))
    expect(worldDirectionOnScreen([1, 0], 0, -89).length).toBeCloseTo(1, 3)
  })

  it('방위각 90°: +x는 화면 오른쪽, 길이는 고도각과 무관하게 1', () => {
    for (const el of [-89, -35, -10]) {
      const a = worldDirectionOnScreen([1, 0], 90, el)
      expect(a.angle_deg).toBeCloseTo(90)
      expect(a.length).toBeCloseTo(1)
    }
  })

  it('방위각 180°: 화면 아래, 270°: 화면 왼쪽', () => {
    expect(worldDirectionOnScreen([1, 0], 180, -35).angle_deg).toBeCloseTo(180)
    expect(worldDirectionOnScreen([1, 0], 270, -35).angle_deg).toBeCloseTo(270)
  })

  it('방위각 105°("옆에서"): 대략 오른쪽에서 조금 아래로', () => {
    const el = -10
    const a = worldDirectionOnScreen([1, 0], 105, el)
    const right = Math.sin((105 * Math.PI) / 180)
    const up = Math.cos((105 * Math.PI) / 180) * sinEl(el)
    expect(a.angle_deg).toBeCloseTo(((Math.atan2(right, up) * 180) / Math.PI + 360) % 360)
    expect(a.angle_deg).toBeGreaterThan(90)
    expect(a.angle_deg).toBeLessThan(135)
    expect(a.length).toBeCloseTo(Math.hypot(right, up))
  })

  it('방위각을 돌리면 같이 돈다: 1°씩 바꾸면 각도도 연속으로 변한다(고도각 -89°에서도)', () => {
    let previous = worldDirectionOnScreen([1, 0], 0, -89).angle_deg
    for (let az = 1; az <= 360; az++) {
      const now = worldDirectionOnScreen([1, 0], az, -89).angle_deg
      const diff = (((now - previous) % 360) + 540) % 360 - 180
      expect(Math.abs(diff - 1)).toBeLessThan(0.01)
      previous = now
    }
  })

  it('각도는 항상 0 이상 360 미만', () => {
    for (let az = -400; az <= 400; az += 7) {
      const { angle_deg } = worldDirectionOnScreen([1, 0], az, -35)
      expect(angle_deg).toBeGreaterThanOrEqual(0)
      expect(angle_deg).toBeLessThan(360)
    }
  })

  it('-y(로봇 오른쪽 옆)는 +x에서 시계 방향으로 90° 떨어진 방향(정확히 수직으로 내려다볼 때)', () => {
    const front = worldDirectionOnScreen([1, 0], 40, -90).angle_deg
    const minusY = worldDirectionOnScreen([0, -1], 40, -90).angle_deg
    expect((minusY - front + 360) % 360).toBeCloseTo(90, 6)
    // -89°에서는 바닥이 sin(89°)만큼 눌려 보여서 0.01° 이내로만 같다
    const front89 = worldDirectionOnScreen([1, 0], 40, -89).angle_deg
    const minusY89 = worldDirectionOnScreen([0, -1], 40, -89).angle_deg
    expect(Math.abs(((minusY89 - front89 + 360) % 360) - 90)).toBeLessThan(0.01)
  })
})

describe('compassArrows', () => {
  it('상대 모드: 손 위/오른쪽은 보기와 상관없이 화면 위/오른쪽(고정), 앞쪽(+x)만 돈다', () => {
    for (const az of [0, 90, 105, 180, 270]) {
      const arrows = compassArrows({ azimuthDeg: az, elevationDeg: -35, mode: 'relative' })
      expect(arrows.handUp).toEqual({ angle_deg: 0, length: 1 })
      expect(arrows.handRight).toEqual({ angle_deg: 90, length: 1 })
      expect(arrows.front.angle_deg).toBeCloseTo(worldDirectionOnScreen([1, 0], az, -35).angle_deg)
    }
  })

  it('상대 모드의 앞쪽 화살표: 방위각 0°이면 위, 90°이면 오른쪽', () => {
    expect(compassArrows({ azimuthDeg: 0, elevationDeg: -35, mode: 'relative' }).front.angle_deg).toBeCloseTo(0)
    expect(compassArrows({ azimuthDeg: 90, elevationDeg: -35, mode: 'relative' }).front.angle_deg).toBeCloseTo(90)
  })

  it('절대 모드: 손 위 = +x(앞), 손 오른쪽 = -y라서 보기와 같이 돈다', () => {
    const arrows = compassArrows({ azimuthDeg: 0, elevationDeg: -89, mode: 'absolute' })
    expect(arrows.handUp.angle_deg).toBeCloseTo(0) // 위에서 보기: 손 위 = 화면 위
    expect(arrows.handRight.angle_deg).toBeCloseTo(90) // 손 오른쪽 = 화면 오른쪽
    const turned = compassArrows({ azimuthDeg: 90, elevationDeg: -89, mode: 'absolute' })
    expect(turned.handUp.angle_deg).toBeCloseTo(90) // 90°에서는 손 위 = 화면 오른쪽: 어긋난다
    expect(turned.handRight.angle_deg).toBeCloseTo(180)
  })

  it('모드를 안 주면 상대 모드처럼 그린다', () => {
    const arrows = compassArrows({ azimuthDeg: 30, elevationDeg: -35 })
    expect(arrows.handUp.angle_deg).toBe(0)
  })
})

describe('visibleLength', () => {
  it('너무 짧으면 보이는 최소 길이로, 1을 넘으면 1로', () => {
    expect(visibleLength(0.01)).toBe(MIN_VISIBLE_LENGTH)
    expect(visibleLength(0.6)).toBe(0.6)
    expect(visibleLength(1.4)).toBe(1)
  })
})
