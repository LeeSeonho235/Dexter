// 옆모습 보조 화면(2-6): 월드→화면 변환, 도형 목록, 그리기 호출, 크기·위치 설정, 나침반과 겹치지 않음.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  CAPTION_HEIGHT_PX,
  COLORS,
  COMPASS_BOX_PX,
  SIDE_MIRROR_CAPTION,
  SIDE_MIRROR_CORNER,
  SIDE_MIRROR_MARGIN_PX,
  SIDE_MIRROR_SIZE_PX,
  SIM_VIEW_SIZE_PX,
  TIP_RADIUS_PX,
  WORLD_RANGE_M,
} from './config.js'
import { boxesOverlap, compassBoxAbs, drawShapes, makeProjection, mirrorBox, sideViewShapes } from './sideView.js'

const proj = () => makeProjection(SIDE_MIRROR_SIZE_PX, CAPTION_HEIGHT_PX)
const SCENE = { block_m: [0.22, -0.05, 0.015], block_half_m: 0.015, target_center_m: [0.2, 0.12], target_half_m: 0.025 }
const CONTROL = { tip_m: [0.23, 0.0, 0.08], gripper_closed: false }
const byKind = (shapes, kind) => shapes.filter((s) => s.kind === kind)

describe('makeProjection', () => {
  it('z가 올라가면 화면 y는 내려가고(위로), x가 커지면 오른쪽이다', () => {
    const p = proj()
    const [x0, y0] = p.toPx(0.1, 0)
    const [x1, y1] = p.toPx(0.2, 0.1)
    expect(x1).toBeGreaterThan(x0)
    expect(y1).toBeLessThan(y0)
  })

  it('x와 z가 같은 배율이라 블록이 정사각형이다', () => {
    const p = proj()
    const [x0, y0] = p.toPx(0.1, 0.0)
    const [x1, y1] = p.toPx(0.2, 0.1)
    expect(x1 - x0).toBeCloseTo(y0 - y1, 6)
  })

  it('그림이 캔버스 안, 문구 영역 위에 들어간다', () => {
    const p = proj()
    expect(p.left).toBeGreaterThanOrEqual(0)
    expect(p.right).toBeLessThanOrEqual(SIDE_MIRROR_SIZE_PX.width)
    expect(p.top).toBeGreaterThanOrEqual(0)
    expect(toBottom(p)).toBeLessThanOrEqual(SIDE_MIRROR_SIZE_PX.height - CAPTION_HEIGHT_PX)
  })

  it('범위 밖 점은 가장자리로 끌어오고 clamped로 알려 준다', () => {
    const p = proj()
    const inside = p.toPxClamped(0.23, 0.08)
    expect(inside.clamped).toBe(false)
    const far = p.toPxClamped(5, 5)
    expect(far.clamped).toBe(true)
    expect(far.px).toBeCloseTo(p.right)
    expect(far.py).toBeCloseTo(p.top)
  })
})

function toBottom(p) {
  return p.toPx(0, WORLD_RANGE_M.z[0])[1]
}

describe('sideViewShapes', () => {
  it('데이터가 전혀 없어도(연결 전) 에러 없이 바닥선과 로봇 밑 선만 낸다', () => {
    for (const input of [undefined, {}, { control: null, scene: null }, { control: {}, scene: {} }]) {
      const shapes = sideViewShapes(input, proj())
      expect(shapes).toHaveLength(2)
      expect(byKind(shapes, 'line').every((s) => [COLORS.floor, COLORS.base].includes(s.color))).toBe(true)
    }
  })

  it('이상한 값(NaN, 문자열, 짧은 배열)은 그리지 않는다', () => {
    const shapes = sideViewShapes(
      { control: { tip_m: [NaN, 0, 0.1] }, scene: { block_m: [0.2, 0], block_half_m: 'a', target_center_m: ['x', 0], target_half_m: 0.02 } },
      proj(),
    )
    expect(shapes).toHaveLength(2)
  })

  it('블록: 한 변이 2 * block_half_m, 중심이 (x, z)', () => {
    const p = proj()
    const [block] = byKind(sideViewShapes({ scene: SCENE }, p), 'rect')
    expect(block.w).toBeCloseTo(2 * SCENE.block_half_m * p.scale)
    expect(block.h).toBeCloseTo(block.w)
    const [cx, cy] = p.toPx(SCENE.block_m[0], SCENE.block_m[2])
    expect(block.x + block.w / 2).toBeCloseTo(cx)
    expect(block.y + block.h / 2).toBeCloseTo(cy)
    expect(block.color).toBe(COLORS.block)
    // 블록 아래면은 바닥선에 닿는다
    expect(block.y + block.h).toBeCloseTo(p.floorY)
  })

  it('블록 크기를 바꾸면(서버 값) 그림도 따라 바뀐다', () => {
    const p = proj()
    const small = byKind(sideViewShapes({ scene: { ...SCENE, block_half_m: 0.01 } }, p), 'rect')[0]
    const big = byKind(sideViewShapes({ scene: { ...SCENE, block_half_m: 0.02 } }, p), 'rect')[0]
    expect(big.w / small.w).toBeCloseTo(2)
  })

  it('목표: 바닥 위 초록 선분, 길이가 2 * target_half_m', () => {
    const p = proj()
    const target = sideViewShapes({ scene: SCENE }, p).find((s) => s.color === COLORS.target)
    expect(target.y1).toBeCloseTo(p.floorY)
    expect(target.x2 - target.x1).toBeCloseTo(2 * SCENE.target_half_m * p.scale)
  })

  it('집게 끝: 점은 (x, z), 바닥까지 점선이 이어진다', () => {
    const p = proj()
    const shapes = sideViewShapes({ control: CONTROL }, p)
    const [tip] = byKind(shapes, 'circle')
    const [tx, ty] = p.toPx(CONTROL.tip_m[0], CONTROL.tip_m[2])
    expect(tip.x).toBeCloseTo(tx)
    expect(tip.y).toBeCloseTo(ty)
    expect(tip.r).toBe(TIP_RADIUS_PX)
    expect(tip.hollow).toBe(false)
    const height = shapes.find((s) => s.color === COLORS.height)
    expect(height.y1).toBeCloseTo(ty)
    expect(height.y2).toBeCloseTo(p.floorY)
    expect(height.dash).not.toBeNull()
  })

  it('집게가 높을수록 점이 위에 그려진다', () => {
    const p = proj()
    const y = (z) => byKind(sideViewShapes({ control: { tip_m: [0.23, 0, z] } }, p), 'circle')[0].y
    expect(y(0.12)).toBeLessThan(y(0.08))
    expect(y(0.08)).toBeLessThan(y(0.02))
  })

  it('열림 / 닫힘 / 모름은 색이 다르다', () => {
    const p = proj()
    const color = (closed) => byKind(sideViewShapes({ control: { tip_m: [0.2, 0, 0.05], gripper_closed: closed } }, p), 'circle')[0].color
    expect(color(false)).toBe(COLORS.tipOpen)
    expect(color(true)).toBe(COLORS.tipClosed)
    expect(color(undefined)).toBe(COLORS.tipUnknown)
    expect(new Set([color(false), color(true), color(undefined)]).size).toBe(3)
  })

  it('범위 밖 집게는 가장자리에 속 빈 점으로 보인다', () => {
    const [tip] = byKind(sideViewShapes({ control: { tip_m: [3, 0, 0.5] } }, proj()), 'circle')
    expect(tip.hollow).toBe(true)
    expect(tip.x).toBeLessThanOrEqual(SIDE_MIRROR_SIZE_PX.width)
  })

  it('y(좌우)는 그림에 영향이 없다', () => {
    const p = proj()
    const a = sideViewShapes({ control: { tip_m: [0.2, -0.1, 0.05] }, scene: { ...SCENE, block_m: [0.22, -0.1, 0.015] } }, p)
    const b = sideViewShapes({ control: { tip_m: [0.2, 0.1, 0.05] }, scene: { ...SCENE, block_m: [0.22, 0.1, 0.015] } }, p)
    expect(a).toEqual(b)
  })
})

describe('drawShapes (가짜 ctx)', () => {
  function fakeCtx() {
    const calls = []
    return new Proxy(
      { calls },
      {
        get: (target, name) => (name === 'calls' ? calls : (...args) => calls.push([name, ...args])),
        set: (target, name, value) => {
          calls.push(['set', name, value])
          return true
        },
      },
    )
  }

  it('지우고 도형 종류마다 맞는 그리기 호출을 하며 예외가 없다', () => {
    const ctx = fakeCtx()
    const shapes = sideViewShapes({ control: CONTROL, scene: SCENE }, proj())
    expect(() => drawShapes(ctx, shapes, SIDE_MIRROR_SIZE_PX)).not.toThrow()
    const names = ctx.calls.map((c) => c[0])
    expect(names[0]).toBe('clearRect')
    expect(names).toContain('fillRect') // 블록
    expect(names).toContain('arc') // 집게 끝
    expect(names.filter((n) => n === 'stroke').length).toBeGreaterThanOrEqual(4)
  })

  it('도형이 없어도 지우기만 한다', () => {
    const ctx = fakeCtx()
    drawShapes(ctx, [], SIDE_MIRROR_SIZE_PX)
    expect(ctx.calls.map((c) => c[0])).toEqual(['clearRect', 'setLineDash'])
  })
})

describe('크기와 위치 설정', () => {
  it('요청한 160x100 px, 기본 위치는 오른쪽 아래', () => {
    expect(SIDE_MIRROR_SIZE_PX).toEqual({ width: 160, height: 100 })
    expect(SIDE_MIRROR_CORNER).toBe('bottom-right')
  })

  it('문구에 "y(좌우)는 안 보임, 높이 확인용"이 들어 있다', () => {
    expect(SIDE_MIRROR_CAPTION.join(' ')).toContain('y(좌우)는 안 보임, 높이 확인용')
  })

  it('보조 화면은 메인 화면 안에 들어가고 나침반(오른쪽 위)과 겹치지 않는다', () => {
    const compass = compassBoxAbs(COMPASS_BOX_PX, SIM_VIEW_SIZE_PX)
    const box = mirrorBox(SIDE_MIRROR_SIZE_PX, SIDE_MIRROR_CORNER, SIDE_MIRROR_MARGIN_PX, SIM_VIEW_SIZE_PX)
    expect(box.left).toBeGreaterThanOrEqual(0)
    expect(box.right).toBeLessThanOrEqual(SIM_VIEW_SIZE_PX.width)
    expect(box.bottom).toBeLessThanOrEqual(SIM_VIEW_SIZE_PX.height)
    expect(boxesOverlap(box, compass)).toBe(false)
  })

  it('화면이 줄어들어도(가로 320~640 px, 4:3) 겹치지 않는다', () => {
    for (let width = 320; width <= 640; width += 20) {
      const sim = { width, height: (width * 3) / 4 }
      const compass = compassBoxAbs(COMPASS_BOX_PX, sim)
      const box = mirrorBox(SIDE_MIRROR_SIZE_PX, SIDE_MIRROR_CORNER, SIDE_MIRROR_MARGIN_PX, sim)
      expect(boxesOverlap(box, compass), `width ${width}`).toBe(false)
    }
  })

  it('겹침 판정이 실제로 겹칠 때는 true를 낸다(빈 확인이 아님)', () => {
    const compass = compassBoxAbs(COMPASS_BOX_PX, SIM_VIEW_SIZE_PX)
    const tall = mirrorBox({ width: 160, height: 460 }, 'bottom-right', 6, SIM_VIEW_SIZE_PX)
    expect(boxesOverlap(tall, compass)).toBe(true)
  })

  it('왼쪽 아래로 바꾸면 왼쪽 가장자리에 붙는다', () => {
    const box = mirrorBox(SIDE_MIRROR_SIZE_PX, 'bottom-left', 6, SIM_VIEW_SIZE_PX)
    expect(box.left).toBe(6)
    expect(box.bottom).toBe(SIM_VIEW_SIZE_PX.height - 6)
  })
})

describe('다른 파일과 같은 값', () => {
  it('COMPASS_BOX_PX == index.css의 .view-compass', () => {
    const css = readFileSync(new URL('../index.css', import.meta.url), 'utf8')
    const block = css.match(/\.view-compass\s*\{([^}]*)\}/)[1]
    const px = (name) => Number(block.match(new RegExp(`${name}:\\s*(\\d+)px`))[1])
    expect({ top: px('top'), right: px('right'), width: px('width'), height: px('height') }).toEqual(COMPASS_BOX_PX)
  })

  it('SIM_VIEW_SIZE_PX == server/config.py의 RENDER_WIDTH_PX / RENDER_HEIGHT_PX', () => {
    const py = readFileSync(new URL('../../../server/config.py', import.meta.url), 'utf8')
    expect(Number(py.match(/^RENDER_WIDTH_PX = (\d+)/m)[1])).toBe(SIM_VIEW_SIZE_PX.width)
    expect(Number(py.match(/^RENDER_HEIGHT_PX = (\d+)/m)[1])).toBe(SIM_VIEW_SIZE_PX.height)
  })
})
