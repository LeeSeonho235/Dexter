// 옆모습 보조 화면의 계산: 월드(m) → 화면(px) 변환과 그릴 도형 목록. React와 캔버스에 의존하지 않는 순수 계산.
// 입력: state.control(tip_m, gripper_closed)와 state.scene(block_m, block_half_m, target_center_m, target_half_m). 없으면 있는 것만 그린다.
// 옆에서 본 모습이라 x(앞뒤)와 z(높이)만 쓴다. y(좌우)는 버린다.

import { COLORS, DRAW_PAD_PX, TIP_RADIUS_PX, WORLD_RANGE_M } from './config.js'

const isNum = (v) => typeof v === 'number' && Number.isFinite(v)
const isVec3 = (v) => Array.isArray(v) && v.length >= 3 && isNum(v[0]) && isNum(v[1]) && isNum(v[2])

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

// size: { width, height }(캔버스 px), captionPx: 아래 문구 영역 높이. x, z 두 방향에 같은 배율을 써서 블록이 정사각형으로 보인다.
export function makeProjection(size, captionPx, pad = DRAW_PAD_PX, range = WORLD_RANGE_M) {
  const areaW = size.width - 2 * pad
  const areaH = size.height - captionPx - 2 * pad
  const [xMin, xMax] = range.x
  const [zMin, zMax] = range.z
  const scale = Math.min(areaW / (xMax - xMin), areaH / (zMax - zMin)) // px/m
  const offsetX = pad + (areaW - scale * (xMax - xMin)) / 2
  const bottomY = pad + areaH - (areaH - scale * (zMax - zMin)) / 2 // z = zMin이 놓이는 y(px)
  const toPx = (x_m, z_m) => [offsetX + (x_m - xMin) * scale, bottomY - (z_m - zMin) * scale]
  // 범위 밖 점을 그림 영역 안으로 끌어온다. 끌어왔으면 clamped = true.
  const toPxClamped = (x_m, z_m) => {
    const [px, py] = toPx(x_m, z_m)
    const cx = clamp(px, offsetX, offsetX + scale * (xMax - xMin))
    const cy = clamp(py, bottomY - scale * (zMax - zMin), bottomY)
    return { px: cx, py: cy, clamped: cx !== px || cy !== py }
  }
  return {
    scale,
    toPx,
    toPxClamped,
    left: offsetX,
    right: offsetX + scale * (xMax - xMin),
    top: bottomY - scale * (zMax - zMin),
    floorY: toPx(0, 0)[1],
  }
}

// 도형 종류: line {x1,y1,x2,y2,color,width,dash}, rect {x,y,w,h,color}, circle {x,y,r,color,hollow}
export function sideViewShapes({ control, scene } = {}, projection) {
  const p = projection
  const shapes = []

  // 항상: 바닥선, 로봇 밑(x = 0) 세로 점선
  shapes.push({ kind: 'line', x1: p.left, y1: p.floorY, x2: p.right, y2: p.floorY, color: COLORS.floor, width: 1.5, dash: null })
  const [baseX, baseTop] = p.toPx(0, WORLD_RANGE_M.z[1])
  shapes.push({ kind: 'line', x1: baseX, y1: baseTop, x2: baseX, y2: p.floorY, color: COLORS.base, width: 1, dash: [3, 3] })

  // 목표: 바닥 위 초록 선분(옆에서는 x 범위만 보인다)
  const target = scene?.target_center_m
  if (Array.isArray(target) && isNum(target[0]) && isNum(scene.target_half_m)) {
    const [x1, y] = p.toPx(target[0] - scene.target_half_m, 0)
    const [x2] = p.toPx(target[0] + scene.target_half_m, 0)
    shapes.push({ kind: 'line', x1, y1: y, x2, y2: y, color: COLORS.target, width: 3, dash: null })
  }

  // 블록: 중심 (x, z), 한 변 = 2 * block_half_m
  if (isVec3(scene?.block_m) && isNum(scene.block_half_m)) {
    const half = scene.block_half_m
    const [left, top] = p.toPx(scene.block_m[0] - half, scene.block_m[2] + half)
    const side = 2 * half * p.scale
    shapes.push({ kind: 'rect', x: left, y: top, w: side, h: side, color: COLORS.block })
  }

  // 집게 끝: 바닥까지 점선(높이)과 점. 닫힘/열림은 색으로. 범위 밖이면 가장자리에 속 빈 점으로 보여 준다.
  if (isVec3(control?.tip_m)) {
    const { px, py, clamped } = p.toPxClamped(control.tip_m[0], control.tip_m[2])
    const closed = control.gripper_closed
    const color = closed === true ? COLORS.tipClosed : closed === false ? COLORS.tipOpen : COLORS.tipUnknown
    shapes.push({ kind: 'line', x1: px, y1: py, x2: px, y2: p.floorY, color: COLORS.height, width: 1, dash: [2, 3] })
    shapes.push({ kind: 'circle', x: px, y: py, r: TIP_RADIUS_PX, color, hollow: clamped })
  }
  return shapes
}

// 캔버스 2D 컨텍스트에 도형을 그린다(좌표는 CSS px. 해상도 배율은 호출하는 쪽이 ctx.setTransform으로 맞춘다).
export function drawShapes(ctx, shapes, size) {
  ctx.clearRect(0, 0, size.width, size.height)
  for (const s of shapes) {
    if (s.kind === 'line') {
      ctx.beginPath()
      ctx.setLineDash(s.dash ?? [])
      ctx.strokeStyle = s.color
      ctx.lineWidth = s.width
      ctx.moveTo(s.x1, s.y1)
      ctx.lineTo(s.x2, s.y2)
      ctx.stroke()
    } else if (s.kind === 'rect') {
      ctx.fillStyle = s.color
      ctx.fillRect(s.x, s.y, s.w, s.h)
    } else if (s.kind === 'circle') {
      ctx.beginPath()
      ctx.setLineDash([])
      ctx.arc(s.x, s.y, s.r, 0, 2 * Math.PI)
      if (s.hollow) {
        ctx.strokeStyle = s.color
        ctx.lineWidth = 2
        ctx.stroke()
      } else {
        ctx.fillStyle = s.color
        ctx.fill()
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 1
        ctx.stroke()
      }
    }
  }
  ctx.setLineDash([])
}

// 보조 화면 상자의 위치(px, 메인 화면 왼쪽 위 기준)와 나침반 상자가 겹치는지. 겹침 확인용 순수 계산.
export function mirrorBox(sizePx, corner, marginPx, simSizePx) {
  const left = corner === 'bottom-left' ? marginPx : simSizePx.width - marginPx - sizePx.width
  const top = simSizePx.height - marginPx - sizePx.height
  return { left, top, right: left + sizePx.width, bottom: top + sizePx.height }
}

export function compassBoxAbs(compass, simSizePx) {
  const right = simSizePx.width - compass.right
  return { left: right - compass.width, top: compass.top, right, bottom: compass.top + compass.height }
}

export function boxesOverlap(a, b) {
  return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom
}
