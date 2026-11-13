// 메인 화면 구석의 옆모습 보조 화면(2-6, 보기 전용). 집게 높이, 블록, 바닥선을 Canvas 2D로 그린다.
// 계산은 scene/sideView.js(순수 함수), 크기·위치·색은 scene/config.js 한 곳. 클릭을 받지 않는다(메인 화면 끌기를 방해하지 않게).

import { useEffect, useRef } from 'react'
import {
  CAPTION_HEIGHT_PX,
  SIDE_MIRROR_CAPTION,
  SIDE_MIRROR_CORNER,
  SIDE_MIRROR_MARGIN_PX,
  SIDE_MIRROR_SIZE_PX,
} from '../scene/config.js'
import { drawShapes, makeProjection, sideViewShapes } from '../scene/sideView.js'

const PROJECTION = makeProjection(SIDE_MIRROR_SIZE_PX, CAPTION_HEIGHT_PX)

// control: 서버 state.control, scene: 서버 state.scene. 연결이 없으면 둘 다 undefined로 받는다(바닥선만 그린다).
// className: 덧붙일 클래스(진짜 보조 화면이 켜져 있을 때는 창이 좁아 그쪽이 숨겨진 경우에만 보이게 하려고 쓴다, index.css).
export default function SideMirror({ control, scene, className = '' }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    const { width, height } = SIDE_MIRROR_SIZE_PX
    if (canvas.width !== Math.round(width * dpr)) {
      canvas.width = Math.round(width * dpr) // 크기를 바꾸면 캔버스가 지워지므로 바뀔 때만 바꾼다
      canvas.height = Math.round(height * dpr)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    drawShapes(ctx, sideViewShapes({ control, scene }, PROJECTION), SIDE_MIRROR_SIZE_PX)
  }, [control, scene])

  const position = {
    bottom: SIDE_MIRROR_MARGIN_PX,
    [SIDE_MIRROR_CORNER === 'bottom-left' ? 'left' : 'right']: SIDE_MIRROR_MARGIN_PX,
    width: SIDE_MIRROR_SIZE_PX.width,
    height: SIDE_MIRROR_SIZE_PX.height,
  }

  return (
    <div className={className ? `side-mirror ${className}` : 'side-mirror'} style={position} role="img" aria-label={SIDE_MIRROR_CAPTION.join('. ')}>
      <canvas ref={canvasRef} style={{ width: SIDE_MIRROR_SIZE_PX.width, height: SIDE_MIRROR_SIZE_PX.height }} />
      <div className="side-mirror-caption" style={{ height: CAPTION_HEIGHT_PX }}>
        {SIDE_MIRROR_CAPTION.map((line) => (
          <div key={line}>{line}</div>
        ))}
      </div>
    </div>
  )
}
