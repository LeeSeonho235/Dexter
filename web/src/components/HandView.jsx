// 웹캠 영상 + 21개 점 그리기. 영상은 화면에만 보여주고 저장하거나 보내지 않는다.

import { useEffect, useRef } from 'react'
import { DrawingUtils } from '@mediapipe/tasks-vision'
import { CONTROL_HANDEDNESS, VIDEO_HEIGHT_PX, VIDEO_WIDTH_PX } from '../hand/config.js'
import { overlayGeometry } from '../hand/edge.js'
import { makeHandMessage } from '../hand/features.js'
import { selectHand } from '../hand/select.js'
import { createHandTracker, HandLandmarker } from '../hand/tracker.js'

// 조종 손은 굵은 노란 선, 나머지 손은 회색 (조종에 쓰지 않는다)
const CONTROL_STYLE = { line: '#ffd400', lineWidth: 6, point: '#00ff66', radius: 5 }
const OTHER_STYLE = { line: '#777777', lineWidth: 2, point: '#999999', radius: 3 }

// 상대 모드 안내: 기준점은 십자(하늘색), 지금 손목은 점(주황), 둘 사이는 선(흰색 점선). 캔버스는 CSS로 영상과 같이 좌우가
// 뒤집혀 보이므로 점 좌표를 그대로 쓰면 영상 위의 손과 같은 자리에 그려진다. 글자는 캔버스가 아니라 HTML로 보여 준다.
function drawReferenceOverlay(ctx, geometry) {
  const { cross, dot, line } = geometry
  ctx.save()
  ctx.setLineDash([6, 5])
  ctx.strokeStyle = '#ffffff'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(line.x1, line.y1)
  ctx.lineTo(line.x2, line.y2)
  ctx.stroke()
  ctx.setLineDash([])
  ctx.strokeStyle = '#00e5ff'
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.moveTo(cross.x - cross.half, cross.y)
  ctx.lineTo(cross.x + cross.half, cross.y)
  ctx.moveTo(cross.x, cross.y - cross.half)
  ctx.lineTo(cross.x, cross.y + cross.half)
  ctx.stroke()
  ctx.fillStyle = '#ff9100'
  ctx.beginPath()
  ctx.arc(dot.x, dot.y, 7, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

// edge: 손이 영상 가장자리 10% 안쪽에 들어왔는가(테두리를 빨갛게). reference: 서버가 알려 준 기준 손 위치 [x, y]
// (정규화, 상대 모드에서 조종 중일 때만 값이 있다). mode: 'relative' | 'absolute'
export default function HandView({ onHand, onStatus, edge = false, reference = null, mode = null }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const overlayRef = useRef({ reference: null, mode: null }) // 그리기 루프가 최신 값을 읽도록 ref에 둔다

  useEffect(() => {
    overlayRef.current = { reference, mode }
  }, [reference, mode])

  useEffect(() => {
    let cancelled = false
    let stream = null
    let landmarker = null
    let rafId = 0

    async function start() {
      onStatus({ phase: 'loading' })
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: VIDEO_WIDTH_PX, height: VIDEO_HEIGHT_PX },
        audio: false,
      })
      if (cancelled) return
      const video = videoRef.current
      video.srcObject = stream
      await video.play()

      const tracker = await createHandTracker()
      if (cancelled) {
        tracker.landmarker.close()
        return
      }
      landmarker = tracker.landmarker

      const canvas = canvasRef.current
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      const drawer = new DrawingUtils(ctx)
      const aspect = video.videoWidth / video.videoHeight

      let lastVideoTime = -1
      let frames = 0
      let fpsStart = performance.now()

      function loop() {
        if (cancelled) return
        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime
          const now = performance.now()
          const result = landmarker.detectForVideo(video, now)
          // 손은 2개까지 인식하되 라벨이 맞는 손 하나만 조종에 쓴다. 다른 손은 서버로 보내지 않는다.
          const selection = selectHand(result, CONTROL_HANDEDNESS)

          ctx.clearRect(0, 0, canvas.width, canvas.height)
          result.landmarks.forEach((landmarks, index) => {
            const style = index === selection.chosen?.index ? CONTROL_STYLE : OTHER_STYLE
            drawer.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
              color: style.line,
              lineWidth: style.lineWidth,
            })
            drawer.drawLandmarks(landmarks, { color: style.point, lineWidth: 1, radius: style.radius })
          })
          const overlay = overlayRef.current
          const wristPoint = selection.chosen?.landmarks[0]
          if (overlay.mode === 'relative' && wristPoint) {
            const geometry = overlayGeometry(
              { reference: overlay.reference, wrist: [wristPoint.x, wristPoint.y] },
              canvas.width,
              canvas.height,
            )
            if (geometry) drawReferenceOverlay(ctx, geometry)
          }
          onHand(makeHandMessage(selection.chosen?.landmarks, { t: Date.now() / 1000, aspect }), selection)

          frames += 1
          if (now - fpsStart >= 1000) {
            onStatus({
              phase: 'running',
              delegate: tracker.delegate,
              fps: (frames * 1000) / (now - fpsStart),
              size: `${video.videoWidth}x${video.videoHeight}`,
            })
            frames = 0
            fpsStart = now
          }
        }
        rafId = requestAnimationFrame(loop)
      }
      onStatus({ phase: 'running', delegate: tracker.delegate, fps: 0, size: `${video.videoWidth}x${video.videoHeight}` })
      loop()
    }

    start().catch((err) => {
      if (!cancelled) onStatus({ phase: 'error', message: String(err?.message ?? err) })
    })

    return () => {
      cancelled = true
      cancelAnimationFrame(rafId)
      stream?.getTracks().forEach((track) => track.stop())
      landmarker?.close()
    }
  }, [onHand, onStatus])

  return (
    <div className={edge ? 'hand-view edge-warning' : 'hand-view'}>
      <video ref={videoRef} className="mirror" playsInline muted />
      <canvas ref={canvasRef} className="mirror" />
    </div>
  )
}
