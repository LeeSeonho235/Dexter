// 웹캠 영상 + 21개 점 그리기. 영상은 화면에만 보여주고 저장하거나 보내지 않는다.

import { useEffect, useRef } from 'react'
import { DrawingUtils } from '@mediapipe/tasks-vision'
import { VIDEO_HEIGHT_PX, VIDEO_WIDTH_PX } from '../hand/config.js'
import { makeHandMessage } from '../hand/features.js'
import { createHandTracker, HandLandmarker } from '../hand/tracker.js'

export default function HandView({ onHand, onStatus }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

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
          const landmarks = result.landmarks[0]

          ctx.clearRect(0, 0, canvas.width, canvas.height)
          if (landmarks) {
            drawer.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
              color: '#00d0ff',
              lineWidth: 3,
            })
            drawer.drawLandmarks(landmarks, { color: '#00ff66', lineWidth: 1, radius: 4 })
          }
          onHand(makeHandMessage(landmarks, { t: Date.now() / 1000, aspect }))

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
    <div className="hand-view">
      <video ref={videoRef} className="mirror" playsInline muted />
      <canvas ref={canvasRef} className="mirror" />
    </div>
  )
}
