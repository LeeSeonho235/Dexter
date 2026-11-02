import { useCallback, useEffect, useRef, useState } from 'react'
import ConnectionPanel from './components/ConnectionPanel.jsx'
import ControlHandPanel from './components/ControlHandPanel.jsx'
import FeaturePanel from './components/FeaturePanel.jsx'
import HandView from './components/HandView.jsx'
import RobotPanel from './components/RobotPanel.jsx'
import SimView from './components/SimView.jsx'
import { STATS_REFRESH_MS } from './net/config.js'
import { SimConnection, wsUrl } from './net/connection.js'

export default function App() {
  const [cameraOn, setCameraOn] = useState(false)
  const [hand, setHand] = useState(null)
  const [selection, setSelection] = useState(null) // 보이는 손들과 조종 손 (HandView → selectHand)
  const [serverState, setServerState] = useState(null) // 서버 state 메시지(10Hz). 막대와 숫자를 부드럽게 보이려고 바로 받는다.
  const [status, setStatus] = useState({ phase: 'idle' })
  const [frameUrl, setFrameUrl] = useState(null)
  const [netStats, setNetStats] = useState({ phase: 'idle' })
  const connRef = useRef(null)
  const frameUrlRef = useRef(null)

  // 페이지를 열면 카메라와 상관없이 서버에 연결한다(시뮬레이션 화면은 카메라 없이도 보인다).
  useEffect(() => {
    // 새 프레임(JPEG Blob)이 오면 주소를 만들고 이전 주소는 해제해 메모리가 쌓이지 않게 한다.
    function onFrame(blob) {
      const prev = frameUrlRef.current
      frameUrlRef.current = URL.createObjectURL(blob)
      setFrameUrl(frameUrlRef.current)
      if (prev) URL.revokeObjectURL(prev)
    }
    const conn = new SimConnection({ url: wsUrl(window.location), onFrame, onState: setServerState })
    connRef.current = conn
    conn.start()
    const timer = setInterval(() => setNetStats(conn.getStats()), STATS_REFRESH_MS)
    return () => {
      clearInterval(timer)
      conn.stop()
      connRef.current = null
      if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current)
      frameUrlRef.current = null
      setFrameUrl(null)
    }
  }, [])

  // 손 추적 결과(숫자)만 서버로 보낸다. 카메라가 꺼져 있으면 HandView가 없어서 보내지 않는다.
  // 서버에는 조종 손 하나의 값만 간다(다른 손은 selection에만 있고 보내지 않는다).
  const onHand = useCallback((msg, sel) => {
    setHand(msg)
    setSelection(sel ?? null)
    connRef.current?.sendHand(msg)
  }, [])
  const onStatus = useCallback((s) => setStatus(s), [])

  function toggleCamera() {
    if (cameraOn) {
      setHand(null)
      setSelection(null)
      setStatus({ phase: 'idle' })
    }
    setCameraOn(!cameraOn)
  }

  return (
    <main>
      <h1>손으로 시뮬레이션 로봇 조종</h1>
      <button onClick={toggleCamera}>{cameraOn ? '카메라 끄기' : '카메라 켜기'}</button>
      <div className="layout">
        {cameraOn ? (
          <HandView onHand={onHand} onStatus={onStatus} />
        ) : (
          <div className="hand-view placeholder">카메라 꺼짐</div>
        )}
        <SimView url={frameUrl} />
      </div>
      <div className="layout">
        <ControlHandPanel selection={selection} cameraOn={cameraOn} />
        <RobotPanel control={serverState?.control} connected={netStats.phase === 'open'} />
      </div>
      <div className="layout">
        <FeaturePanel hand={hand} status={status} />
        <ConnectionPanel stats={netStats} />
      </div>
    </main>
  )
}
