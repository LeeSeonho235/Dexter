import { useCallback, useState } from 'react'
import FeaturePanel from './components/FeaturePanel.jsx'
import HandView from './components/HandView.jsx'

export default function App() {
  const [cameraOn, setCameraOn] = useState(false)
  const [hand, setHand] = useState(null)
  const [status, setStatus] = useState({ phase: 'idle' })

  const onHand = useCallback((msg) => setHand(msg), [])
  const onStatus = useCallback((s) => setStatus(s), [])

  function toggleCamera() {
    if (cameraOn) {
      setHand(null)
      setStatus({ phase: 'idle' })
    }
    setCameraOn(!cameraOn)
  }

  return (
    <main>
      <h1>손 추적 확인</h1>
      <button onClick={toggleCamera}>{cameraOn ? '카메라 끄기' : '카메라 켜기'}</button>
      <div className="layout">
        {cameraOn ? (
          <HandView onHand={onHand} onStatus={onStatus} />
        ) : (
          <div className="hand-view placeholder">카메라 꺼짐</div>
        )}
        <FeaturePanel hand={hand} status={status} />
      </div>
    </main>
  )
}
