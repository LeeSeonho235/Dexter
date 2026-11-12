import { useCallback, useEffect, useRef, useState } from 'react'
import AutoPanel from './components/AutoPanel.jsx'
import ConnectionPanel from './components/ConnectionPanel.jsx'
import ControlHandPanel from './components/ControlHandPanel.jsx'
import FeaturePanel from './components/FeaturePanel.jsx'
import HandView from './components/HandView.jsx'
import RecordPanel from './components/RecordPanel.jsx'
import RobotPanel from './components/RobotPanel.jsx'
import SimView from './components/SimView.jsx'
import { handEdgeStatus } from './hand/edge.js'
import { useHeightKeys } from './input/useHeightKeys.js'
import { useTakeoverKey } from './input/useTakeoverKey.js'
import { STATS_REFRESH_MS } from './net/config.js'
import { SimConnection, wsUrl } from './net/connection.js'
import { applyDrag, applyWheel, pickDisplayedView, presetView, viewsEqual } from './view/camera.js'

export default function App() {
  const [cameraOn, setCameraOn] = useState(false)
  const [hand, setHand] = useState(null)
  const [selection, setSelection] = useState(null) // 보이는 손들과 조종 손 (HandView → selectHand)
  const [edge, setEdge] = useState(false) // 손이 영상 가장자리 10% 안쪽에 들어왔는가(같은 값이면 다시 그리지 않는다)
  const [serverState, setServerState] = useState(null) // 서버 state 메시지(10Hz). 막대와 숫자를 부드럽게 보이려고 바로 받는다.
  const [status, setStatus] = useState({ phase: 'idle' })
  const [frameUrl, setFrameUrl] = useState(null)
  const [netStats, setNetStats] = useState({ phase: 'idle' })
  // 보기 방향(자유 카메라). 서버에는 하나뿐이라 화면 숫자는 서버가 알려 주는 값을 따라간다(직접 조작한 직후 제외).
  const [view, setView] = useState(() => presetView('default'))
  const viewRef = useRef(view) // 드래그가 연달아 들어와도 항상 최신 값에서 계산하려고 ref도 둔다
  const lastViewInteractionRef = useRef(-Infinity) // 마지막으로 직접 조작한 시각(ms, performance.now)
  const connRef = useRef(null)
  const frameUrlRef = useRef(null)
  const autoRef = useRef(null) // 서버 state.auto의 최신 값(Space 키가 지금 상태를 보고 넘겨받기/돌려주기를 고른다)
  const connectedRef = useRef(false)

  // 페이지를 열면 카메라와 상관없이 서버에 연결한다(시뮬레이션 화면은 카메라 없이도 보인다).
  useEffect(() => {
    // 새 프레임(JPEG Blob)이 오면 주소를 만들고 이전 주소는 해제해 메모리가 쌓이지 않게 한다.
    function onFrame(blob) {
      const prev = frameUrlRef.current
      frameUrlRef.current = URL.createObjectURL(blob)
      setFrameUrl(frameUrlRef.current)
      if (prev) URL.revokeObjectURL(prev)
    }
    // 서버 state(10Hz): 막대와 숫자용으로 보관하고, 서버의 보기 방향이 내 값과 다르면(다른 브라우저가 바꿨을 때) 따라간다.
    function onState(msg) {
      autoRef.current = msg.auto ?? null
      setServerState(msg)
      if (!msg.view) return
      const shown = pickDisplayedView({
        local: viewRef.current,
        server: msg.view,
        lastInteractionMs: lastViewInteractionRef.current,
        nowMs: performance.now(),
      })
      if (!viewsEqual(shown, viewRef.current)) {
        viewRef.current = shown
        setView(shown)
      }
    }
    const conn = new SimConnection({ url: wsUrl(window.location), onFrame, onState })
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
    setEdge(handEdgeStatus(sel?.chosen?.landmarks).near)
    connRef.current?.sendHand(msg)
  }, [])
  const onStatus = useCallback((s) => setStatus(s), [])

  // 보기 방향을 바꾸고 서버에 알린다(최대 15Hz, 마지막 값은 반드시 도착: net/throttle.js).
  const changeView = useCallback((next) => {
    viewRef.current = next
    lastViewInteractionRef.current = performance.now()
    setView(next)
    connRef.current?.requestView(next)
  }, [])
  const onViewDrag = useCallback((dx, dy) => changeView(applyDrag(viewRef.current, dx, dy)), [changeView])
  const onViewWheel = useCallback((deltaY) => changeView(applyWheel(viewRef.current, deltaY)), [changeView])
  const onViewPreset = useCallback((name) => changeView(presetView(name)), [changeView])

  function toggleCamera() {
    if (cameraOn) {
      setHand(null)
      setSelection(null)
      setEdge(false)
      setStatus({ phase: 'idle' })
    }
    setCameraOn(!cameraOn)
  }

  // 서버가 알려 주는 매핑 방식과 기준 손 위치(상대 모드에서 조종 중일 때만). 연결이 끊기면 오래된 값을 쓰지 않는다.
  const connected = netStats.phase === 'open'
  useEffect(() => {
    connectedRef.current = connected // Space 키가 연결 여부를 본다(렌더 중에 ref를 쓰지 않으려고 effect에서)
  }, [connected])
  const control = connected ? serverState?.control : undefined
  const mappingMode = control?.mapping_mode ?? null
  const referenceHand = control?.reference_hand ?? null

  // 키보드 높이(↓/S 내리기, ↑/W 올리기, Shift = 천천히). 눌린 키 상태를 돌려받아 화면에 표시한다.
  const heightKeys = useHeightKeys({ connRef, connected })
  // Space = 넘겨받기/돌려주기 토글(AUTO 상자의 큰 버튼과 같은 일). 버튼에 포커스가 있어도 동작하고 페이지는 스크롤되지 않는다.
  useTakeoverKey({ connRef, autoRef, connectedRef })

  // 녹화 버튼(record/view.js commandFor가 만든 메시지)을 서버로 보낸다. 서버가 거절하면 state.record.last_event에 이유가 온다.
  const onRecordCommand = useCallback((msg) => connRef.current?.sendRecord(msg), [])
  // AUTO 버튼(auto/view.js commandFor가 만든 메시지). 서버가 거절하면 state.auto.last_event에 이유가 온다.
  const onAutoCommand = useCallback((msg) => connRef.current?.sendAuto(msg), [])

  return (
    <main>
      <h1>손으로 시뮬레이션 로봇 조종</h1>
      <button onClick={toggleCamera}>{cameraOn ? '카메라 끄기' : '카메라 켜기'}</button>
      <div className="layout">
        {cameraOn ? (
          <HandView onHand={onHand} onStatus={onStatus} edge={edge} reference={referenceHand} mode={mappingMode} />
        ) : (
          <div className="hand-view placeholder">카메라 꺼짐</div>
        )}
        <SimView
          url={frameUrl}
          view={view}
          mode={mappingMode}
          onDrag={onViewDrag}
          onWheel={onViewWheel}
          onPreset={onViewPreset}
        />
      </div>
      <div className="layout">
        <RecordPanel record={serverState?.record} connected={connected} onCommand={onRecordCommand} />
        <AutoPanel
          auto={serverState?.auto}
          record={serverState?.record}
          control={serverState?.control}
          connected={connected}
          onCommand={onAutoCommand}
        />
      </div>
      <div className="layout">
        <ControlHandPanel
          selection={selection}
          cameraOn={cameraOn}
          control={serverState?.control}
          connected={connected}
          edge={edge}
          mode={mappingMode}
        />
        <RobotPanel control={serverState?.control} connected={connected} keys={heightKeys} />
      </div>
      <div className="layout">
        <FeaturePanel hand={hand} status={status} />
        <ConnectionPanel stats={netStats} />
      </div>
    </main>
  )
}
