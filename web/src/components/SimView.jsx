// 서버가 보낸 시뮬레이션 화면 + 보기 방향 조작. url은 JPEG Blob 주소(App이 만들고 해제한다).
//
// 마우스: 화면 위에서 끌면(좌우 = 방위각, 상하 = 고도각) 돌고, 휠은 거리(확대/축소). 프리셋 버튼 3개.
// 각도 계산은 view/camera.js(순수 함수)에서 하고 여기서는 이벤트만 넘긴다.

import { useEffect, useRef } from 'react'
import { formatView } from '../view/camera.js'
import { PRESET_LABELS } from '../view/config.js'
import ViewCompass from './ViewCompass.jsx'

const WHEEL_LINE_TO_PX = 33 // deltaMode가 "줄"인 마우스(주로 Firefox)를 픽셀 단위로 맞춘다

// mode: 서버의 매핑 방식('relative' | 'absolute'). 아직 모르면 기본인 상대 모드로 본다.
export default function SimView({ url, view, mode, onDrag, onWheel, onPreset }) {
  const relative = mode !== 'absolute'
  const frameRef = useRef(null)
  const lastPointerRef = useRef(null) // 끄는 중이면 마지막 포인터 위치 {x, y}

  // 휠은 페이지가 같이 스크롤되면 안 돼서 preventDefault가 필요하다. React의 onWheel은 passive라 직접 등록한다.
  useEffect(() => {
    const el = frameRef.current
    if (!el) return undefined
    function handleWheel(event) {
      event.preventDefault()
      onWheel(event.deltaY * (event.deltaMode === 1 ? WHEEL_LINE_TO_PX : 1))
    }
    el.addEventListener('wheel', handleWheel, { passive: false })
    return () => el.removeEventListener('wheel', handleWheel)
  }, [onWheel])

  function handlePointerDown(event) {
    event.currentTarget.setPointerCapture(event.pointerId)
    lastPointerRef.current = { x: event.clientX, y: event.clientY }
  }

  function handlePointerMove(event) {
    const last = lastPointerRef.current
    if (!last) return
    lastPointerRef.current = { x: event.clientX, y: event.clientY }
    onDrag(event.clientX - last.x, event.clientY - last.y)
  }

  function handlePointerEnd() {
    lastPointerRef.current = null
  }

  return (
    <div className="sim-column">
      <div className="preset-row">
        <span className="preset-title">보기 방향:</span>
        {PRESET_LABELS.map(([name, label]) => (
          <button key={name} type="button" onClick={() => onPreset(name)}>
            {label}
          </button>
        ))}
      </div>

      <div
        ref={frameRef}
        className="sim-view sim-view-interactive"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerEnd}
        onPointerCancel={handlePointerEnd}
        title="끌어서 돌리기, 휠로 확대/축소"
      >
        {url ? (
          <img src={url} alt="시뮬레이션 화면" draggable={false} />
        ) : (
          <div className="placeholder">시뮬레이션 화면 기다리는 중</div>
        )}
        <ViewCompass view={view} mode={mode} />
      </div>

      <div className="legend">
        <span className="legend-item">
          <span className="legend-swatch legend-block" /> 빨강 = 블록
        </span>
        <span className="legend-item">
          <span className="legend-swatch legend-target" /> 초록 = 목표
        </span>
        <span className="view-readout">{formatView(view)}</span>
      </div>
      <p className="note">화면 위에서 마우스로 끌면 돌고, 휠로 확대/축소합니다.</p>
      <p className="hand-direction-note">
        {relative ? '손 방향은 현재 보기 기준' : '손 방향은 위에서 본 화면 기준 (절대 모드)'}
      </p>
      <p className="note">
        보기 방향은 서버에 하나뿐입니다. 브라우저를 여러 개 열어 두면 마지막에 바꾼 값이 모두에게 적용됩니다.
      </p>
    </div>
  )
}
