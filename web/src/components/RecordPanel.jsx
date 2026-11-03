// 녹화 상자: 시연을 녹화하는 버튼과 상태(서버 state.record). 로직은 record/view.js(순수 계산)에 있고 여기서는 그리기와 이벤트만 한다.
//
// 키보드 높이(↑↓, W/S)와 충돌하지 않게: 높이 키는 window에서 받으므로 버튼에 포커스가 있어도 동작한다. 반대로 버튼을 누른 뒤
// 포커스가 남아 Enter/Space가 같은 버튼을 다시 누르는 일이 없도록 클릭 직후 blur()한다.

import { useEffect, useRef, useState } from 'react'
import { createConfirm } from '../record/confirm.js'
import { buttonStates, commandFor, eventLevel, statusText, stillIsHigh, summaryRows } from '../record/view.js'

export default function RecordPanel({ record, connected, onCommand }) {
  const rec = connected ? (record ?? null) : null
  const states = buttonStates(rec, connected)
  const [armed, setArmed] = useState(false) // 삭제 확인 상태
  const confirmRef = useRef(null)

  useEffect(() => {
    const confirm = createConfirm({ onChange: setArmed })
    confirmRef.current = confirm
    return () => confirm.cancel()
  }, [])

  // 삭제 버튼이 눌릴 수 없는 상태가 되면(녹화 시작, 연결 끊김 등) 확인 상태를 푼다.
  const deleteEnabled = states.delete_last.enabled
  useEffect(() => {
    if (!deleteEnabled) confirmRef.current?.cancel()
  }, [deleteEnabled])

  function click(id, event) {
    event.currentTarget.blur()
    if (id === 'delete_last') {
      if (!confirmRef.current.press()) return // 첫 번째 누름은 확인 상태로만 바꾼다
    } else {
      confirmRef.current.cancel()
    }
    const msg = commandFor(id, rec)
    if (msg) onCommand(msg)
  }

  function button(id, label, extraClass = '') {
    const s = states[id]
    return (
      <button
        key={id}
        type="button"
        className={`record-button ${extraClass}`.trim()}
        disabled={!s.enabled}
        title={s.enabled ? '' : s.reason}
        onClick={(event) => click(id, event)}
      >
        {label}
      </button>
    )
  }

  const event = rec?.last_event
  const high = rec?.recording && stillIsHigh(rec.still_ratio, rec.still_longest_s)
  const lastIndex = rec?.last_episode?.index

  return (
    <div className="panel record-panel">
      <h2 className="record-title">시연 녹화</h2>
      <p className={rec?.recording ? 'record-status recording' : 'record-status'}>{statusText(rec, connected)}</p>

      <div className="record-buttons">
        {button('reset', '블록 리셋(다음 시드)')}
        {button('start', '녹화 시작', 'record-start')}
        {button('stop_success', '끝: 성공', 'record-success')}
        {button('stop_fail', '끝: 실패', 'record-fail')}
        {button('discard', '취소(버리기)')}
        {button(
          'delete_last',
          armed ? `정말 삭제? (에피소드 ${lastIndex}) 한 번 더 누르세요` : '마지막 에피소드 삭제',
          armed ? 'record-delete armed' : 'record-delete',
        )}
      </div>

      {rec?.success_condition_met && <p className="record-flag success">성공 조건 달성 (블록이 목표 위에 놓였고 집게가 열려 있음). 끝: 성공을 눌러 저장하세요</p>}
      {high && (
        <p className="record-flag warn">
          정지 구간이 많습니다: 손이 화면 밖이거나 멈춰 있으면 로봇이 서 있는 구간이 길어져 학습에 나쁩니다. 망설였다면 "취소"하고 다시 찍으세요
        </p>
      )}
      {event && <p className={`record-event ${eventLevel(event)}`}>{event.text}</p>}
      {rec && rec.save_error_count > 0 && (
        <p className="record-event error">저장 실패가 {rec.save_error_count}번 있었습니다. 실패한 에피소드는 성공/실패 수에 들어 있지 않습니다.</p>
      )}

      <table>
        <tbody>
          {summaryRows(rec).map(([name, value]) => (
            <tr key={name}>
              <th>{name}</th>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        순서: 블록 리셋 → 녹화 시작 → (블록을 집어 목표에 놓기) → 끝: 성공 또는 끝: 실패. 틀렸거나 망설였으면 "취소"로 버립니다. 높이 키(↑↓, W/S)는 버튼에
        포커스가 있어도 동작합니다.
      </p>
    </div>
  )
}
