// AUTO(학습된 정책) 상자: 학습된 정책이 혼자 로봇을 움직이게 하는 버튼과 상태(서버 state.auto). 로직은 auto/view.js(순수 계산)에 있다.
//
// 키보드 높이(↑↓, W/S)와 충돌하지 않게: 높이 키는 window에서 받으므로 버튼에 포커스가 있어도 동작한다. 반대로 버튼을 누른 뒤
// 포커스가 남아 Enter/Space가 같은 버튼을 다시 누르는 일이 없도록 클릭 직후 blur()한다.
// AUTO 중에는 손·키보드 입력이 로봇에 닿지 않는다(서버가 버린다). 블록 리셋은 시연 녹화 상자의 버튼을 쓴다.

import { AUTO_BUTTON_IDS, buttonStates, commandFor, eventLevel, policyText, statusClass, statusText, summaryRows } from '../auto/view.js'

const LABELS = {
  start: 'AUTO 시작',
  stop: '중지',
  shutdown_policy: '정책 프로세스 지금 끄기',
}

export default function AutoPanel({ auto, record, connected, onCommand }) {
  const a = connected ? (auto ?? null) : null
  const states = buttonStates(a, record, connected)

  function click(id, event) {
    event.currentTarget.blur()
    const msg = commandFor(id)
    if (msg) onCommand(msg)
  }

  function button(id, extraClass = '') {
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
        {LABELS[id]}
      </button>
    )
  }

  const event = a?.last_event
  return (
    <div className="panel record-panel auto-panel">
      <h2 className="record-title">AUTO (학습된 정책)</h2>
      <p className={`record-status auto-status ${statusClass(a)}`}>{statusText(a, record, connected)}</p>

      <div className="record-buttons">
        {button('start', 'record-start')}
        {button('stop', 'record-fail')}
      </div>

      {a?.status === 'success' && <p className="record-flag success">성공! 블록이 목표 위에 놓였고 집게가 열려 있습니다</p>}
      {event && <p className={`record-event ${eventLevel(event)}`}>{event.text}</p>}

      <table>
        <tbody>
          {summaryRows(a).map(([name, value]) => (
            <tr key={name}>
              <th>{name}</th>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="auto-policy">
        <span className="note">{policyText(a, connected)}</span>
        {button('shutdown_policy', 'auto-shutdown')}
      </div>
      <p className="note">
        순서: 시연 녹화 상자에서 블록 리셋 → AUTO 시작(첫 시작은 정책을 불러오느라 몇 초 걸립니다) → 결과 확인. AUTO 중에는 손과 키보드 입력이 무시되고,
        녹화와 블록 리셋은 쓸 수 없습니다. 쓰지 않으면 정책 프로세스는 스스로 꺼집니다(공용 서버라서 바로 끄려면 위 버튼).
      </p>
    </div>
  )
}

// 버튼 종류가 바뀌면 LABELS도 같이 바꾼다(테스트가 확인한다).
export const LABELED_BUTTONS = AUTO_BUTTON_IDS
