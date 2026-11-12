// AUTO(학습된 정책) 상자: 학습된 정책이 혼자 로봇을 움직이게 하는 버튼과 상태(서버 state.auto). 로직은 auto/view.js, auto/correction.js(순수 계산)에 있다.
//
// 교정(4-2): 정책이 틀리려는 순간 사람이 넘겨받고(Space 또는 큰 토글 버튼), 고친 뒤 다시 정책에게 돌려준다. 지금 누가 조종하는지는 큰 문구와
// 배경색·가장자리 색으로 함께 알린다(색만으로 구분하지 않는다). 끝난 뒤 사람이 고친 구간이 있으면 "교정 저장 / 버리기"를 고를 때까지 저장하지 않는다.
//
// 키보드와 충돌하지 않게: 높이 키(↑↓, W/S)와 Space(넘겨받기)는 window에서 받으므로 버튼에 포커스가 있어도 동작한다. 반대로 버튼을 누른 뒤
// 포커스가 남아 Enter/Space가 같은 버튼을 다시 누르는 일이 없도록 클릭 직후 blur()한다(입력창은 쓰지 않는다).
// AUTO 중에는 손·키보드 입력이 로봇에 닿지 않는다(서버가 버린다). 넘겨받아 HUMAN이 되면 손과 높이 키가 다시 로봇을 움직인다.

import {
  correctionRows,
  decisionPending,
  decisionText,
  gripperHint,
  ownerBanner,
  ownerKind,
  seedPurposeWarning,
  toggleButton,
} from '../auto/correction.js'
import { AUTO_BUTTON_IDS, buttonStates, commandFor, eventLevel, policyText, statusClass, statusText, summaryRows } from '../auto/view.js'

const LABELS = {
  start: 'AUTO 시작',
  stop: '중지',
  shutdown_policy: '정책 프로세스 지금 끄기',
}

export default function AutoPanel({ auto, record, control, connected, onCommand }) {
  const a = connected ? (auto ?? null) : null
  const states = buttonStates(a, record, connected)
  const kind = ownerKind(a, connected)
  const banner = ownerBanner(a, connected)
  const toggle = toggleButton(a, connected)
  const hint = gripperHint(a, connected ? control : null, connected)
  const pending = decisionPending(a)
  const warning = seedPurposeWarning(a)

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
    <div className={`panel record-panel auto-panel owner-${kind}`}>
      <h2 className="record-title">AUTO (학습된 정책)</h2>

      {banner && (
        // key가 바뀌면(주체가 바뀌면) 다시 만들어져서 깜빡이는 효과가 한 번 재생된다
        <div key={banner.kind} className={`owner-banner owner-banner-${banner.kind}`} role="status" aria-live="polite">
          <div className="owner-banner-title">
            <span className="owner-banner-mark" aria-hidden="true">
              {banner.mark}
            </span>{' '}
            {banner.title}
          </div>
          {banner.detail && <div className="owner-banner-detail">{banner.detail}</div>}
        </div>
      )}
      {hint && (
        <div className="gripper-hint" role="alert">
          <strong>{hint.text}</strong>
          <span>{hint.detail}</span>
        </div>
      )}

      <p className={`record-status auto-status ${statusClass(a)}`}>{statusText(a, record, connected)}</p>

      <div className="record-buttons">
        {button('start', 'record-start')}
        {button('stop', 'record-fail')}
      </div>
      <div className="record-buttons">
        <button
          type="button"
          className={`takeover-button takeover-${toggle.id}`}
          disabled={!toggle.enabled}
          title={toggle.enabled ? '' : toggle.reason}
          onClick={(e) => click(toggle.id, e)}
        >
          {toggle.label}
        </button>
      </div>

      {pending && (
        <div className="decision-box" role="alert">
          <p className="decision-text">{decisionText(a)}</p>
          <div className="record-buttons">
            <button type="button" className="record-button record-start" onClick={(e) => click('save_correction', e)}>
              교정 저장
            </button>
            <button type="button" className="record-button record-fail" onClick={(e) => click('discard_correction', e)}>
              버리기
            </button>
          </div>
          <p className="note">고르기 전에는 저장하지 않고, 새 AUTO와 블록 리셋도 할 수 없습니다.</p>
        </div>
      )}

      {a?.status === 'success' && <p className="record-flag success">성공! 블록이 목표 위에 놓였고 집게가 열려 있습니다</p>}
      {warning && <p className="record-flag warn">{warning}</p>}
      {event && <p className={`record-event ${eventLevel(event)}`}>{event.text}</p>}

      <table>
        <tbody>
          {summaryRows(a).map(([name, value]) => (
            <tr key={name}>
              <th>{name}</th>
              <td>{value}</td>
            </tr>
          ))}
          {correctionRows(a).map(([name, value]) => (
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
        순서: 시연 녹화 상자에서 블록 리셋 → AUTO 시작(첫 시작은 정책을 불러오느라 몇 초 걸립니다) → 로봇이 틀리려 하면 Space로 넘겨받기 → 손을 화면 중앙에
        보이기 → 고치기 → Space로 돌려주기 → 끝나면 교정 저장/버리기. AUTO 중에는 손과 키보드 입력이 무시되고, 녹화와 블록 리셋은 쓸 수 없습니다. 쓰지 않으면
        정책 프로세스는 스스로 꺼집니다(공용 서버라서 바로 끄려면 위 버튼).
      </p>
    </div>
  )
}

// 버튼 종류가 바뀌면 LABELS도 같이 바꾼다(테스트가 확인한다).
export const LABELED_BUTTONS = AUTO_BUTTON_IDS
