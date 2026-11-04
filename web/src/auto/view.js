// AUTO 상자의 표시 규칙(React에 의존하지 않는 순수 계산). 서버 state.auto 형식은 docs/12-auto.md.
//
// 버튼을 누를 수 있는지는 서버가 같은 규칙으로 다시 검사한다(화면은 안내일 뿐이다).

export const AUTO_BUTTON_IDS = ['start', 'stop', 'shutdown_policy']

const BUSY = ['loading', 'running']

// 버튼 id → 서버로 보낼 메시지. 모르는 id면 null.
export function commandFor(buttonId) {
  return AUTO_BUTTON_IDS.includes(buttonId) ? { type: 'auto', action: buttonId } : null
}

// 각 버튼이 지금 눌릴 수 있는지와 안 될 때의 이유.
// auto가 없으면(연결 전, AUTO를 모르는 서버) 모두 꺼진다. 시작은 블록 리셋 직후(record.block_ready)에만 켜진다.
export function buttonStates(auto, record, connected) {
  const off = (reason) => ({ enabled: false, reason })
  const on = { enabled: true, reason: '' }
  if (!connected) return Object.fromEntries(AUTO_BUTTON_IDS.map((id) => [id, off('서버에 연결되지 않음')]))
  if (!auto) return Object.fromEntries(AUTO_BUTTON_IDS.map((id) => [id, off('서버가 AUTO 상태를 보내지 않음')]))

  const busy = BUSY.includes(auto.status)
  return {
    start: busy
      ? off('AUTO 실행 중')
      : record?.recording
        ? off('녹화 중에는 시작할 수 없음')
        : record?.resetting
          ? off('블록 리셋 중')
          : record?.block_ready
            ? on
            : off('블록 리셋을 먼저 하세요'),
    stop: busy ? on : off('AUTO가 실행 중이 아님'),
    shutdown_policy: busy
      ? off('AUTO 실행 중에는 끌 수 없음')
      : auto.policy?.state === 'on'
        ? on
        : off('정책 프로세스가 꺼져 있음'),
  }
}

// 시간 표시: 초 → "N초" (60초 이상이면 "M분 N초")
export function formatSeconds(seconds) {
  if (seconds == null) return '—'
  const s = Math.max(0, Math.round(seconds))
  return s >= 60 ? `${Math.floor(s / 60)}분 ${s % 60}초` : `${s}초`
}

// 상태 한 줄
export function statusText(auto, record, connected) {
  if (!connected) return '서버에 연결되지 않음'
  if (!auto) return 'AUTO 상태를 모름 (서버가 AUTO 기능을 지원하지 않거나 아직 상태를 안 보냄)'
  switch (auto.status) {
    case 'loading': {
      const n = auto.load_estimate_s ?? 3
      const waited = auto.loading_s == null ? '' : ` (${auto.loading_s.toFixed(0)}초 지남)`
      return `정책 불러오는 중, 약 ${n}초${waited}`
    }
    case 'running':
      return `● AUTO 실행 중: 시드 ${auto.seed}`
    case 'success':
      return `성공! 시드 ${auto.result?.seed ?? auto.seed}, ${auto.result?.time_s ?? '—'}초`
    case 'timeout':
      return `시간 초과: 시드 ${auto.result?.seed ?? auto.seed}, ${auto.max_s}초 안에 성공하지 못함`
    case 'stopped':
      return `중지함: 시드 ${auto.result?.seed ?? auto.seed ?? '—'}`
    case 'error':
      return `오류: ${auto.error || '알 수 없는 오류'}`
    default:
      return record?.block_ready
        ? `준비됨: 시드 ${record.current_seed}의 블록이 놓여 있습니다. "AUTO 시작"을 누르세요`
        : '"블록 리셋"(녹화 상자)을 누른 뒤 시작할 수 있습니다'
  }
}

// 상태 문구의 CSS 클래스 접미사: running | success | fail | error | idle
export function statusClass(auto) {
  switch (auto?.status) {
    case 'loading':
    case 'running':
      return 'running'
    case 'success':
      return 'success'
    case 'timeout':
    case 'stopped':
      return 'fail'
    case 'error':
      return 'error'
    default:
      return 'idle'
  }
}

// 정책 프로세스 표시(작게): "정책 프로세스 켜짐 (598초 후 종료)" / "꺼짐" / "불러오는 중"
export function policyText(auto, connected) {
  if (!connected || !auto?.policy) return '정책 프로세스: —'
  const { state, idle_remaining_s: remaining } = auto.policy
  if (state === 'loading') return '정책 프로세스: 불러오는 중'
  if (state === 'on') {
    return remaining == null ? '정책 프로세스: 켜짐 (AUTO 실행 중)' : `정책 프로세스: 켜짐 (${formatSeconds(remaining)} 후 종료)`
  }
  return '정책 프로세스: 꺼짐'
}

// 마지막 이벤트 표시용 CSS 클래스 접미사: info | warn | error
export function eventLevel(event) {
  return event && ['info', 'warn', 'error'].includes(event.level) ? event.level : 'info'
}

// 경과 시간 "12.3 / 45 s"
export function formatElapsed(elapsedS, maxS) {
  if (elapsedS == null) return '—'
  const max = maxS == null ? '' : ` / ${maxS.toFixed(0)}`
  return `${elapsedS.toFixed(1)}${max} s`
}

// 표에 보여줄 줄들 [이름, 값]
export function summaryRows(auto) {
  if (!auto) return []
  const running = auto.status === 'running'
  const sample = running && auto.sample != null ? String(auto.sample) : '—'
  const chunk = running && auto.chunk_pos != null && auto.chunk_len != null ? `${auto.chunk_pos + 1} / ${auto.chunk_len}` : '—'
  return [
    ['시드', auto.seed ?? '—'],
    ['경과 시간 (시뮬레이션)', auto.elapsed_s == null ? '—' : formatElapsed(auto.elapsed_s, auto.max_s)],
    ['샘플 번호', sample],
    ['청크 안 위치', chunk],
    ['마지막 추론 총 지연 (렌더+추론, ms)', auto.last_latency_ms == null ? '—' : auto.last_latency_ms.toFixed(0)],
    ['이번 세션 성공 / 시도', `${auto.session_success ?? 0} / ${auto.session_attempts ?? 0}`],
  ]
}
