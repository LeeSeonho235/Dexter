// 녹화 상자의 표시 규칙(React에 의존하지 않는 순수 계산). 서버 state.record 형식은 docs/08-recording.md.
//
// 버튼을 누를 수 있는지는 서버가 같은 규칙으로 다시 검사한다(화면은 안내일 뿐이다).

import { STILL_WARN_LONGEST_S, STILL_WARN_RATIO } from './config.js'

export const BUTTON_IDS = ['start', 'stop_success', 'stop_fail', 'discard', 'reset', 'delete_last']

// 버튼 id → 서버로 보낼 메시지. delete_last는 화면이 알고 있는 가장 최근 에피소드 번호를 같이 보낸다(오래된 화면 방지).
// 보낼 수 없으면(번호를 모르는 delete_last, 모르는 id) null.
export function commandFor(buttonId, record) {
  switch (buttonId) {
    case 'start':
    case 'stop_success':
    case 'stop_fail':
    case 'discard':
      return { type: 'record', action: buttonId }
    case 'reset':
      return { type: 'reset' }
    case 'delete_last': {
      const index = record?.last_episode?.index
      return Number.isInteger(index) && index >= 0 ? { type: 'record', action: 'delete_last', episode: index } : null
    }
    default:
      return null
  }
}

// 각 버튼이 지금 눌릴 수 있는지와 안 될 때의 이유. record가 없으면(연결 전, 녹화를 모르는 서버) 모두 꺼진다.
export function buttonStates(record, connected) {
  const off = (reason) => ({ enabled: false, reason })
  const on = { enabled: true, reason: '' }
  if (!connected) return Object.fromEntries(BUTTON_IDS.map((id) => [id, off('서버에 연결되지 않음')]))
  if (!record) return Object.fromEntries(BUTTON_IDS.map((id) => [id, off('서버가 녹화 상태를 보내지 않음')]))

  const recording = Boolean(record.recording)
  const resetting = Boolean(record.resetting)
  const busyWrites = (record.pending_writes ?? 0) > 0
  return {
    start: recording
      ? off('이미 녹화 중')
      : resetting
        ? off('블록 리셋 중')
        : record.block_ready
          ? on
          : off('블록 리셋을 먼저 하세요'),
    stop_success: recording ? on : off('녹화 중이 아님'),
    stop_fail: recording ? on : off('녹화 중이 아님'),
    discard: recording ? on : off('녹화 중이 아님'),
    reset: recording
      ? off('녹화 중에는 리셋할 수 없음')
      : resetting
        ? off('블록 리셋 중')
        : record.next_seed == null
          ? off('시연 시드를 모두 사용함')
          : on,
    delete_last: recording
      ? off('녹화 중에는 삭제할 수 없음')
      : resetting
        ? off('블록 리셋 중')
        : busyWrites
          ? off('저장 중')
          : record.last_episode
            ? on
            : off('삭제할 에피소드 없음'),
  }
}

// 상태 한 줄: 녹화 중 / 리셋 중 / 시작 가능 / 리셋 필요
export function statusText(record, connected) {
  if (!connected) return '서버에 연결되지 않음'
  if (!record) return '녹화 상태를 모름 (서버가 녹화 기능을 지원하지 않거나 아직 상태를 안 보냄)'
  if (record.recording) return '● 녹화 중'
  if (record.resetting) return '블록 리셋 중… (팔이 올라가고 블록이 옮겨집니다)'
  if (record.block_ready) return `준비됨: 시드 ${record.current_seed}의 블록이 놓여 있습니다. "녹화 시작"을 누르세요`
  return '"블록 리셋"을 눌러 다음 시드의 블록을 놓으세요'
}

// 경과 시간 "12.3 / 60 s"
export function formatElapsed(elapsedS, maxS) {
  if (elapsedS == null) return '—'
  const max = maxS == null ? '' : ` / ${maxS.toFixed(0)}`
  return `${elapsedS.toFixed(1)}${max} s`
}

// 정지 구간 표시: "정지 구간 31% (가장 긴 정지 1.2 s)". 값이 아직 없으면 "—".
export function stillText(ratio, longestS) {
  if (ratio == null) return '—'
  const longest = longestS == null ? '' : ` (가장 긴 정지 ${longestS.toFixed(1)} s)`
  return `${(ratio * 100).toFixed(0)}%${longest}`
}

// 정지 구간이 많은가(로봇이 멈춰 있는 구간이 길면 학습에 나쁘다)
export function stillIsHigh(ratio, longestS) {
  return (ratio != null && ratio >= STILL_WARN_RATIO) || (longestS != null && longestS >= STILL_WARN_LONGEST_S)
}

// 마지막 이벤트 표시용 CSS 클래스 접미사: info | warn | error
export function eventLevel(event) {
  return event && ['info', 'warn', 'error'].includes(event.level) ? event.level : 'info'
}

// 표에 보여줄 줄들 [이름, 값]
export function summaryRows(record) {
  if (!record) return []
  const last = record.last_episode
  return [
    ['지금 시드 / 다음 시드', `${record.current_seed ?? '—'} / ${record.next_seed ?? '없음'} (남은 시드 ${record.seeds_left ?? 0})`],
    ['저장된 에피소드 (성공 / 실패)', `${record.saved_success ?? 0} / ${record.saved_fail ?? 0}`],
    ['경과 시간', record.recording ? formatElapsed(record.elapsed_s, record.max_s) : '—'],
    ['기록한 틱 수', record.recording ? String(record.ticks ?? 0) : '—'],
    ['정지 구간 비율', record.recording ? stillText(record.still_ratio, record.still_longest_s) : '—'],
    [
      '마지막 저장',
      last
        ? `에피소드 ${last.index} (시드 ${last.seed}, ${last.success ? '성공' : '실패'}, ${last.ticks}틱, 정지 ${stillText(last.still_ratio, last.still_longest_s)})`
        : '—',
    ],
    ['저장 중 / 저장 실패 횟수', `${record.pending_writes ?? 0} / ${record.save_error_count ?? 0}`],
    ['틱 간격 ms (최근 평균 / 최대)', tickText(record.tick_ms)],
  ]
}

function tickText(tick) {
  if (!tick || tick.recent_mean_ms == null) return '—'
  return `${tick.recent_mean_ms.toFixed(1)} / ${tick.recent_max_ms.toFixed(1)}`
}
