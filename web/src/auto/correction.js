// 교정(4-2) 표시 규칙(React에 의존하지 않는 순수 계산). 서버 state.auto / state.control 형식은 docs/13-correction.md.
//
// 제어 주체는 서버 state.auto.owner(AUTO, HUMAN_WAIT, HUMAN, IDLE)가 정한다. 화면은 안내일 뿐이고 서버가 같은 규칙으로 다시 검사한다.

import { commandFor } from './view.js'

// 화면에 보이는 제어 주체 종류: auto | wait | human | loading | idle
export function ownerKind(auto, connected) {
  if (!connected || !auto) return 'idle'
  if (auto.status === 'loading') return 'loading'
  if (auto.status !== 'running') return 'idle'
  if (auto.owner === 'HUMAN') return 'human'
  if (auto.owner === 'HUMAN_WAIT') return 'wait'
  if (auto.owner === 'AUTO') return 'auto'
  return 'idle'
}

// 큰 표시 문구. 색만으로 구분하지 않으려고 문구와 기호(mark)를 함께 둔다. idle이면 null(표시하지 않음).
export function ownerBanner(auto, connected) {
  switch (ownerKind(auto, connected)) {
    case 'auto':
      return {
        kind: 'auto',
        mark: '▶',
        title: '로봇이 혼자 하는 중(AUTO)',
        detail: '로봇이 블록 위에서 멈칫하거나 엉뚱한 곳으로 가려는 순간 Space(또는 아래 버튼)로 넘겨받으세요',
      }
    case 'wait':
      return {
        kind: 'wait',
        mark: '✋',
        title: '손을 보여주세요(대기)',
        detail: '손을 화면 중앙에 보여주세요. 조종 손이 3프레임 연속 보이면 넘겨받습니다. 로봇은 그 자리에 멈춰 있습니다',
      }
    case 'human':
      return {
        kind: 'human',
        mark: '●',
        title: '내가 조종 중',
        detail: '다 고쳤으면 Space(또는 아래 버튼)로 정책에게 돌려주세요. 높이는 ↑↓ 또는 W/S',
      }
    case 'loading':
      return { kind: 'loading', mark: '…', title: '정책 불러오는 중', detail: '' }
    default:
      return null
  }
}

// 넘겨받은 직후 집게 안내: 사용자의 pinch가 로봇 집게 상태와 아직 일치하지 않을 때만(HUMAN 중에만) 뜬다. 일치하면 사라진다.
export function gripperHint(auto, control, connected) {
  if (ownerKind(auto, connected) !== 'human') return null
  const sync = control?.gripper_sync
  if (!sync || !sync.active || !sync.waiting) return null
  const state = sync.robot_closed ? '닫힘' : '열림'
  const action = sync.robot_closed ? '손을 쥐세요(엄지와 검지를 붙이기)' : '손을 펴세요'
  return { text: `집게를 로봇(${state})에 맞추세요`, detail: `${action}. 맞출 때까지 집게는 지금 상태 그대로 있습니다` }
}

// Space와 같은 일을 하는 토글 버튼의 모양: {id, label, enabled, reason}
export function toggleButton(auto, connected) {
  const base = { id: 'takeover', label: '넘겨받기 (Space)', enabled: false, reason: '' }
  if (!connected) return { ...base, reason: '서버에 연결되지 않음' }
  if (!auto) return { ...base, reason: '서버가 AUTO 상태를 보내지 않음' }
  switch (ownerKind(auto, connected)) {
    case 'auto':
      return { ...base, enabled: true }
    case 'wait':
      return { ...base, label: '손을 기다리는 중…', reason: '손을 화면 중앙에 보여주세요' }
    case 'human':
      return { id: 'handback', label: '정책에게 돌려주기 (Space)', enabled: true, reason: '' }
    case 'loading':
      return { ...base, reason: '정책을 불러오는 중' }
    default:
      return { ...base, reason: 'AUTO가 실행 중이 아님' }
  }
}

// Space를 눌렀을 때 보낼 메시지(없으면 null). 버튼과 같은 규칙이다.
export function spaceCommand(auto, connected) {
  const button = toggleButton(auto, connected)
  return button.enabled ? commandFor(button.id) : null
}

// 끝난 뒤 "교정 저장 / 버리기"를 고르는 중인가
export function decisionPending(auto) {
  return Boolean(auto?.correction?.pending)
}

export function decisionText(auto) {
  const p = auto?.correction?.pending
  if (!p) return ''
  const how = { success: '성공', timeout: '시간 초과', stopped: '중지', error: '오류' }[p.outcome] ?? p.outcome
  return `끝남(${how}): 사람이 고친 구간 ${p.interventions}번, 사람 시간 ${p.human_time_s.toFixed(1)}초. 저장할까요?`
}

function fmt(value, digits = 1, unit = '초') {
  return value == null ? '—' : `${Number(value).toFixed(digits)}${unit}`
}

// 시드 안내: 평가 시드는 화면에 나오지 않는다(서버가 리셋에서 시연 시드만 낸다). 용도는 안내일 뿐 서버가 막지 않는다.
export const PURPOSE_LABELS = { correction: '교정용', extra_demo: '추가 시연용' }

export function seedGuideText(auto) {
  if (!auto) return '—'
  const label = (purpose) => PURPOSE_LABELS[purpose] ?? '배분표에 없는 시드'
  if (auto.ready_seed != null) return `준비된 블록: 시드 ${auto.ready_seed} → ${label(auto.ready_seed_purpose)}`
  if (auto.next_seed != null) return `다음 시드: ${auto.next_seed} → ${label(auto.next_seed_purpose)}`
  return '시연 시드를 모두 썼습니다'
}

// 준비된 시드가 추가 시연용이면 교정 대신 시연용으로 남겨 두라는 경고(막지는 않는다). 아니면 null.
export function seedPurposeWarning(auto) {
  if (auto?.ready_seed != null && auto.ready_seed_purpose === 'extra_demo') {
    return '이 시드는 추가 시연용입니다. 교정 실험용으로 쓰려면 다음 시드로 리셋하세요(막지는 않습니다)'
  }
  return null
}

// 표 줄 [이름, 값]. 사람 시간은 HUMAN 구간만(06 문서의 "같은 시간 비교"에 쓰는 값)이다.
export function correctionRows(auto) {
  if (!auto) return []
  const c = auto.correction ?? {}
  return [
    ['사람 시간 (HUMAN 구간만)', fmt(auto.human_s)],
    ['개입 횟수', auto.interventions == null ? '—' : `${auto.interventions}번`],
    ['AUTO 시간 / 제한', auto.auto_s == null ? '—' : `${auto.auto_s.toFixed(1)} / ${Number(auto.max_s).toFixed(0)} 초`],
    ['손 기다린 시간', fmt(auto.wait_s)],
    ['전체 시간 / 상한', auto.total_s == null ? '—' : `${auto.total_s.toFixed(1)} / ${Number(auto.total_max_s).toFixed(0)} 초`],
    ['저장한 교정', c.saved_count == null ? '—' : `${c.saved_count}개 (사람 시간 합계 ${fmt(c.saved_human_s)})`],
    ['시드 안내', seedGuideText(auto)],
  ]
}
