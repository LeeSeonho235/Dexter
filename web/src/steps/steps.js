// 3단계 마법사의 단계 목록과 이동 규칙(React에 의존하지 않는 순수 계산).

export const STEPS = [
  { id: 'record', no: 1, label: '녹화' },
  { id: 'train', no: 2, label: '학습' },
  { id: 'test', no: 3, label: '테스트·교정' },
]

export const STEP_IDS = STEPS.map((s) => s.id)

// 서버 state.auto.status가 이 값이면 정책이 불러오는 중이거나 돌아가는 중이다(auto/view.js의 BUSY와 같은 기준).
const AUTO_BUSY = ['loading', 'running']

// 녹화 중이거나 AUTO(교정 포함)가 진행 중이면 true. 이때는 단계를 옮기지 못하게 한다.
// 서버 상태를 모르면(연결 전) 막지 않는다: 진행 중인 일이 없다고 본다.
export function isBusy(record, auto) {
  return Boolean(record?.recording) || AUTO_BUSY.includes(auto?.status)
}

// 지금 busy인 이유(버튼 옆 안내용). busy가 아니면 ''.
export function busyReason(record, auto) {
  if (record?.recording) return '녹화 중에는 단계를 옮길 수 없습니다'
  if (AUTO_BUSY.includes(auto?.status)) return 'AUTO 진행 중에는 단계를 옮길 수 없습니다'
  return ''
}
