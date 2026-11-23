// 학습 스텝 수 입력 검사와 예상 시간 계산(순수 함수). 값은 여기 한 곳에서 정한다.
// 0.14초/스텝은 Colab 무료 T4 기준 추정치다. GPU가 다르면 시간이 달라진다.

export const STEPS_DEFAULT = 20_000
export const STEPS_MIN = 1_000
export const STEPS_MAX = 100_000
export const SECONDS_PER_STEP = 0.14 // 초/스텝 (T4 기준)
export const OVERHEAD_MINUTES = 10 // 설치·확인 등 학습 밖 시간(분)

// 입력 문자열(또는 숫자)을 정수로 읽는다. 정수 모양이 아니면 null(소수, 빈 값, 글자, 지수 표기 불가).
export function parseSteps(value) {
  const text = String(value ?? '').trim()
  return /^[0-9]+$/.test(text) ? Number(text) : null
}

// 통과하면 null, 아니면 이유 문구.
export function stepsError(value) {
  const n = parseSteps(value)
  if (n === null) return '숫자(정수)를 입력하세요'
  if (n < STEPS_MIN || n > STEPS_MAX) return `${STEPS_MIN.toLocaleString('en-US')} ~ ${STEPS_MAX.toLocaleString('en-US')} 사이로 입력하세요`
  return null
}

// 학습 자체의 예상 시간(분, 올림한 정수). 입력이 안전하지 않으면 null.
export function estimateMinutes(value) {
  if (stepsError(value)) return null
  return Math.ceil((parseSteps(value) * SECONDS_PER_STEP) / 60)
}

// 화면 문구. 입력이 안전하지 않으면 null.
export function estimateText(value) {
  const minutes = estimateMinutes(value)
  return minutes === null ? null : `약 ${minutes}분 (설치·확인 포함 약 +${OVERHEAD_MINUTES}분)`
}
