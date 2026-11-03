// 녹화 상자 설정값. (서버 쪽 녹화 규칙은 server/config.py의 RECORD_*, docs/08-recording.md)

// "마지막 에피소드 삭제"는 한 번 누르면 확인 상태가 되고, 이 시간(ms) 안에 한 번 더 눌러야 실제로 지운다.
export const CONFIRM_TIMEOUT_MS = 3000

// 정지 구간이 이만큼 많으면 화면에 경고를 띄운다. 임시 기준이다(시연 20개를 찍어 본 뒤 조정한다).
//   비율: ctrl이 거의 안 변한 샘플의 비율 / 가장 긴 연속 정지: 초
export const STILL_WARN_RATIO = 0.5
export const STILL_WARN_LONGEST_S = 3.0
