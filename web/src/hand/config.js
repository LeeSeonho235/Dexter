// 손 추적 설정값을 한 곳에 모은다. (서버 쪽 매핑 설정은 server/config.py)

// 웹캠 요청 해상도 (px). 브라우저가 다른 크기를 줄 수 있으므로 계산에는 실제 video 크기를 쓴다.
export const VIDEO_WIDTH_PX = 640
export const VIDEO_HEIGHT_PX = 480

// 로컬에서 제공하는 MediaPipe 파일 경로 (외부 CDN을 쓰지 않는다)
export const WASM_BASE_PATH = '/mediapipe/wasm' // scripts/copy-wasm.mjs가 복사
export const MODEL_PATH = '/models/hand_landmarker.task' // README의 curl 명령으로 받음

// 손은 2개까지 인식하지만 로봇 조종에는 CONTROL_HANDEDNESS인 손 하나만 쓴다(서버에는 그 손만 보낸다).
export const NUM_HANDS = 2

// 조종할 손을 고르는 MediaPipe handedness 라벨: 'Right' 또는 'Left'.
// 주의: MediaPipe는 거울처럼 뒤집은(셀카) 영상을 기준으로 라벨을 붙인다. 이 앱은 뒤집지 않은 원본을 넣으므로
// 라벨이 실제 손과 반대로 나올 수 있다(내 오른손이 'Left'로). 화면의 "조종 중인 손" 표시와 노란 선으로 확인하고,
// 반대면 이 값을 바꾼다.
export const CONTROL_HANDEDNESS = 'Right'

// palm_open 판정: 손목→손끝 거리가 손목→둘째 마디(PIP) 거리 × 이 값보다 크면 "펴짐" (단위 없음)
export const FINGER_OPEN_MARGIN = 1.1
