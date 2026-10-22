// 손 추적 설정값을 한 곳에 모은다. (서버 쪽 매핑 설정은 server/config.py)

// 웹캠 요청 해상도 (px). 브라우저가 다른 크기를 줄 수 있으므로 계산에는 실제 video 크기를 쓴다.
export const VIDEO_WIDTH_PX = 640
export const VIDEO_HEIGHT_PX = 480

// 로컬에서 제공하는 MediaPipe 파일 경로 (외부 CDN을 쓰지 않는다)
export const WASM_BASE_PATH = '/mediapipe/wasm' // scripts/copy-wasm.mjs가 복사
export const MODEL_PATH = '/models/hand_landmarker.task' // README의 curl 명령으로 받음

export const NUM_HANDS = 1

// palm_open 판정: 손목→손끝 거리가 손목→둘째 마디(PIP) 거리 × 이 값보다 크면 "펴짐" (단위 없음)
export const FINGER_OPEN_MARGIN = 1.1
