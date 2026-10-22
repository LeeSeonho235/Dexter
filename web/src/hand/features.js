// 21개 손 점 → 서버로 보낼 hand 메시지 (docs/04-architecture.md 형식 + pinch_ratio)
// React, MediaPipe에 의존하지 않는 순수 계산만 둔다.
//
// 좌표: MediaPipe 정규화 좌표 (x, y는 화면 너비·높이를 1로 본 비율, 좌우 뒤집지 않음).
// 거리(pinch, hand_scale): x, y 평면 거리. x에 화면 비율(너비/높이)을 곱해 가로·세로를 같은 척도로 맞추므로
// 단위는 "화면 높이 = 1". z는 추정값이라 흔들려서 거리 계산에서 뺀다.

import { FINGER_OPEN_MARGIN } from './config.js'

// MediaPipe Hand Landmarker 점 번호
export const LM = {
  WRIST: 0,
  THUMB_TIP: 4,
  INDEX_PIP: 6,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_TIP: 12,
  RING_PIP: 14,
  RING_TIP: 16,
  PINKY_PIP: 18,
  PINKY_TIP: 20,
}

// palm_open 판정에 쓰는 손가락 4개 (엄지 제외): [PIP, TIP]
const FINGERS = [
  [LM.INDEX_PIP, LM.INDEX_TIP],
  [LM.MIDDLE_PIP, LM.MIDDLE_TIP],
  [LM.RING_PIP, LM.RING_TIP],
  [LM.PINKY_PIP, LM.PINKY_TIP],
]

const NUM_LANDMARKS = 21

// aspect = 화면 너비 / 높이 (640x480이면 4/3)
export function distance2d(a, b, aspect = 1) {
  const dx = (a.x - b.x) * aspect
  const dy = a.y - b.y
  return Math.hypot(dx, dy)
}

export function pinchRatio(pinch, handScale) {
  if (pinch == null || handScale == null || handScale === 0) return null
  return pinch / handScale
}

export function isPalmOpen(landmarks, aspect = 1, margin = FINGER_OPEN_MARGIN) {
  const wrist = landmarks[LM.WRIST]
  return FINGERS.every(
    ([pip, tip]) =>
      distance2d(wrist, landmarks[tip], aspect) > distance2d(wrist, landmarks[pip], aspect) * margin,
  )
}

function toArray(p) {
  return [p.x, p.y, p.z]
}

// landmarks: [{x, y, z}, ...] 21개, 손이 없으면 null/undefined
// t: 초 단위 시각
export function makeHandMessage(landmarks, { t, aspect = 1, margin = FINGER_OPEN_MARGIN } = {}) {
  if (!landmarks || landmarks.length < NUM_LANDMARKS) {
    return {
      type: 'hand',
      t,
      detected: false,
      wrist: null,
      index_tip: null,
      thumb_tip: null,
      pinch: null,
      pinch_ratio: null,
      palm_open: null,
      hand_scale: null,
    }
  }

  const wrist = landmarks[LM.WRIST]
  const indexTip = landmarks[LM.INDEX_TIP]
  const thumbTip = landmarks[LM.THUMB_TIP]
  const pinch = distance2d(thumbTip, indexTip, aspect)
  const handScale = distance2d(wrist, landmarks[LM.MIDDLE_MCP], aspect)

  return {
    type: 'hand',
    t,
    detected: true,
    wrist: toArray(wrist),
    index_tip: toArray(indexTip),
    thumb_tip: toArray(thumbTip),
    pinch,
    pinch_ratio: pinchRatio(pinch, handScale),
    palm_open: isPalmOpen(landmarks, aspect, margin),
    hand_scale: handScale,
  }
}
