// 손이 영상 가장자리에 가까운지 판정, 상대 모드 안내(기준점 십자, 현재 손 점, 연결선)의 좌표 계산.
// React, MediaPipe에 의존하지 않는 순수 계산. 좌표는 MediaPipe 정규화 좌표(0~1, 좌우 뒤집지 않은 원본).

import { EDGE_MARGIN } from './config.js'

// landmarks: [{x, y, z}, ...] 손의 21개 점. 점 전체가 들어가는 상자가 영상 가장자리 margin 이내에 닿거나 넘으면 가장자리다.
// 조종점(손목)만이 아니라 손 전체를 보는 이유: 손가락 끝이 먼저 화면 밖으로 나가도 인식이 끊긴다.
// 돌려주는 값: { near, left, right, top, bottom } (left/right/top/bottom은 영상 원본 기준 어느 쪽이 가까운지)
export function handEdgeStatus(landmarks, margin = EDGE_MARGIN) {
  const none = { near: false, left: false, right: false, top: false, bottom: false }
  if (!landmarks || landmarks.length === 0) return none
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  const left = minX <= margin
  const right = maxX >= 1 - margin
  const top = minY <= margin
  const bottom = maxY >= 1 - margin
  return { near: left || right || top || bottom, left, right, top, bottom }
}

// 상대 모드 안내 도형의 캔버스 좌표(px).
//   reference: 서버가 알려 준 기준 손 위치 [x, y] (정규화), wrist: 지금 손목 [x, y] (정규화)
// 둘 중 하나라도 없으면 null(그리지 않는다). 십자는 reference에, 점은 손목에, 선은 둘 사이.
export function overlayGeometry({ reference, wrist }, widthPx, heightPx, crossHalfPx = 10) {
  if (!reference || !wrist) return null
  const cross = { x: reference[0] * widthPx, y: reference[1] * heightPx, half: crossHalfPx }
  const dot = { x: wrist[0] * widthPx, y: wrist[1] * heightPx }
  return { cross, dot, line: { x1: cross.x, y1: cross.y, x2: dot.x, y2: dot.y } }
}

// 가장자리 안내 문구. 상대 모드는 클러치(마우스를 들어 옮기기)로 이어서 움직이는 방법을 알려 준다.
export function edgeMessage(mode) {
  if (mode === 'absolute') {
    return '화면 가장자리: 손이 화면 밖으로 나가면 인식이 끊겨요. 화면 안쪽으로 옮기세요'
  }
  return '화면 가장자리: 손을 화면 밖으로 뺐다가 중앙에서 다시 넣으면 이어서 움직여요'
}
