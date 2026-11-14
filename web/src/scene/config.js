// 옆모습 보조 화면(2-6, 보기 전용)의 설정값 한 곳. 크기와 위치는 여기서만 바꾼다(CSS에 같은 숫자를 따로 적지 않는다).
// 단위: 화면 px, 월드 m. 월드 좌표는 로봇 밑이 원점, x = 로봇 앞쪽, z = 위. 서버 값(블록·목표 크기)은 state.scene으로 받는다.

// 보조 화면 크기(px)
export const SIDE_MIRROR_SIZE_PX = { width: 160, height: 100 }
// 메인 화면 안의 위치. 'bottom-right'(기본) 또는 'bottom-left'.
// 오른쪽 위 나침반(index.css .view-compass)과 겹치면 'bottom-left'로 바꾼다. 겹치는지는 sideView.test.js가 확인한다.
export const SIDE_MIRROR_CORNER = 'bottom-right'
export const SIDE_MIRROR_MARGIN_PX = 6

// 메인 화면(시뮬레이션 JPEG) 크기(px). server/config.py의 RENDER_WIDTH_PX / RENDER_HEIGHT_PX와 같다. 겹침 확인용.
export const SIM_VIEW_SIZE_PX = { width: 640, height: 480 }
// 나침반 상자(px). index.css의 .view-compass와 같아야 한다(sideView.test.js가 CSS와 비교한다).
export const COMPASS_BOX_PX = { top: 6, right: 6, width: 150, height: 128 }

// 보조 화면 안 문구(두 줄). 줄 하나가 160px 안에 들어가게 짧게 쓴다.
export const SIDE_MIRROR_CAPTION = ['옆모습 (앞뒤 x, 높이 z)', 'y(좌우)는 안 보임, 높이 확인용']
// 문구가 차지하는 아래쪽 높이(px). 그림은 이 위쪽 영역에만 그린다.
export const CAPTION_HEIGHT_PX = 28

// 그리는 월드 범위(m). 작업 상자는 x 0.18~0.28, z 0.015~0.12(docs/Handoff.md)이고 로봇 밑은 x = 0.
export const WORLD_RANGE_M = { x: [-0.02, 0.34], z: [-0.01, 0.16] }
// 그림 가장자리 여백(px)
export const DRAW_PAD_PX = 6

// 색
export const COLORS = {
  floor: '#9aa0a6',
  base: '#6b7075',
  block: '#e04040', // 빨강 = 블록(메인 화면 범례와 같다)
  target: '#2fb350', // 초록 = 목표
  tipOpen: '#4fd0ff',
  tipClosed: '#ffb020',
  tipUnknown: '#cfd3d6',
  height: '#ffffff',
}
export const TIP_RADIUS_PX = 4
