// 보조 카메라 화면 두 개(2-6, 보기 전용)의 설정값 한 곳. 크기와 위치는 여기서만 바꾼다(CSS에 같은 숫자를 따로 적지 않는다. 예외는 좁은 창에서 숨기는 CSS 한 줄이고 view.test.js가 같은지 확인한다).
// 단위: 화면 px, 시간 ms. 서버 쪽 원본은 server/aux_view_config.py(시점, fps, 크기)와 server/app.py(aux_frame 메시지).

// 화면 하나의 크기(px)
export const AUX_TILE_SIZE_PX = { width: 160, height: 120 }

// 화면 두 개. ★ 순서는 서버가 한 장으로 이어 붙인 이미지의 가로 순서(왼쪽부터)와 같아야 한다:
//   server/aux_view_config.py AUX_CAMERA_VIEWS = (옆에서 105°/-10°, 비스듬히 45°/-25°). view.test.js가 서버 파일과 순서를 비교한다.
// corner: 메인 화면 안의 위치. 옆에서 = 왼쪽 아래, 비스듬히 = 오른쪽 아래(나침반은 오른쪽 위라서 겹치지 않는다).
export const AUX_TILES = [
  { id: 'side', label: '옆에서', corner: 'bottom-left' },
  { id: 'oblique', label: '비스듬히', corner: 'bottom-right' },
]
export const AUX_MARGIN_PX = 6

// 메인 화면(.sim-view) 가로가 이 값(px)보다 좁으면 진짜 보조 화면 두 개를 숨기고 2D 그림(SideMirror)만 보여 준다.
// 나침반(오른쪽 위 150x128)과 겹치지 않는 최소 가로: 4:3 화면에서 아래 화면 위쪽(0.75 x 가로 - 126)이 나침반 아래(134)보다 아래여야 하고 두 화면이 안 겹쳐야 하므로 약 347 이상.
// index.css의 @container (max-width: 이 값 - 1px) 한 줄과 같아야 한다(view.test.js가 비교한다).
export const AUX_MIN_VIEW_WIDTH_PX = 360

// 마지막 aux_frame을 받은 지 이 시간(ms)이 지나면 "프레임이 끊겼다"고 보고 2D 그림으로 돌아간다.
// 서버가 느려서 fps를 바닥(1fps, server/aux_view_config.py AUX_MIN_FPS)까지 낮춰도 끊긴 것으로 보지 않게 1초 주기보다 길게 잡았다.
export const AUX_STALE_MS = 2500

// aux_frame의 base64 JPEG 글자 수 상한(이보다 크면 버린다). 정상 프레임은 약 8,000자(320x120, 약 6KB).
export const AUX_MAX_JPEG_BASE64_CHARS = 2_000_000
