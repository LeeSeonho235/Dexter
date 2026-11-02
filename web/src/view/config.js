// 보기 방향(자유 카메라) 설정값. 서버 쪽 원본은 server/config.py의 VIEW_* (src/view/serverView.test.js가 같은지 확인한다).
// 각도는 도(deg), 거리는 m. 방위각 90°에서는 로봇 앞쪽(+x)이 화면 오른쪽, 0°에서는 +x가 화면 위쪽이다.

// 서버가 받아 주는 범위(서버도 같은 범위로 자른다). 방위각은 -180~180으로 돌려서 쓴다.
export const ELEVATION_RANGE_DEG = [-89, -3]
export const DISTANCE_RANGE_M = [0.3, 2.0]

// 마우스 감도. MuJoCo 기본 뷰어와 같은 부호: 오른쪽으로 끌면 방위각이 줄고(장면이 끈 방향으로 도는 느낌),
// 아래로 끌면 고도각이 줄어(더 위에서 내려다봄) 든다. 어색하면 부호만 바꾼다.
export const AZIMUTH_DEG_PER_PX = -0.35
export const ELEVATION_DEG_PER_PX = -0.35
// 휠: 거리 *= exp(deltaY * 이 값). 아래로 굴리면(deltaY > 0) 멀어진다.
export const WHEEL_LOG_DISTANCE_PER_DELTA = 0.001

// (보기 방향 메시지를 서버로 보내는 최대 빈도는 src/net/config.js의 VIEW_SEND_MAX_HZ)

// 직접 조작한 뒤 이 시간(ms) 동안은 서버가 알려 주는 값으로 화면 숫자를 덮어쓰지 않는다(조작이 되돌아 튀는 것을 막는다).
export const SERVER_VIEW_HOLD_MS = 500

// 프리셋. 원본은 server/config.py의 VIEW_PRESETS (같은지 serverView.test.js가 확인).
export const PRESETS = {
  default: { azimuth_deg: 110, elevation_deg: -35, distance_m: 0.8 },
  top: { azimuth_deg: 0, elevation_deg: -89, distance_m: 0.85 },
  side: { azimuth_deg: 105, elevation_deg: -10, distance_m: 0.8 },
}

export const PRESET_LABELS = [
  ['top', '위에서'],
  ['side', '옆에서'],
  ['default', '비스듬히 (기본)'],
]
