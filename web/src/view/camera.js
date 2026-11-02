// 마우스 드래그, 휠, 프리셋 → 보기 방향 계산. React에 의존하지 않는 순수 함수.
//
// view = { azimuth_deg, elevation_deg, distance_m } (서버 state.view와 같은 모양)

import {
  AZIMUTH_DEG_PER_PX,
  DISTANCE_RANGE_M,
  ELEVATION_DEG_PER_PX,
  ELEVATION_RANGE_DEG,
  PRESETS,
  SERVER_VIEW_HOLD_MS,
  WHEEL_LOG_DISTANCE_PER_DELTA,
} from './config.js'

const clamp = (value, [lo, hi]) => Math.min(hi, Math.max(lo, value))

// 방위각은 한 바퀴 돌면 같은 방향이므로 자르지 않고 -180 이상 180 미만으로 돌려 쓴다.
export function wrapAzimuthDeg(azimuthDeg) {
  return ((((azimuthDeg + 180) % 360) + 360) % 360) - 180
}

// 서버가 받아 주는 범위로 맞춘다.
export function clampView(view) {
  return {
    azimuth_deg: wrapAzimuthDeg(view.azimuth_deg),
    elevation_deg: clamp(view.elevation_deg, ELEVATION_RANGE_DEG),
    distance_m: clamp(view.distance_m, DISTANCE_RANGE_M),
  }
}

// 드래그: dxPx(오른쪽 +), dyPx(아래 +) 만큼 끌었을 때의 새 보기 방향
export function applyDrag(view, dxPx, dyPx) {
  return clampView({
    ...view,
    azimuth_deg: view.azimuth_deg + dxPx * AZIMUTH_DEG_PER_PX,
    elevation_deg: view.elevation_deg + dyPx * ELEVATION_DEG_PER_PX,
  })
}

// 휠: deltaY(아래로 굴리면 +, 브라우저 WheelEvent.deltaY)
export function applyWheel(view, deltaY) {
  return clampView({ ...view, distance_m: view.distance_m * Math.exp(deltaY * WHEEL_LOG_DISTANCE_PER_DELTA) })
}

export function presetView(name) {
  const preset = PRESETS[name]
  if (!preset) throw new Error(`unknown preset: ${name}`)
  return clampView(preset)
}

// 서버로 보내는 view 메시지 (server/protocol.py validate_view 형식)
export function viewMessage(view) {
  return {
    type: 'view',
    azimuth: view.azimuth_deg,
    elevation: view.elevation_deg,
    distance: view.distance_m,
  }
}

// 직접 조작한 지 얼마 안 됐으면 내 값을, 아니면 서버가 알려 준 값을 쓴다.
// (서버 보기 방향은 하나뿐이라 다른 브라우저가 바꾼 것도 반영해야 한다.)
export function pickDisplayedView({ local, server, lastInteractionMs, nowMs, holdMs = SERVER_VIEW_HOLD_MS }) {
  if (!server) return local
  return nowMs - lastInteractionMs < holdMs ? local : server
}

// 서버가 10Hz로 보내는 값이 지금 값과 (표시 자리수 기준으로) 같으면 화면을 다시 그리지 않으려고 쓴다.
export function viewsEqual(a, b, epsilon = 0.01) {
  return (
    Math.abs(a.azimuth_deg - b.azimuth_deg) < epsilon &&
    Math.abs(a.elevation_deg - b.elevation_deg) < epsilon &&
    Math.abs(a.distance_m - b.distance_m) < epsilon / 100
  )
}

// 화면 숫자 표시: "방위각 110° · 고도각 -35° · 거리 0.80 m"
export function formatView(view) {
  if (!view) return '—'
  return `방위각 ${view.azimuth_deg.toFixed(0)}° · 고도각 ${view.elevation_deg.toFixed(0)}° · 거리 ${view.distance_m.toFixed(2)} m`
}
