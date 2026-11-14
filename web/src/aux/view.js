// 보조 카메라 화면(2-6)의 계산: aux_frame 메시지 검사, 진짜 화면/2D 폴백 고르기, 화면 스타일, 켜기 메시지를 언제 보낼지. React와 DOM에 의존하지 않는 순수 계산.
//
// 서버 → 브라우저 aux_frame: { type:'aux_frame', seq, w, h, views, jpeg:<base64> }. 두 시점을 가로로 이은 JPEG 한 장(w x h)이다.
// 서버 state.aux: { enabled, running, viewers, fps, slowed, error, ... } (server/aux_view.py state()).

import { AUX_MARGIN_PX, AUX_MAX_JPEG_BASE64_CHARS, AUX_STALE_MS, AUX_TILE_SIZE_PX, AUX_TILES } from './config.js'

const BASE64 = /^[A-Za-z0-9+/]+={0,2}$/
const isPositiveInt = (v) => Number.isInteger(v) && v > 0

// aux_frame 메시지를 검사해서 { seq, w, h, views, src }로 바꾼다. src는 <img>나 CSS에 바로 쓸 수 있는 data 주소다. 이상하면 null.
export function auxFrameFromMessage(msg) {
  if (!msg || msg.type !== 'aux_frame') return null
  const { seq, w, h, views, jpeg } = msg
  if (!Number.isInteger(seq) || !isPositiveInt(w) || !isPositiveInt(h) || !isPositiveInt(views)) return null
  if (typeof jpeg !== 'string' || jpeg.length === 0 || jpeg.length > AUX_MAX_JPEG_BASE64_CHARS || !BASE64.test(jpeg)) return null
  return { seq, w, h, views, src: `data:image/jpeg;base64,${jpeg}` }
}

// 진짜 보조 화면 두 개('mirrors')를 보여 줄지, 2D 그림('fallback')으로 대신할지.
//   mirrors 조건: 사용자가 켰고(wanted), 연결이 열려 있고, 서버 aux가 켜져 있고 돌고 있고 오류가 없고, 마지막 프레임이 최근(staleMs 안)이고 시점 수가 맞다.
// frame: { views, receivedMs }(App이 받은 시각을 붙인다), nowMs: 지금(ms, 같은 시계).
export function auxMode({ wanted, connected, auxState, frame, nowMs, staleMs = AUX_STALE_MS }) {
  if (!wanted || !connected) return 'fallback'
  if (!auxState || auxState.enabled !== true || auxState.running !== true || auxState.error) return 'fallback'
  if (!frame || frame.views !== AUX_TILES.length) return 'fallback'
  if (!(nowMs - frame.receivedMs <= staleMs)) return 'fallback'
  return 'mirrors'
}

// 화면 하나(tile, 그림 이미지에서 index번째 칸)의 인라인 스타일. 이미지는 시점 수만큼 가로로 이어져 있고,
// 서버가 작은 크기(128x96)로 보내도 칸 크기(160x120)에 맞게 늘어난다(같은 4:3).
export function auxTileStyle(tile, index, frame) {
  const { width, height } = AUX_TILE_SIZE_PX
  return {
    bottom: AUX_MARGIN_PX,
    [tile.corner === 'bottom-left' ? 'left' : 'right']: AUX_MARGIN_PX,
    width,
    height,
    backgroundImage: `url("${frame.src}")`,
    backgroundSize: `${width * frame.views}px ${height}px`,
    backgroundPosition: `${-index * width}px 0px`,
  }
}

// 메인 화면(가로 x 세로 px) 안에서 화면 하나가 차지하는 상자(왼쪽 위 기준). 나침반과 안 겹치는지 확인하는 데 쓴다.
export function auxTileBox(tile, simSizePx) {
  const { width, height } = AUX_TILE_SIZE_PX
  const left = tile.corner === 'bottom-left' ? AUX_MARGIN_PX : simSizePx.width - AUX_MARGIN_PX - width
  const top = simSizePx.height - AUX_MARGIN_PX - height
  return { left, top, right: left + width, bottom: top + height }
}

// 켜기/끄기 체크박스를 보여 줄지: 서버에서 aux가 켜져 있을 때만(AUX_ENABLED=1). 꺼져 있으면 체크박스도 없다.
export function auxToggleVisible({ connected, auxState }) {
  return connected === true && auxState?.enabled === true
}

// 체크박스 옆 한 줄 안내. 보여 줄 것이 없으면 null.
export function auxStatusText({ connected, auxState, wanted, mode }) {
  if (!auxToggleVisible({ connected, auxState })) return null
  if (auxState.error) return `보조 화면 오류: ${auxState.error} — 2D 그림으로 대신 보여 줍니다`
  if (auxState.running !== true) return '보조 카메라가 멈춰 있습니다 — 2D 그림으로 보여 줍니다'
  if (!wanted) return '보조 화면 꺼짐 — 2D 그림으로 보여 줍니다'
  if (mode !== 'mirrors') return '보조 화면 기다리는 중 — 그동안 2D 그림으로 보여 줍니다'
  if (auxState.slowed) return `보조 화면이 느려서 ${Number(auxState.fps).toFixed(1)}fps로 낮췄습니다`
  return '왼쪽 아래 = 옆에서, 오른쪽 아래 = 비스듬히 (보기 전용)'
}

// 서버에 { type:'aux', enabled }를 언제 보낼지. 서버는 연결마다 "보조 화면을 받겠다"는 표시를 따로 갖고 있고 기본은 꺼짐이다.
//   그래서 (1) 연결이 새로 열릴 때마다(재연결 포함) 켜 두었으면 다시 보내고, (2) 체크박스를 바꾸면 보낸다.
//   서버에서 aux가 꺼져 있으면(serverEnabled false) 아무것도 안 보낸다. 연결이 끊기면 기억을 지운다.
// input: { connected, serverEnabled, wanted, connectCount }, last: 지난번에 보낸 기록({ connectCount, enabled }) 또는 null.
// 반환: { message: 보낼 메시지 또는 null, last: 새 기록 }. 실제로 못 보냈으면 호출하는 쪽이 last를 바꾸지 않는다.
export function planAuxSend({ connected, serverEnabled, wanted, connectCount }, last) {
  if (!connected) return { message: null, last: null }
  if (!serverEnabled) return { message: null, last }
  const sameConnection = last !== null && last.connectCount === connectCount
  if (sameConnection && last.enabled === wanted) return { message: null, last }
  // 새 연결에서 끄려는 경우: 서버 기본이 꺼짐이라 보낼 필요 없다(기록만 한다).
  if (!sameConnection && !wanted) return { message: null, last: { connectCount, enabled: false } }
  return { message: { type: 'aux', enabled: wanted }, last: { connectCount, enabled: wanted } }
}
