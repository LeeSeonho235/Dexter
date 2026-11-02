// 파이썬 서버와의 WebSocket 연결 (React에 의존하지 않음).
//
// 보내는 것: hand 메시지(JSON 텍스트, 최대 HAND_SEND_MAX_HZ), view(보기 방향, 최대 VIEW_SEND_MAX_HZ),
// height(키보드 높이, 누르는 동안 HEIGHT_REPEAT_HZ), ping.
// 웹캠 영상·이미지는 보내지 않는다.
// 받는 것: 시뮬레이션 화면 JPEG(바이너리) → onFrame(blob), state/pong/error(JSON 텍스트).
// 끊기면 RECONNECT_DELAY_MS 뒤 다시 연결한다.

import { viewMessage } from '../view/camera.js'
import {
  HAND_SEND_MAX_HZ,
  PING_INTERVAL_MS,
  RATE_WINDOW_MS,
  RECONNECT_DELAY_MS,
  VIEW_SEND_MAX_HZ,
  WS_PATH,
} from './config.js'
import { createRateCounter, createRateLimiter } from './rate.js'
import { createTrailingThrottle } from './throttle.js'

// http 페이지면 ws://, https면 wss://. 주소는 지금 페이지와 같은 곳(Vite 프록시)이다.
export function wsUrl(location, path = WS_PATH) {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${scheme}://${location.host}${path}`
}

// 브라우저 WebSocket.readyState 값
const OPEN = 1

export class SimConnection {
  constructor({
    url,
    WebSocketImpl = globalThis.WebSocket,
    now = () => performance.now(),
    onFrame = () => {},
    onState = () => {},
    maxHandHz = HAND_SEND_MAX_HZ,
    maxViewHz = VIEW_SEND_MAX_HZ,
    setTimer = (fn, ms) => setTimeout(fn, ms),
    clearTimer = (id) => clearTimeout(id),
    pingIntervalMs = PING_INTERVAL_MS,
    reconnectDelayMs = RECONNECT_DELAY_MS,
  }) {
    this.url = url
    this.WebSocketImpl = WebSocketImpl
    this.now = now
    this.onFrame = onFrame
    this.onState = onState
    this.pingIntervalMs = pingIntervalMs
    this.reconnectDelayMs = reconnectDelayMs

    this.shouldSendHand = createRateLimiter(maxHandHz)
    // 보기 방향은 최대 maxViewHz로 보내되 마지막 값은 반드시 보낸다(드래그가 끝난 자리에서 화면이 멈추게).
    this.viewThrottle = createTrailingThrottle((view) => this.sendViewNow(view), {
      maxHz: maxViewHz,
      now,
      setTimer,
      clearTimer,
    })
    this.handRate = createRateCounter(RATE_WINDOW_MS)
    this.frameRate = createRateCounter(RATE_WINDOW_MS)

    this.ws = null
    this.stopped = true
    this.phase = 'idle' // idle | connecting | open | closed
    this.connectCount = 0
    this.reconnectTimer = null
    this.pingTimer = null
    this.nextPingId = 1
    this.pingSentAt = new Map() // ping id → 보낸 시각 (ms)
    this.rttMs = null
    this.serverState = null
    this.lastServerError = null
  }

  start() {
    if (!this.stopped) return
    this.stopped = false
    this.connect()
  }

  stop() {
    this.stopped = true
    clearTimeout(this.reconnectTimer)
    this.reconnectTimer = null
    this.viewThrottle.cancel()
    this.teardownSocket()
    this.phase = 'idle'
  }

  // hand 메시지를 보낸다. 연결 전이거나 빈도 제한에 걸리면 false.
  sendHand(msg) {
    if (!this.ws || this.ws.readyState !== OPEN) return false
    if (!msg || msg.type !== 'hand') return false
    const nowMs = this.now()
    if (!this.shouldSendHand(nowMs)) return false
    this.ws.send(JSON.stringify(msg))
    this.handRate.add(nowMs)
    return true
  }

  // 보기 방향({azimuth_deg, elevation_deg, distance_m})을 서버에 알린다. 빈도 제한(VIEW_SEND_MAX_HZ)이 있고,
  // 너무 자주 부르면 최신 값만 남겨 두었다가 허용 시각에 보낸다.
  requestView(view) {
    this.viewThrottle.push(view)
  }

  sendViewNow(view) {
    if (!this.ws || this.ws.readyState !== OPEN) return false
    this.ws.send(JSON.stringify(viewMessage(view)))
    return true
  }

  // 키보드 높이 메시지({type:'height', dir, slow})를 보낸다. 빈도는 호출하는 쪽(input/heightKeys.js)이 20Hz로 맞춘다.
  // 연결이 열려 있지 않으면 false(끊긴 연결로는 보낼 수 없다. 서버는 0.3초 타임아웃으로 멈춘다).
  sendHeight(msg) {
    if (!this.ws || this.ws.readyState !== OPEN) return false
    if (!msg || msg.type !== 'height') return false
    this.ws.send(JSON.stringify(msg))
    return true
  }

  getStats() {
    const nowMs = this.now()
    return {
      phase: this.phase,
      connectCount: this.connectCount,
      handPerSec: this.handRate.perSecond(nowMs),
      frameFps: this.frameRate.perSecond(nowMs),
      rttMs: this.rttMs,
      serverState: this.serverState,
      lastServerError: this.lastServerError,
    }
  }

  // --- 내부 ---

  connect() {
    this.phase = 'connecting'
    let ws
    try {
      ws = new this.WebSocketImpl(this.url)
    } catch (err) {
      // 예: 보안 정책(CSP)이 막으면 생성 단계에서 예외가 날 수 있다.
      console.warn('WebSocket 생성 실패:', err)
      this.scheduleReconnect()
      return
    }
    ws.binaryType = 'blob'
    this.ws = ws

    ws.onopen = () => {
      if (this.ws !== ws) return
      this.phase = 'open'
      this.connectCount += 1
      this.sendPing()
      this.pingTimer = setInterval(() => this.sendPing(), this.pingIntervalMs)
    }
    ws.onmessage = (event) => {
      if (this.ws !== ws) return
      this.handleMessage(event.data)
    }
    ws.onclose = () => {
      if (this.ws !== ws) return
      this.teardownSocket()
      this.scheduleReconnect()
    }
    // error 뒤에는 브라우저가 close를 부르므로 재연결은 onclose에서만 한다.
    ws.onerror = () => {}
  }

  teardownSocket() {
    clearInterval(this.pingTimer)
    this.pingTimer = null
    this.pingSentAt.clear()
    this.handRate.reset()
    this.frameRate.reset()
    const ws = this.ws
    this.ws = null
    if (ws) {
      ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null
      try {
        ws.close()
      } catch {
        // 이미 닫힘
      }
    }
  }

  scheduleReconnect() {
    if (this.stopped) return
    this.phase = 'closed'
    clearTimeout(this.reconnectTimer)
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      if (!this.stopped) this.connect()
    }, this.reconnectDelayMs)
  }

  sendPing() {
    if (!this.ws || this.ws.readyState !== OPEN) return
    const id = this.nextPingId++
    this.pingSentAt.set(id, this.now())
    // 응답이 안 온 ping이 쌓이지 않게 오래된 것은 버린다.
    if (this.pingSentAt.size > 10) this.pingSentAt.delete(this.pingSentAt.keys().next().value)
    this.ws.send(JSON.stringify({ type: 'ping', id }))
  }

  handleMessage(data) {
    if (typeof data !== 'string') {
      this.frameRate.add(this.now())
      this.onFrame(data)
      return
    }
    let msg
    try {
      msg = JSON.parse(data)
    } catch {
      return
    }
    if (msg.type === 'pong') {
      const sentAt = this.pingSentAt.get(msg.id)
      if (sentAt !== undefined) {
        this.pingSentAt.delete(msg.id)
        this.rttMs = this.now() - sentAt
      }
    } else if (msg.type === 'state') {
      this.serverState = msg
      this.onState(msg)
    } else if (msg.type === 'error') {
      this.lastServerError = msg.reason
    }
  }
}
