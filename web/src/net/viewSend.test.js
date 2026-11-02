// SimConnection.requestView: 보기 방향 메시지를 최대 15Hz로 보내고, 마지막 값은 반드시 보낸다.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { SimConnection } from './connection.js'

class FakeWebSocket {
  static instances = []

  constructor(url) {
    this.url = url
    this.readyState = 0
    this.sent = []
    FakeWebSocket.instances.push(this)
  }

  send(data) {
    this.sent.push(data)
  }

  close() {
    this.readyState = 3
  }

  open() {
    this.readyState = 1
    this.onopen?.()
  }

  views() {
    return this.sent.map((s) => JSON.parse(s)).filter((m) => m.type === 'view')
  }
}

const view = (azimuth) => ({ azimuth_deg: azimuth, elevation_deg: -35, distance_m: 0.8 })

function makeOpenConnection(overrides = {}) {
  const conn = new SimConnection({
    url: 'ws://localhost:3007/ws',
    WebSocketImpl: FakeWebSocket,
    now: () => Date.now(),
    pingIntervalMs: 100000,
    ...overrides,
  })
  conn.start()
  const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1]
  ws.open()
  return { conn, ws }
}

describe('SimConnection.requestView', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
    vi.useFakeTimers()
    vi.setSystemTime(0)
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('view 메시지 형식으로 보낸다', () => {
    const { conn, ws } = makeOpenConnection()
    conn.requestView(view(30))
    expect(ws.views()).toEqual([{ type: 'view', azimuth: 30, elevation: -35, distance: 0.8 }])
  })

  it('60Hz로 끌어도 1초에 15개 이하로 보내고 마지막 값은 도착한다', () => {
    const { conn, ws } = makeOpenConnection()
    for (let i = 0; i < 120; i++) {
      // 2초 동안 60Hz
      conn.requestView(view(i))
      vi.advanceTimersByTime(1000 / 60)
    }
    vi.advanceTimersByTime(500)
    const views = ws.views()
    expect(views.length).toBeLessThanOrEqual(2 * 15 + 1)
    expect(views.length).toBeGreaterThanOrEqual(25)
    expect(views[views.length - 1].azimuth).toBe(119) // 마지막 값
  })

  it('연결이 열려 있지 않으면 아무것도 보내지 않고 오류도 없다', () => {
    const conn = new SimConnection({ url: 'ws://x/ws', WebSocketImpl: FakeWebSocket, now: () => Date.now() })
    conn.start() // 아직 open 전
    expect(() => conn.requestView(view(1))).not.toThrow()
    expect(FakeWebSocket.instances[0].views()).toEqual([])
  })

  it('stop하면 기다리던 값을 보내지 않는다', () => {
    const { conn, ws } = makeOpenConnection()
    conn.requestView(view(1))
    conn.requestView(view(2))
    conn.stop()
    vi.advanceTimersByTime(1000)
    expect(ws.views().map((v) => v.azimuth)).toEqual([1])
  })
})
