// SimConnection.sendHeight: 연결이 열려 있을 때만 height 메시지를 JSON으로 보낸다.
import { beforeEach, describe, expect, it } from 'vitest'
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

  heights() {
    return this.sent.map((s) => JSON.parse(s)).filter((m) => m.type === 'height')
  }
}

function makeConnection() {
  const conn = new SimConnection({ url: 'ws://localhost:3007/ws', WebSocketImpl: FakeWebSocket, now: () => Date.now(), pingIntervalMs: 100000 })
  conn.start()
  return { conn, ws: FakeWebSocket.instances[FakeWebSocket.instances.length - 1] }
}

describe('SimConnection.sendHeight', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
  })

  it('열려 있으면 height 메시지를 그대로 보낸다', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    expect(conn.sendHeight({ type: 'height', dir: -1, slow: true })).toBe(true)
    expect(ws.heights()).toEqual([{ type: 'height', dir: -1, slow: true }])
  })

  it('연결 전, 연결이 끊긴 뒤에는 보내지 않고 false', () => {
    const { conn, ws } = makeConnection()
    expect(conn.sendHeight({ type: 'height', dir: 1, slow: false })).toBe(false) // 아직 open 전
    ws.open()
    ws.readyState = 3 // 끊김
    expect(conn.sendHeight({ type: 'height', dir: 1, slow: false })).toBe(false)
    expect(ws.heights()).toEqual([])
  })

  it('height가 아닌 메시지는 보내지 않는다', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    expect(conn.sendHeight({ type: 'hand' })).toBe(false)
    expect(conn.sendHeight(null)).toBe(false)
    expect(ws.heights()).toEqual([])
  })
})
