// SimConnection.sendAuto: AUTO 버튼 메시지를 연결이 열려 있을 때만 JSON으로 보낸다. 버튼 → 메시지 전체 경로도 확인한다.
import { beforeEach, describe, expect, it } from 'vitest'
import { SimConnection } from './connection.js'
import { AUTO_BUTTON_IDS, commandFor } from '../auto/view.js'

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

  messagesOf(type) {
    return this.sent.map((s) => JSON.parse(s)).filter((m) => m.type === type)
  }
}

function makeConnection() {
  const conn = new SimConnection({ url: 'ws://localhost:3007/ws', WebSocketImpl: FakeWebSocket, now: () => Date.now(), pingIntervalMs: 100000 })
  conn.start()
  return { conn, ws: FakeWebSocket.instances[FakeWebSocket.instances.length - 1] }
}

describe('SimConnection.sendAuto', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
  })

  it('연결 전에는 보내지 않고 false', () => {
    const { conn, ws } = makeConnection()
    expect(conn.sendAuto({ type: 'auto', action: 'start' })).toBe(false)
    expect(ws.messagesOf('auto')).toEqual([])
  })

  it('연결되면 JSON 텍스트로 보낸다', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    expect(conn.sendAuto({ type: 'auto', action: 'stop' })).toBe(true)
    expect(ws.messagesOf('auto')).toEqual([{ type: 'auto', action: 'stop' }])
  })

  it('auto가 아닌 메시지와 빈 값은 보내지 않는다', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    expect(conn.sendAuto({ type: 'record', action: 'start' })).toBe(false)
    expect(conn.sendAuto({ type: 'reset' })).toBe(false)
    expect(conn.sendAuto(null)).toBe(false)
    expect(conn.sendAuto(undefined)).toBe(false)
    expect(ws.messagesOf('auto')).toEqual([])
  })

  it('버튼 → 메시지 → 전송: 세 버튼 모두 서버가 받는 형식이다', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    for (const id of AUTO_BUTTON_IDS) expect(conn.sendAuto(commandFor(id))).toBe(true)
    expect(ws.messagesOf('auto').map((m) => m.action)).toEqual(['start', 'stop', 'shutdown_policy'])
  })

  it('끊기면 보내지 않는다', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    ws.close()
    expect(conn.sendAuto({ type: 'auto', action: 'start' })).toBe(false)
  })
})
