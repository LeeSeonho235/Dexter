// SimConnection.sendRecord: 녹화 버튼 메시지를 연결이 열려 있을 때만 JSON으로 보낸다. 버튼 → 메시지 전체 경로도 확인한다.
import { beforeEach, describe, expect, it } from 'vitest'
import { SimConnection } from './connection.js'
import { BUTTON_IDS, commandFor } from '../record/view.js'

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

const RECORD = { last_episode: { index: 7 } }

describe('SimConnection.sendRecord', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
  })

  it('연결 전에는 보내지 않고 false', () => {
    const { conn, ws } = makeConnection()
    expect(conn.sendRecord({ type: 'record', action: 'start' })).toBe(false)
    expect(ws.messagesOf('record')).toEqual([])
  })

  it('열려 있으면 JSON 텍스트로 보낸다', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    expect(conn.sendRecord({ type: 'record', action: 'start' })).toBe(true)
    expect(conn.sendRecord({ type: 'reset' })).toBe(true)
    expect(ws.messagesOf('record')).toEqual([{ type: 'record', action: 'start' }])
    expect(ws.messagesOf('reset')).toEqual([{ type: 'reset' }])
  })

  it('다른 종류의 메시지나 빈 값은 보내지 않는다(손 메시지를 흉내 내도 안 된다)', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    const before = ws.sent.length
    for (const bad of [null, undefined, {}, { type: 'hand' }, { type: 'height', dir: 1, slow: false }, { type: 'ping', id: 1 }, 'record']) {
      expect(conn.sendRecord(bad)).toBe(false)
    }
    expect(ws.sent.length).toBe(before)
  })

  it('끊긴 뒤에는 보내지 않는다', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    ws.readyState = 3
    expect(conn.sendRecord({ type: 'record', action: 'discard' })).toBe(false)
  })

  it('버튼 하나를 누를 때마다 서버가 받는 메시지가 정확히 한 개 생긴다', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    const expected = {
      start: { type: 'record', action: 'start' },
      stop_success: { type: 'record', action: 'stop_success' },
      stop_fail: { type: 'record', action: 'stop_fail' },
      discard: { type: 'record', action: 'discard' },
      reset: { type: 'reset' },
      delete_last: { type: 'record', action: 'delete_last', episode: 7 },
    }
    expect(Object.keys(expected).sort()).toEqual([...BUTTON_IDS].sort())
    for (const id of BUTTON_IDS) {
      const before = ws.sent.length
      expect(conn.sendRecord(commandFor(id, RECORD))).toBe(true)
      expect(ws.sent.length).toBe(before + 1)
      expect(JSON.parse(ws.sent.at(-1))).toEqual(expected[id])
    }
  })

  it('보낸 메시지는 서버 검사 규칙(필드 이름·값)과 맞는다', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    for (const id of BUTTON_IDS) conn.sendRecord(commandFor(id, RECORD))
    const msgs = ws.sent.map((s) => JSON.parse(s))
    for (const m of msgs.filter((m) => m.type === 'record')) {
      expect(['start', 'stop_success', 'stop_fail', 'discard', 'delete_last']).toContain(m.action)
      expect(Object.keys(m).sort()).toEqual(m.action === 'delete_last' ? ['action', 'episode', 'type'] : ['action', 'type'])
    }
    expect(msgs.find((m) => m.type === 'reset')).toEqual({ type: 'reset' })
  })
})
