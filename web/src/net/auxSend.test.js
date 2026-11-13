// SimConnection의 보조 카메라(2-6): sendAux(켜기/끄기 메시지)와 aux_frame 수신(onAuxFrame). 메인 화면 바이너리 프레임 처리는 그대로여야 한다.
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

  receive(data) {
    this.onmessage?.({ data })
  }

  messagesOf(type) {
    return this.sent.map((s) => JSON.parse(s)).filter((m) => m.type === type)
  }
}

function makeConnection(handlers = {}) {
  const conn = new SimConnection({
    url: 'ws://localhost:3007/ws',
    WebSocketImpl: FakeWebSocket,
    now: () => Date.now(),
    pingIntervalMs: 100000,
    ...handlers,
  })
  conn.start()
  return { conn, ws: FakeWebSocket.instances[FakeWebSocket.instances.length - 1] }
}

describe('SimConnection.sendAux', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
  })

  it('연결 전에는 보내지 않고 false', () => {
    const { conn, ws } = makeConnection()
    expect(conn.sendAux({ type: 'aux', enabled: true })).toBe(false)
    expect(ws.messagesOf('aux')).toEqual([])
  })

  it('연결되면 {type:"aux", enabled}만 JSON 텍스트로 보낸다', () => {
    const { conn, ws } = makeConnection()
    ws.open()
    expect(conn.sendAux({ type: 'aux', enabled: true })).toBe(true)
    expect(conn.sendAux({ type: 'aux', enabled: false, extra: 'x' })).toBe(true)
    expect(ws.messagesOf('aux')).toEqual([
      { type: 'aux', enabled: true },
      { type: 'aux', enabled: false }, // 다른 필드는 서버가 거부하므로 보내지 않는다
    ])
  })

  it.each([null, undefined, {}, { type: 'auto', action: 'start' }, { type: 'aux' }, { type: 'aux', enabled: 1 }, { type: 'aux', enabled: 'true' }])(
    '잘못된 메시지는 보내지 않는다: %j',
    (msg) => {
      const { conn, ws } = makeConnection()
      ws.open()
      expect(conn.sendAux(msg)).toBe(false)
      expect(ws.messagesOf('aux')).toEqual([])
    },
  )
})

describe('SimConnection aux_frame 수신', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
  })

  const MESSAGE = { type: 'aux_frame', seq: 3, w: 320, h: 120, views: 2, jpeg: '/9j/4AAQSkZJRg==' }

  it('aux_frame은 onAuxFrame으로 가고 메인 화면 onFrame이나 state로는 가지 않는다', () => {
    const aux = []
    const frames = []
    const states = []
    const { ws, conn } = makeConnection({ onAuxFrame: (f) => aux.push(f), onFrame: (b) => frames.push(b), onState: (s) => states.push(s) })
    ws.open()
    ws.receive(JSON.stringify(MESSAGE))
    expect(aux).toEqual([{ seq: 3, w: 320, h: 120, views: 2, src: 'data:image/jpeg;base64,/9j/4AAQSkZJRg==' }])
    expect(frames).toEqual([])
    expect(states).toEqual([])
    expect(conn.getStats().frameFps).toBe(0) // 메인 화면 fps 통계에 섞이지 않는다
  })

  it('이상한 aux_frame은 버린다(예외 없이)', () => {
    const aux = []
    const { ws } = makeConnection({ onAuxFrame: (f) => aux.push(f) })
    ws.open()
    ws.receive(JSON.stringify({ ...MESSAGE, jpeg: 'not base64 !!' }))
    ws.receive(JSON.stringify({ ...MESSAGE, w: 0 }))
    ws.receive('{broken json')
    expect(aux).toEqual([])
  })

  it('메인 화면 바이너리 프레임과 state는 그대로 처리된다', () => {
    const frames = []
    const states = []
    const aux = []
    const { ws, conn } = makeConnection({ onFrame: (b) => frames.push(b), onState: (s) => states.push(s), onAuxFrame: (f) => aux.push(f) })
    ws.open()
    const blob = new Blob([new Uint8Array([0xff, 0xd8])])
    ws.receive(blob)
    ws.receive(JSON.stringify({ type: 'state', aux: { enabled: true } }))
    ws.receive(JSON.stringify(MESSAGE))
    expect(frames).toEqual([blob])
    expect(states).toEqual([{ type: 'state', aux: { enabled: true } }])
    expect(aux).toHaveLength(1)
    expect(conn.getStats().serverState).toEqual({ type: 'state', aux: { enabled: true } })
  })

  it('onAuxFrame을 안 주면(기본) 받아도 아무 일 없다', () => {
    const { ws } = makeConnection()
    ws.open()
    expect(() => ws.receive(JSON.stringify(MESSAGE))).not.toThrow()
  })
})
