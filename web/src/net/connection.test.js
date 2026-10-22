import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { SimConnection, wsUrl } from './connection.js'

// 테스트용 가짜 WebSocket. 만들어진 것을 FakeWebSocket.instances에 모은다.
class FakeWebSocket {
  static instances = []

  constructor(url) {
    this.url = url
    this.readyState = 0
    this.sent = []
    this.closed = false
    FakeWebSocket.instances.push(this)
  }

  send(data) {
    this.sent.push(data)
  }

  close() {
    this.closed = true
    this.readyState = 3
  }

  // --- 서버 쪽 흉내 ---
  open() {
    this.readyState = 1
    this.onopen?.()
  }

  receive(data) {
    this.onmessage?.({ data })
  }

  drop() {
    this.readyState = 3
    this.onclose?.()
  }

  sentJson() {
    return this.sent.map((s) => JSON.parse(s))
  }
}

const HAND = {
  type: 'hand',
  t: 1,
  detected: true,
  wrist: [0.5, 0.6, 0],
  index_tip: [0.5, 0.5, 0],
  thumb_tip: [0.45, 0.55, 0],
  pinch: 0.05,
  pinch_ratio: 0.3,
  palm_open: false,
  hand_scale: 0.17,
}

function makeConnection(overrides = {}) {
  const conn = new SimConnection({
    url: 'ws://localhost:3007/ws',
    WebSocketImpl: FakeWebSocket,
    now: () => Date.now(),
    pingIntervalMs: 1000,
    reconnectDelayMs: 2000,
    ...overrides,
  })
  conn.start()
  return conn
}

function last() {
  return FakeWebSocket.instances.at(-1)
}

beforeEach(() => {
  FakeWebSocket.instances = []
  vi.useFakeTimers()
  vi.setSystemTime(0)
})

afterEach(() => {
  vi.useRealTimers()
})

describe('wsUrl', () => {
  it('지금 페이지와 같은 주소의 /ws', () => {
    expect(wsUrl({ protocol: 'http:', host: 'localhost:3007' })).toBe('ws://localhost:3007/ws')
    expect(wsUrl({ protocol: 'https:', host: 'example.test' })).toBe('wss://example.test/ws')
  })
})

describe('SimConnection', () => {
  it('start하면 연결하고, 열리면 ping을 보낸다', () => {
    const conn = makeConnection()
    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(last().url).toBe('ws://localhost:3007/ws')
    expect(conn.getStats().phase).toBe('connecting')

    last().open()
    expect(conn.getStats().phase).toBe('open')
    expect(last().sentJson()).toEqual([{ type: 'ping', id: 1 }])

    vi.advanceTimersByTime(1000)
    expect(last().sentJson().at(-1)).toEqual({ type: 'ping', id: 2 })
  })

  it('pong이 오면 왕복 시간(ms)을 잰다', () => {
    const conn = makeConnection()
    last().open() // ping id 1을 시각 0에 보냄
    vi.advanceTimersByTime(25)
    last().receive(JSON.stringify({ type: 'pong', id: 1 }))
    expect(conn.getStats().rttMs).toBe(25)

    // 모르는 id의 pong은 무시
    last().receive(JSON.stringify({ type: 'pong', id: 99 }))
    expect(conn.getStats().rttMs).toBe(25)
  })

  it('연결 전에는 hand를 보내지 않는다', () => {
    const conn = makeConnection()
    expect(conn.sendHand(HAND)).toBe(false)
    expect(last().sent).toEqual([])
  })

  it('hand는 JSON 텍스트로만 보내고, hand가 아닌 것은 보내지 않는다', () => {
    const conn = makeConnection()
    last().open()
    last().sent = []
    expect(conn.sendHand(HAND)).toBe(true)
    expect(conn.sendHand({ type: 'video', data: 'x' })).toBe(false)
    expect(conn.sendHand(null)).toBe(false)
    expect(last().sent).toHaveLength(1)
    expect(typeof last().sent[0]).toBe('string')
    expect(JSON.parse(last().sent[0])).toEqual(HAND)
  })

  it('60FPS로 넣어도 1초에 30개까지만 보낸다', () => {
    const conn = makeConnection({ pingIntervalMs: 100000 })
    last().open()
    last().sent = []
    for (let i = 0; i < 60; i++) {
      conn.sendHand({ ...HAND, t: i })
      vi.advanceTimersByTime(1000 / 60)
    }
    expect(last().sent).toHaveLength(30)
    expect(conn.getStats().handPerSec).toBe(30)
  })

  it('바이너리는 화면 프레임으로 넘기고 fps를 센다', () => {
    const frames = []
    const conn = makeConnection({ onFrame: (f) => frames.push(f) })
    last().open()
    for (let i = 0; i < 15; i++) {
      last().receive(new Uint8Array([0xff, 0xd8, i]))
      vi.advanceTimersByTime(1000 / 15)
    }
    expect(frames).toHaveLength(15)
    expect(conn.getStats().frameFps).toBeGreaterThanOrEqual(14)
    expect(conn.getStats().frameFps).toBeLessThanOrEqual(15)
  })

  it('state와 error 메시지를 기억한다', () => {
    const states = []
    const conn = makeConnection({ onState: (s) => states.push(s) })
    last().open()
    const state = { type: 'state', hand_count: 3, invalid_count: 1, last_hand: null, sim_time_s: 1.2 }
    last().receive(JSON.stringify(state))
    last().receive(JSON.stringify({ type: 'error', reason: 'bad' }))
    last().receive('not json')
    expect(states).toEqual([state])
    expect(conn.getStats().serverState).toEqual(state)
    expect(conn.getStats().lastServerError).toBe('bad')
  })

  it('끊기면 2초 뒤 다시 연결한다', () => {
    const conn = makeConnection()
    last().open()
    last().drop()
    expect(conn.getStats().phase).toBe('closed')
    expect(FakeWebSocket.instances).toHaveLength(1)

    vi.advanceTimersByTime(1999)
    expect(FakeWebSocket.instances).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(FakeWebSocket.instances).toHaveLength(2)
    expect(conn.getStats().phase).toBe('connecting')

    last().open()
    expect(conn.getStats().phase).toBe('open')
    expect(conn.getStats().connectCount).toBe(2)
  })

  it('처음 연결이 실패해도 계속 다시 시도한다', () => {
    makeConnection()
    last().drop() // 서버가 꺼져 있음
    vi.advanceTimersByTime(2000)
    last().drop()
    vi.advanceTimersByTime(2000)
    expect(FakeWebSocket.instances).toHaveLength(3)
  })

  it('끊긴 뒤에는 ping을 보내지 않는다', () => {
    makeConnection()
    const first = last()
    first.open()
    first.drop()
    const sentBefore = first.sent.length
    vi.advanceTimersByTime(1500)
    expect(first.sent.length).toBe(sentBefore)
  })

  it('WebSocket 생성이 예외를 던져도 다시 시도한다', () => {
    let calls = 0
    class Throwing {
      constructor() {
        calls++
        throw new Error('blocked')
      }
    }
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const conn = makeConnection({ WebSocketImpl: Throwing })
    expect(conn.getStats().phase).toBe('closed')
    vi.advanceTimersByTime(2000)
    expect(calls).toBe(2)
    conn.stop()
    warn.mockRestore()
  })

  it('stop하면 소켓을 닫고 다시 연결하지 않는다', () => {
    const conn = makeConnection()
    last().open()
    const ws = last()
    conn.stop()
    expect(ws.closed).toBe(true)
    expect(conn.getStats().phase).toBe('idle')
    vi.advanceTimersByTime(10000)
    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(ws.sent.filter((s) => JSON.parse(s).type === 'ping')).toHaveLength(1)
  })
})
