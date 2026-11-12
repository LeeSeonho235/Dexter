// Space 키(넘겨받기/돌려주기 토글): 버튼에 포커스가 있어도 동작, 스크롤 막기, 입력창과 Ctrl/Cmd/Alt 조합 무시, 반복과 연타, 높이 키와 겹치지 않음.
import { describe, expect, it } from 'vitest'
import { createHeightController } from './heightKeys.js'
import { createTakeoverKeyController } from './takeoverKey.js'

const AUTO = { status: 'running', owner: 'AUTO' }
const WAIT = { status: 'running', owner: 'HUMAN_WAIT' }
const HUMAN = { status: 'running', owner: 'HUMAN' }
const IDLE = { status: 'idle', owner: 'IDLE' }

function setup(auto = AUTO, { connected = true } = {}) {
  const sent = []
  const clock = { t: 1000 }
  const state = { auto, connected }
  const controller = createTakeoverKeyController({
    send: (m) => sent.push(m),
    getAuto: () => state.auto,
    isConnected: () => state.connected,
    now: () => clock.t,
    minIntervalMs: 400,
  })
  return { controller, sent, clock, state }
}

function key(code = 'Space', extra = {}) {
  const event = { code, target: { tagName: 'BODY' }, repeat: false, prevented: 0, preventDefault() { this.prevented += 1 }, ...extra }
  return event
}

function onButton() {
  const target = { tagName: 'BUTTON', blurred: 0, blur() { this.blurred += 1 } }
  return key('Space', { target })
}

describe('Space로 넘겨받기와 돌려주기', () => {
  it('AUTO에서 누르면 takeover, HUMAN에서 누르면 handback을 보낸다', () => {
    const t = setup(AUTO)
    expect(t.controller.keyDown(key())).toBe(true)
    expect(t.sent).toEqual([{ type: 'auto', action: 'takeover' }])
    t.state.auto = HUMAN
    t.clock.t += 500
    t.controller.keyDown(key())
    expect(t.sent.at(-1)).toEqual({ type: 'auto', action: 'handback' })
  })

  it('손을 기다리는 중이거나 AUTO가 아닐 때, 연결이 없을 때는 아무것도 보내지 않는다(그래도 스크롤은 막는다)', () => {
    for (const [auto, opts] of [[WAIT, {}], [IDLE, {}], [null, {}], [AUTO, { connected: false }]]) {
      const t = setup(auto, opts)
      const event = key()
      expect(t.controller.keyDown(event)).toBe(true)
      expect(t.sent).toEqual([])
      expect(event.prevented).toBe(1)
    }
  })

  it('페이지가 스크롤되지 않게 keydown과 keyup 모두 기본 동작을 막는다', () => {
    const t = setup(AUTO)
    const down = key()
    const up = key()
    t.controller.keyDown(down)
    expect(t.controller.keyUp(up)).toBe(true)
    expect(down.prevented).toBe(1)
    expect(up.prevented).toBe(1)
  })
})

describe('버튼에 포커스가 있을 때', () => {
  it('동작하고, 포커스를 빼서 같은 버튼이 다시 눌리지 않게 한다', () => {
    const t = setup(AUTO)
    const event = onButton()
    expect(t.controller.keyDown(event)).toBe(true)
    expect(t.sent).toEqual([{ type: 'auto', action: 'takeover' }])
    expect(event.target.blurred).toBe(1)
    expect(event.prevented).toBe(1)
  })
})

describe('무시해야 하는 경우', () => {
  it.each([['ctrlKey'], ['metaKey'], ['altKey']])('%s 조합은 보내지 않고 기본 동작도 막지 않는다', (modifier) => {
    const t = setup(AUTO)
    const event = key('Space', { [modifier]: true })
    expect(t.controller.keyDown(event)).toBe(false)
    expect(t.controller.keyUp(event)).toBe(false)
    expect(t.sent).toEqual([])
    expect(event.prevented).toBe(0)
  })

  it.each([['INPUT'], ['TEXTAREA'], ['SELECT']])('%s에 글자를 치는 중이면 무시한다(공백 입력을 막지 않는다)', (tagName) => {
    const t = setup(AUTO)
    const event = key('Space', { target: { tagName } })
    expect(t.controller.keyDown(event)).toBe(false)
    expect(t.sent).toEqual([])
    expect(event.prevented).toBe(0)
  })

  it('contentEditable도 무시한다', () => {
    const t = setup(AUTO)
    const event = key('Space', { target: { tagName: 'DIV', isContentEditable: true } })
    expect(t.controller.keyDown(event)).toBe(false)
    expect(t.sent).toEqual([])
  })

  it('Space가 아닌 키는 처리하지 않는다', () => {
    const t = setup(AUTO)
    for (const code of ['Enter', 'KeyW', 'ArrowUp', 'KeyS', 'ShiftLeft']) expect(t.controller.keyDown(key(code))).toBe(false)
    expect(t.sent).toEqual([])
  })
})

describe('반복과 연타', () => {
  it('키를 누르고 있어서 생기는 반복은 보내지 않는다', () => {
    const t = setup(AUTO)
    t.controller.keyDown(key())
    for (let i = 0; i < 5; i += 1) {
      t.clock.t += 500
      t.controller.keyDown(key('Space', { repeat: true }))
    }
    expect(t.sent.length).toBe(1)
  })

  it('최소 간격 안의 연타는 한 번만 보낸다(서버 응답을 기다리지 않고 토글이 뒤집히는 사고를 막는다)', () => {
    const t = setup(AUTO)
    t.controller.keyDown(key())
    t.clock.t += 100
    t.controller.keyDown(key())
    expect(t.sent.length).toBe(1)
    t.clock.t += 400
    t.controller.keyDown(key())
    expect(t.sent.length).toBe(2)
  })
})

describe('높이 키와 겹치지 않는다', () => {
  it('Space는 높이 컨트롤러가 처리하지 않고, ↑↓ W S는 Space 컨트롤러가 처리하지 않는다', () => {
    const heightSent = []
    const height = createHeightController({ send: (m) => heightSent.push(m), setTimer: () => 1, clearTimer: () => {} })
    const t = setup(AUTO)
    expect(height.keyDown(key('Space'))).toBe(false)
    expect(heightSent).toEqual([])
    for (const code of ['ArrowUp', 'ArrowDown', 'KeyW', 'KeyS']) expect(t.controller.keyDown(key(code))).toBe(false)
    expect(t.sent).toEqual([])
    // 같은 순간 ↓를 누른 채 Space를 눌러도 서로의 동작을 바꾸지 않는다
    height.keyDown(key('ArrowDown'))
    t.controller.keyDown(key('Space'))
    expect(heightSent.at(-1)).toEqual({ type: 'height', dir: -1, slow: false })
    expect(t.sent).toEqual([{ type: 'auto', action: 'takeover' }])
  })
})
