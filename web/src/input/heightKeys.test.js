import { describe, expect, it, vi } from 'vitest'
import {
  createHeightController,
  directionFromKeys,
  isHeightKey,
  isTypingTarget,
} from './heightKeys.js'

// 시계와 타이머를 직접 움직이는 테스트 도구
function makeClock() {
  let nowMs = 0
  let timers = []
  let nextId = 1
  return {
    now: () => nowMs,
    setTimer: (fn, ms) => {
      const id = nextId++
      timers.push({ id, at: nowMs + ms, fn })
      return id
    },
    clearTimer: (id) => {
      timers = timers.filter((t) => t.id !== id)
    },
    pending: () => timers.length,
    advanceTo(toMs) {
      for (;;) {
        const due = timers.filter((t) => t.at <= toMs).sort((a, b) => a.at - b.at)[0]
        if (!due) break
        timers = timers.filter((t) => t !== due)
        nowMs = Math.max(nowMs, due.at)
        due.fn()
      }
      nowMs = toMs
    },
  }
}

function setup(options = {}) {
  const clock = makeClock()
  const sent = [] // { msg, at }
  const changes = []
  const controller = createHeightController({
    send: (msg) => sent.push({ msg, at: clock.now() }),
    setTimer: clock.setTimer,
    clearTimer: clock.clearTimer,
    onChange: (state) => changes.push(state),
    ...options,
  })
  return { clock, sent, changes, controller }
}

function keyEvent(code, extra = {}) {
  return { code, repeat: false, ctrlKey: false, metaKey: false, altKey: false, target: { tagName: 'BODY' }, preventDefault: vi.fn(), ...extra }
}

const msgs = (sent) => sent.map((s) => s.msg)
const HEIGHT = (dir, slow = false) => ({ type: 'height', dir, slow })

describe('directionFromKeys', () => {
  it('↓/S는 -1, ↑/W는 +1', () => {
    expect(directionFromKeys(new Set(['ArrowDown'])).dir).toBe(-1)
    expect(directionFromKeys(new Set(['KeyS'])).dir).toBe(-1)
    expect(directionFromKeys(new Set(['ArrowUp'])).dir).toBe(1)
    expect(directionFromKeys(new Set(['KeyW'])).dir).toBe(1)
  })
  it('둘 다 누르면 정지(0): 방향키끼리도, 방향키와 W/S도', () => {
    expect(directionFromKeys(new Set(['ArrowDown', 'ArrowUp'])).dir).toBe(0)
    expect(directionFromKeys(new Set(['KeyS', 'KeyW'])).dir).toBe(0)
    expect(directionFromKeys(new Set(['ArrowDown', 'KeyW'])).dir).toBe(0)
  })
  it('같은 방향의 두 키는 그냥 그 방향', () => {
    expect(directionFromKeys(new Set(['ArrowDown', 'KeyS'])).dir).toBe(-1)
  })
  it('Shift는 방향이 있을 때만 slow, 혼자서는 아무것도 아니다', () => {
    expect(directionFromKeys(new Set(['ArrowDown', 'ShiftLeft']))).toMatchObject({ dir: -1, slow: true })
    expect(directionFromKeys(new Set(['KeyW', 'ShiftRight']))).toMatchObject({ dir: 1, slow: true })
    expect(directionFromKeys(new Set(['ShiftLeft']))).toMatchObject({ dir: 0, slow: false, shift: true })
    expect(directionFromKeys(new Set(['ArrowDown']))).toMatchObject({ slow: false })
  })
  it('아무 키도 없으면 정지', () => {
    expect(directionFromKeys(new Set())).toEqual({ dir: 0, slow: false, down: false, up: false, shift: false })
  })
})

describe('isHeightKey / isTypingTarget', () => {
  it('높이에 쓰는 키만 true', () => {
    for (const code of ['ArrowDown', 'ArrowUp', 'KeyS', 'KeyW', 'ShiftLeft', 'ShiftRight']) expect(isHeightKey(code)).toBe(true)
    for (const code of ['KeyA', 'ArrowLeft', 'Space', 'Enter']) expect(isHeightKey(code)).toBe(false)
  })
  it('입력창이면 true', () => {
    expect(isTypingTarget({ tagName: 'INPUT' })).toBe(true)
    expect(isTypingTarget({ tagName: 'textarea' })).toBe(true)
    expect(isTypingTarget({ tagName: 'SELECT' })).toBe(true)
    expect(isTypingTarget({ tagName: 'DIV', isContentEditable: true })).toBe(true)
    expect(isTypingTarget({ tagName: 'BODY' })).toBe(false)
    expect(isTypingTarget(null)).toBe(false)
  })
})

describe('키 → 메시지', () => {
  it('↓를 누르면 dir -1 한 번, 떼면 dir 0 한 번', () => {
    const { sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    expect(msgs(sent)).toEqual([HEIGHT(-1)])
    controller.keyUp(keyEvent('ArrowDown'))
    expect(msgs(sent)).toEqual([HEIGHT(-1), HEIGHT(0)])
  })

  it('S, ↑, W', () => {
    for (const [code, dir] of [['KeyS', -1], ['ArrowUp', 1], ['KeyW', 1]]) {
      const { sent, controller } = setup()
      controller.keyDown(keyEvent(code))
      expect(msgs(sent)).toEqual([HEIGHT(dir)])
    }
  })

  it('Shift를 같이 누르면 slow true', () => {
    const { sent, controller } = setup()
    controller.keyDown(keyEvent('ShiftLeft'))
    expect(sent).toEqual([]) // Shift 혼자는 아무것도 안 보낸다
    controller.keyDown(keyEvent('ArrowDown'))
    expect(msgs(sent)).toEqual([HEIGHT(-1, true)])
  })

  it('누르는 도중 Shift를 누르거나 떼면 slow가 바뀐 메시지를 바로 보낸다', () => {
    const { sent, controller } = setup()
    controller.keyDown(keyEvent('KeyW'))
    controller.keyDown(keyEvent('ShiftRight'))
    controller.keyUp(keyEvent('ShiftRight'))
    expect(msgs(sent)).toEqual([HEIGHT(1), HEIGHT(1, true), HEIGHT(1, false)])
  })

  it('둘 다 누르면 정지(dir 0), 하나를 떼면 남은 키 방향', () => {
    const { sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    controller.keyDown(keyEvent('ArrowUp'))
    expect(msgs(sent)).toEqual([HEIGHT(-1), HEIGHT(0)])
    controller.keyUp(keyEvent('ArrowDown'))
    expect(msgs(sent)).toEqual([HEIGHT(-1), HEIGHT(0), HEIGHT(1)])
  })

  it('같은 방향의 두 키 중 하나만 떼면 계속 그 방향(메시지 없음), 둘 다 떼면 0', () => {
    const { sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    controller.keyDown(keyEvent('KeyS'))
    controller.keyUp(keyEvent('ArrowDown'))
    expect(msgs(sent)).toEqual([HEIGHT(-1)])
    controller.keyUp(keyEvent('KeyS'))
    expect(msgs(sent)).toEqual([HEIGHT(-1), HEIGHT(0)])
  })

  it('높이와 상관없는 키는 무시', () => {
    const { sent, controller } = setup()
    const event = keyEvent('KeyA')
    expect(controller.keyDown(event)).toBe(false)
    expect(controller.keyUp(keyEvent('KeyA'))).toBe(false)
    expect(sent).toEqual([])
    expect(event.preventDefault).not.toHaveBeenCalled()
  })

  it('키 자동 반복(event.repeat)은 상태를 바꾸지 않는다', () => {
    const { sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    for (let i = 0; i < 5; i++) controller.keyDown(keyEvent('ArrowDown', { repeat: true }))
    expect(msgs(sent)).toEqual([HEIGHT(-1)])
  })
})

describe('20Hz 반복', () => {
  it('누르고 있는 동안 50ms마다 같은 메시지를 되풀이한다(1초에 20~21개)', () => {
    const { clock, sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    clock.advanceTo(1000)
    expect(sent.length).toBeGreaterThanOrEqual(20)
    expect(sent.length).toBeLessThanOrEqual(21)
    for (const { msg } of sent) expect(msg).toEqual(HEIGHT(-1))
    for (let i = 1; i < sent.length; i++) expect(sent[i].at - sent[i - 1].at).toBeGreaterThanOrEqual(50 - 1e-9)
  })

  it('slow 상태로 반복하면 slow도 같이 되풀이', () => {
    const { clock, sent, controller } = setup()
    controller.keyDown(keyEvent('ShiftLeft'))
    controller.keyDown(keyEvent('KeyW'))
    clock.advanceTo(300)
    expect(msgs(sent).every((m) => m.dir === 1 && m.slow === true)).toBe(true)
    expect(sent.length).toBeGreaterThanOrEqual(6)
  })

  it('키를 떼면 반복이 멈추고 dir 0은 한 번만 간다', () => {
    const { clock, sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowUp'))
    clock.advanceTo(205)
    controller.keyUp(keyEvent('ArrowUp'))
    const count = sent.length
    expect(msgs(sent).at(-1)).toEqual(HEIGHT(0))
    clock.advanceTo(3000)
    expect(sent.length).toBe(count) // 그 뒤로는 아무것도 안 보낸다
    expect(clock.pending()).toBe(0)
  })

  it('정지(dir 0) 상태에서는 반복하지 않는다', () => {
    const { clock, sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    controller.keyDown(keyEvent('ArrowUp')) // 둘 다 → 정지
    const count = sent.length
    clock.advanceTo(1000)
    expect(sent.length).toBe(count)
  })

  it('방향이 바뀌면 그 순간 보내고 반복 주기를 새로 시작한다', () => {
    const { clock, sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    clock.advanceTo(30)
    controller.keyUp(keyEvent('ArrowDown'))
    controller.keyDown(keyEvent('ArrowUp'))
    expect(msgs(sent)).toEqual([HEIGHT(-1), HEIGHT(0), HEIGHT(1)])
    clock.advanceTo(30 + 49)
    expect(sent.length).toBe(3) // 새 주기가 50ms
    clock.advanceTo(30 + 50)
    expect(msgs(sent).at(-1)).toEqual(HEIGHT(1))
    expect(sent.length).toBe(4)
  })
})

describe('방향키 스크롤 방지, 입력창, 조합 키', () => {
  it('↑/↓는 preventDefault(페이지가 스크롤되지 않게), Shift는 하지 않는다', () => {
    const { controller } = setup()
    const down = keyEvent('ArrowDown')
    const up = keyEvent('ArrowUp')
    const shift = keyEvent('ShiftLeft')
    controller.keyDown(down)
    controller.keyDown(up)
    controller.keyDown(shift)
    expect(down.preventDefault).toHaveBeenCalled()
    expect(up.preventDefault).toHaveBeenCalled()
    expect(shift.preventDefault).not.toHaveBeenCalled()
  })

  it('자동 반복 keydown에서도 스크롤을 막는다', () => {
    const { controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    const repeated = keyEvent('ArrowDown', { repeat: true })
    controller.keyDown(repeated)
    expect(repeated.preventDefault).toHaveBeenCalled()
  })

  for (const tag of ['INPUT', 'TEXTAREA', 'SELECT']) {
    it(`${tag}에 포커스가 있으면 무시하고 스크롤도 막지 않는다`, () => {
      const { sent, controller } = setup()
      const event = keyEvent('ArrowDown', { target: { tagName: tag } })
      expect(controller.keyDown(event)).toBe(false)
      expect(controller.keyDown(keyEvent('KeyS', { target: { tagName: tag } }))).toBe(false)
      expect(sent).toEqual([])
      expect(event.preventDefault).not.toHaveBeenCalled()
    })
  }

  it('contenteditable도 무시', () => {
    const { sent, controller } = setup()
    expect(controller.keyDown(keyEvent('KeyW', { target: { tagName: 'DIV', isContentEditable: true } }))).toBe(false)
    expect(sent).toEqual([])
  })

  it('입력창으로 옮겨 가기 전에 누른 키를 입력창에서 떼도 반영된다(눌린 채 남지 않는다)', () => {
    const { sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    controller.keyUp(keyEvent('ArrowDown', { target: { tagName: 'INPUT' } }))
    expect(msgs(sent)).toEqual([HEIGHT(-1), HEIGHT(0)])
  })

  it('Ctrl/Meta/Alt와 같이 누른 키(Ctrl+S 저장 등)는 무시하고 기본 동작도 막지 않는다', () => {
    for (const modifier of ['ctrlKey', 'metaKey', 'altKey']) {
      const { sent, controller } = setup()
      const event = keyEvent('KeyS', { [modifier]: true })
      expect(controller.keyDown(event)).toBe(false)
      expect(sent).toEqual([])
      expect(event.preventDefault).not.toHaveBeenCalled()
    }
  })
})

describe('blur, 연결 끊김, 재연결', () => {
  it('누르고 있다가 blur하면 dir 0을 한 번 보내고 반복을 멈춘다', () => {
    const { clock, sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    clock.advanceTo(120)
    controller.blur()
    expect(msgs(sent).at(-1)).toEqual(HEIGHT(0))
    const count = sent.length
    clock.advanceTo(3000)
    expect(sent.length).toBe(count)
    expect(clock.pending()).toBe(0)
  })

  it('blur 뒤에 늦게 오는 keyup은 아무것도 보내지 않는다(눌린 채 남는 사고 없음)', () => {
    const { sent, controller } = setup()
    controller.keyDown(keyEvent('KeyS'))
    controller.blur()
    const count = sent.length
    controller.keyUp(keyEvent('KeyS'))
    expect(sent.length).toBe(count)
    expect(controller.getState()).toMatchObject({ dir: 0, keys: [] })
  })

  it('blur 뒤 다시 누르면 처음부터 동작한다', () => {
    const { sent, controller } = setup()
    controller.keyDown(keyEvent('KeyS'))
    controller.blur()
    controller.keyDown(keyEvent('KeyS'))
    expect(msgs(sent)).toEqual([HEIGHT(-1), HEIGHT(0), HEIGHT(-1)])
  })

  it('아무것도 안 누른 채 blur하면 아무것도 보내지 않는다', () => {
    const { sent, controller } = setup()
    controller.blur()
    expect(sent).toEqual([])
  })

  it('Shift만 누른 채 blur해도 상태가 비워진다', () => {
    const { sent, controller, changes } = setup()
    controller.keyDown(keyEvent('ShiftLeft'))
    controller.blur()
    expect(sent).toEqual([])
    expect(changes.at(-1)).toMatchObject({ shift: false })
    controller.keyDown(keyEvent('ArrowDown'))
    expect(msgs(sent)).toEqual([HEIGHT(-1)]) // 지워진 Shift가 남아 slow가 되지 않는다
  })

  it('연결이 끊기면(reset) 보내지 않고 키 상태와 타이머만 비운다', () => {
    const { clock, sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    const count = sent.length
    controller.reset()
    expect(sent.length).toBe(count)
    expect(clock.pending()).toBe(0)
    clock.advanceTo(2000)
    expect(sent.length).toBe(count)
    expect(controller.getState()).toMatchObject({ dir: 0, keys: [] })
  })

  it('재연결 직후 dir 0을 한 번 보낸다', () => {
    const { sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    controller.reset()
    sent.length = 0
    controller.connectionOpened()
    expect(msgs(sent)).toEqual([HEIGHT(0)])
  })

  it('재연결 때 키가 눌려 있으면(반복이 곧 보낼 것이므로) 따로 0을 보내지 않는다', () => {
    const { sent, controller } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    sent.length = 0
    controller.connectionOpened()
    expect(sent).toEqual([])
  })
})

describe('onChange (화면의 눌린 키 표시)', () => {
  it('눌린 키 상태가 바뀔 때만 알려 준다', () => {
    const { controller, changes } = setup()
    controller.keyDown(keyEvent('ArrowDown'))
    controller.keyDown(keyEvent('ShiftLeft'))
    controller.keyUp(keyEvent('ShiftLeft'))
    controller.keyUp(keyEvent('ArrowDown'))
    expect(changes.map((c) => [c.down, c.shift, c.dir, c.slow])).toEqual([
      [true, false, -1, false],
      [true, true, -1, true],
      [true, false, -1, false],
      [false, false, 0, false],
    ])
  })

  it('Shift만 눌러도 키 표시는 켜진다(메시지는 없다)', () => {
    const { controller, changes, sent } = setup()
    controller.keyDown(keyEvent('ShiftRight'))
    expect(changes.at(-1)).toMatchObject({ shift: true, dir: 0 })
    expect(sent).toEqual([])
  })
})
