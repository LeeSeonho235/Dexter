// 녹화 상자 버튼에 포커스가 있어도 높이 키(↑↓, W/S)가 그대로 동작하는지, 입력창에서만 무시되는지 확인한다.
// 높이 키는 window에서 받으므로 이벤트의 target이 버튼이어도 처리되어야 한다(RecordPanel은 입력창을 쓰지 않는다).
import { describe, expect, it } from 'vitest'
import { createHeightController } from './heightKeys.js'

function setup() {
  const sent = []
  const controller = createHeightController({ send: (m) => sent.push(m), setTimer: () => 1, clearTimer: () => {} })
  return { controller, sent }
}

const onButton = (code, extra = {}) => ({ code, target: { tagName: 'BUTTON' }, preventDefault() {}, ...extra })

describe('버튼에 포커스가 있을 때', () => {
  it.each([
    ['ArrowUp', 1],
    ['KeyW', 1],
    ['ArrowDown', -1],
    ['KeyS', -1],
  ])('%s 키는 처리되고 dir %i를 보낸다', (code, dir) => {
    const { controller, sent } = setup()
    expect(controller.keyDown(onButton(code))).toBe(true)
    expect(sent.at(-1)).toEqual({ type: 'height', dir, slow: false })
    expect(controller.keyUp(onButton(code))).toBe(true)
    expect(sent.at(-1)).toEqual({ type: 'height', dir: 0, slow: false })
  })

  it('Shift + 방향키는 천천히(slow)로 보낸다', () => {
    const { controller, sent } = setup()
    controller.keyDown(onButton('ShiftLeft'))
    controller.keyDown(onButton('ArrowDown'))
    expect(sent.at(-1)).toEqual({ type: 'height', dir: -1, slow: true })
  })

  it('방향키가 페이지를 스크롤하지 않게 preventDefault를 부른다', () => {
    const { controller } = setup()
    let prevented = false
    controller.keyDown(onButton('ArrowUp', { preventDefault: () => (prevented = true) }))
    expect(prevented).toBe(true)
  })

  it('Enter와 Space는 높이 키가 아니라서 처리하지 않는다(버튼 동작을 가로채지 않는다)', () => {
    const { controller, sent } = setup()
    let prevented = false
    const event = (code) => onButton(code, { preventDefault: () => (prevented = true) })
    expect(controller.keyDown(event('Enter'))).toBe(false)
    expect(controller.keyDown(event('Space'))).toBe(false)
    expect(prevented).toBe(false)
    expect(sent).toEqual([])
  })
})

describe('입력창에서는 무시된다(기존 동작 그대로)', () => {
  it.each(['INPUT', 'TEXTAREA', 'SELECT'])('%s에 포커스가 있으면 높이 키를 처리하지 않는다', (tagName) => {
    const { controller, sent } = setup()
    expect(controller.keyDown({ code: 'ArrowUp', target: { tagName }, preventDefault() {} })).toBe(false)
    expect(sent).toEqual([])
  })
})
