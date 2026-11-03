// 삭제 2단계 확인: 첫 누름은 확인 상태, 시간 안의 두 번째 누름이 실행, 시간이 지나면 풀린다.
import { describe, expect, it } from 'vitest'
import { createConfirm } from './confirm.js'

function fakeTimers() {
  let next = 1
  const timers = new Map()
  return {
    setTimer: (fn, ms) => {
      const id = next++
      timers.set(id, { fn, ms })
      return id
    },
    clearTimer: (id) => timers.delete(id),
    fireAll: () => {
      for (const [id, t] of [...timers]) {
        timers.delete(id)
        t.fn()
      }
    },
    pending: () => timers.size,
    lastMs: () => [...timers.values()].at(-1)?.ms,
  }
}

function make(extra = {}) {
  const timers = fakeTimers()
  const changes = []
  const confirm = createConfirm({ ...timers, timeoutMs: 3000, onChange: (v) => changes.push(v), ...extra })
  return { confirm, timers, changes }
}

describe('createConfirm', () => {
  it('첫 번째 누름은 실행하지 않고 확인 상태로만 바꾼다', () => {
    const { confirm, changes, timers } = make()
    expect(confirm.press()).toBe(false)
    expect(confirm.isArmed()).toBe(true)
    expect(changes).toEqual([true])
    expect(timers.pending()).toBe(1)
    expect(timers.lastMs()).toBe(3000)
  })

  it('두 번째 누름이 실행이고 확인 상태가 풀린다', () => {
    const { confirm, changes, timers } = make()
    confirm.press()
    expect(confirm.press()).toBe(true)
    expect(confirm.isArmed()).toBe(false)
    expect(changes).toEqual([true, false])
    expect(timers.pending()).toBe(0)
    expect(confirm.press()).toBe(false) // 다음 삭제는 다시 두 번 눌러야 한다
  })

  it('시간이 지나면 확인 상태가 풀리고 다음 누름은 다시 첫 번째 누름이다', () => {
    const { confirm, timers, changes } = make()
    confirm.press()
    timers.fireAll()
    expect(confirm.isArmed()).toBe(false)
    expect(changes).toEqual([true, false])
    expect(confirm.press()).toBe(false)
  })

  it('cancel은 확인 상태와 타이머를 지운다(없을 때는 아무 일도 없다)', () => {
    const { confirm, timers, changes } = make()
    confirm.cancel()
    expect(changes).toEqual([])
    confirm.press()
    confirm.cancel()
    expect(confirm.isArmed()).toBe(false)
    expect(timers.pending()).toBe(0)
    expect(changes).toEqual([true, false])
  })

  it('기본 시간은 설정값(3초)이다', () => {
    const timers = fakeTimers()
    const confirm = createConfirm({ setTimer: timers.setTimer, clearTimer: timers.clearTimer })
    confirm.press()
    expect(timers.lastMs()).toBe(3000)
  })
})
