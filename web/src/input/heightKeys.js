// 키보드 높이 조작(↓/S 내리기, ↑/W 올리기, Shift = 천천히) → 서버로 보낼 height 메시지. React에 의존하지 않는다.
//
// 보내는 메시지: { type: 'height', dir: -1 | 0 | 1, slow: boolean }
//   - 방향이나 slow가 바뀔 때 바로 한 번 보낸다.
//   - dir이 0이 아닌 동안은 HEIGHT_REPEAT_HZ(20Hz)로 같은 메시지를 되풀이해 보낸다(서버는 0.3초 동안 메시지가 없으면 정지).
//   - 키를 모두 떼면 dir 0을 한 번 보낸다. blur(창/탭이 포커스를 잃음)나 탭이 숨겨질 때도 키 상태를 비우고 dir 0을 보낸다.
//   - 연결이 끊기면 보낼 수 없으므로 키 상태와 반복 타이머만 비우고(reset), 다시 연결되면 dir 0을 한 번 보낸다.
// 키는 event.code(물리 키 위치)로 본다: 한글 입력기가 켜져 있어도 W/S가 같다.

import { HEIGHT_REPEAT_HZ } from '../net/config.js'

export const DOWN_CODES = ['ArrowDown', 'KeyS']
export const UP_CODES = ['ArrowUp', 'KeyW']
export const SHIFT_CODES = ['ShiftLeft', 'ShiftRight']
const SCROLL_CODES = ['ArrowDown', 'ArrowUp'] // 눌렀을 때 페이지가 스크롤되면 안 되는 키
const RELEVANT = new Set([...DOWN_CODES, ...UP_CODES, ...SHIFT_CODES])

export function isHeightKey(code) {
  return RELEVANT.has(code)
}

// 입력창에 글자를 치는 중이면 키보드 높이를 무시한다.
export function isTypingTarget(element) {
  if (!element) return false
  const tag = String(element.tagName ?? '').toUpperCase()
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || element.isContentEditable === true
}

// 눌린 키 집합(code) → { dir, slow, down, up, shift }. 둘 다 누르면 dir 0(정지). Shift 혼자는 아무것도 아니다(slow는 dir이 있을 때만).
export function directionFromKeys(pressed) {
  const down = DOWN_CODES.some((code) => pressed.has(code))
  const up = UP_CODES.some((code) => pressed.has(code))
  const shift = SHIFT_CODES.some((code) => pressed.has(code))
  const dir = (up ? 1 : 0) - (down ? 1 : 0)
  return { dir, slow: dir !== 0 && shift, down, up, shift }
}

export function createHeightController({
  send,
  repeatHz = HEIGHT_REPEAT_HZ,
  setTimer = (fn, ms) => setTimeout(fn, ms),
  clearTimer = (id) => clearTimeout(id),
  onChange = () => {},
}) {
  const repeatMs = 1000 / repeatHz
  const pressed = new Set()
  let current = { dir: 0, slow: false, down: false, up: false, shift: false }
  let timer = null

  function emit() {
    send({ type: 'height', dir: current.dir, slow: current.slow })
  }

  function stopRepeat() {
    if (timer !== null) clearTimer(timer)
    timer = null
  }

  function scheduleRepeat() {
    stopRepeat()
    timer = setTimer(() => {
      timer = null
      if (current.dir === 0) return
      emit()
      scheduleRepeat()
    }, repeatMs)
  }

  // 눌린 키가 바뀐 뒤 부른다: 방향이나 slow가 바뀌었으면 바로 보내고 반복을 새로 시작한다.
  function update() {
    const next = directionFromKeys(pressed)
    const sendChanged = next.dir !== current.dir || next.slow !== current.slow
    const displayChanged = sendChanged || next.down !== current.down || next.up !== current.up || next.shift !== current.shift
    current = next
    if (sendChanged) {
      emit()
      if (current.dir !== 0) scheduleRepeat()
      else stopRepeat()
    }
    if (displayChanged) onChange(current)
  }

  // 모든 키를 뗀 것으로 하고 상태를 비운다. notify면 dir이 0이 아니었을 때 dir 0을 한 번 보낸다.
  function clearAll(notify) {
    const wasActive = current.dir !== 0
    pressed.clear()
    stopRepeat()
    const changedDisplay = current.down || current.up || current.shift
    current = { dir: 0, slow: false, down: false, up: false, shift: false }
    if (notify && wasActive) emit()
    if (changedDisplay) onChange(current)
  }

  return {
    // keydown. 처리하는 키면 true. Ctrl/Meta/Alt 조합(Ctrl+S 저장 등)과 입력창은 무시(false)하고 기본 동작도 막지 않는다.
    keyDown(event) {
      if (event.ctrlKey || event.metaKey || event.altKey) return false
      if (isTypingTarget(event.target)) return false
      if (!RELEVANT.has(event.code)) return false
      if (SCROLL_CODES.includes(event.code)) event.preventDefault?.() // 방향키가 페이지를 스크롤하지 않게
      if (event.repeat) return true // 키 자동 반복은 상태가 아니라서 무시한다
      pressed.add(event.code)
      update()
      return true
    },
    // keyup. 입력창에 포커스가 있어도 떼는 것은 항상 반영한다(눌린 채 남는 사고 방지).
    keyUp(event) {
      if (!RELEVANT.has(event.code)) return false
      pressed.delete(event.code)
      update()
      return true
    },
    // 창/탭이 포커스를 잃거나 숨겨질 때: keyup이 오지 않을 수 있으므로 모두 뗀 것으로 하고 dir 0을 보낸다.
    blur() {
      clearAll(true)
    },
    // 연결이 끊김: 보낼 수 없으니 상태와 타이머만 비운다.
    reset() {
      clearAll(false)
    },
    // 연결(재연결 포함)이 열린 직후: 눌린 키가 없는 깨끗한 상태임을 서버에 한 번 알린다.
    connectionOpened() {
      if (current.dir === 0) emit()
    },
    getState() {
      return { ...current, keys: [...pressed] }
    },
  }
}
