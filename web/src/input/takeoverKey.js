// Space 키 = "넘겨받기 / 돌려주기" 토글(AUTO 상자의 큰 버튼과 같은 일). React에 의존하지 않는다.
//
// 규칙
// - 키는 event.code === 'Space'(물리 키 위치)로 본다. 한글 입력기가 켜져 있어도 같다.
// - Ctrl/Meta/Alt 조합과 입력창(INPUT, TEXTAREA, SELECT, contentEditable)에서는 아무것도 하지 않고 기본 동작도 막지 않는다.
// - 그 외에는 항상 기본 동작을 막는다(페이지가 스크롤되면 안 된다). 버튼에 포커스가 있어도 동작해야 하므로, 버튼이 target이면 포커스를 빼서
//   keyup 때 Space가 그 버튼을 다시 누르는 일(카메라 켜기/끄기 등)이 없게 한다. keydown과 keyup 둘 다 기본 동작을 막는다.
// - 키를 누르고 있어서 생기는 반복(event.repeat)은 보내지 않는다. 연달아 눌러도 minIntervalMs 안에는 한 번만 보낸다.
// - 보낼 메시지는 서버가 알려 준 지금 상태로 정한다(correction.js spaceCommand): AUTO면 takeover, HUMAN이면 handback, 그 외에는 보내지 않는다.

import { spaceCommand } from '../auto/correction.js'
import { isTypingTarget } from './heightKeys.js'

export const TAKEOVER_MIN_INTERVAL_MS = 400

function ignoredByContext(event) {
  return event.ctrlKey || event.metaKey || event.altKey || isTypingTarget(event.target)
}

export function createTakeoverKeyController({
  send,
  getAuto,
  isConnected = () => true,
  now = () => performance.now(),
  minIntervalMs = TAKEOVER_MIN_INTERVAL_MS,
}) {
  let lastSentMs = -Infinity

  return {
    // keydown. Space를 처리하면 true(기본 동작을 막았다). 메시지를 실제로 보냈는지는 send의 호출로 본다.
    keyDown(event) {
      if (event.code !== 'Space') return false
      if (ignoredByContext(event)) return false
      event.preventDefault?.()
      const target = event.target
      if (target && String(target.tagName ?? '').toUpperCase() === 'BUTTON') target.blur?.()
      if (event.repeat) return true
      const message = spaceCommand(getAuto(), isConnected())
      if (!message) return true
      const t = now()
      if (t - lastSentMs < minIntervalMs) return true
      lastSentMs = t
      send(message)
      return true
    },
    // keyup. 버튼 활성화(클릭)와 스크롤을 막으려고 기본 동작만 막는다.
    keyUp(event) {
      if (event.code !== 'Space') return false
      if (ignoredByContext(event)) return false
      event.preventDefault?.()
      return true
    },
  }
}
