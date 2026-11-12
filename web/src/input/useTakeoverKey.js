// Space 키(넘겨받기/돌려주기 토글)를 브라우저 이벤트에 연결하는 React 훅. 규칙은 takeoverKey.js(순수 계산)에 있다.
// 높이 키(useHeightKeys)와 같은 window 이벤트를 쓰지만 키가 겹치지 않는다(높이는 ↑↓ W S, 여기는 Space).

import { useEffect } from 'react'
import { createTakeoverKeyController } from './takeoverKey.js'

// autoRef: 서버 state.auto의 최신 값을 담은 ref, connectedRef: 연결이 열려 있는지 담은 ref
export function useTakeoverKey({ connRef, autoRef, connectedRef }) {
  useEffect(() => {
    const controller = createTakeoverKeyController({
      send: (msg) => connRef.current?.sendAuto(msg),
      getAuto: () => autoRef.current,
      isConnected: () => connectedRef.current === true,
    })
    const onKeyDown = (event) => controller.keyDown(event)
    const onKeyUp = (event) => controller.keyUp(event)
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
    }
  }, [connRef, autoRef, connectedRef])
}
