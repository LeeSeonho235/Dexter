// 키보드 높이 조작을 브라우저 이벤트에 연결하는 React 훅. 변환 규칙은 heightKeys.js(순수 계산)에 있다.
//
// window의 keydown/keyup/blur, 탭이 숨겨질 때(visibilitychange)를 듣는다. connected가 false가 되면(연결 끊김) 키 상태를
// 비우고, 다시 true가 되면(재연결) dir 0을 한 번 보낸다. 눌린 키 상태({dir, slow, down, up, shift})를 돌려줘 화면에 표시한다.

import { useEffect, useRef, useState } from 'react'
import { createHeightController } from './heightKeys.js'

const IDLE = { dir: 0, slow: false, down: false, up: false, shift: false }

export function useHeightKeys({ connRef, connected }) {
  const [keys, setKeys] = useState(IDLE)
  const controllerRef = useRef(null)

  useEffect(() => {
    const controller = createHeightController({
      send: (msg) => connRef.current?.sendHeight(msg),
      onChange: setKeys,
    })
    controllerRef.current = controller

    const onKeyDown = (event) => controller.keyDown(event)
    const onKeyUp = (event) => controller.keyUp(event)
    const onBlur = () => controller.blur()
    const onVisibility = () => {
      if (document.visibilityState === 'hidden') controller.blur()
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('blur', onBlur)
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', onBlur)
      document.removeEventListener('visibilitychange', onVisibility)
      controller.blur() // 페이지를 떠날 때 눌린 키가 남지 않게
      controller.reset()
      controllerRef.current = null
    }
  }, [connRef])

  useEffect(() => {
    const controller = controllerRef.current
    if (!controller) return
    if (connected) controller.connectionOpened()
    else controller.reset()
  }, [connected])

  return keys
}
