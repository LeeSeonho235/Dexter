// 보조 카메라 화면 두 개(2-6): aux_frame 검사, 진짜 화면/2D 폴백 선택, 화면 스타일, 위치(나침반과 안 겹침), 켜기 메시지 시점, 안내 문구,
// 그리고 서버 파일·CSS와 같은 값인지 확인한다.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { boxesOverlap, compassBoxAbs } from '../scene/sideView.js'
import { COMPASS_BOX_PX, SIM_VIEW_SIZE_PX } from '../scene/config.js'
import {
  AUX_MARGIN_PX,
  AUX_MAX_JPEG_BASE64_CHARS,
  AUX_MIN_VIEW_WIDTH_PX,
  AUX_STALE_MS,
  AUX_TILE_SIZE_PX,
  AUX_TILES,
} from './config.js'
import { auxFrameFromMessage, auxMode, auxStatusText, auxTileBox, auxTileStyle, auxToggleVisible, planAuxSend } from './view.js'

const read = (relative) => readFileSync(new URL(relative, import.meta.url), 'utf8')

const MESSAGE = { type: 'aux_frame', seq: 7, w: 320, h: 120, views: 2, jpeg: '/9j/4AAQSkZJRg==' }

describe('auxFrameFromMessage', () => {
  it('올바른 메시지는 data 주소가 있는 프레임으로 바꾼다', () => {
    expect(auxFrameFromMessage(MESSAGE)).toEqual({ seq: 7, w: 320, h: 120, views: 2, src: 'data:image/jpeg;base64,/9j/4AAQSkZJRg==' })
  })

  it.each([
    ['null', null],
    ['다른 type', { ...MESSAGE, type: 'state' }],
    ['seq 없음', { ...MESSAGE, seq: undefined }],
    ['seq 소수', { ...MESSAGE, seq: 1.5 }],
    ['w 0', { ...MESSAGE, w: 0 }],
    ['h 음수', { ...MESSAGE, h: -1 }],
    ['views 0', { ...MESSAGE, views: 0 }],
    ['views 문자열', { ...MESSAGE, views: '2' }],
    ['jpeg 없음', { ...MESSAGE, jpeg: undefined }],
    ['jpeg 빈 문자열', { ...MESSAGE, jpeg: '' }],
    ['jpeg 숫자', { ...MESSAGE, jpeg: 12 }],
    ['jpeg가 base64가 아님', { ...MESSAGE, jpeg: 'abc$%^' }],
    ['jpeg에 따옴표(CSS 주입 시도)', { ...MESSAGE, jpeg: 'abcd")' }],
    ['jpeg가 너무 큼', { ...MESSAGE, jpeg: 'A'.repeat(AUX_MAX_JPEG_BASE64_CHARS + 1) }],
  ])('이상한 메시지는 버린다: %s', (_name, msg) => {
    expect(auxFrameFromMessage(msg)).toBeNull()
  })
})

describe('auxMode: 진짜 화면 두 개(mirrors) 또는 2D 그림(fallback)', () => {
  const nowMs = 100_000
  const good = {
    wanted: true,
    connected: true,
    auxState: { enabled: true, running: true, error: null, slowed: false, fps: 10 },
    frame: { views: 2, receivedMs: nowMs - 200 },
    nowMs,
  }

  it('켜져 있고, 오류 없고, 프레임이 최근이면 진짜 화면', () => {
    expect(auxMode(good)).toBe('mirrors')
  })

  it.each([
    ['체크박스를 껐다', { wanted: false }],
    ['연결이 없다', { connected: false }],
    ['state.aux가 없다', { auxState: undefined }],
    ['서버에서 aux가 꺼져 있다', { auxState: { ...good.auxState, enabled: false } }],
    ['aux 스레드가 안 돈다', { auxState: { ...good.auxState, running: false } }],
    ['오류가 있다', { auxState: { ...good.auxState, error: 'RuntimeError: EGL' } }],
    ['프레임이 아직 없다', { frame: null }],
    ['시점 수가 맞지 않는다', { frame: { views: 1, receivedMs: nowMs } }],
    ['프레임이 끊겼다', { frame: { views: 2, receivedMs: nowMs - AUX_STALE_MS - 1 } }],
  ])('2D 그림으로 돌아간다: %s', (_name, change) => {
    expect(auxMode({ ...good, ...change })).toBe('fallback')
  })

  it('끊김 경계: 정확히 AUX_STALE_MS 전이면 아직 진짜 화면', () => {
    expect(auxMode({ ...good, frame: { views: 2, receivedMs: nowMs - AUX_STALE_MS } })).toBe('mirrors')
  })

  it('느려져서 fps가 낮아져도(감속) 프레임이 계속 오면 진짜 화면', () => {
    expect(auxMode({ ...good, auxState: { ...good.auxState, slowed: true, fps: 1 }, frame: { views: 2, receivedMs: nowMs - 1100 } })).toBe('mirrors')
  })

  it('시계가 거꾸로 가거나 값이 이상해도 진짜 화면으로 잘못 고르지 않는다', () => {
    expect(auxMode({ ...good, nowMs: Number.NaN })).toBe('fallback')
    expect(auxMode({ ...good, frame: { views: 2, receivedMs: undefined } })).toBe('fallback')
  })
})

describe('auxTileStyle: 한 장의 이미지를 칸 둘로', () => {
  const frame = { src: 'data:image/jpeg;base64,AAAA', views: 2 }

  it('옆에서 = 왼쪽 아래, 비스듬히 = 오른쪽 아래, 각 200x150, 여백은 상수', () => {
    const [side, oblique] = AUX_TILES.map((tile, index) => auxTileStyle(tile, index, frame))
    expect(AUX_TILES.map((t) => [t.id, t.label, t.corner])).toEqual([
      ['side', '옆에서', 'bottom-left'],
      ['oblique', '비스듬히', 'bottom-right'],
    ])
    expect(AUX_TILE_SIZE_PX).toEqual({ width: 200, height: 150 })
    expect(side).toMatchObject({ bottom: AUX_MARGIN_PX, left: AUX_MARGIN_PX, width: 200, height: 150 })
    expect(side.right).toBeUndefined()
    expect(oblique).toMatchObject({ bottom: AUX_MARGIN_PX, right: AUX_MARGIN_PX, width: 200, height: 150 })
    expect(oblique.left).toBeUndefined()
  })

  it('이미지의 왼쪽 절반이 옆에서, 오른쪽 절반이 비스듬히', () => {
    const [side, oblique] = AUX_TILES.map((tile, index) => auxTileStyle(tile, index, frame))
    expect(side.backgroundSize).toBe('400px 150px')
    expect(side.backgroundPosition).toBe('0px 0px')
    expect(oblique.backgroundSize).toBe('400px 150px')
    expect(oblique.backgroundPosition).toBe('-200px 0px')
    expect(side.backgroundImage).toBe('url("data:image/jpeg;base64,AAAA")')
  })

  it('서버가 작은 크기(160x120 x 2시점 = 320x120)로 보내도 칸 크기(200x150)에 맞게 늘린다', () => {
    const small = { src: 'data:image/jpeg;base64,AAAA', views: 2, w: 320, h: 120 }
    const [side, oblique] = AUX_TILES.map((tile, index) => auxTileStyle(tile, index, small))
    expect(side.backgroundSize).toBe('400px 150px') // 이미지 자체 크기가 아니라 시점 수 x 칸 크기
    expect(oblique.backgroundPosition).toBe('-200px 0px')
  })
})

describe('위치: 나침반과 겹치지 않는다', () => {
  it('640x480에서 두 화면은 메인 화면 안, 서로와 나침반을 피한다', () => {
    const compass = compassBoxAbs(COMPASS_BOX_PX, SIM_VIEW_SIZE_PX)
    const [left, right] = AUX_TILES.map((tile) => auxTileBox(tile, SIM_VIEW_SIZE_PX))
    for (const box of [left, right]) {
      expect(box.left).toBeGreaterThanOrEqual(0)
      expect(box.right).toBeLessThanOrEqual(SIM_VIEW_SIZE_PX.width)
      expect(box.bottom).toBeLessThanOrEqual(SIM_VIEW_SIZE_PX.height)
      expect(boxesOverlap(box, compass)).toBe(false)
    }
    expect(boxesOverlap(left, right)).toBe(false)
    expect(left.left).toBe(AUX_MARGIN_PX)
    expect(right.right).toBe(SIM_VIEW_SIZE_PX.width - AUX_MARGIN_PX)
  })

  it('보여 주는 가장 좁은 창(AUX_MIN_VIEW_WIDTH_PX)부터 640 px까지(4:3) 겹치지 않는다', () => {
    for (let width = AUX_MIN_VIEW_WIDTH_PX; width <= 640; width += 10) {
      const sim = { width, height: (width * 3) / 4 }
      const compass = compassBoxAbs(COMPASS_BOX_PX, sim)
      const [left, right] = AUX_TILES.map((tile) => auxTileBox(tile, sim))
      expect(boxesOverlap(left, compass), `왼쪽 ${width}`).toBe(false)
      expect(boxesOverlap(right, compass), `오른쪽 ${width}`).toBe(false)
      expect(boxesOverlap(left, right), `서로 ${width}`).toBe(false)
    }
  })

  it('검사가 빈 것이 아님: 숨기는 기준보다 훨씬 좁은 창(320 px)에서는 오른쪽 화면이 나침반과 겹친다', () => {
    const sim = { width: 320, height: 240 }
    const compass = compassBoxAbs(COMPASS_BOX_PX, sim)
    const [, right] = AUX_TILES.map((tile) => auxTileBox(tile, sim))
    expect(boxesOverlap(right, compass)).toBe(true)
  })
})

describe('planAuxSend: 서버에 aux 켜기/끄기를 언제 보내나', () => {
  const on = { connected: true, serverEnabled: true, wanted: true, connectCount: 1 }

  it('연결이 열리면 한 번 켜기를 보내고, 같은 연결에서는 다시 보내지 않는다', () => {
    const first = planAuxSend(on, null)
    expect(first.message).toEqual({ type: 'aux', enabled: true })
    expect(planAuxSend(on, first.last)).toEqual({ message: null, last: first.last })
  })

  it('체크박스를 끄면 끄기를, 다시 켜면 켜기를 보낸다', () => {
    const first = planAuxSend(on, null)
    const off = planAuxSend({ ...on, wanted: false }, first.last)
    expect(off.message).toEqual({ type: 'aux', enabled: false })
    expect(planAuxSend({ ...on, wanted: true }, off.last).message).toEqual({ type: 'aux', enabled: true })
  })

  it('재연결하면(connectCount가 늘면) 켜 두었던 것을 다시 보낸다', () => {
    const first = planAuxSend(on, null)
    expect(planAuxSend({ ...on, connectCount: 2 }, first.last).message).toEqual({ type: 'aux', enabled: true })
  })

  it('연결이 끊기면 기억을 지운다(같은 번호로 다시 열려도 다시 보낸다)', () => {
    const first = planAuxSend(on, null)
    const down = planAuxSend({ ...on, connected: false }, first.last)
    expect(down).toEqual({ message: null, last: null })
    expect(planAuxSend(on, down.last).message).toEqual({ type: 'aux', enabled: true })
  })

  it('서버에서 aux가 꺼져 있으면 아무것도 보내지 않는다(체크박스를 바꿔도)', () => {
    expect(planAuxSend({ ...on, serverEnabled: false }, null)).toEqual({ message: null, last: null })
    expect(planAuxSend({ ...on, serverEnabled: false, wanted: false }, null).message).toBeNull()
  })

  it('새 연결에서 처음부터 꺼 두었으면 보내지 않는다(서버 기본이 꺼짐). 기록만 한다', () => {
    const plan = planAuxSend({ ...on, wanted: false }, null)
    expect(plan).toEqual({ message: null, last: { connectCount: 1, enabled: false } })
    expect(planAuxSend({ ...on, wanted: true }, plan.last).message).toEqual({ type: 'aux', enabled: true })
  })
})

describe('auxToggleVisible / auxStatusText', () => {
  const auxState = { enabled: true, running: true, error: null, slowed: false, fps: 10 }
  const base = { connected: true, auxState, wanted: true, mode: 'mirrors' }

  it('체크박스는 서버에서 aux가 켜져 있고 연결됐을 때만', () => {
    expect(auxToggleVisible({ connected: true, auxState })).toBe(true)
    expect(auxToggleVisible({ connected: false, auxState })).toBe(false)
    expect(auxToggleVisible({ connected: true, auxState: { ...auxState, enabled: false } })).toBe(false)
    expect(auxToggleVisible({ connected: true, auxState: undefined })).toBe(false)
  })

  it('상태별 안내 문구', () => {
    expect(auxStatusText({ ...base, auxState: { ...auxState, enabled: false } })).toBeNull()
    expect(auxStatusText({ ...base, connected: false })).toBeNull()
    expect(auxStatusText({ ...base, auxState: { ...auxState, error: 'RuntimeError: EGL' } })).toContain('RuntimeError: EGL')
    expect(auxStatusText({ ...base, auxState: { ...auxState, running: false } })).toContain('멈춰')
    expect(auxStatusText({ ...base, wanted: false, mode: 'fallback' })).toContain('꺼짐')
    expect(auxStatusText({ ...base, mode: 'fallback' })).toContain('기다리는 중')
    expect(auxStatusText({ ...base, auxState: { ...auxState, slowed: true, fps: 2.5 } })).toContain('2.5fps')
    expect(auxStatusText(base)).toContain('옆에서')
    expect(auxStatusText(base)).toContain('비스듬히')
  })
})

describe('서버와 CSS에 적은 값과 같다', () => {
  it('화면 순서(옆에서, 비스듬히)가 서버 이미지의 가로 순서와 같다 (server/aux_view_config.py)', () => {
    const py = read('../../../server/aux_view_config.py')
    const block = py.match(/^AUX_CAMERAS = \(([\s\S]*?)^\)/m)[1]
    expect([...block.matchAll(/\{"name": "(\w+)"/g)].map((m) => m[1])).toEqual(['side', 'oblique']) // 왼쪽이 옆에서, 오른쪽이 비스듬히
    expect(AUX_TILES.map((t) => t.id)).toEqual(['side', 'oblique'])
    expect(block).toMatch(/"name": "side", "azimuth_deg": _SIDE_AZIMUTH_DEG, "elevation_deg": -12\.0, "distance_m": 0\.68/)
    expect(block).toMatch(/"name": "oblique", "azimuth_deg": 45\.0, "elevation_deg": -25\.0, "distance_m": 1\.0/)
    expect(py).toMatch(/^AUX_FOVY_DEG = 20\.0/m)
  })

  it('칸 크기가 서버 기본 시점 크기(normal)와 같다', () => {
    const py = read('../../../server/aux_view_config.py')
    const normal = py.match(/AUX_SIZE_LEVELS = \{[^}]*"normal": \((\d+), (\d+)\)/)
    expect([Number(normal[1]), Number(normal[2])]).toEqual([AUX_TILE_SIZE_PX.width, AUX_TILE_SIZE_PX.height])
  })

  it('aux_frame 필드 이름이 서버(app.py)와 같다', () => {
    const app = read('../../../server/app.py')
    const block = app.match(/def aux_frame_text[\s\S]*?return json\.dumps\(([\s\S]*?)\n    \)/)[1]
    for (const key of ['type', 'seq', 'w', 'h', 'views', 'jpeg']) expect(block).toContain(`"${key}"`)
    expect(block).toContain('"aux_frame"')
  })

  it('서버가 보내는 안 보임 판정에 쓰는 끊김 시간이 서버 최저 fps의 주기보다 길다', () => {
    const py = read('../../../server/aux_view_config.py')
    const minFps = Number(py.match(/^AUX_MIN_FPS = ([\d.]+)$/m)[1])
    expect(AUX_STALE_MS).toBeGreaterThan(1000 / minFps)
  })

  it('좁은 창 기준(AUX_MIN_VIEW_WIDTH_PX)이 index.css의 @container 한 줄과 같다', () => {
    const css = read('../index.css')
    const query = css.match(/@container \(max-width: (\d+)px\)/)
    expect(Number(query[1])).toBe(AUX_MIN_VIEW_WIDTH_PX - 1)
    const simViewRules = [...css.matchAll(/\.sim-view\s*\{([^}]*)\}/g)].map((m) => m[1])
    expect(simViewRules.some((rule) => /container-type:\s*inline-size/.test(rule))).toBe(true)
  })

  it('보조 화면 칸은 클릭을 받지 않는다(pointer-events: none)', () => {
    const css = read('../index.css')
    const rule = css.match(/\.aux-mirrors\s*\{([^}]*)\}/)[1]
    expect(rule).toMatch(/pointer-events:\s*none/)
    expect(rule).toMatch(/position:\s*absolute/)
  })
})
