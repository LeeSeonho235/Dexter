// SimView와 AuxMirrors를 서버 렌더(react-dom/server)로 그려서 DOM 연결을 확인한다: 모드별로 무엇이 보이는지, 체크박스 표시 조건, 위치 스타일.
// (브라우저 없이 문자열로만 확인하므로 효과(useEffect, 캔버스 그리기)는 돌지 않는다. 눈으로 보는 확인은 사용자가 한다.)
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { AUX_MARGIN_PX, AUX_TILES } from '../aux/config.js'
import AuxMirrors from './AuxMirrors.jsx'
import SimView from './SimView.jsx'

const VIEW = { azimuth_deg: 110, elevation_deg: -35, distance_m: 0.8 }
const FRAME = { seq: 1, w: 320, h: 120, views: 2, src: 'data:image/jpeg;base64,AAAA', receivedMs: 0 }
const AUX_STATE = { enabled: true, running: true, error: null, slowed: false, fps: 10, viewers: 1 }

function render(props = {}) {
  return renderToStaticMarkup(
    <SimView url={null} view={VIEW} mode="absolute" onDrag={() => {}} onWheel={() => {}} onPreset={() => {}} connected {...props} />,
  )
}

const count = (html, text) => html.split(text).length - 1

describe('AuxMirrors', () => {
  it('프레임이 없으면 아무것도 그리지 않는다', () => {
    expect(renderToStaticMarkup(<AuxMirrors frame={null} />)).toBe('')
  })

  it('칸 둘: 옆에서(왼쪽 아래), 비스듬히(오른쪽 아래), 이미지의 왼쪽/오른쪽 절반', () => {
    const html = renderToStaticMarkup(<AuxMirrors frame={FRAME} />)
    expect(count(html, 'class="aux-tile"')).toBe(2)
    expect(html).toContain('class="aux-mirrors"')
    const [side, oblique] = AUX_TILES.map((t) => t.label)
    expect(html.indexOf(side)).toBeLessThan(html.indexOf(oblique)) // 왼쪽 칸이 먼저
    expect(html).toContain(`bottom:${AUX_MARGIN_PX}px;left:${AUX_MARGIN_PX}px;width:160px;height:120px`)
    expect(html).toContain(`bottom:${AUX_MARGIN_PX}px;right:${AUX_MARGIN_PX}px;width:160px;height:120px`)
    expect(html).toContain('background-position:0px 0px')
    expect(html).toContain('background-position:-160px 0px')
    expect(html).toContain('background-size:320px 120px')
    expect(html).toContain('data:image/jpeg;base64,AAAA')
  })
})

describe('SimView 보조 화면 모드', () => {
  it('mirrors: 진짜 화면 두 개 + (좁은 창용으로 숨겨 둔) 2D 그림', () => {
    const html = render({ auxMode: 'mirrors', auxFrame: FRAME, auxState: AUX_STATE })
    expect(count(html, 'class="aux-tile"')).toBe(2)
    expect(html).toContain('side-mirror side-mirror-narrow-only')
    expect(count(html, 'side-mirror-narrow-only')).toBe(1) // 2D 그림은 하나(좁은 창 전용)뿐이다
    expect(html).not.toContain('class="side-mirror"') // 평소 모양의 2D 그림은 없다
  })

  it('fallback: 2D 그림만(평소 모양 그대로), 진짜 화면은 없다', () => {
    const html = render({ auxMode: 'fallback', auxFrame: null, auxState: AUX_STATE })
    expect(html).not.toContain('aux-mirrors')
    expect(html).toContain('class="side-mirror"')
    expect(html).not.toContain('side-mirror-narrow-only')
  })

  it('aux 관련 props를 안 주면(기본) 지금까지와 같다: 2D 그림만, 체크박스 없음', () => {
    const html = renderToStaticMarkup(<SimView url={null} view={VIEW} mode="absolute" onDrag={() => {}} onWheel={() => {}} onPreset={() => {}} />)
    expect(html).toContain('class="side-mirror"')
    expect(html).not.toContain('aux-toggle')
    expect(html).not.toContain('aux-mirrors')
  })

  it('메인 화면 요소(나침반, 범례)는 그대로 있다', () => {
    const html = render({ auxMode: 'mirrors', auxFrame: FRAME, auxState: AUX_STATE })
    expect(html).toContain('view-compass')
    expect(html).toContain('legend')
    expect(html).toContain('시뮬레이션 화면 기다리는 중')
  })
})

describe('SimView 체크박스', () => {
  it('서버에서 aux가 켜져 있으면 보이고 기본은 체크됨', () => {
    const html = render({ auxMode: 'mirrors', auxFrame: FRAME, auxState: AUX_STATE })
    expect(html).toContain('aux-toggle')
    expect(html).toMatch(/<input type="checkbox" checked=""/)
    expect(html).toContain('옆에서')
  })

  it('끈 상태(auxWanted=false)는 체크가 풀리고 안내가 바뀐다', () => {
    const html = render({ auxMode: 'fallback', auxState: AUX_STATE, auxWanted: false })
    expect(html).not.toMatch(/<input type="checkbox" checked=""/)
    expect(html).toContain('보조 화면 꺼짐')
  })

  it.each([
    ['서버 aux 꺼짐', { auxState: { ...AUX_STATE, enabled: false } }],
    ['state.aux 없음', { auxState: undefined }],
    ['연결 없음', { connected: false, auxState: AUX_STATE }],
  ])('체크박스가 없다: %s', (_name, props) => {
    expect(render({ auxMode: 'fallback', ...props })).not.toContain('aux-toggle')
  })

  it('오류는 안내에 이유가 보이고(HTML 이스케이프됨) 2D 그림으로 대신 보여 준다', () => {
    const html = render({ auxMode: 'fallback', auxState: { ...AUX_STATE, error: 'RuntimeError: <EGL> 실패' } })
    expect(html).toContain('RuntimeError: &lt;EGL&gt; 실패')
    expect(html).toContain('class="side-mirror"')
  })

  it('느려서 fps를 낮췄으면 안내에 보인다', () => {
    const html = render({ auxMode: 'mirrors', auxFrame: FRAME, auxState: { ...AUX_STATE, slowed: true, fps: 2.5 } })
    expect(html).toContain('2.5fps')
  })
})
