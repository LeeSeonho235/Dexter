// StepBar와 TrainStep을 서버 렌더(react-dom/server)로 그려 구조를 확인한다(브라우저 없이 문자열로만).
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { COLAB_NOTEBOOK_URL, DRIVE_FOLDER_URL } from '../train/links.js'
import StepBar from './StepBar.jsx'
import TrainStep from './TrainStep.jsx'

const noop = () => {}
const count = (html, text) => html.split(text).length - 1

describe('StepBar', () => {
  it('단계 3개와 순환 안내가 보이고, 지금 단계만 current', () => {
    const html = renderToStaticMarkup(<StepBar step="train" onStep={noop} />)
    expect(count(html, 'class="step-button"')).toBe(3)
    expect(count(html, 'step-item current')).toBe(1)
    expect(html).toContain('aria-current="step"')
    expect(html).toContain('테스트·교정')
    expect(html).toContain('다시 1 녹화')
    expect(html).not.toContain('disabled')
  })

  it('3단계에서만 "1 녹화로 돌아가기" 버튼이 있다', () => {
    expect(renderToStaticMarkup(<StepBar step="test" onStep={noop} />)).toContain('1 녹화로 돌아가기')
    expect(renderToStaticMarkup(<StepBar step="record" onStep={noop} />)).not.toContain('1 녹화로 돌아가기')
  })

  it('busyText가 있으면 다른 단계 버튼은 비활성, 지금 단계는 활성이고 이유가 보인다', () => {
    const html = renderToStaticMarkup(<StepBar step="record" onStep={noop} busyText="녹화 중에는 단계를 옮길 수 없습니다" />)
    expect(count(html, 'disabled=""')).toBe(2)
    expect(html).toContain('녹화 중에는 단계를 옮길 수 없습니다')
  })

  it('3단계 + busy면 돌아가기 버튼도 비활성', () => {
    const html = renderToStaticMarkup(<StepBar step="test" onStep={noop} busyText="AUTO 진행 중" />)
    expect(count(html, 'disabled=""')).toBe(3) // 다른 단계 2개 + 돌아가기
  })
})

describe('TrainStep', () => {
  it('처음에는 "처음 시범" 기본값으로 4줄 코드가 보인다', () => {
    const html = renderToStaticMarkup(<TrainStep />)
    expect(html).toContain('DATASET_TAR')
    expect(html).toContain('REPO_ID = &#x27;local/dexter_pick_place_v1&#x27;')
    expect(html).toContain('DATASET_ROOT')
    expect(html).toContain('코드 복사')
    expect(count(html, 'type="radio"')).toBe(3)
  })

  it('STEPS 안내: 기본 20000 → 약 47분, 경고문이 있고 셀 코드에는 STEPS가 없다', () => {
    const html = renderToStaticMarkup(<TrainStep />)
    expect(html).toContain('value="20000"')
    expect(html).toContain('약 47분 (설치·확인 포함 약 +10분)')
    expect(html).toContain('탭을 닫지 마세요')
    const code = html.slice(html.indexOf('class="code-box"'), html.indexOf('</pre>'))
    expect(code).not.toContain('STEPS')
  })

  it('링크 2개가 새 탭으로 열리고 코드 상자보다 위에 있다', () => {
    const html = renderToStaticMarkup(<TrainStep />)
    for (const url of [DRIVE_FOLDER_URL, COLAB_NOTEBOOK_URL]) {
      expect(html).toContain(`href="${url}" target="_blank" rel="noopener noreferrer"`)
      expect(html.indexOf(url)).toBeLessThan(html.indexOf('class="code-box"'))
    }
    expect(html).toContain('① Drive 폴더 열기 (tar 올리기)')
    expect(html).toContain('② Colab 노트북 열기')
    expect(html).toContain('모두 실행 금지')
  })
})
