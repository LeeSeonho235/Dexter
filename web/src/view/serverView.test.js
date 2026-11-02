// 보기 방향 설정의 원본은 server/config.py다. 브라우저 설정(범위, 프리셋)이 같은지 확인한다.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DISTANCE_RANGE_M, ELEVATION_RANGE_DEG, PRESETS } from './config.js'

const configPy = readFileSync(new URL('../../../server/config.py', import.meta.url), 'utf8')
const NUM = '(-?\\d+(?:\\.\\d+)?)'

function pair(name) {
  const match = configPy.match(new RegExp(`^${name} = \\(${NUM}, ${NUM}\\)`, 'm'))
  expect(match, `${name} not found in server/config.py`).not.toBeNull()
  return [Number(match[1]), Number(match[2])]
}

function dict(literal) {
  const match = literal.match(
    new RegExp(`\\{"azimuth_deg": ${NUM}, "elevation_deg": ${NUM}, "distance_m": ${NUM}\\}`),
  )
  expect(match, `view dict not found: ${literal}`).not.toBeNull()
  return { azimuth_deg: Number(match[1]), elevation_deg: Number(match[2]), distance_m: Number(match[3]) }
}

describe('server/config.py와 같은 보기 방향 설정', () => {
  it('고도각, 거리 범위', () => {
    expect(ELEVATION_RANGE_DEG).toEqual(pair('VIEW_ELEVATION_RANGE_DEG'))
    expect(DISTANCE_RANGE_M).toEqual(pair('VIEW_DISTANCE_RANGE_M'))
  })

  it('기본 보기 방향', () => {
    const match = configPy.match(/^VIEW_DEFAULT = (\{.*\})$/m)
    expect(match).not.toBeNull()
    expect(PRESETS.default).toEqual(dict(match[1]))
  })

  it('위에서, 옆에서 프리셋', () => {
    for (const name of ['top', 'side']) {
      const match = configPy.match(new RegExp(`^    "${name}": (\\{.*\\}),$`, 'm'))
      expect(match, `${name} preset not found`).not.toBeNull()
      expect(PRESETS[name]).toEqual(dict(match[1]))
    }
  })
})
