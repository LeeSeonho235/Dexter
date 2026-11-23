import { describe, expect, it } from 'vitest'
import { buildColabCell, CONDITIONS, nameError } from './colabCell.js'

describe('조건 3종 기본값', () => {
  it('처음 시범 / 교정 / 추가 시연 이름이 정해진 값이다', () => {
    expect(CONDITIONS.map((c) => [c.label, c.defaultName])).toEqual([
      ['처음 시범', 'dexter_pick_place_v1'],
      ['교정', 'dexter_r1_correction'],
      ['추가 시연', 'dexter_r1_demo'],
    ])
  })

  it('기본 이름은 모두 입력 검사를 통과한다', () => {
    for (const c of CONDITIONS) expect(nameError(c.defaultName)).toBeNull()
  })
})

describe('buildColabCell', () => {
  it('노트북 1번 셀 형식의 4줄을 만든다(tar 이름 = name)', () => {
    expect(buildColabCell('dexter_r1_correction')).toBe(
      [
        "DRIVE_DIR = '/content/drive/MyDrive/dexter'",
        "DATASET_TAR = f'{DRIVE_DIR}/dexter_r1_correction.tar'",
        "REPO_ID = 'local/dexter_r1_correction'",
        "DATASET_ROOT = '/content/data/dexter_r1_correction'",
      ].join('\n'),
    )
  })

  it('tar 이름을 따로 주면 DATASET_TAR에만 반영된다', () => {
    const code = buildColabCell('abc', 'my-tar')
    expect(code).toContain("f'{DRIVE_DIR}/my-tar.tar'")
    expect(code).toContain("'local/abc'")
    expect(code).toContain("'/content/data/abc'")
  })

  it.each([
    ['빈 값', ''],
    ["따옴표", "a'b"],
    ['줄바꿈', 'a\nb'],
    ['공백', 'a b'],
    ['점', 'a.tar'],
    ['경로 구분', '../x'],
    ['64자 초과', 'a'.repeat(65)],
  ])('name이 안전하지 않으면 null: %s', (_label, bad) => {
    expect(buildColabCell(bad)).toBeNull()
    expect(buildColabCell('ok', bad)).toBeNull()
  })

  it('64자는 허용한다', () => {
    expect(buildColabCell('a'.repeat(64))).not.toBeNull()
  })
})

describe('nameError', () => {
  it('문자열이 아니면 이유를 돌려준다', () => {
    expect(nameError(undefined)).toBeTruthy()
    expect(nameError(null)).toBeTruthy()
  })
})
