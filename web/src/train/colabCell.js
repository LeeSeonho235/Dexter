// Colab 노트북(train/colab/dexter_train.ipynb) 1번 셀의 데이터셋 설정 줄을 문자열로 만든다(서버와 상관없는 순수 계산).
// 사용자가 입력한 이름이 코드에 그대로 들어가므로, 따옴표·줄바꿈·공백 같은 것은 입력 검사로 막는다.

export const CONDITIONS = [
  { id: 'first', label: '처음 시범', defaultName: 'dexter_pick_place_v1' },
  { id: 'correction', label: '교정', defaultName: 'dexter_r1_correction' },
  { id: 'extra', label: '추가 시연', defaultName: 'dexter_r1_demo' },
]

const NAME_PATTERN = /^[A-Za-z0-9_-]{1,64}$/

// 이름이 안전한가: 영문·숫자·밑줄(_)·하이픈(-)만, 1~64자. 통과하면 null, 아니면 이유 문구.
export function nameError(value) {
  if (typeof value !== 'string' || value === '') return '이름을 입력하세요'
  if (value.length > 64) return '64자 이하로 입력하세요'
  if (!NAME_PATTERN.test(value)) return '영문, 숫자, _ , - 만 쓸 수 있습니다(공백·점·따옴표 불가)'
  return null
}

// 1번 셀에서 바꿀 4줄. DRIVE_DIR은 노트북 값 그대로이고, 나머지 설정(STEPS 등)은 노트북에 있는 값을 그대로 둔다.
// name 또는 tarName이 검사를 통과하지 못하면 null(복사할 코드를 만들지 않는다).
export function buildColabCell(name, tarName = name) {
  if (nameError(name) || nameError(tarName)) return null
  return [
    "DRIVE_DIR = '/content/drive/MyDrive/dexter'",
    `DATASET_TAR = f'{DRIVE_DIR}/${tarName}.tar'`,
    `REPO_ID = 'local/${name}'`,
    `DATASET_ROOT = '/content/data/${name}'`,
  ].join('\n')
}
