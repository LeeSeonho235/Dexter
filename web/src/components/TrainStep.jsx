import { useState } from 'react'
import { buildColabCell, CONDITIONS, nameError } from '../train/colabCell.js'
import { COLAB_NOTEBOOK_URL, DRIVE_FOLDER_URL } from '../train/links.js'

// 2단계 "학습": Colab 1번 셀에 붙여 넣을 설정 줄을 만든다. 서버와 주고받는 것은 없다(웹 안의 문자열 생성뿐).
export default function TrainStep() {
  const [conditionId, setConditionId] = useState(CONDITIONS[0].id)
  // 조건마다 이름을 따로 기억한다(다른 조건으로 갔다 와도 입력이 남는다). 손대지 않았으면 기본값을 쓴다.
  const [names, setNames] = useState({})
  const [tarNames, setTarNames] = useState({}) // 없으면 이름과 같게
  const [copyState, setCopyState] = useState('idle') // idle | done | failed

  const condition = CONDITIONS.find((c) => c.id === conditionId)
  const name = names[conditionId] ?? condition.defaultName
  const tarName = tarNames[conditionId] ?? name // 손대지 않으면 이름을 따라가고, 한 번 고치면 입력값 그대로(비워도 그대로: 검사에서 걸린다)
  const nameProblem = nameError(name)
  const tarProblem = nameError(tarName)
  const code = buildColabCell(name, tarName)

  async function copy() {
    if (!code) return
    try {
      await navigator.clipboard.writeText(code)
      setCopyState('done')
    } catch {
      setCopyState('failed') // 주소가 https나 localhost가 아니면 막힌다: 아래 칸에서 직접 선택해 복사한다
    }
  }

  return (
    <section className="card train-step">
      <h2 className="card-title">2 학습 (Colab)</h2>
      <p className="note">
        Colab 노트북(<code>train/colab/dexter_train.ipynb</code>)의 1번 셀에서 아래 4줄만 바꿔 붙여 넣으세요. STEPS 등 나머지 설정은 그대로 둡니다.
      </p>

      <fieldset className="condition-group">
        <legend>어떤 데이터로 학습시키실 건가요?</legend>
        {CONDITIONS.map((c) => (
          <label key={c.id} className="condition-option">
            <input
              type="radio"
              name="condition"
              value={c.id}
              checked={c.id === conditionId}
              onChange={() => {
                setConditionId(c.id)
                setCopyState('idle')
              }}
            />
            {c.label}
          </label>
        ))}
      </fieldset>

      <div className="field-row">
        <label className="field">
          <span>데이터셋 이름 (name)</span>
          <input
            type="text"
            value={name}
            spellCheck={false}
            onChange={(e) => {
              setNames({ ...names, [conditionId]: e.target.value })
              setCopyState('idle')
            }}
          />
          {nameProblem && <span className="field-error">{nameProblem}</span>}
        </label>
        <label className="field">
          <span>tar 파일 이름 (.tar 제외)</span>
          <input
            type="text"
            value={tarName}
            spellCheck={false}
            onChange={(e) => {
              setTarNames({ ...tarNames, [conditionId]: e.target.value })
              setCopyState('idle')
            }}
          />
          {tarProblem && <span className="field-error">{tarProblem}</span>}
        </label>
      </div>

      <div className="link-row">
        <a className="link-button" href={DRIVE_FOLDER_URL} target="_blank" rel="noopener noreferrer">
          ① Drive 폴더 열기 (tar 올리기)
        </a>
        <a className="link-button" href={COLAB_NOTEBOOK_URL} target="_blank" rel="noopener noreferrer">
          ② Colab 노트북 열기
        </a>
      </div>
      <p className="note">③ 아래 셀 코드를 복사해 노트북 1번 셀에 붙여넣고, 셀을 위에서부터 하나씩 실행(모두 실행 금지)</p>

      <pre className="code-box" aria-label="Colab 1번 셀 코드">
        {code ?? '# 이름을 먼저 바로잡으세요'}
      </pre>
      <div className="copy-row">
        <button type="button" className="primary" disabled={!code} onClick={copy}>
          코드 복사
        </button>
        <span className="status" role="status">
          {copyState === 'done' && '복사했습니다'}
          {copyState === 'failed' && '자동 복사가 막혔습니다. 위 코드를 직접 선택해 복사하세요'}
        </span>
      </div>
    </section>
  )
}
