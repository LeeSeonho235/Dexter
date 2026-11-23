import { STEPS } from '../steps/steps.js'

// 상단 단계 표시줄: 1 녹화 → 2 학습 → 3 테스트·교정, 그리고 3에서 1로 돌아가는 순환 표시.
// busyText가 있으면(녹화·AUTO 진행 중) 다른 단계 버튼을 누를 수 없고 이유를 보여 준다. 서버 검사는 따로 없고 화면 안내다.
export default function StepBar({ step, onStep, busyText = '' }) {
  const busy = busyText !== ''
  return (
    <nav className="step-bar" aria-label="작업 단계">
      <ol className="step-list">
        {STEPS.map((s, i) => {
          const current = s.id === step
          return (
            <li key={s.id} className={current ? 'step-item current' : 'step-item'}>
              <button
                type="button"
                className="step-button"
                aria-current={current ? 'step' : undefined}
                disabled={busy && !current}
                onClick={() => onStep(s.id)}
              >
                <span className="step-no">{s.no}</span>
                <span className="step-label">{s.label}</span>
              </button>
              {i < STEPS.length - 1 && (
                <span className="step-arrow" aria-hidden="true">
                  →
                </span>
              )}
            </li>
          )
        })}
      </ol>
      <p className="step-loop">
        <span aria-hidden="true">↺</span> 교정 데이터가 쌓이면 다시 1 녹화·2 학습으로 돌아갑니다
        {step === 'test' && (
          <button type="button" className="step-loop-button" disabled={busy} onClick={() => onStep('record')}>
            1 녹화로 돌아가기
          </button>
        )}
      </p>
      {busy && <p className="step-busy">{busyText}</p>}
    </nav>
  )
}
