// hand 메시지 숫자 표시

function fmt(v, digits = 3) {
  if (v == null) return '—'
  if (typeof v === 'boolean') return String(v)
  if (Array.isArray(v)) return `[${v.map((x) => x.toFixed(digits)).join(', ')}]`
  return v.toFixed(digits)
}

export default function FeaturePanel({ hand, status }) {
  const rows = [
    ['detected', hand ? String(hand.detected) : '—'],
    ['wrist (x, y, z)', fmt(hand?.wrist)],
    ['pinch', fmt(hand?.pinch)],
    ['pinch_ratio', fmt(hand?.pinch_ratio)],
    ['palm_open', fmt(hand?.palm_open)],
    ['hand_scale', fmt(hand?.hand_scale)],
  ]
  return (
    <div className="panel">
      <table>
        <tbody>
          {rows.map(([name, value]) => (
            <tr key={name}>
              <th>{name}</th>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="status">
        {status.phase === 'running' &&
          `FPS ${status.fps.toFixed(1)} · ${status.delegate} · 영상 ${status.size}`}
        {status.phase === 'loading' && '카메라와 모델 준비 중…'}
        {status.phase === 'error' && `오류: ${status.message}`}
      </p>
      <p className="note">
        좌표는 MediaPipe 원래 값(좌우 뒤집지 않음). 거리 단위는 화면 높이 = 1.
      </p>
    </div>
  )
}
