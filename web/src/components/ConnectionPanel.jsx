// 서버 연결 상태와 숫자 표시

const PHASE_LABEL = {
  idle: '시작 전',
  connecting: '연결 중…',
  open: '연결됨',
  closed: '끊김 → 곧 다시 연결',
}

function num(v, digits = 1) {
  return v == null ? '—' : v.toFixed(digits)
}

export default function ConnectionPanel({ stats }) {
  const server = stats.serverState
  const lastHand = server?.last_hand
  const rows = [
    ['연결', PHASE_LABEL[stats.phase] ?? stats.phase],
    ['보낸 hand / 초', num(stats.handPerSec, 0)],
    ['왕복 지연 (ms)', num(stats.rttMs)],
    ['시뮬 화면 fps', num(stats.frameFps, 0)],
    ['서버가 받은 hand', server ? String(server.hand_count) : '—'],
    ['서버가 거부한 메시지', server ? String(server.invalid_count) : '—'],
    ['서버 마지막 pinch_ratio', lastHand ? num(lastHand.pinch_ratio, 3) : '—'],
    ['서버 마지막 detected', lastHand ? String(lastHand.detected) : '—'],
    ['시뮬 시간 (s)', server ? num(server.sim_time_s) : '—'],
  ]
  return (
    <div className="panel">
      <table>
        <tbody>
          {rows.map(([name, value]) => (
            <tr key={name}>
              <th>{name}</th>
              <td className={name === '연결' ? `phase-${stats.phase}` : undefined}>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {stats.lastServerError && <p className="status">서버가 마지막으로 거부한 이유: {stats.lastServerError}</p>}
    </div>
  )
}
