// 로봇 상태: 집게 높이 막대, 집게 열림/닫힘, 목표 위치, 실제 집게 위치, 거리, 측정값.
// 위에서 본 화면만으로는 집게 높이를 알기 어려워서(렌더링을 더 하면 CPU를 더 쓴다) 숫자와 막대로 보여준다.

import { HEIGHT_BAR_MAX_M } from '../robot/config.js'
import { barFraction, controlStatusText, fmtNum, fmtStats, fmtVec } from '../robot/view.js'

export default function RobotPanel({ control, connected }) {
  const c = connected ? control : null
  const tipZ = c?.tip_m?.[2]
  const targetZ = c?.target_m?.[2]
  const tipFrac = barFraction(tipZ, HEIGHT_BAR_MAX_M)
  const targetFrac = barFraction(targetZ, HEIGHT_BAR_MAX_M)
  const closed = c?.gripper_closed

  const rows = [
    ['목표 위치 x, y, z (m)', fmtVec(c?.target_m)],
    ['실제 집게 위치 (m)', fmtVec(c?.tip_m)],
    ['목표와 집게 거리 (m)', fmtNum(c?.error_m, 3)],
    ['IK 실패 횟수', c ? String(c.ik_fail_count) : '—'],
    ['IK 1회 계산 ms (마지막/평균/최대)', fmtStats(c?.ik_ms)],
    ['손 메시지→ctrl ms (마지막/평균/최대)', fmtStats(c?.latency_ms)],
    ['목표 흔들림 1초 표준편차 (mm), 필터 후', fmtNum(c?.jitter_mm?.filtered, 2)],
    ['목표 흔들림 1초 표준편차 (mm), 필터 전', fmtNum(c?.jitter_mm?.raw, 2)],
  ]

  return (
    <div className="panel robot-panel">
      <p className="robot-status">{controlStatusText(c, connected)}</p>

      <div className="robot-indicators">
        <div className="height-bar" aria-label="집게 높이">
          <div className="height-bar-title">집게 높이</div>
          <div className="height-bar-track">
            {tipFrac != null && <div className="height-bar-fill" style={{ height: `${tipFrac * 100}%` }} />}
            {targetFrac != null && <div className="height-bar-target" style={{ bottom: `${targetFrac * 100}%` }} />}
          </div>
          <div className="height-bar-value">{tipZ == null ? '—' : `${(tipZ * 100).toFixed(1)} cm`}</div>
          <div className="height-bar-scale">0 ~ {(HEIGHT_BAR_MAX_M * 100).toFixed(0)} cm, 가로선 = 목표</div>
        </div>

        <div className="gripper-state">
          <div className="height-bar-title">집게</div>
          <div className={closed == null ? 'gripper-badge' : closed ? 'gripper-badge closed' : 'gripper-badge open'}>
            {closed == null ? '—' : closed ? '닫힘' : '열림'}
          </div>
          <div className="height-bar-scale">
            목표 {fmtNum(c?.gripper_goal_rad, 2)} rad
            <br />
            실제 {fmtNum(c?.gripper_rad, 2)} rad
          </div>
        </div>
      </div>

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
      {c?.ik_last_failure && <p className="status">IK 마지막 실패 이유: {c.ik_last_failure}</p>}
      <p className="note">
        손 메시지→ctrl은 서버가 메시지를 받은 때부터 ctrl에 반영할 때까지의 시간입니다. 브라우저에서 서버까지는 왕복 지연이
        따로 보여주고, ctrl이 반영된 뒤 실제로 움직이기까지(물리 틱 최대 33ms, 모터 응답)는 들어 있지 않습니다.
      </p>
    </div>
  )
}
