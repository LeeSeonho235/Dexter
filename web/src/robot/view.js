// 서버 state.control 값을 화면에 보여주기 좋게 바꾼다. React에 의존하지 않는 순수 계산.
//
// control 형식(server/control.py state()): controlling, consecutive_frames, start_frames, target_m [x,y,z] (m),
// tip_m [x,y,z] (m), error_m (m), gripper_closed, gripper_goal_rad, gripper_rad, ik_fail_count, ik_ms, latency_ms,
// jitter_mm { filtered, raw } ...

// 값이 0~max(m) 중 어디쯤인지 0~1로. 없으면 null, 범위 밖은 자른다.
export function barFraction(value, max) {
  if (value == null || !(max > 0)) return null
  return Math.min(1, Math.max(0, value / max))
}

// [x, y, z] (m) → "(0.230, 0.000, 0.080)". 없으면 "—".
export function fmtVec(v, digits = 3) {
  if (!Array.isArray(v)) return '—'
  return `(${v.map((x) => x.toFixed(digits)).join(', ')})`
}

export function fmtNum(v, digits = 1) {
  return v == null ? '—' : v.toFixed(digits)
}

// {last, mean, max} 통계를 "마지막 / 평균 / 최대" 한 줄로
export function fmtStats(stats, digits = 2) {
  if (!stats || stats.n === 0) return '—'
  return `${fmtNum(stats.last, digits)} / ${fmtNum(stats.mean, digits)} / ${fmtNum(stats.max, digits)}`
}

// 지금 로봇이 어떤 상태인지 한 줄
export function controlStatusText(control, connected) {
  if (!connected) return '서버에 연결되지 않음'
  if (!control) return '서버 상태 기다리는 중'
  if (control.controlling) return '손을 따라가는 중'
  if (control.consecutive_frames > 0) {
    return `조종 손 확인 중 (${control.consecutive_frames}/${control.start_frames}프레임 연속)`
  }
  if (control.target_m) return '조종 손이 안 보임 → 마지막 목표에서 멈춤'
  return '조종 시작 전 (조종 손을 기다리는 중)'
}
