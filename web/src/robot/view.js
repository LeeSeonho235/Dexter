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

// 키보드 높이 안내 한 줄(서버가 확인한 상태 기준). control은 서버 state.control.
//   조종 시작 전: 키보드가 아직 안 먹는다는 안내 / 상자 맨 위·맨 아래 / 움직이는 중 / 그 밖에는 빈 문자열
export function heightHint(control) {
  if (!control) return ''
  if (control.target_z_m == null) return '손으로 조종을 시작한 뒤부터 동작합니다'
  if (control.height_active) {
    return `${control.height_dir < 0 ? '내려가는 중' : '올라가는 중'}${control.height_slow ? ' (천천히)' : ''}`
  }
  if (control.height_limit === 'top') return '상자 맨 위'
  if (control.height_limit === 'bottom') return '상자 맨 아래'
  return ''
}

// 웹캠 화면 아래에 보여 주는 조종 상태 문구(사용자가 무엇을 하면 되는지 중심). 로봇 상태 칸은 아래 controlStatusText를 쓴다.
//   "손 기준점 잡는 중 (n/3)" / "손을 따라가는 중" / "손 안 보임 → 정지 (중앙에서 다시 넣으세요)"
export function handStatusText(control, connected) {
  if (!connected) return '서버에 연결되지 않음'
  if (!control) return '서버 상태 기다리는 중'
  if (control.controlling) return '손을 따라가는 중'
  if (control.consecutive_frames > 0) {
    return `손 기준점 잡는 중 (${control.consecutive_frames}/${control.start_frames})`
  }
  if (control.target_m) return '손 안 보임 → 정지 (중앙에서 다시 넣으세요)'
  return '조종 시작 전 (조종 손을 화면 중앙에 넣으세요)'
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
