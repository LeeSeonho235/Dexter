// 시뮬레이션 화면 모서리 나침반: 현재 보기에서 로봇 앞쪽(+x)과 손의 "위/오른쪽" 방향이 화면에서 어느 쪽인지.
// React에 의존하지 않는 순수 계산. 규칙은 서버 server/mapping.py의 view_basis와 같다(MuJoCo 자유 카메라):
//   바닥 위 화면 방향: forward = (cos az, sin az) (화면 안쪽/위), right = (sin az, -cos az) (화면 오른쪽).
//   비스듬히 볼수록 바닥 위 앞뒤 방향은 화면에서 짧게 눌려 보인다: 화면 위 성분 = (forward 성분) × sin|고도각|.
//
// 각도(angle_deg)는 화면 위쪽에서 시계 방향으로 잰 값(0 = 위, 90 = 오른쪽, 180 = 아래, 270 = 왼쪽), 0 이상 360 미만.
// length는 화살표 길이 비율(0~1): 화면에서 실제로 보이는 길이.

const toRad = (deg) => (deg * Math.PI) / 180

// 화살표가 너무 짧아지면(거의 화면 안쪽을 향할 때) 보이지 않으므로 그릴 때 이 길이 이상으로 그린다.
export const MIN_VISIBLE_LENGTH = 0.35

export function visibleLength(length) {
  return Math.max(MIN_VISIBLE_LENGTH, Math.min(1, length))
}

// 월드 바닥 방향 [wx, wy]가 현재 보기에서 화면 어느 쪽인지: { angle_deg, length }
export function worldDirectionOnScreen([wx, wy], azimuthDeg, elevationDeg) {
  const az = toRad(azimuthDeg)
  const rightPart = wx * Math.sin(az) - wy * Math.cos(az)
  const upPart = (wx * Math.cos(az) + wy * Math.sin(az)) * Math.sin(toRad(Math.abs(elevationDeg)))
  const angle = (((Math.atan2(rightPart, upPart) * 180) / Math.PI) % 360 + 360) % 360
  return { angle_deg: angle, length: Math.hypot(rightPart, upPart) }
}

// 모드별 나침반 화살표 세 개.
//   front: 로봇 앞쪽(+x)  — 보기를 돌리면 같이 돈다.
//   handUp, handRight: 손을 위/오른쪽으로 움직이면 로봇이 가는 방향.
//     relative: 늘 화면 위/오른쪽(고정). 보기를 돌려도 손 방향과 화면 방향이 같다.
//     absolute: 월드 고정(손 위 = +x, 손 오른쪽 = -y)이라 보기를 돌리면 같이 돈다.
export function compassArrows({ azimuthDeg, elevationDeg, mode }) {
  const front = worldDirectionOnScreen([1, 0], azimuthDeg, elevationDeg)
  if (mode === 'absolute') {
    return {
      front,
      handUp: worldDirectionOnScreen([1, 0], azimuthDeg, elevationDeg),
      handRight: worldDirectionOnScreen([0, -1], azimuthDeg, elevationDeg),
    }
  }
  return {
    front,
    handUp: { angle_deg: 0, length: 1 },
    handRight: { angle_deg: 90, length: 1 },
  }
}
