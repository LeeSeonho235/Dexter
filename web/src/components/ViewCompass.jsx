// 시뮬레이션 화면 모서리의 작은 나침반(SVG). 현재 보기에서 로봇 앞쪽(+x)과 손의 "위/오른쪽" 방향이 화면 어느 쪽인지.
// 각도 계산은 view/compass.js(순수 함수). 보기 방위각이 바뀌면 같이 돈다. 클릭을 받지 않는다(드래그를 방해하지 않게).

import { compassArrows, visibleLength } from '../view/compass.js'

// viewBox 150x128. 화살표 끝 + 라벨(글자 폭 약 55px)이 오른쪽/왼쪽 끝에서 잘리지 않도록 길이와 중심을 잡았다
// (오른쪽 끝 라벨 중심 75 + 32 + 10 = 117, 글자 절반 28 더해도 145 < 150).
const CENTER_X = 75
const CENTER_Y = 64
const MAX_LENGTH_PX = 32
const LABEL_OFFSET_PX = 10
const HEAD_PX = 8

// 화살표 하나: 위쪽을 향해 그린 뒤 angle_deg만큼 돌린다. 라벨은 돌리지 않고 화살표 끝 바깥에 둔다(글자가 눕지 않게).
function Arrow({ arrow, color, width, label, labelDy = 0 }) {
  const length = visibleLength(arrow.length) * MAX_LENGTH_PX
  const rad = (arrow.angle_deg * Math.PI) / 180
  const labelX = CENTER_X + (length + LABEL_OFFSET_PX) * Math.sin(rad)
  const labelY = CENTER_Y - (length + LABEL_OFFSET_PX) * Math.cos(rad) + 4 + labelDy
  return (
    <g>
      <g transform={`rotate(${arrow.angle_deg} ${CENTER_X} ${CENTER_Y})`}>
        <line x1={CENTER_X} y1={CENTER_Y} x2={CENTER_X} y2={CENTER_Y - length + 1} stroke={color} strokeWidth={width} strokeLinecap="round" />
        <polygon
          points={`${CENTER_X},${CENTER_Y - length} ${CENTER_X - HEAD_PX / 2},${CENTER_Y - length + HEAD_PX} ${CENTER_X + HEAD_PX / 2},${CENTER_Y - length + HEAD_PX}`}
          fill={color}
        />
      </g>
      <text x={labelX} y={labelY} textAnchor="middle" fontSize="11" fontWeight="700" fill={color} stroke="#000" strokeWidth="0.4" paintOrder="stroke">
        {label}
      </text>
    </g>
  )
}

export default function ViewCompass({ view, mode }) {
  if (!view) return null
  const arrows = compassArrows({ azimuthDeg: view.azimuth_deg, elevationDeg: view.elevation_deg, mode })
  const relative = mode !== 'absolute'
  return (
    <svg
      className="view-compass"
      viewBox="0 0 150 128"
      role="img"
      aria-label={`나침반: 로봇 앞쪽은 화면에서 ${Math.round(arrows.front.angle_deg)}도 방향`}
    >
      <title>
        {relative
          ? '앞 = 로봇 앞쪽(+x). 손을 위/오른쪽으로 움직이면 화면에서도 위/오른쪽으로 갑니다.'
          : '앞 = 로봇 앞쪽(+x). 절대 모드: 손 위 = 로봇 앞쪽, 손 오른쪽 = 로봇 오른쪽 옆(월드 고정).'}
      </title>
      <rect x="1" y="1" width="148" height="126" rx="10" fill="#000" opacity="0.45" />
      <circle cx={CENTER_X} cy={CENTER_Y} r="3" fill="#fff" />
      <Arrow arrow={arrows.handUp} color="#6ec6ff" width={2} label="손 위" />
      <Arrow arrow={arrows.handRight} color="#6ec6ff" width={2} label="손 오른쪽" />
      <Arrow arrow={arrows.front} color="#ffb300" width={4} label="앞" labelDy={relative ? 0 : -2} />
    </svg>
  )
}
