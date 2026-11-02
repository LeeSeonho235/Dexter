// "조종 중인 손" 표시 + 조종 상태 문구 + 화면 가장자리 안내.
// MediaPipe 라벨은 뒤집지 않은 원본 영상 기준이라 실제 손과 반대로 나올 수 있으므로,
// 직접 확인할 수 있게 라벨과 점수를 그대로 보여준다. 글자는 HTML이다(캔버스는 CSS로 좌우가 뒤집혀 있어서 글자를 그리면 거울 글자가 된다).

import { CONTROL_HANDEDNESS } from '../hand/config.js'
import { edgeMessage } from '../hand/edge.js'
import { handStatusText } from '../robot/view.js'

export default function ControlHandPanel({ selection, cameraOn, control, connected, edge, mode }) {
  const chosen = selection?.chosen
  const hands = selection?.hands ?? []
  let headline = '카메라 꺼짐'
  if (cameraOn) {
    headline = chosen
      ? `조종 중인 손: ${chosen.label} (노란 선, 신뢰도 ${chosen.score.toFixed(2)})`
      : `조종 손 안 보임 (찾는 라벨: ${CONTROL_HANDEDNESS})`
  }
  return (
    <div className="panel control-hand">
      <p className={chosen ? 'control-hand-on' : 'control-hand-off'}>{headline}</p>
      <p className="control-state">{handStatusText(control, connected)}</p>
      {cameraOn && edge && <p className="edge-message">{edgeMessage(mode)}</p>}
      <p className="status">
        보이는 손: {hands.length === 0 ? '없음' : hands.map((h) => `${h.label ?? '?'} ${h.score.toFixed(2)}${h.controlling ? ' ← 조종' : ''}`).join(' · ')}
      </p>
      {mode === 'relative' && (
        <p className="note">
          상대 모드: 하늘색 십자 = 기준점(손을 처음 잡은 자리), 주황 점 = 지금 손목, 점선 = 움직인 양. 손이 화면 가장자리에 가까우면
          손을 빼서 중앙에서 다시 넣으면 그 자리가 새 기준점이 됩니다.
        </p>
      )}
      <p className="note">
        조종 손은 MediaPipe 라벨 '{CONTROL_HANDEDNESS}'인 손입니다(src/hand/config.js CONTROL_HANDEDNESS). 영상을 좌우 뒤집지
        않은 원본으로 넣어서 라벨이 실제 손과 반대일 수 있습니다. 손을 하나씩 들어 노란 선이 어느 손인지 확인하고, 반대면
        설정값을 바꾸세요.
      </p>
    </div>
  )
}
