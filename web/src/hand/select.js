// 인식된 손 여러 개 중 로봇을 조종할 손 하나를 고른다. React, MediaPipe에 의존하지 않는 순수 계산.
//
// result: MediaPipe HandLandmarkerResult. result.landmarks[i]는 i번째 손의 21개 점,
//         result.handedness[i][0]은 그 손의 라벨 { categoryName: 'Left' | 'Right', score }.
// label:  조종할 손의 라벨(hand/config.js CONTROL_HANDEDNESS).
//
// 같은 라벨의 손이 둘이면(MediaPipe가 가끔 그렇게 본다) 신뢰도(score)가 높은 쪽을 고른다.
// 조종 손이 안 보이면 chosen은 null이다. 다른 손이 화면에 있어도 조종은 그쪽으로 넘어가지 않는다.
//
// 돌려주는 값: { chosen, hands }
//   chosen: { index, landmarks, label, score } 또는 null
//   hands:  보이는 모든 손의 { index, label, score, controlling } (화면 표시용)

export function selectHand(result, label) {
  const landmarks = result?.landmarks ?? []
  const handedness = result?.handedness ?? []
  const all = landmarks.map((points, index) => {
    const category = handedness[index]?.[0]
    return { index, landmarks: points, label: category?.categoryName ?? null, score: category?.score ?? 0 }
  })

  let chosen = null
  for (const hand of all) {
    if (hand.label !== label) continue
    if (chosen === null || hand.score > chosen.score) chosen = hand
  }

  const hands = all.map(({ index, label: handLabel, score }) => ({
    index,
    label: handLabel,
    score,
    controlling: chosen !== null && index === chosen.index,
  }))
  return { chosen, hands }
}
