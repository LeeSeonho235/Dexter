// MediaPipe HandLandmarker 만들기 (@mediapipe/tasks-vision 1.1.1 vision.d.ts 기준)

import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'
import { MODEL_PATH, NUM_HANDS, WASM_BASE_PATH } from './config.js'

function createWithDelegate(fileset, delegate) {
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate },
    runningMode: 'VIDEO',
    numHands: NUM_HANDS,
  })
}

// GPU로 먼저 시도하고, 실패하면 CPU로 만든다.
export async function createHandTracker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE_PATH)
  try {
    return { landmarker: await createWithDelegate(fileset, 'GPU'), delegate: 'GPU' }
  } catch (err) {
    console.warn('GPU delegate 실패, CPU로 다시 시도:', err)
    return { landmarker: await createWithDelegate(fileset, 'CPU'), delegate: 'CPU' }
  }
}

export { HandLandmarker }
