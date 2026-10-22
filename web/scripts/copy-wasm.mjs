// node_modules의 MediaPipe wasm 파일을 public/mediapipe/wasm으로 복사한다.
// 설치된 @mediapipe/tasks-vision 버전과 항상 같은 파일을 로컬에서 제공하기 위함 (외부 CDN 안 씀).

import { cpSync, rmSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const src = fileURLToPath(new URL('../node_modules/@mediapipe/tasks-vision/wasm', import.meta.url))
const dst = fileURLToPath(new URL('../public/mediapipe/wasm', import.meta.url))

rmSync(dst, { recursive: true, force: true })
cpSync(src, dst, { recursive: true })
console.log(`copied MediaPipe wasm → ${dst}`)
