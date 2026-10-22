# web — 브라우저 손 추적

웹캠으로 손 하나를 추적해(MediaPipe Tasks Hand Landmarker, `@mediapipe/tasks-vision` 1.1.1) 21개 점을 그리고,
서버로 보낼 `hand` 메시지 값(wrist, pinch, pinch_ratio, palm_open, hand_scale)을 화면에 표시한다.
영상은 화면에만 보여주고 저장하거나 전송하지 않는다.

## 처음 한 번

```bash
cd ~/handover/web
npm install
mkdir -p public/models
curl -fL -o public/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

모델 주소는 `node_modules/@mediapipe/tasks-vision/README.md`에 적힌 것과 같다. `public/models/`와
`public/mediapipe/`(wasm 복사본)는 git에서 제외한다.

## 실행

```bash
npm run dev
```

- 주소와 포트는 `vite.config.js`에 있다: `127.0.0.1:3007`(내 계정에 배정된 포트), `strictPort: true`라서 포트가 사용 중이면 다른 포트로 넘어가지 않고 오류로 멈춘다.
- `npm run dev`/`npm run build` 전에 `scripts/copy-wasm.mjs`가 wasm을 `public/mediapipe/wasm`으로 복사한다.
- VS Code **Ports**에서 3007을 **Private**으로 전달하고 노트북 브라우저에서 `http://localhost:3007`을 연다.

## 확인

```bash
npm test        # features.js 계산 테스트 (vitest)
npm run build
```

## 외부 전송 차단

`@mediapipe/tasks-vision` 1.1.1은 HandLandmarker를 만들면 사용 통계를 60초마다
`https://odml.pa.googleapis.com/v1/log`로 보내는 코드가 들어 있다(`vision_bundle.mjs`). 끄는 옵션이 없어서
`index.html`에 `Content-Security-Policy: connect-src 'self'`를 넣어 브라우저가 같은 주소(localhost) 밖으로
연결하지 못하게 막았다. 나중에 서버 WebSocket(예: `ws://localhost:8780`)에 연결할 때는 그 주소를 `connect-src`에 추가해야 한다.
