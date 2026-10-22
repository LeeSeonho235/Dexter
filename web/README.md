# web — 브라우저 손 추적 + 시뮬레이션 연결

웹캠으로 손 하나를 추적해(MediaPipe Tasks Hand Landmarker, `@mediapipe/tasks-vision` 1.1.1) 21개 점을 그리고,
`hand` 메시지(wrist, pinch, pinch_ratio, palm_open, hand_scale)를 파이썬 서버로 최대 30Hz 보낸다.
서버가 보내는 시뮬레이션 화면(top 카메라 JPEG)을 웹캠 옆에 보여준다.
웹캠 영상은 화면에만 보여주고 저장하거나 전송하지 않는다. 서버로 가는 것은 숫자(JSON)뿐이다.

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

터미널 두 개를 쓴다.

```bash
# 1) 파이썬 서버 (127.0.0.1:18007, 포트는 server/config.py)
cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.app

# 2) 웹
cd ~/handover/web && npm run dev
```

- 주소와 포트는 `vite.config.js`에 있다: `127.0.0.1:3007`(내 계정에 배정된 포트), `strictPort: true`라서 포트가 사용 중이면 다른 포트로 넘어가지 않고 오류로 멈춘다.
- `npm run dev`/`npm run build` 전에 `scripts/copy-wasm.mjs`가 wasm을 `public/mediapipe/wasm`으로 복사한다.
- VS Code **Ports**에서 **3007만** **Private**으로 전달하고 노트북 브라우저에서 `http://localhost:3007`을 연다.
  18007(파이썬 서버)은 전달하지 않는다.

## 서버 연결 구조

```
브라우저 ws://localhost:3007/ws ──(Vite 프록시, vite.config.js)──▶ ws://127.0.0.1:18007/ws (FastAPI)
```

- 브라우저 → 서버: `hand`(JSON, 최대 30Hz), `ping`(1초마다, 왕복 지연 측정)
- 서버 → 브라우저: 시뮬레이션 화면 JPEG(바이너리, 약 15fps), `state`(0.5초마다), `pong`, `error`
- 서버는 연결된 브라우저가 없으면 렌더링을 멈춘다(시뮬레이션 계산은 계속).
- 연결이 끊기면 2초 뒤 자동으로 다시 연결한다. 설정값은 `src/net/config.js`.
- 서버 포트를 바꾸면 `server/config.py`의 `PORT`와 `vite.config.js`의 `SIM_SERVER_PORT`를 같이 바꾼다
  (`src/net/serverPort.test.js`가 둘이 같은지 확인한다).

## 확인

```bash
npm test        # vitest: features.js 계산, 30Hz 제한, 연결/재연결/ping, 포트 일치
npm run lint
npm run build
```

## 외부 전송 차단

`@mediapipe/tasks-vision` 1.1.1은 HandLandmarker를 만들면 사용 통계를 60초마다
`https://odml.pa.googleapis.com/v1/log`로 보내는 코드가 들어 있다(`vision_bundle.mjs`). 끄는 옵션이 없어서
`index.html`에 `Content-Security-Policy: connect-src 'self'`를 넣어 브라우저가 같은 주소(localhost) 밖으로
연결하지 못하게 막았다. 서버 WebSocket은 같은 주소(`localhost:3007/ws`)로 붙고 Vite가 파이썬 서버로 넘겨주므로
CSP는 바꾸지 않는다.
