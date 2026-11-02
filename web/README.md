# web — 브라우저 손 추적 + 시뮬레이션 연결 + 손으로 로봇 조종

웹캠으로 손을 최대 2개 추적해(MediaPipe Tasks Hand Landmarker, `@mediapipe/tasks-vision` 1.1.1) 21개 점을 그리고,
그중 **조종 손 하나**의 `hand` 메시지(wrist, pinch, pinch_ratio, palm_open, hand_scale)만 파이썬 서버로 최대 30Hz 보낸다.
서버가 보내는 시뮬레이션 화면(top 카메라 JPEG)을 웹캠 옆에 보여준다.
웹캠 영상은 화면에만 보여주고 저장하거나 전송하지 않는다. 서버로 가는 것은 숫자(JSON)뿐이다.

## 조종 손 고르기

- 손은 2개까지 인식하지만 로봇 조종에는 MediaPipe handedness 라벨이 `CONTROL_HANDEDNESS`(`src/hand/config.js`, 기본 `'Right'`)인
  손 하나만 쓴다. 다른 손은 화면에 회색으로만 그리고 서버로 보내지 않는다. 조종 손이 안 보이면 다른 손이 있어도
  `detected: false`를 보낸다(조종이 다른 손으로 넘어가지 않는다).
- 같은 라벨의 손이 둘로 잡히면 신뢰도(score)가 높은 쪽을 쓴다(`src/hand/select.js`).
- **영상을 좌우 뒤집지 않은 원본으로 넣기 때문에 라벨이 실제 손과 반대로 나올 수 있다**(내 오른손이 `Left`). 화면의
  "조종 중인 손" 표시와 노란 굵은 선으로 어느 손인지 직접 확인하고, 반대면 `CONTROL_HANDEDNESS`를 바꾼다.

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
- 서버 → 브라우저: 시뮬레이션 화면 JPEG(바이너리, 약 15fps), `state`(0.1초마다, `control` 항목에 로봇 상태), `pong`, `error`
- 서버는 연결된 브라우저가 없으면 렌더링을 멈춘다(시뮬레이션 계산은 계속).
- 연결이 끊기면 2초 뒤 자동으로 다시 연결한다. 설정값은 `src/net/config.js`.
- 서버 포트를 바꾸면 `server/config.py`의 `PORT`와 `vite.config.js`의 `SIM_SERVER_PORT`를 같이 바꾼다
  (`src/net/serverPort.test.js`가 둘이 같은지 확인한다).

## 화면에 보이는 것 (로봇 상태, 서버 `state.control`)

- **조종 중인 손:** 라벨(MediaPipe 기준)과 신뢰도, 보이는 손 목록. 조종 손은 노란 굵은 선.
- **집게 높이 막대:** 실제 집게 끝 높이(0~15cm, 파란 막대)와 목표 높이(초록 가로선). 위에서 본 화면만으로는 높이를 알기
  어려워서 렌더링을 더 하지 않고 숫자와 막대로 보여준다.
- **집게 열림/닫힘:** 상태, 목표 각도와 실제 각도(rad).
- **목표 위치 x, y, z (m), 실제 집게 위치 (m), 둘 사이 거리 (m).**
- **IK 실패 횟수**(실패하면 ctrl을 바꾸지 않고 직전 값 유지), **IK 1회 계산 시간 ms**(마지막/평균/최대, 최근 300개).
- **손 메시지→ctrl ms:** 서버가 hand 메시지를 받은 때부터 ctrl에 반영할 때까지(서버 안 시간). 브라우저→서버는 왕복 지연,
  ctrl 반영 뒤 실제로 움직이기까지(물리 틱 최대 33ms, 모터 응답)는 포함되지 않는다.
- **목표 흔들림:** 최근 1초 동안 목표 위치의 표준편차(mm). 손을 가만히 두고 집게만 열고 닫을 때 값이 크면 손목 기준점이
  집게 동작에 흔들린다는 뜻이다(손바닥 중심으로 바꾸는 것을 다음 작업으로 검토). "필터 후"는 로봇이 받는 목표, "필터 전"은
  필터를 거치기 전 값이다.

## 브라우저에서 확인하기

1. 서버와 웹을 띄우고(위 실행) `http://localhost:3007`을 연다. 시뮬레이션 화면이 나오고 "서버에 연결되지 않음"이 아닌지 본다.
2. **카메라 켜기** → 양손을 보여준다. 점이 둘 다 그려지고 한 손만 노란 선이다. "조종 중인 손" 라벨이 실제와 반대면
   `CONTROL_HANDEDNESS`를 바꾼다. 조종하지 않는 손을 흔들어도 로봇이 움직이지 않아야 한다.
3. 조종 손을 처음 보이면 "조종 손 확인 중 (n/3프레임 연속)" 뒤 "손을 따라가는 중"이 된다. 로봇이 홈 자세에서 튀지 않고
   부드럽게 손 위치로 온다.
4. 손을 오른쪽/위로 → 시뮬레이션 화면 속 집게도 오른쪽/위로. 손을 카메라에 가까이 → 집게 높이 막대가 줄어든다.
5. 손을 화면 밖으로 → "마지막 목표에서 멈춤", 로봇이 그 자리에 선다.
6. 엄지와 검지를 붙이면 집게가 "닫힘", 펴면 "열림". 반쯤 붙인 채로는 직전 상태가 유지된다.
7. 목표와 집게 거리는 가만히 있을 때 몇 mm, 빠르게 움직일 때는 더 커진다. IK 실패 횟수는 보통 0이다.

## 확인

```bash
npm test        # vitest: features.js 계산, 조종 손 고르기(select.js), 로봇 상태 표시 계산, 30Hz 제한, 연결/재연결/ping, 포트 일치
npm run lint
npm run build
```

## 외부 전송 차단

`@mediapipe/tasks-vision` 1.1.1은 HandLandmarker를 만들면 사용 통계를 60초마다
`https://odml.pa.googleapis.com/v1/log`로 보내는 코드가 들어 있다(`vision_bundle.mjs`). 끄는 옵션이 없어서
`index.html`에 `Content-Security-Policy: connect-src 'self'`를 넣어 브라우저가 같은 주소(localhost) 밖으로
연결하지 못하게 막았다. 서버 WebSocket은 같은 주소(`localhost:3007/ws`)로 붙고 Vite가 파이썬 서버로 넘겨주므로
CSP는 바꾸지 않는다.
