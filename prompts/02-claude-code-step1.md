# Claude Code 요청문 — 1단계

VS Code 안 Claude Code **새 대화**에 붙여 넣는다. 보내기 전에 `git branch`로 내 브랜치(`feature/step1-scene`)인지 확인.

## 1-1. MuJoCo 장면

```
CLAUDE.md와 docs/Handoff.md, docs/04-architecture.md를 먼저 읽어줘. 계획만 먼저 보여줘(코드 쓰지 마).

목표: third_party/mujoco_menagerie/robotstudio_so101 모델을 써서 sim/scene.xml을 만든다. 장면에는 SO-101, 바닥, 집을 수 있는 블록(약 3cm, 자유롭게 움직임), 목표 위치 표시(충돌 없는 얇은 사각형), 위에서 내려다보는 카메라(top)가 있어야 한다.
그리고 sim/render_check.py를 만들어 MUJOCO_GL 환경변수로 화면 없이 렌더링해 /tmp/scene_top.png로 저장하고, 관절 각도를 몇 가지로 바꿔 /tmp/pose_1.png ~ /tmp/pose_3.png를 저장한다.

바꿔도 되는 곳: sim/, server/tests/
금지: third_party 안 파일 수정, mujoco.viewer 사용, git commit, docs/Handoff.md 수정.
확인 방법: 실행 명령과 정상일 때 출력, 이미지에서 무엇이 보여야 하는지 적어줘. 블록이 바닥을 뚫고 떨어지지 않는지 시뮬레이션 1초 후 블록 높이를 출력해서 확인.
```

## 1-2. 브라우저 손 추적

```
CLAUDE.md와 docs/04-architecture.md의 메시지 형식을 읽어줘. 계획만 먼저 보여줘(코드 쓰지 마).

목표: web/ (React + Vite)에서 웹캠을 켜고 @mediapipe/tasks-vision의 HandLandmarker(예전 @mediapipe/hands 아님)로 손 하나를 추적한다. 영상 위에 21개 점을 그리고, 화면에 wrist 좌표, pinch(엄지 끝-검지 끝 거리), palm_open(손가락 4개가 펴졌는지), hand_scale(손목-중지 뿌리 거리)을 숫자로 표시한다. 계산은 web/src/hand/features.js 같은 별도 파일로 분리해서 나중에 서버로 보낼 수 있게 한다.

바꿔도 되는 곳: web/
금지: 웹캠 영상이나 이미지를 파일로 저장, 외부 서버로 전송, git commit.
설치된 @mediapipe/tasks-vision 버전의 사용법을 node_modules의 타입 정의나 문서로 먼저 확인해줘.
확인 방법: npm run dev 실행 명령과 포트, 브라우저에서 무엇이 보여야 하는지 적어줘.
```
