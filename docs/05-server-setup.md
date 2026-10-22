# 05. dev-server-01 설정

> 8팀 개발 가이드와 같은 서버, 같은 방식(VS Code SSH)으로 작업한다. 팀 프로젝트(`~/Team-project`)와 폴더를 섞지 않는다.

## 0. 먼저 확인할 것

```bash
nvidia-smi                 # GPU가 보이면 렌더링 egl, 학습도 서버에서 가능
python3 --version          # 3.10 이상
df -h ~                    # 남은 용량 (데이터셋, 모델 저장)
```

| 결과 | 렌더링 | 학습 |
|---|---|---|
| GPU 있음 | `MUJOCO_GL=egl` | 서버에서. 다른 팀원 작업과 시간 겹치지 않게 |
| GPU 없음 | `MUJOCO_GL=osmesa` (느림, 해상도 낮게) | Colab 등 외부 GPU. 데이터셋을 Hugging Face Hub에 올려서 옮김 |

서버는 여러 명이 같이 쓴다. 오래 걸리는 학습을 돌리기 전에 팀에 말한다.

## 1. 저장소

```bash
cd ~
mkdir handover && cd handover
git init
git config --global core.pager cat
```

GitHub에 새 저장소(handover)를 만들고 연결한다. 공개 여부는 발표 전에 정한다.

## 2. 파이썬 환경

```bash
cd ~/handover
mkdir -p server && cd server
python3 -m venv .venv
.venv/bin/pip install mujoco mink fastapi "uvicorn[standard]" numpy pillow pytest
```

LeRobot은 3단계(시연 녹화)에 들어갈 때 설치한다(의존성이 크다). 설치 명령은 LeRobot 문서의 현재 버전을 따른다.

## 3. SO-101 모델 가져오기

```bash
cd ~/handover
mkdir -p third_party && cd third_party
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie rev-parse HEAD > menagerie_commit.txt
```

`third_party/README.md`에 출처, 라이선스(Apache-2.0), 커밋 번호를 적는다. 저장소에는 `robotstudio_so101` 폴더만 남기거나 `.gitignore`로 제외하고 설치 스크립트로 받는다.

## 4. 화면 없이 렌더링 확인

```bash
cd ~/handover/server
MUJOCO_GL=egl .venv/bin/python - <<'PY'
import mujoco
from PIL import Image
m = mujoco.MjModel.from_xml_path("../third_party/mujoco_menagerie/robotstudio_so101/scene.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 480, 640)
r.update_scene(d)
Image.fromarray(r.render()).save("/tmp/so101.png")
print("ok, joints:", m.nq)
PY
```

`ok`가 나오면 VS Code 탐색기에서 `/tmp/so101.png`를 열어 로봇팔이 보이는지 눈으로 확인한다. egl 오류가 나면 `MUJOCO_GL=osmesa`로 다시 해본다(필요하면 서버 관리자에게 `libosmesa6` 설치 요청).

## 5. 웹(브라우저)

```bash
cd ~/handover
npm create vite@latest web -- --template react
cd web && npm install && npm install @mediapipe/tasks-vision
npm run dev -- --port 5180
```

- 포트가 사용 중이면 다른 번호를 쓴다.
- VS Code **Ports** 탭에서 해당 포트를 **Private**으로 전달하고, 내 노트북 브라우저에서 `localhost:포트`로 연다.
- 웹캠은 `localhost`나 https 주소에서만 켜진다. Ports 전달로 `localhost`로 열면 된다.
- 서버 WebSocket 포트(예: 8780)도 같은 방식으로 Private 전달한다.

## 6. 자주 막히는 곳

| 증상 | 해결 |
|---|---|
| `mujoco.viewer` 실행 시 오류 | 서버엔 화면이 없다. viewer 쓰지 말고 4번처럼 Renderer로 이미지 저장 |
| 웹캠 권한 창이 안 뜸 | 주소가 `localhost`인지 확인. 서버 IP로 직접 열면 브라우저가 카메라를 막는다 |
| 손 추적이 느림 | MediaPipe를 GPU delegate로, 영상 해상도 640x480으로 |
| WebSocket 연결 실패 | 서버 포트도 Ports에 전달했는지, 브라우저 코드의 주소가 `ws://localhost:포트`인지 |
