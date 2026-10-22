# Handoff — 현재 상태와 규칙의 원본

> 이 파일은 **나만 수정한다.** 다른 문서와 내용이 다르면 이 파일이 맞다.
> 마지막 수정: 2026-10-22 (2-1 완료)

## 한 줄

브라우저에서 웹캠으로 손을 인식해 가상 SO-101 로봇팔을 가르치고, 로봇이 틀리면 손으로 바로 잡아주는(교정) 시스템.

## 확정된 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 로봇 | SO-101 (MuJoCo Menagerie `robotstudio_so101`) | 실물이 저가라 나중에 실물로 이어갈 수 있음, LeRobot 지원 |
| 시뮬레이터 | MuJoCo | 로봇 학습 표준, LeRobot 생태계와 맞음 |
| 손 인식 | MediaPipe **Tasks** Hand Landmarker (예전 Solutions API 아님) | 브라우저에서 GPU 없이 실시간 |
| 역기구학 | mink | MuJoCo 기반, 직접 수식 안 짬 |
| 학습 | LeRobot ACT | 기본 모방학습, 설정 한 줄로 학습 |
| 과제 | 블록 하나 집어서 표시된 목표 위에 놓기 | 5주 안에 검증 가능한 최소 과제 |
| 센서 | **사용 안 함** | 일정 위험. "향후 계획"으로만 언급 |
| 기존 코드 | handrobot, webcam-teach-robot 코드는 **복사하지 않음** | 직접 구현. 막혔을 때 참고만 하고 출처 표기 |

## 지금 상태

- [x] 기획, 스토리, 기존 프로젝트 조사
- [x] 1단계: MuJoCo에 SO-101 띄우기(화면 없이 렌더링), 브라우저 손 추적
  - [x] 1-1 장면: sim/scene.xml (블록 3cm, 목표, top 카메라, home 자세)
  - [x] 1-2 브라우저 손 추적: web/ (MediaPipe 1.1.1, GPU 약 60FPS)
- [ ] 2단계: 손 → 로봇 연결
  - [x] 2-1 연결: 브라우저 ↔ 서버 WebSocket (hand 30Hz, 시뮬 화면 15fps, 왕복 16~30ms)
  - [ ] 2-2 매핑: 손 → 로봇 움직임 + 집게
- [ ] 3단계: 시연 녹화, 1차 학습
- [ ] 4단계: 교정 모드, 2차 학습
- [ ] 5단계: 실험, 발표

## 확인이 필요한 것 (시작 전)

- [x] GPU 없음. 렌더링은 MUJOCO_GL=egl(Mesa, CPU)로 됨. osmesa는 없음. Renderer는 끝날 때 close()를 직접 불러야 에러가 안 남. 학습은 외부 GPU(Colab 등)
- [ ] "webcam correction imitation learning", "HG-DAgger hand tracking" 관련 논문이 있는지 검색 → 있으면 02-related-work.md에 추가
- [ ] LeRobot 현재 버전에서 ACT 학습 명령과 데이터셋 형식 확인 (버전마다 바뀜)

## 규칙

1. 작업 전 `git checkout main && git pull`, 작업은 `feature/작업이름` 브랜치에서.
2. `git add .` 금지. 경로를 직접 적는다.
3. 웹캠 영상은 저장하지 않는다. 손 좌표 숫자와 시뮬레이션 화면만 저장.
4. Claude Code 보고는 그대로 믿지 않고, 숫자(테스트 결과, 바뀐 파일)를 내 터미널에서 직접 확인한다.
5. 화면이 바뀌는 작업은 결과를 내 눈으로 본다.
6. 포트 전달은 Private만.

## 실행 방법

```bash
cd ~/handover
MUJOCO_GL=egl server/.venv/bin/python sim/render_check.py   # 이미지 → /tmp
server/.venv/bin/python -m pytest server/tests -q
```

서버 띄우기 (웹보다 먼저):

```bash
cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.app
```

메모 (2-1 렌더링 비용, 640x480):
- 그림자+반사 켬 151ms(약 6fps, CPU 3코어) / 반사만 끔 120ms / 둘 다 끔 54ms(15fps)
- 실시간 화면은 그림자·반사 끔(server/config.py). 학습 이미지는 저장한 관절 값으로 3단계에서 다시 그릴 수 있으니 그때 결정
- 브라우저 연결 있을 때 CPU 약 1.6코어, 없으면 0.2%. 공용 서버라 안 볼 땐 탭 닫기

메모: shoulder_pan 값이 +이면 로봇이 오른쪽(-y, top 화면 오른쪽)으로 돈다.

```bash
cd ~/handover/web
npm test
npm run dev -- --host 127.0.0.1 --port 3007 --strictPort   # Ports에서 3007 Private 전달 → localhost:3007
```

메모 (1-2 측정):
- pinch_ratio = pinch / hand_scale. 폄 0.8~1.0, 붙임 0.1~0.2, 주먹 0.2 (거리 상관없이 비슷). 그리퍼 기준 후보: 0.3 아래 닫힘, 0.5 위 열림
- wrist의 z는 항상 0 (MediaPipe 기준점). 앞뒤 거리는 hand_scale로 추정 (가까이 0.19, 멀리 0.07)
- MediaPipe가 통계를 외부로 보내는 코드가 있어 index.html의 CSP(connect-src 'self')로 막음. 서버는 Vite 프록시(/ws)로 붙어서 CSP 수정 필요 없음
- 포트: web 3007 (Private 전달), 서버 18007 (127.0.0.1 전용, 전달 안 함)
