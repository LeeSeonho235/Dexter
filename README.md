# Dexter

**브라우저에서 웹캠으로 손을 인식해 가상 로봇팔을 가르치고, 로봇이 틀리면 손으로 바로 잡아주는 시스템**

*Teach a robot arm with your bare hand in the browser, and take over the moment it goes wrong.*

> 이름의 뜻: 라틴어 dexter는 "오른손"이며 dexterity(손재주)의 어원이다. 오른손으로 로봇을 가르친다.

---

## 왜 만들었나

천안 AX 피지컬 AI 수업에서 "로봇에게 가장 어려운 일 중 하나가 수건 접기"라는 말을 들었다.
천은 잡을 때마다 모양이 바뀌어서 규칙으로 프로그래밍할 수 없다. 결국 **사람이 직접 보여주고 로봇이 따라 배우는 방식(모방학습)** 이 필요한데, 그 시연에는 조종용 로봇팔이나 VR 같은 비싼 장비가 든다. 나도 로봇팔이 없어서 시작조차 못 했다.

**누구나 웹캠만으로 로봇에게 시연을 보여줄 수 있다면**, 수건 접기 같은 어려운 일도 더 많은 사람이 가르칠 수 있다.
5주 안에 이 방식이 작동하는지 검증하려고 **블록 옮기기**부터 시작한다. 최종 목표는 수건 접기 같은 어려운 작업이다.

## 무엇이 다른가

웹캠으로 로봇을 가르치는 공개 프로젝트는 이미 있다(→ [docs/02-related-work.md](docs/02-related-work.md)). 둘 다 노트북에 설치하는 프로그램이고, 흐름이 "시연 → 학습 → 끝"이다.

| | 기존 프로젝트 | Dexter |
|---|---|---|
| 사용 방법 | 노트북에 설치 | **브라우저 링크로 접속** |
| 로봇이 실패하면 | 처음부터 시연을 더 해서 재학습 | **틀리는 순간 손을 내밀어 바로잡고, 그 순간만 학습** |
| 필요한 장비 | 웹캠 | 웹캠 |

## 어떻게 동작하나

```
[브라우저] 웹캠 + MediaPipe Hand Landmarker → 손 위치, 집기, 손바닥 제스처
     │ WebSocket
     ▼
[서버] 손 좌표 → 로봇 목표 위치 → 역기구학(mink) → MuJoCo 속 SO-101
     │ 시뮬레이션 화면은 다시 브라우저로
     ▼
[기록] LeRobot 데이터셋 형식으로 저장 (교정 구간 표시 포함)
     ▼
[학습] LeRobot ACT → 로봇이 혼자 작업 → 성공률 측정
```

1. 손으로 블록 옮기기를 20번 보여준다.
2. 그 데이터로 학습한 로봇(1호)이 혼자 해본다.
3. 틀리려는 순간 **손바닥을 보여주면** 조종권이 나에게 넘어온다. 바로잡고 **손을 내리면** 로봇이 이어서 한다. 이 교정 구간만 따로 저장된다.
4. 원래 시연 + 교정 기록으로 다시 학습한 로봇(2호)이 약했던 부분을 배운다.

## 증명할 것

같은 시간을 들였을 때 **"처음부터 다시 시연" vs "틀린 부분만 교정"** 중 어느 쪽이 성공률을 더 올리는가. 자세한 설계는 [docs/06-experiment.md](docs/06-experiment.md).

## 문서

| 파일 | 내용 |
|---|---|
| [docs/Handoff.md](docs/Handoff.md) | **현재 상태와 규칙의 원본.** 다른 문서와 다르면 이게 맞다 |
| [CLAUDE.md](CLAUDE.md) | Claude Code가 지킬 규칙 |
| [docs/01-story.md](docs/01-story.md) | 문제 정의, 발표 스토리 |
| [docs/02-related-work.md](docs/02-related-work.md) | 기존 프로젝트 비교 |
| [docs/03-plan.md](docs/03-plan.md) | 진행 순서와 단계별 완료 기준 |
| [docs/04-architecture.md](docs/04-architecture.md) | 구조, 데이터 형식, 교정 모드 |
| [docs/05-server-setup.md](docs/05-server-setup.md) | dev-server-01 설정 |
| [docs/06-experiment.md](docs/06-experiment.md) | 실험 설계와 측정 |
| [prompts/](prompts/) | 웹 Claude, Claude Code에 붙여 넣을 요청문 |

## 기술 스택

React + Vite, MediaPipe Tasks(Hand Landmarker), FastAPI + WebSocket, MuJoCo, mink, LeRobot(ACT)

## 출처

- SO-101 모델: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `robotstudio_so101` (Apache-2.0)
- 손 추적: [MediaPipe](https://github.com/google-ai-edge/mediapipe) Hand Landmarker (Apache-2.0)
- 학습: [LeRobot](https://github.com/huggingface/lerobot) ACT (Apache-2.0)
- 역기구학: [mink](https://github.com/kevinzakka/mink)
