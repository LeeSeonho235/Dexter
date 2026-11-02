# 02. 기존 프로젝트 비교

> 조사: 2026-10-22, GitHub 저장소 검색 기준. **논문 검색은 아직 안 함**(Handoff.md 확인 목록 참고).
> 이 표는 발표의 "기존 방식의 한계" 슬라이드가 된다. 발표 전에 각 저장소를 다시 열어 내용이 바뀌었는지 확인한다.

## 손동작으로 로봇 조종 (조종에서 끝남)

"mediapipe robot arm"으로 검색하면 200개 이상 나오고, 별이 많은 상위 30개는 대부분 **손을 따라 움직이는 조종**에서 끝난다(학습 없음). 예:

- e-candeloro/Vision-Robotic-Arm-Gesture-Recognition — 웹캠 + ROS로 로봇 조종
- iotdesignshop/mediapipe-robot-arm-controller — MediaPipe Holistic으로 로봇팔 조종 데모
- 아두이노 + 서보모터 제스처 로봇팔 다수

## 웹캠 시연 → 학습 (가장 가까운 것)

| 프로젝트 | 하는 것 | 없는 것 |
|---|---|---|
| [guptabhishekumar/handrobot](https://github.com/guptabhishekumar/handrobot) | 웹캠 → MuJoCo → ACT, SO-101·Franka, 100번 중 95번 성공, 하드웨어 $0 | 웹(브라우저) 없음. 실패 시 교정 학습 없음 |
| [kheechan04/webcam-teach-robot](https://github.com/kheechan04/webcam-teach-robot) | 웹캠 → SO-101 → ACT, **웹캠 깊이 오차가 학습 성공률에 주는 영향** 연구(한국어) | 웹 없음. 교정 학습 없음 |
| [cadenpage/TeleOp-BC](https://github.com/cadenpage/TeleOp-BC) | MediaPipe 조종 → MuJoCo 상자 밀기 → 행동 복제 | 단순 1차원 과제. 웹·교정 없음 |
| [learnermaker/physical-ai-lab](https://github.com/learnermaker/physical-ai-lab) | 웹캠 → MuJoCo → 강화학습 등 3시간 실습 교재 | 교육용. 시연 수집·교정 목적 아님 |

참고할 결과 (webcam-teach-robot README): 웹캠 시연으로 학습하면 스크립트 시연보다 성공률이 약 17%p 낮고, 그중 깊이 오차의 몫이 약 9%p. → 웹캠 시연의 약점 근거로 인용 가능.

## 브라우저로 로봇 조종 (VR 필요)

WebXR로 브라우저에서 로봇팔을 조종하는 프로젝트는 있다(예: Dream-Machines-Robotics/vr-teleop-kit, oscardvs/vr-teleop-kit). 모두 **Meta Quest 같은 VR 헤드셋이 필요**하다.

## 공식 도구

- [LeRobot](https://github.com/huggingface/lerobot): 데이터셋 형식, ACT 학습. 휴대폰·키보드·게임패드 조종 지원
- LeLab (LeRobot README에 소개): 브라우저에서 **실물 SO 팔**로 조종·녹화·학습

## 검색에서 못 찾은 것 (= Dexter의 자리)

- 웹캠 손 추적으로 **로봇 실패 순간을 교정**하고 그 교정을 학습에 쓰는 공개 프로젝트: 0개
- 브라우저에서 웹캠 손 추적으로 시연 → 모방학습: 0개

주의: 교정 학습 방법 자체(HG-DAgger, Kelly et al. 2019)는 연구에 이미 있다. 기여는 "방법"이 아니라 "웹캠 + 브라우저로 누구나 하게 만든 것".
