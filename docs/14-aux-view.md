# 14. 보조 카메라 (2-6, 보기 전용)

메인 화면(위에서 보는 큰 화면) 구석에 작은 화면 두 개를 항상 보여 준다: **옆에서**(왼쪽 아래)와 **비스듬히**(오른쪽 아래). 집게 높이와 블록 위치를 눈으로 보기 쉽게 하려는 것이다(사이드 미러 같은 보기 전용).
**학습·녹화·AUTO·평가에는 아무 영향이 없어야 한다.** 상태: 구현과 라이브 확인 완료(2026-11-13). **기본은 꺼짐**(`AUX_ENABLED=1`로 켠다).
측정값은 이 서버(공용, CPU 렌더링)에서 잰 것이고, "추정"이라고 적은 것은 확인하지 않은 설명이다.

## 방향이 바뀐 경위

1. **서버에서 따로 그리기(후보 A)**: 별도 스레드가 자기만의 `MjModel` / `Renderer`로 두 번째 화면을 그려 보낸다. bench 결과 한 장에 렌더 50~65 ms가 고정으로 들고 해상도를 줄여도 안 빨라져서 aux가 1 fps 수준밖에 안 나왔다(기준 미달).
2. **원인 확인(프로브, `server/aux_view_probe.py`)**: 렌더 시간은 로봇 메시의 삼각형 수에 거의 비례했다. 삼각형 40만 개 50.5 ms → 약 4만 개(10%) 9.6 ms → 로봇을 끈 빈 장면 4.3 ms(고정비용). 해상도(160×120 → 320×240)에는 거의 영향이 없었다. 삼각형을 CPU가 처리하는 비용 때문이라는 설명은 추정이다.
3. **해결**: 보조 화면 전용 모델에서 로봇의 보이는 메시를 10%로 줄였다(`server/aux_mesh.py`, 순수 numpy 꼭짓점 묶기). 시작할 때 한 번 만든다(약 0.6 s). 원본 STL(`third_party`)은 읽기만 하고, 줄인 모델은 메모리에만 있다. 모양 변화는 가장자리 최대 약 2 mm(측정).
4. 그 사이에 집게 높이·블록·바닥선만 그리는 2D 그림(`SideMirror`)을 먼저 만들었다. 지금은 **폴백**으로 남아 있다.
5. 검토만 하고 채택하지 않은 방안: 브라우저에서 qpos로 로봇을 직접 그리기(3D 라이브러리가 없어 작업량 큼), 서버 메인 렌더에 시점 추가(메인 fps가 4~5로 떨어질 것으로 추정).

## 구조

```
SimWorker(틱마다 qpos 사본 13개) → AuxViewWorker(스레드 하나, 줄인 모델, 두 시점을 한 장으로) → JPEG → app.py가 base64 JSON 텍스트(aux_frame)로 → 브라우저
```

- `server/aux_view.py` `AuxViewWorker`: 자기만의 `MjModel`(줄인 메시, 시야각 20°) / `MjData` / GL 컨텍스트를 쓴다. 시뮬레이션의 `MjData`, 학습용 `StateRenderer`와 아무것도 공유하지 않는다.
- **두 시점 M1 방식**: GL 컨텍스트 하나, 오프스크린 버퍼 하나에 viewport 둘을 가로로 나란히 그리고, 읽기와 JPEG 인코딩은 한 번만 한다. 컨텍스트를 둘 따로 쓰는 방식(S2)보다 약 15~20% 빨랐다(측정: 두 시점 합계 p50 85.1 vs 104.2 ms는 원본 메시, 줄인 메시 13.3 vs 17.0 ms). 이미지 한 장 = 400×150(시점 하나 200×150).
- `server/sim_worker.py` `attach_aux(aux)`: 물리 루프가 끝난 뒤 `aux.offer_qpos(data.qpos.copy(), auto_running)`만 부른다. 사본이라 aux가 값을 바꿔도 시뮬레이션은 영향이 없고, `offer_qpos`가 예외를 던지면 aux를 떼고 시뮬레이션은 계속 돈다. `sim_worker.py`는 `aux_view`를 import하지 않는다.
- `server/app.py` + `server/protocol.py`: 브라우저 → 서버 `{"type":"aux","enabled":bool}`(연결마다 받을지, 기본 꺼짐), 서버 → 브라우저 `{"type":"aux_frame","seq","w","h","views","jpeg":<base64>}`(텍스트). 메인 화면 바이너리 프레임의 형식은 바꾸지 않았다. `state.aux`에 상태(`enabled`, `running`, `viewers`, `fps`, `slowed`, `error` …)가 실린다. 한 프레임은 약 9.4 KB(JPEG), base64로 연결당 약 125 KB/s(10 fps).
- 설정은 `server/aux_view_config.py` 한 곳. `config.py`와 겹치는 값(장면 경로, 그림자·반사, 옆에서 방위각, 바라보는 점의 x, y)은 import해서 쓴다(`test_aux_isolation.py`가 확인).
- 웹: `web/src/aux/config.js`(크기·위치·순서·숨김 기준), `aux/view.js`(순수 계산), `components/AuxMirrors.jsx`, `SimView.jsx`, `App.jsx`, `net/connection.js`(`aux_frame` 수신, `sendAux`).

## 카메라: C2 고정 망원

| | 옆에서(왼쪽 아래) | 비스듬히(오른쪽 아래) |
|---|---|---|
| 방위각 | 105° (`config.VIEW_PRESETS["side"]`) | 45° |
| 고도 | −12° | −25° |
| 거리 | 0.68 m | 1.00 m |
| 바라보는 점 | (x, y는 `config.VIEW_LOOKAT_M`, z 0.05) | (같은 x, y, z 0.03) |

- 시야각은 두 시점 모두 **20°**(`AUX_FOVY_DEG`). 이 값은 aux 전용 모델에만 적용되고 메인과 `StateRenderer`의 모델은 45° 그대로다(`test_aux_view.py`가 확인).
- 이전 카메라(시야각 45°, 거리 0.8 m, 고도 −10°)는 블록이 칸 높이의 6% 남짓이었고 위쪽 27%가 검은 하늘이었다. 시야 반각(10°)보다 고도가 가파르면 하늘이 화면 밖으로 나가서 검은 띠가 없어진다.
- 시연 24개의 작업 프레임(집게 끝과 블록이 15 cm 안) 120개로 잰 블록 높이(빨간 픽셀 상자 / 칸 높이): 옆에서 중앙값 **15.3%**(블록이 보이는 프레임 100%), 비스듬히 **12.7%**(보임 87.5%). 기준은 사용자가 메인 화면을 직접 확대한 모습의 12.5%였다.
- 더 나은 대안(적용 안 함): 비스듬히를 방위 70°·거리 0.80 m로 하면 보임 100%, 중앙값 15.0%. 거리만 0.80 m로 줄이면 중앙값 15.3%, 보임 90.8%. `AUX_CAMERAS`의 값만 고치면 된다.

## 켜는 법

```bash
cd ~/handover && AUX_ENABLED=1 MUJOCO_GL=egl server/.venv/bin/python -m server.app    # 서버 (127.0.0.1:18007, 전달 안 함)
cd ~/handover/web && npm run dev -- --host 127.0.0.1 --port 3007 --strictPort          # 웹 (Ports 3007은 Private으로만)
```

`AUX_ENABLED`가 없으면 아무것도 만들지 않고 화면에는 2D 그림만 보인다.

## 끄는 장치

1. `AUX_ENABLED`(기본 **False**): 꺼져 있으면 스레드도 GL 컨텍스트도 만들지 않고 `offer_qpos`는 바로 돌아온다. 웹에는 체크박스도 안 나온다.
2. 보는 연결이 0개이면 렌더하지 않는다(보조 화면을 켠 연결 수만 센다. 브라우저 탭을 닫으면 0이 된다). bench F 조건(aux 켬, 보는 연결 0)은 aux 없음(B)과 같았다.
3. 브라우저 체크박스(기본 켬): 끄면 서버에 `aux enabled:false`를 보내 렌더를 멈추고 2D 그림으로 바뀐다.
4. 렌더(+JPEG)가 `AUX_MAX_RENDER_MS`(40 ms)를 넘으면 fps를 반으로 줄인다(바닥 `AUX_MIN_FPS` 1). AUTO 중 멈추는 옵션(`AUX_PAUSE_DURING_AUTO`)은 **기본 False**(교정은 AUTO 중에 봐야 하므로).
5. aux 스레드가 시작에 실패하거나 죽으면 서버는 계속 돌고 `state.aux.error`에 이유가 남는다. 웹은 2D 그림으로 돌아간다.

## 2D SideMirror 폴백

진짜 화면 두 개는 다음이 모두 맞을 때만 보인다: 체크박스 켬, 연결됨, `state.aux`가 켜져 있고 돌고 오류 없음, 마지막 프레임이 2.5초 안. 하나라도 어긋나면(서버 꺼짐, 오류, 프레임 끊김, 체크 해제) 기존 2D 그림(`SideMirror`, 바닥선·블록·집게 끝 높이)이 오른쪽 아래에 보인다. 메인 화면 가로가 440 px보다 좁으면 나침반과 겹치므로 진짜 화면은 숨기고 2D 그림을 보인다(CSS 컨테이너 쿼리).

## 격리: 학습·녹화 경로는 바이트 단위로 그대로

- `server/tests/test_aux_isolation.py`(31개): 학습용 `StateRenderer` 이미지 sha256이 aux 꺼짐 → 켜짐(렌더 중) → 꺼짐에서 같음, 결정적 롤아웃·시연 재생 결과 같음, 학습·녹화·AUTO·평가 코드가 `aux_view`/`aux_mesh`를 import하지 않음, `RECORD_*`와 `config_sha256` 불변, 녹화 npz 바이트 같음.
- `server/tests/test_aux_wiring_isolation.py`(15개): 실제 `SimWorker`에 실제 두 시점 aux(렌더 중)를 붙이고 물리 스텝별 qpos 바이트가 aux 없을 때와 같음(값을 망가뜨리는 가짜 aux를 붙여도 같음), 예외를 던지는 aux는 떼고 계속 돎, `sim_worker.py`의 aux 호출은 물리 루프 뒤에서 `qpos.copy()`로만, 서버 통합(aux_frame 형식, 메인 프레임 그대로, aux 메시지가 로봇·녹화·AUTO 상태를 안 바꿈).
- 라이브 서버는 시계에 의존해 비트 비교가 안 되므로 위처럼 "스텝 번호별 궤적"과 가짜 틱 녹화로 확인한다. 녹화 npz 전체의 라이브 비트 비교는 하지 못했다.

## 측정 결과 (`server/aux_view_bench.py`)

합격 기준(사용자와 확정): 틱 p95 +3 ms, 50 ms 초과 비율 +3%p, 메인 fps −0.5, CPU +0.5코어, aux 렌더 p95 ≤ 40 ms, aux fps ≥ 8.

| 비교 | 구성 | 틱 p95 | 50 ms 초과 | 메인 fps | CPU |
|---|---|---|---|---|---|
| C−B (aux 켬 − 메인만) | 처음(시야 45°, 160×120) | +2.1 ms | +2.2%p | 15.0 그대로 | +0.35 |
| C−B | **최종(C2, 200×150)** 정방향 / 역방향 | +0.96 / +0.74 ms | +0.50 / +0.33%p | 15.0 그대로 | +0.25 |
| H−G (AUTO 중 aux 켬 − 끔, 시드 0,1,2 합침) | 최종 정방향 / 역방향 | +0.92 / +2.28 ms | +0.08 / +1.08%p | +0.06 / −0.11 | +0.28 / +0.21 |

- 최종 구성은 모든 기준을 통과했다. aux 렌더 p50은 약 10 ms(AUTO 중 p95는 29~32 ms), aux 실제 fps는 10(역방향 한 구간 제외).
- 가까운 카메라가 처음보다 렌더가 빨라서 틱 영향이 줄었다.
- 틱 p95 영향이 처음보다 줄었다는 설명(가까운 카메라 때문)은 추정이다. 카메라와 크기를 따로 바꿔 보지는 않았다.

## 알려진 한계

- **비스듬히 칸은 블록이 가려진다**: 시연 프레임의 12.5%에서 팔이나 집게 뒤에 숨어 안 보인다. 중앙값 여유도 0.2%p뿐이다(12.7% vs 기준 12.5%). 위 "더 나은 대안" 참고.
- **감속 규칙이 프레임 하나에 민감하다**: 렌더 한 프레임이 40 ms를 넘으면 fps를 절반으로 낮추고, 보는 연결이 0에서 다시 생기기 전에는 안 돌아온다. AUTO 중 렌더 p95가 약 30 ms라 여유가 약 10 ms뿐이고, 역방향 측정에서 한 구간(약 20초)이 평균 5.9 fps로 떨어졌다. 규칙은 바꾸지 않았다. 연속 몇 프레임일 때만 낮추고 회복하는 규칙이 필요할 수 있다(추정).
- **정책 프로세스의 CPU는 bench의 CPU 열에 없다**: AUTO는 별도 프로세스(nice 19, 스레드 2개)라 서버 전체 부하는 표보다 더 크다.
- AUTO 측정 구간이 짧다(시드 0은 약 8초에 성공, 시드 1·2는 20초 창). 시드별 H−G는 방향마다 부호가 흔들릴 수 있다.
- 측정 중 다른 사용자의 부하(load 4~6)가 있으면 p95가 45~65 ms로 튄다. 틱 p95는 load 약 1~2에서 잰 값이다.
- 그리퍼 몸통 전체가 칸 안에 들어오는지는 재지 않았다(집게 끝 점과 블록 중심 기준). 팔 윗부분은 일부 잘린다.
- 컨테이너 쿼리(좁은 창 숨김)는 최신 브라우저(Chrome 105+, Firefox 110+, Safari 16+)만 지원한다.
- 라이브 확인(사용자): 두 화면과 체크박스 동작, 손 추적 정상. 손 입력 지연의 체감(+1~2 ms 효과)은 사용자 판단에 맡긴다.

## 파일

- 서버: `aux_mesh.py`, `aux_view.py`, `aux_view_config.py`, `aux_view_bench.py`(측정), `aux_view_probe.py`(렌더 비용 가설 검증), 연결: `sim_worker.py`, `app.py`, `protocol.py`
- 웹: `web/src/aux/`(config, view, view.test), `components/AuxMirrors.jsx`, `SimView.jsx`, `App.jsx`, `net/connection.js`, `index.css`
- 테스트: `test_aux_view.py`, `test_aux_mesh.py`, `test_aux_isolation.py`, `test_aux_wiring_isolation.py`, `test_aux_view_probe.py`, `test_aux_view_bench.py`, `test_protocol_aux.py`, `test_app_scene.py`, 웹 `aux/view.test.js`, `net/auxSend.test.js`, `components/auxRender.test.jsx`
