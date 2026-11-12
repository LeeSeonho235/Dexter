# 14. 보조 카메라 (2-6, 보기 전용)

옆에서 본 작은 화면을 메인 화면 구석에 따로 보여 주는 기능. 집게 높이를 눈으로 보기 쉽게 하려는 것이다. **학습·녹화·AUTO·평가에는 아무 영향이 없어야 한다.**
이 문서는 서버 쪽(1단계)까지만 쓴다. 웹 쪽(`aux_frame` 메시지, 오버레이)은 서버 보고를 확인한 뒤 추가한다.

## 구조 (후보 A: 별도 스레드가 두 번째로 그려서 따로 보낸다)

- `server/aux_view.py` `AuxViewWorker`: 스레드 하나. 자기만의 `MjModel` / `MjData` / `Renderer`로 그린다. 시뮬레이션의 `MjData`, 학습용 `StateRenderer`와 아무것도 공유하지 않는다.
- 시뮬레이션 스레드는 틱마다 `offer_qpos(qpos, auto_running)`으로 qpos 13개의 사본만 넘긴다(락 안에서 복사만). 렌더 시간이 물리 틱을 늘리지 않는다.
- 설정은 `server/aux_view_config.py` 한 곳. `config.py`의 값(장면 경로, `VIEW_PRESETS["side"]`, 그림자·반사)은 import해서 쓰고 다시 정하지 않는다.
- 카메라: 옆에서 본 고정 카메라(방위각 105°, 고도 -10°, 거리 0.8 m). 마우스 보기와 연결하지 않는다.

## 끄는 장치

1. `AUX_ENABLED`(기본 **False**, 환경변수 `AUX_ENABLED=1`로 켬)가 꺼져 있으면 스레드도 렌더러도 만들지 않고 `offer_qpos`는 바로 돌아온다.
2. 보는 연결이 0개(`set_viewer_count(0)`)이면 렌더하지 않는다.
3. 크기 두 단계(`small` 128×96 / `normal` 160×120), fps는 설정(`AUX_FPS`).
4. 렌더(+JPEG)가 `AUX_MAX_RENDER_MS`를 넘으면 fps를 반으로 줄인다(`AUX_MIN_FPS` 밑으로는 안 내림). AUTO 중에는 멈춘다(`AUX_PAUSE_DURING_AUTO`).

## 격리 테스트 (`server/tests/test_aux_isolation.py`)

1. `StateRenderer` 이미지 바이트가 aux 꺼짐 → 켜짐(실제 렌더 중) → 꺼짐에서 같다.
2. 결정적 롤아웃(`EvalEnv`, 시연 정답 재생 `replay_episode`) 결과가 aux 켜고 끌 때 같다.
3. 학습·녹화·AUTO·평가 코드가 `aux_view`를 import하지 않는다. aux는 `StateRenderer`/`make_record_camera`/`record_view`를 쓰지 않는다. `RECORD_*` 값과 `config_sha256`이 그대로다.
4. `EpisodeRecorder` 에피소드와 저장한 npz 바이트가 aux 켬/끔에서 같다.

하나라도 실패하면 구현을 멈추고 원인부터 본다(테스트를 고쳐서 통과시키지 않는다).

## 아직 안 한 것 / 모르는 것

- fps, 크기, JPEG 품질, `AUX_MAX_RENDER_MS`는 측정 전의 임시값이다(추측). 렌더 비용 측정(`server/aux_view_bench.py`) 뒤에 정한다.
- `sim_worker.py`에 `offer_qpos`를 붙이는 일, `app.py`/`protocol.py`의 `aux` 메시지와 `aux_frame` 전송은 아직 하지 않았다.
- 두 번째 EGL 컨텍스트가 라이브 틱 간격에 주는 영향은 측정 전에는 모른다(격리 테스트 1은 이미지 바이트만 확인한다).
