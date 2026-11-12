# 13. 교정 — AUTO 중에 사람이 넘겨받아 고치고 돌려주기 (4-2)

학습된 정책(ACT 1호)이 화면 속에서 혼자 움직이다가 틀리려는 순간, 사람이 손으로 **넘겨받아** 고치고, 다시 정책에게 **돌려주고**, 그 구간을 학습 데이터로 저장한다
(HG-DAgger 방식: 사람이 필요할 때만 끼어든다). 4-3에서 이 데이터를 합쳐 Colab으로 2호를 학습한다. **이번 단계에서는 데이터셋 변환과 학습은 하지 않았다.**
**이 문서의 숫자는 이 서버에서 직접 실행해 얻은 값이다.** 측정 도구는 `server/correction_headless.py`.

## 구조

```
브라우저 ──ws──▶ app.py ─ auto 메시지(takeover / handback / save_correction / discard_correction) ─▶ AutoRunner(auto.py)
   손 30Hz, 높이 키 ─▶ ControlWorker(control.py)            AutoEpisode(auto.py): 시뮬레이션 스레드에서 전환·기록·성공 판정
                                                            ControlOwner(owner.py): IDLE / AUTO / HUMAN_WAIT / HUMAN
                                                            CorrectionStore(correction_store.py) ─▶ data/corrections/
```

| 파일 | 하는 일 |
|---|---|
| `server/correction_config.py` | 교정 설정값 한 곳(총 상한 120초, 집게 일치 기준, 시드 배분표 `SEED_BLOCKS`, 저장 위치). `server/config.py`는 건드리지 않았다 |
| `server/owner.py` | 제어 주체 4가지와 허용 전환. 한 순간에 ctrl을 쓰는 주체는 하나 |
| `server/auto.py` | `AutoEpisode`: AUTO / HUMAN_WAIT / HUMAN 전환, 기록, 성공·시간 제한. `AutoRunner`: 명령 처리, 끝난 뒤 저장/버리기 선택, 화면용 상태 |
| `server/control.py` | HUMAN_WAIT에서 손 3프레임 세기 → HUMAN, 항상 상대 이동 + 지금 집게 끝 기준, 집게 일치 판정(`HandoffGripper`), IK 시작 자세 |
| `server/correction.py` | 집게 일치 판정, 기록(`CorrectionRecording`), 개입 구간·점프 계산(순수 함수) |
| `server/correction_store.py` | `data/corrections/` 저장(npz + json), `auto_runs.csv`, 체크포인트 SHA-256 |
| `server/recording.py`, `server/sim_worker.py`, `server/protocol.py`, `server/app.py` | 다음 시드 계산에 교정 폴더 포함, 바쁠 때 녹화·리셋 거절, SimWorker의 ctrl 적용 규칙, 새 `auto` 동작, 연결 |
| `server/correction_headless.py` | 브라우저 없이 전체 경로 검증과 측정(가짜 정책, 가짜 사람, 실제 체크포인트) |
| `web/src/auto/correction.js`, `web/src/input/takeoverKey.js`, `web/src/components/AutoPanel.jsx` | 화면 표시 규칙, Space 키, AUTO 상자 |

## 전환 규칙

```
IDLE ──(AUTO 시작)──▶ AUTO ──takeover──▶ HUMAN_WAIT ──손 3프레임 연속──▶ HUMAN ──handback──▶ AUTO
                       └──────── 성공 / 시간 초과 / 중지 / 오류: 어느 상태에서든 ────────▶ IDLE
```

허용하지 않는 것(서버가 거부, `invalid_count`는 올리지 않고 이유를 `error`로 돌려줌): IDLE에서 takeover, AUTO에서 handback, HUMAN_WAIT에서 handback(손을 보인 뒤에만 돌려줄 수 있다),
이미 대기/조종 중에 takeover, AUTO 아닌 때 save/discard 등. 모르는 `action` 값이나 다른 필드가 붙은 메시지는 `ProtocolError`(연결 유지, `invalid_count` +1).

| 상태 | ctrl을 쓰는 주체 | 로봇 | 화면 |
|---|---|---|---|
| AUTO | 정책(시뮬레이션 스레드) | 정책 행동을 그대로 | ▶ 로봇이 혼자 하는 중(AUTO) |
| HUMAN_WAIT | 아무도 안 씀 | **직전 ctrl 그대로 멈춤** | ✋ 손을 보여주세요(대기). "손을 화면 중앙에 보여주세요" |
| HUMAN | 사람(ControlWorker: 손 → 매핑 → IK → 변화량 제한) | 손을 따라감 | ● 내가 조종 중 |

### AUTO → HUMAN_WAIT (넘겨받기 요청)
- 전환은 **시뮬레이션 스레드가 다음 물리 스텝 직전에** 한다(요청을 받은 스레드가 하지 않는다). 그래서 ctrl을 쓰는 주체가 둘이 되는 순간이 없다.
- 순서: ① 대기 중인 청크 큐를 버린다 ② 지금 `data.ctrl`을 ControlWorker와 SimWorker에 같이 넘긴다 ③ 소유자를 HUMAN_WAIT로 바꾼다. SimWorker는 HUMAN_WAIT에서 ControlWorker의 ctrl을 적용하지 않는다(낡은 값이 덮어쓰지 않게).
- 정책을 기다리는 호출(약 0.1초)이 진행 중일 때 눌러도 안전하다: 받아 온 청크는 통째로 버리고 직전 ctrl을 유지한다(`test_takeover_during_a_policy_call_discards_the_chunk_and_holds_ctrl`).

### HUMAN_WAIT → HUMAN (손 3프레임)
- 조종 손이 `START_CONSECUTIVE_FRAMES`(3)프레임 **연속**으로 보이면 그 순간의 손 위치가 기준점이 된다. 중간에 손이 사라지면 처음부터 다시 센다.
- **항상 상대 이동(클러치)으로 시작한다.** 서버의 `MAPPING_MODE`(지금 작업 트리에서 "absolute")는 읽지 않는다. 로봇 쪽 기준은 작업 상자 중심이 아니라 **지금 실제 집게 끝(gripperframe)** 이다.
  방향 규칙은 기존 상대 모드(보기 방위각 기준)와 같다. 작업 상자 clamp, 필터, IK, 변화량 제한, 키보드 높이(↑↓, W/S)는 HUMAN 중 기존대로 동작한다.
- 손이 화면 밖으로 나가면 정지하고, 다시 들어와 3프레임 보이면 새 기준점(기존 클러치와 같은 동작).
- 넘어가는 순간 IK를 **지금 관절 각도에서 시작**하게 한다(`ik.set_arm`, `correction_config.HUMAN_IK_WARM_START`). 측정 결과는 아래 "측정".

### 집게: "로봇과 맞을 때까지 그대로, 맞은 뒤부터 손을 따라간다"
- 넘겨받는 순간 집게 ctrl은 AUTO가 마지막으로 낸 값 그대로다. 로봇 집게가 닫힘인지는 그 **ctrl 값**이 닫힘/열림 목표의 가운데(0.4 rad)보다 작은가로 정한다(블록을 쥔 qpos는 약 0.17이라 qpos로는 애매하다).
- 사용자의 pinch 판정(0.3 아래 쥠, 0.5 위 폄, 사이는 아직 모름)이 로봇 상태와 **처음 같아진 뒤부터** 손을 따라간다.
- 이유: 블록을 쥔 채 넘겨받았는데 손이 펴져 있으면 바로 따라가는 순간 블록을 떨어뜨린다. 반대로 열린 채 손이 쥐어져 있으면 허공을 쥐며 튄다.
- 화면: 맞기 전까지 **"집게를 로봇(닫힘)에 맞추세요"**(또는 열림)가 뜨고, 맞으면 사라진다. 설정: `GRIPPER_SYNC_MID_RAD`, `GRIPPER_SYNC_REQUIRE_MATCH`.

### HUMAN → AUTO (돌려주기)
- 청크 큐를 비우고, 다음 샘플에서 **새 관측(지금 qpos와 학습 카메라 렌더링)** 으로 즉시 청크를 요청한다. 이때 요청에 `reset=True`를 실어 정책 프로세스가 `policy.reset()`을 부른다.
- 돌려주는 순간의 **ctrl 점프** = (정책 첫 행동 − 직전 사람 ctrl)의 최대 관절 차이(rad). 에피소드 json의 `ctrl_jumps.handback_rad`에 기록한다.
  `HANDBACK_JUMP_WARN_RAD`(0.15)를 넘으면 보간을 넣을지 판단하라는 기준이다. **보간은 아직 구현하지 않았다**(제안: 점프가 크면 첫 N개 행동을 사람 마지막 ctrl과 직선으로 섞기. 실제 정책 점프는 아래 측정에서 평상시보다 작았다).

### 끝내기
- 성공 판정(`is_success` + `SuccessHold`, 0.5초 유지)은 AUTO, HUMAN_WAIT, HUMAN **모두에서** 돈다. 성공하면 어디서든 에피소드를 끝낸다.
- **AUTO 시간**(AUTO 주체로 행동을 낸 샘플만)은 기존 45초 제한을 그대로 쓴다. 손을 기다린 시간과 사람 시간은 빠진다.
- **전체 상한 120초**(`TOTAL_MAX_S`): AUTO + 대기 + 사람 시간 합. 넘으면 "시간 초과"로 자동 종료하고 저장 선택을 묻는다. 손을 영영 안 보여도 120초 뒤에 끝난다.
- 중지 버튼은 어느 상태에서든 끝낸다. 정책 프로세스가 죽으면(error) 로봇은 마지막 ctrl 그대로 멈추고, 사람 구간이 있었으면 저장 선택을 묻는다.
- 끝나면 마지막 ctrl을 SimWorker와 ControlWorker에 같이 넘기고 소유자를 IDLE로 돌린다(4-1과 같은 이어받기).

## 사용법과 요령

서버와 웹은 내가 직접 띄운다.

```bash
cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.app                # 서버 (먼저)
cd ~/handover/web && npm run dev -- --host 127.0.0.1 --port 3007 --strictPort       # 웹. Ports에서 3007만 Private 전달 → http://localhost:3007
```

1. 카메라를 켜고 손이 화면 중앙에 오는지 본다. 시연 녹화 상자의 **블록 리셋** → 화면의 "다음 시드 → 교정용/추가 시연용" 안내를 본다(**교정용 시드일 때만** 교정 실험에 쓴다).
2. AUTO 상자의 **AUTO 시작**. 기록은 AUTO를 시작하면 자동으로 시작된다(저장할지는 끝난 뒤 내가 정한다).
3. 로봇이 틀리려 하면 **Space**(또는 "넘겨받기 (Space)" 버튼). 로봇이 그 자리에 멈추고 "손을 보여주세요(대기)"가 뜬다.
4. **조종 손을 화면 중앙에 보인다**(3프레임 뒤 "내가 조종 중"). 블록을 쥔 상태면 **집게 안내**에 맞춰 손을 쥔 채 보인다.
5. 손으로 앞뒤·좌우, **↑↓ 또는 W/S**로 높이를 고친다. 한 번에 고치고 돌려주는 것이 좋다(오래 끌면 사람 시간만 늘어난다).
6. 다 고쳤으면 **Space 한 번 더**(정책에게 돌려주기). 정책이 이어서 한다. 또 틀리면 3번부터 반복.
7. 성공/시간 초과/중지로 끝나면: 사람이 고친 구간이 있으면 **"교정 저장" / "버리기"** 를 고른다(고르기 전에는 저장하지 않고 새 AUTO와 블록 리셋도 안 된다). 고친 적이 없으면(순수 AUTO) 저장 없이 `auto_runs.csv`에 한 줄만 남는다.

**어느 순간에 넘겨받을까 (요령)**
- 로봇이 **블록 위에서 멈칫**할 때(내리지도 올리지도 못하고 맴돎): 블록 양옆에 맞추고 내려 주거나, 집게를 닫아 준다.
- **엉뚱한 곳으로 갈 때**(블록과 먼 곳으로 가거나 목표를 빗나갈 때): 알아챈 즉시. 늦을수록 고쳐야 할 양이 늘어난다.
- 블록을 **쥔 채 떨어뜨리려 할 때**: 집게 안내에 따라 손을 쥔 채 넘겨받는다.
- 정책이 알아서 잘하는 구간에는 끼어들지 않는다(교정 데이터는 "틀린 순간"이 가치가 있다).
- 넘겨받은 직후 로봇이 크게 움직이면(집게 끝이 작업 상자 밖일 때, 아래 "알려진 한계") 놀라지 말고 손을 가만히 둔다.

## 저장 형식

위치 `data/corrections/` (`.gitignore`의 `data/`에 걸려 git에 올라가지 않는다. **`data/raw`와 섞이지 않는 새 폴더**이고 `train/build_dataset.py`는 `data/raw`만 읽는다).
웹캠 영상과 이미지는 저장하지 않는다. 숫자만 저장한다. 첫 저장 때 폴더가 만들어진다.

```
data/corrections/correction_000003.npz    배열 (numpy만으로 읽힘)
data/corrections/correction_000003.json   메타
data/corrections/auto_runs.csv            AUTO를 한 번 돌릴 때마다 한 줄
```

**npz** (길이 T = 샘플 수, 30Hz 격자는 3-2와 같다 `server/grid.py`):

| 이름 | shape | 내용 |
|---|---|---|
| `sim_time_s` | (T,) | 샘플을 찍은 시뮬레이션 시각(s) |
| `qpos` | (T, 13) | **행동을 적용하기 전** 전체 qpos(팔 6 + 블록 7). `observation.state = qpos[:6]` |
| `ctrl` | (T, 6) | **그 샘플에서 실제로 로봇에 적용 중인 ctrl** = `action`. AUTO 샘플은 정책 행동, HUMAN 샘플은 사람의 ctrl, HUMAN_WAIT 샘플은 직전 값 |
| `is_intervention` | (T,) bool | `owner == HUMAN`인 샘플만 True |
| `owner` | (T,) int8 | 0 = AUTO, 1 = HUMAN, 2 = HUMAN_WAIT |

**틱 정렬**: 샘플 k의 `qpos[k]`와 `ctrl[k]`는 같은 물리 스텝 직전에 한꺼번에 복사한다(3-2와 같은 규칙). 소유자가 바뀌는 샘플도 같다: AUTO→WAIT 샘플은 유지 중인 ctrl,
HUMAN→AUTO의 첫 AUTO 샘플은 새 정책 행동이다. 마지막 샘플(성공/시간 초과를 판정한 샘플)의 ctrl은 직전 적용값이다. `is_intervention`이 HUMAN만인 이유: HUMAN_WAIT는 로봇이 멈춰 있는 구간이라 학습에 넣으면 "멈춰라"를 가르친다.

**json** 주요 항목: `episode_index`, `seed`, `block_start_m`, `source`("correction"), `base_checkpoint`(`model_safetensors_sha256`, `model_safetensors_path`),
`interventions`(`[{start_sample, end_sample(포함), n_samples}]`), `intervention_count`, `human_time_s`(**HUMAN 구간만**: 06 문서의 "같은 시간 비교"에 쓰는 값),
`auto_time_s`, `human_wait_time_s`, `total_s`, `auto_result`(`outcome` success/timeout/stopped/error, `end_reason` success/auto_limit/total_limit/user_stop/error, `time_to_success_s`, `error`),
`user_decision`("saved"), `ctrl_jumps`(`handback_rad`, `takeover_first_second_max_step_rad`, `typical_step_rad`), `scene_sha256`, `robot_model_sha256`, `config_sha256`+`config`(server/config.py),
`correction_config_sha256`+`correction_config`, `created_at`, `n_ticks`, `fps`, `qpos_layout`, `mujoco_version`, `format_version`.
파일 쓰기는 3-2와 같다(임시 이름 → 바꿔치기, npz → json 순서, json이 있어야 저장된 것, 쓰기 스레드). **저장이 실패하면** 화면에 빨간 안내가 뜨고 저장 수에 넣지 않는다(데이터는 사라진다).

**`auto_runs.csv`** 열: `seed, kind, outcome, end_reason, auto_time_s, human_time_s, total_s, intervention_count, episode_index, checkpoint_sha256, created_at`.
`kind`: `auto_only`(개입 없이 끝난 AUTO), `correction_saved`, `correction_discarded`(버린 교정), `correction_save_failed`(저장하려다 실패). 한 번 쓴 시드를 서버를 다시 켠 뒤에도 알기 위한 장부다.

크기(실측, 아래 "측정"): 샘플 242개 에피소드 npz 약 23KB + json 약 7KB, 1382개(46초) npz 약 135KB.

## 시드 배분 (교정 조건과 추가 시연 조건이 겹치지 않게)

06 문서의 비교 실험(교정 B vs 추가 시연 A)은 두 조건이 같은 블록 위치를 쓰면 안 된다. 학습에 쓴 시연 시드는 **0~27**(저장된 시연 24개의 최댓값 27, 4개는 비어 있음)이다.
시연 시드 범위 `DEMO_SEED_RANGE = (0, 100)`에서 **28~99, 모두 72개**를 쓸 수 있다(리셋 버튼은 시드를 순서대로 하나씩만 낸다. 리셋만 하고 안 쓴 시드도 하나 소모된다).

06 문서는 A와 B를 **번갈아** 녹화해 연습 효과를 섞으라고 하므로, 한 덩어리씩 나누지 않고 12개씩 번갈아 쓴다(`server/correction_config.py`의 `SEED_BLOCKS`):

| 시드 | 용도 |
|---|---|
| 28~39 | 교정 |
| 40~51 | 추가 시연 |
| 52~63 | 교정 |
| 64~75 | 추가 시연 |
| 76~87 | 교정 |
| 88~99 | 추가 시연 |

교정 36개, 추가 시연 36개(같은 개수), 남는 시드 0. 평가 시드(1000~1049)는 여기에 없고 규칙도 그대로다: 화면의 AUTO에서 못 쓰고(리셋이 시연 시드만 낸다), 시드 안내는 평가 시드에 용도를 붙이지 않는다(`test_seed_purposes_do_not_clash_with_eval_or_auto_seed_rules`).

- **서버는 이 규칙을 강제하지 않는다.** 화면이 "다음 시드: 40 → 추가 시연용", 블록이 놓인 뒤 "준비된 블록: 시드 28 → 교정용"을 안내하고, 추가 시연용 시드로 AUTO를 시작하려 하면 노란 경고만 띄운다.
- 다음 시드는 `data/raw`, `data/corrections`의 저장본, `auto_runs.csv`의 모든 줄에서 가장 큰 시드 + 1이다. 그래서 서버를 다시 켜도 교정, 버린 교정, 순수 AUTO가 쓴 시드를 다시 내지 않는다.
- **주의(숫자로 본 한계)**: 조건당 36개는 "개수"의 상한이다. 06 문서는 사람 시간 T분을 같게 하라고 한다. 교정 한 번에 쓰는 사람 시간이 짧으면(예: 5초) T=10분을 채우려면 120개가 필요하다. 이 숫자는 가정이고 측정값이 아니다.
  36개로 모자라면 사용자가 `server/config.py`의 `DEMO_SEED_RANGE` 끝값을 늘려야 한다(이번 작업에서는 못 고친다). 실제 사람 교정 시간은 사용해 보고 정한다.

## 측정 (이 서버, 16코어 공용, 정책은 가짜 또는 실제 체크포인트)

**가짜 정책 + 가짜 사람은 실제 경로와 같은 입력만 쓴다**(손 메시지 30Hz, 높이 키). 가짜 정책(`ScriptedPolicy`)은 블록 위치를 아는 되먹임 제어기(IK로 접근 → 내림 → 쥠 → 들기 → 목표 위 → 내려놓기)다.

### 전체 경로 (`--scenario handback --seed 28`)
AUTO가 블록을 들고 가다가 → 넘겨받기 → 손을 **편 채** 시작(집게 안 맞음) → 쥐어서 일치 → 1.5cm 교정 → 돌려주기 → 정책이 이어서 성공.

- 결과: 성공, 242샘플(8.07초): AUTO 6.37초 + 손 기다림 0.10초 + 사람 1.60초, 개입 1번(샘플 134~181). 저장된 npz/json의 정합 검사(`check_alignment`) 문제 0개.
- 같은 경로를 WebSocket으로(`server/tests/test_app_correction.py`), 사람이 끝까지 해서 HUMAN에서 성공(`--scenario human_finishes`, 6.57초 사람 시간)도 확인했다.

### 지연 (요청 → 전환)

| 항목 | 화면 없음 | 화면 있음(`--viewers 1`) | 이번 재측정(2026-11-12, 실제 체크포인트 seed 1000, 각 1회, 서버 load 0.2~1.5) |
|---|---|---|---|
| 넘겨받기 요청 → HUMAN_WAIT(로봇 정지) | 에피소드 안 15.6 ms / 밖에서 잰 값 20.2 ms | 40.7 ms (밖에서 잰 값) | 화면 없음: 에피소드 안 31.9 / 밖 35.4 ms(`real`, `handback` 모드 29.0 ms), 화면 켬: 41.2 ms |
| 손 3번째 프레임을 보낸 순간 → HUMAN(ControlWorker가 ctrl을 쓰기 시작) | 0.2 ms | 0.3 ms | 측정 안 함(`real` 시나리오는 이 값을 재지 않는다) |
| 돌려주기 요청 → AUTO | 28.5 ms / 30.3 ms | 5.1 ms | 화면 없음: 31.8 ms(`real --real-mode handback`) |

- **넘겨받기 지연(화면 없음)**: 이번 약 30~35 ms, 문서 15.6~20.2 ms. 약 2배 높게 나왔지만 **원인 미확인, 1회 측정**이다. 시나리오(가짜 정책 vs 실제 체크포인트)와 시드가 달라 직접 비교는 못 한다. 화면 켠 값(40.7 → 41.2 ms)은 같은 수준이다.
- **돌려주기 지연 조건(확인 결과)**: 위 표의 세 값은 같은 코드 경로다. `--scenario handback --seed 28`에서 `request_handback()`을 부른 순간부터 owner가 AUTO가 될 때까지를 잰다(`server/correction_headless.py`의 `scenario_handback`). 전환은 `server/auto.py`의 `before_step`(물리 스텝 직전)에서 일어나므로, 요청이 한 틱(33.3 ms) 중 어느 시점에 왔느냐에 따라 0~약 35 ms 사이 값이 나온다. 5.1 ms(화면 있음)와 28.5/30.3 ms(화면 없음, 두 값은 이 문서에 어떤 실행인지 적혀 있지 않다)는 **조건이 다른 측정이라기보다 한 틱 안의 위상 차이일 가능성이 크다(추정, 각 1~2회라 확인하지 못했다)**. 화면 켬/끔의 차이 때문이라는 근거는 없다. 이번 31.8 ms도 한 틱 범위(35 ms) 안이다.

(요청을 받은 스레드가 아니라 시뮬레이션 스레드가 다음 물리 스텝 직전에 전환하므로, 정책을 기다리는 중(약 0.1초)에 눌렀다면 그만큼 늦어질 수 있다.)

### ctrl 점프와 평상시 (rad, 한 샘플 = 한 틱 사이 관절 중 최대 변화)

| 항목 | 가짜 정책 | 실제 체크포인트(seed 1000) | 이번 재측정(2026-11-12, seed 1000, 각 1회) |
|---|---|---|---|
| 돌려주는 순간 점프(정책 첫 행동 − 직전 사람 ctrl) | 0.0503 | **0.0234** | 0.0234 (`handback` 모드, 같음) |
| 넘겨받은 직후 1초 동안 샘플 사이 최대 변화(집게 포함) | 0.0176 (화면 켜면 0.0199) | 0.0234 (**`handback` 모드** 값) | `handback` 모드 0.0234(같음) / **`pick` 모드 0.0589(화면 켬 0.1151)** |
| 평상시 샘플 사이 변화 평균 / 95번째 백분위 | 0.029 / 0.050 | 0.016 / 0.062 | `handback` 모드 0.0166 / 0.0622 / `pick` 모드 0.022 / 0.098(화면 켬 0.022 / 0.107) |

- 문서의 0.0234(넘겨받은 직후 1초)는 **`handback` 모드**(집게만 맞추고 곧바로 돌려줌) 값이다. 이번에 처음 잰 `pick` 모드(사람이 집어서 놓음)는 사람이 실제로 움직이는 구간이라 값이 더 크고, 두 모드는 조건이 달라 서로 대응하는 값이 아니다. `pick` 모드의 0.0589, 0.1151이 문제인지는 판단하지 않았다(변화량 제한 0.08 rad/틱 기준은 팔 관절 앞 5개에만 적용되고 이 값은 집게를 포함하며, 화면 켠 값은 집게 이동일 수 있으나 확인하지 않았다).

돌려주는 점프(0.0503, 0.0234)는 경고 기준 0.15보다 작고, 평상시 95번째 백분위(0.050, 0.062)와 같거나 작았다. 그래서 **보간은 필요해 보이지 않는다**(이 두 측정 기준. 사람이 크게 움직인 직후에 돌려주는 경우는 측정하지 않았다).
평상시 최대(0.52, 0.25)는 정책이 집게를 한 번에 크게 움직이는 AUTO 구간 값이다.

### IK 시작 자세: `ik.set_arm`(지금 관절에서) vs `ik.reset`(기본 시작 자세) — 넘겨받은 직후 1초, 팔 관절 앞 5개의 최대 ctrl 변화(rad/틱)
변화량 제한은 0.08 rad/틱이다. 손은 가만히, 집게는 로봇과 맞춤.

| 정책과 시점 | `set_arm` | `reset` |
|---|---|---|
| 가짜 정책(IK가 만든 자세), 3번 | 0.0009, 0.0025, 0.0009 | 0.0005, 0.0020, 0.0020 |
| 실제 체크포인트, seed 1000, 3초에 넘겨받기 | 0.0151 | 0.0115 |
| 실제 체크포인트, 8초에 넘겨받기(2번) | 0.0052, 0.0174 | 0.0047, 0.0047 |
| 실제 체크포인트, 15초에 넘겨받기 | 0.0078 | 0.0089 |

**결론: 이 측정에서는 두 방식의 차이가 없었다**(둘 다 변화량 제한 0.08의 22% 이하). 처음에 걱정한 "기본 시작 자세에서 풀면 같은 집게 끝 위치라도 다른 관절 조합이 나와 팔이 휜다"는 이 자세들에서는 나타나지 않았다
(정책이 사람 시연을 따라 배워서 IK로 만든 자세와 가깝기 때문으로 보이지만 원인은 확인하지 않았다). `set_arm`을 기본으로 둔 것은 이점을 확인해서가 아니라 처음 계획 때문이고, 끄려면 `HUMAN_IK_WARM_START = False`.

### 넘겨받는 순간 집게 끝이 작업 상자 밖일 때 (`--scenario outside`)
AUTO를 시작하자마자(home 자세: 집게 끝 x 0.059, z 0.377 m, 작업 상자는 z 0.015~0.12) 넘겨받고 손은 가만히 두었다.
- 기준점이 상자 안으로 잘려서 첫 목표가 (0.18, -0.001, 0.12) m가 됐다. 로봇이 **지금 자리에서 0.288 m를 약 2초 동안** 움직였다.
- 집게 끝 최대 속도 **1.08 m/s**(`set_arm`) / 1.03 m/s(`reset`), 처음 1초 평균 0.46 m/s. 팔 관절은 변화량 제한(0.08 rad/틱)에 계속 걸렸다. 키보드 높이 속도(0.08 m/s)의 약 13배다.
- 블록 근처(실제 체크포인트 seed 1000에서 3, 8, 15초에 넘겨받고 1.6초 뒤 집게 끝 z 0.038~0.072 m)는 상자 안이라서 이 문제가 없었다(위 IK 표의 값이 모두 작다).
- 따라서 **AUTO를 시작한 직후(집게가 아직 위에 있을 때) 넘겨받으면** 로봇이 상자 안으로 빠르게 내려온다. 지금은 고치지 않고 한계로 둔다(제안: 상자 밖이면 기준점을 자르지 않고 천천히 들어오게 하거나, 그 구간에서는 넘겨받기를 막기).

### 집게를 쥔 채 넘겨받기: 블록이 떨어지지 않는가 (`--scenario handback`, 가짜 손의 pinch를 **펴짐**으로 시작)
- 블록 높이(m): 손이 펴져 있는 동안(HUMAN 중 24프레임, 약 0.8초) **최소 0.0788, 최대 0.0788**(들린 높이 그대로). 화면 켠 측정은 0.0786. 쥔 손(pinch 0.1)으로 바꾸면 일치하고 그 뒤 따라간다.
- "집게를 로봇에 맞추세요" 조건(가짜 손 메시지): 대기 시작부터 HUMAN 중 손이 안 맞는 동안 켜져 있고(`gripper_sync.waiting`), 쥠 판정(0.3 아래)이 나온 프레임에 꺼진다(그 사이 0.3~0.5 애매한 구간에서는 계속 켜져 있다). 로봇이 열려 있고 손도 펴져 있으면 바로 꺼진다.
  HUMAN이 끝나 AUTO로 돌아가면 `active`도 꺼진다. 화면 문구가 뜨는 조건은 `web/src/auto/correction.test.js`가 확인한다.

### HUMAN 중 틱 간격 (공칭 33.3 ms, 화면 렌더링 틱은 약 55 ms라 40 ms를 넘는 것이 원래 있다)

| 조건 | 틱 수 | 평균(ms) | 최대(ms) | >40 ms | >50 ms |
|---|---|---|---|---|---|
| 화면 없음(handback) | 49 | 33.33 | 33.34 | 0 | 0 |
| 화면 없음(human_finishes) | 197 | 33.33 | 33.37 | 0 | 0 |
| 화면 켬(human_finishes) | 191 | 33.37 | 48.18 | 56 | 0 |
| 화면 켬(handback) | 49 | 33.41 | 43.57 | 6 | 0 |
| (이번 재측정 2026-11-12, 각 1회) 화면 없음(`real`, pick) | 205 | 33.33 | 33.35 | 0 | 0 |
| (이번 재측정) 화면 켬(`real`, pick) | 209 | 33.36 | 49.93 | 44 | 0 |
| (이번 재측정) 화면 없음(`real --real-mode handback`) | 28 | 33.33 | 33.34 | 0 | 0 |

AUTO 틱(참고): 정책을 기다린 틱이 50 ms를 넘는다(4-1과 같다. 요청 횟수와 같은 수). HUMAN 중에는 정책을 부르지 않으므로 50 ms를 넘는 틱이 없었다. 40 ms 초과는 화면 렌더링 틱이다.

### 실제 체크포인트 (`--scenario real --seed 1000`, 평가 시드라 내부 함수에서만)
- AUTO 8.03초 뒤 넘겨받아 사람이(스크립트) 집어서 목표에 놓았다: **성공**(총 15.0초: AUTO 8.03 + 대기 0.13 + 사람 6.83, 개입 1번). 사람이 없었으면 이 시드는 평가에서 `grasp_fail`(블록이 안 들림)이다. 정책이 로드되는 데 5.2초(첫 실행)/2.3초.
- 같은 시드에서 사람이 집게만 맞추고 곧바로 돌려주면 정책이 다시 같은 실패를 되풀이해 45초 시간 초과가 된다(`--real-mode handback`, 돌려주기 점프 0.0234 rad, 청크 요청 15번). 성공 여부는 보고만 한다.

### 파일 크기
| 에피소드 | 샘플 | npz | json |
|---|---|---|---|
| 가짜 정책 handback | 242 | 23,095 B | 7,137 B |
| 가짜 정책 human_finishes | 260 | 28,010 B | 7,096 B |
| 실제 체크포인트 사람이 집기 | 450 | 44,180 B | 7,103 B |
| 실제 체크포인트 곧바로 돌려주기(45초) | 1382 | 135,163 B | 7,137 B |

json이 약 7KB인 것은 설정 전체(`config`)를 같이 담기 때문이다(3-2와 같다). 교정 36개면 약 1~2MB.

## 4-3에서 쓸 학습 구간 후보 (결정은 4-3에서)

| 후보 | 내용 | 장점 | 단점 |
|---|---|---|---|
| **개입 구간만** | `is_intervention == True`인 샘플(HUMAN)만 | 06 문서 B 조건의 정의("HUMAN 구간만 학습")와 같다. 정책이 틀린 상황의 올바른 행동만 들어간다. 사람 시간과 학습 데이터 양이 같은 축이라 비교가 깔끔하다 | 개입 직전 "틀리기 시작한 맥락"과 고친 뒤 정책이 이어가는 구간이 빠진다. 구간이 짧으면(1~2초) 샘플이 적다. ACT는 청크(약 100스텝=3.3초)를 예측하므로 짧은 구간은 청크 앞부분만 감독된다 |
| **개입 + 이어진 AUTO 구간** | HUMAN 구간과, 돌려준 뒤 에피소드 끝까지 AUTO가 한 행동 | 돌려준 직후 정책이 어떻게 이어가는지를 포함해 분포가 더 자연스럽다 | AUTO 구간은 정책 자신의 행동이라 "사람이 시킨 것"이 아니다(자기 모방). 틀린 AUTO 행동이 섞이면 같은 실수를 강화할 수 있다. 사람 시간은 적은데 데이터 양은 늘어 06의 "같은 시간" 비교가 흐려진다 |

판단에 필요한 정보는 저장되어 있다(`owner`, `interventions`, `auto_result`). 어느 쪽이든 `data/raw` 시연 20개와 합쳐 같은 학습 설정으로 학습한다. 둘 다 학습해 비교하는 것도 가능하다(학습 시간이 늘어난다).

## 알려진 한계
- 집게 끝이 작업 상자 밖일 때 넘겨받으면 첫 목표가 잘려 로봇이 최대 1.08 m/s로 움직인다(위 측정). 블록 근처에서 넘겨받을 때는 상자 안이라 문제가 없었다.
- 돌려주는 순간 보간은 없다. 점프는 측정한 두 경우에서 평상시보다 작았지만 사람이 크게 움직인 직후는 재지 않았다.
- 서버를 끄면 저장/버리기를 고르지 않은 교정은 버려진다. 브라우저를 닫아도 서버는 들고 있다.
- 저장 실패한 교정은 다시 시도하지 않는다(3-2와 같다). 시드는 `auto_runs.csv`에 `correction_save_failed`로 남는다.
- **장부 기록 실패 알림**(`server/correction_store.py` `_handle_csv`, `server/auto.py` `_on_csv_failed`): 교정 저장은 성공했는데 `auto_runs.csv`만 실패하면 빨간 안내 "auto_runs.csv에 기록하지 못했습니다(시드 N)"가 뜬다. 교정 파일은 저장된 것으로 센다. 교정 저장과 장부가 **둘 다** 실패하면 화면에는 저장 실패 안내만 뜨고(안내를 덮어쓰지 않으려는 선택) 장부 실패는 서버 로그에만 남는다. **(추정, 확인하지 않음)** 이 경우 그 시드가 `auto_runs.csv`에 없어서 다음 시드 안내가 같은 시드를 다시 줄 수 있다. 개선안(사용자 결정): 둘 다 실패한 경우의 안내문에 "장부(auto_runs.csv)도 기록하지 못했습니다"를 덧붙인다.
- `server/config.py`의 `MAPPING_MODE`는 지금 "absolute"인데 이 값은 바꾸지 않았다. 넘겨받은 조종은 항상 상대 이동이라 상관없지만, 평소(IDLE)의 매핑은 그 값을 따른다.
- 시드는 72개뿐이다(위 "시드 배분").
- 실제 브라우저에서의 화면 모양, Space 포커스 동작, 깜빡임은 자동 테스트가 없다(DOM 테스트 도구 없음). 순수 로직만 테스트했다.
- 측정은 모두 스크립트된 가짜 사람이다. 실제 사람의 교정 시간, 넘겨받는 요령은 사용해 보고 정한다.
- 기존 서버 테스트 중 `MAPPING_MODE`가 작업 트리에서 "absolute"라 relative를 기대하는 2개(`test_control_relative.py::test_app_default_mapper_follows_the_config_mode`, `test_mapping_relative.py::test_defaults_modes`)는 이번 작업과 상관없는 이미 아는 실패다.
- `server/tests/test_owner.py`의 두 곳은 새 전환 규칙에 맞게 고쳤다(허락받음).
- `docs/Handoff.md`에 "4-2 구현 완료, 측정은 13-correction.md"를 반영할 것 — 이 작업에서는 수정하지 않았다. `docs/12-auto.md`의 "HUMAN 전환은 이름만 있다"는 낡은 문장이 되었다.

## 검증 명령

```bash
cd ~/handover
server/.venv/bin/python -m pytest server/tests -q                                   # 아는 실패 2개(MAPPING_MODE) 외에 모두 통과해야 한다
(cd web && npm test && npm run lint)
server/.venv/bin/python -m server.correction_headless --scenario handback --seed 28            # 가짜 정책 → 넘겨받기 → 교정 → 돌려주기 → 성공
server/.venv/bin/python -m server.correction_headless --scenario human_finishes --seed 29      # 사람이 끝까지 해서 HUMAN에서 성공
server/.venv/bin/python -m server.correction_headless --scenario ik --repeat 3                 # set_arm vs reset (가짜 정책)
server/.venv/bin/python -m server.correction_headless --scenario ik_real --seed 1000 --repeat 2 --takeover-at-s 8   # 같은 비교를 실제 체크포인트에서
server/.venv/bin/python -m server.correction_headless --scenario outside --seed 30             # 집게 끝이 작업 상자 밖일 때
server/.venv/bin/python -m server.correction_headless --scenario real --seed 1000              # 실제 체크포인트 + 스크립트된 사람
server/.venv/bin/python -m server.correction_headless --scenario handback --viewers 1          # 화면 렌더링을 켜고 틱 간격 재기
MUJOCO_GL=egl server/.venv/bin/python -m server.correction_headless --scenario gripper --seed 28   # 쥔 채 넘겨받기: 손이 펴져 있어도 블록이 안 떨어지는가
```

`--scenario gripper`는 `handback`과 같은 함수를 부르는 별칭이다. **정상 기준**: `outcome`이 `success`, `alignment_problems`가 `[]`, `block_z_min_while_hand_open_m`가 바닥 위 블록 중심 높이(0.015 m)보다 한참 위. 실측(2026-11-12): 0.0788(최대 0.0789, 최소 전체 0.0788). 블록이 떨어졌다면 이 값이 바닥 높이(0.015) 근처까지 내려갈 것이다(추정, 떨어뜨려 본 적은 없다).

## 설정값 위치
`server/correction_config.py`: `CORRECTION_DIR`, `TOTAL_MAX_S`, `GRIPPER_SYNC_MID_RAD`, `GRIPPER_SYNC_REQUIRE_MATCH`, `HUMAN_IK_WARM_START`, `HANDBACK_JUMP_WARN_RAD`, `SEED_BLOCKS`.
AUTO 시간 제한(45초)은 `server/auto_config.py`의 `AUTO_MAX_S`, 3프레임은 `server/config.py`의 `START_CONSECUTIVE_FRAMES`, 집게 pinch 기준은 `PINCH_CLOSE_BELOW`/`PINCH_OPEN_ABOVE`. 화면 쪽 Space 연타 간격은 `web/src/input/takeoverKey.js`.

## 프로토콜 (docs/12 외에 추가된 것)
- 브라우저 → 서버: `{"type":"auto","action":"start"|"stop"|"shutdown_policy"|"takeover"|"handback"|"save_correction"|"discard_correction"}`.
- `state.auto` 추가 필드: `owner`(`IDLE`/`AUTO`/`HUMAN_WAIT`/`HUMAN`), `auto_s`, `human_s`, `wait_s`, `total_s`, `total_max_s`, `interventions`,
  `correction`(`pending`: 선택을 기다리는 교정 요약 또는 null, `saved_count`, `saved_human_s`, `save_error_count`, `last_saved`), `ready_seed`/`ready_seed_purpose`, `next_seed`/`next_seed_purpose`.
- `state.control` 추가 필드: `human_session`, `gripper_sync`(`active`, `waiting`, `robot_closed`).
