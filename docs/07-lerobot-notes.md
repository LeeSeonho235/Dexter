# 07. LeRobot 메모 (3-1)

작성일: 2026-11-03. 아래 내용은 **설치된 소스를 직접 읽거나 직접 실행해서 확인한 것**이다. 확인 못 한 것은 맨 아래 "아직 모르는 것"에 모았다.
소스 경로의 `SP`는 `train/.venv/lib/python3.12/site-packages/lerobot`이다.

## 1. 확인한 버전

| 항목 | 값 |
|---|---|
| lerobot | 0.6.1 (`lerobot[training]`) |
| 데이터셋 형식 | **v3.0** (`SP/datasets/dataset_metadata.py:60` `CODEBASE_VERSION = "v3.0"`) |
| Python | 3.12.3 (lerobot은 `>=3.12` 필요) |
| torch / torchvision | 2.11.0+cpu / 0.26.0+cpu (lerobot 요구: torch `>=2.7,<2.12`) |
| numpy / pillow | 2.2.6 / 12.3.0 (lerobot 요구: numpy `<2.3`) |
| av / torchcodec / pyarrow / datasets | 15.1.0 / 0.11.1+cpu / 26.0.0 / 4.8.5 |
| accelerate / wandb | 1.15.0 / 0.27.2 (학습 extra에 포함됨. 쓰지 않음) |

**환경을 분리한 이유(실제로 충돌함):** `server/.venv`는 numpy 2.5.4, pillow 13.0.0, mujoco 3.16.0이다. lerobot은 numpy `<2.3`을 요구하므로 같은 환경에 설치할 수 없다. `train/.venv`에는 mujoco가 없다(lerobot이 요구하지 않음). 그래서 **시뮬레이션 평가(MuJoCo + 정책)를 한 프로세스에서 돌리려면 환경을 하나로 합쳐야 하는데 지금은 안 된다.** 4~5단계에서 정책 실행을 서버에 넣을 때 해결해야 한다(아래 "아직 모르는 것").

설치 크기: 다운로드 기본 327MB + 학습 extra 145MB, `train/.venv` 디스크 1.8GB. 패키지에 문서 파일은 들어 있지 않다(docstring과 소스 주석만 읽음).

## 2. 데이터셋 형식과 디스크 구조 (v3.0)

경로 상수: `SP/datasets/utils.py:88-111`. 실제로 만든 가짜 데이터셋(2에피소드 × 30프레임)의 파일:

```
<root>/
├─ meta/info.json                         # codebase_version, fps, features, 총 개수, 경로 패턴
├─ meta/stats.json                        # feature별 평균/표준편차/최소/최대 (정규화에 쓰임)
├─ meta/tasks.parquet                     # task 문자열 → task_index
├─ meta/episodes/chunk-000/file-000.parquet   # 에피소드별 길이, 파일 위치, 통계
├─ data/chunk-000/file-000.parquet        # 프레임 표 (state, action, timestamp, frame_index, episode_index, index, task_index)
└─ videos/observation.images.main/chunk-000/file-000.mp4  # 에피소드가 이어붙은 영상 (3-1 가짜 데이터셋은 키가 top이었고, 실제 데이터셋(3-3)은 main)
```

- v3.0은 에피소드마다 파일을 만들지 않고 **여러 에피소드를 한 파일에 이어붙인다**(파일 크기 상한: data 100MB, video 200MB, `datasets/utils.py:89-90`).
- 이미지를 영상 대신 PNG로 저장하는 방식도 있다(`use_videos=False`, 경로 `images/{key}/episode-XXXXXX/frame-XXXXXX.png`).
- 실제 크기: 가짜 데이터(64×64, 60프레임) 128KB.

## 3. 데이터셋 만들기 API

호출 순서(`SP/datasets/lerobot_dataset.py`의 `create` 684행, `add_frame` 418행, `save_episode` 434행, `finalize` 474행):

```python
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset.create(repo_id="local/handover_v1", fps=30, features=FEATURES,
                           root="<아직 없는 폴더>", robot_type="so101_sim", use_videos=True)
for each episode:
    for each frame:
        ds.add_frame({"observation.state": ..., "action": ..., "observation.images.top": ..., "task": "..."})
    ds.save_episode()        # 에피소드 하나를 디스크에 씀(영상 인코딩 포함)
ds.finalize()                # 반드시 호출. 안 하면 parquet 마무리 정보가 빠져 데이터셋이 깨진다(소스 docstring)
```

- 실패한 에피소드를 버릴 때: `ds.clear_episode_buffer()` (`lerobot_dataset.py:452`).
- `add_frame` 규칙(`SP/datasets/dataset_writer.py:202`, `SP/datasets/feature_utils.py:221`): `task` 키 **필수**, `timestamp`와 `frame_index`는 **넣으면 안 됨**(자동 계산, `frame_index / fps`), 정의한 feature 외의 키는 거부, dtype과 shape가 다르면 `ValueError`.
- `shape`는 **튜플**(`(6,)`)이어야 하고, numpy 배열의 dtype이 정확히 같아야 한다(float32 자리에 float64를 넣으면 오류).
- 이미지는 `(H, W, 3)` uint8 또는 `(3, H, W)`를 받는다(`feature_utils.py:347-350`).
- 실제로 돌려 확인: 2에피소드 × 30프레임 생성 0.15초, 같은 구조로 위 파일이 만들어짐 (`train/smoke_test.py make`).

## 4. 우리 feature 설계

| 이름 | shape | dtype | 내용 |
|---|---|---|---|
| `observation.state` | (6,) | float32 | 관절 각도(rad). 순서: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper (`third_party/.../so101.xml` actuator 순서) |
| `action` | (6,) | float32 | 모터 목표 각도(rad), 위와 같은 순서. 서버가 `ctrl`에 쓰는 값 그대로 |
| `observation.images.main` | (240, 320, 3) | video | 시뮬레이션 학습 카메라(`RECORD_CAMERA_*`: 방위각 110°, 고도 -35°, 거리 0.8 m). 위에서 내려다보는 `top`이 아니라 비스듬한 시점이라 `main`으로 부른다(웹캠 아님). 3-3에서 확정 |
| fps | 30 | | 서버의 제어·시뮬레이션 주기(`SIM_TICK_HZ`, `CONTROL_HZ`)와 같게 |
| task | 문자열 | | 한 문장으로 고정. 예: `"pick the red block and place it on the green target"` |

- 블록 위치는 관측에 넣지 않는다(이미지로만 알게 함).
- task 문자열은 `meta/tasks.parquet`에 저장되고 프레임에는 `task_index`(숫자)로 들어간다. 데이터셋 전체에서 같은 문자열을 쓴다.
- **확정(3-2, 3-3):** `observation.state[k] = qpos[k][:6]`(행동을 적용하기 전 팔 관절), `action[k] = ctrl[k][:6]`, 이미지[k] = `qpos[k]` 13개(블록 포함)로 다시 그린 그림. 규칙은 `docs/08-recording.md`의 "틱 정렬 규칙".
- **4단계용 추가 칼럼(seed, is_intervention, source, success)을 데이터셋 안에 넣는 방법: 확인 결과(적용은 안 함).** 소스(`SP/datasets/feature_utils.py:221`, `dataset_writer.py:202`, `dataset_metadata.py:834`)를 읽고, 가짜 데이터셋(2에피소드 × 10프레임)으로 직접 돌려 확인했다.
  - **프레임 단위 추가 feature는 된다.** `features`에 `"seed": {"dtype": "int64", "shape": (1,), "names": None}`, `"is_intervention": {"dtype": "bool", "shape": (1,), "names": None}`를 넣고 `add_frame`에 같은 키를 주면 저장된다. 다시 불러 `ds[i]["seed"]`, `ds[i]["is_intervention"]`가 나오고(`tensor(7)`, `tensor(True)`) `meta/stats.json`에도 통계가 생긴다. 값이 프레임마다 반복되므로 `seed`는 에피소드 안에서 같은 값을 계속 넣어야 한다.
  - **에피소드 단위 메타(에피소드당 값 하나)를 넣는 공개 API는 못 찾았다.** `DatasetMetadata.save_episode(..., episode_metadata)`는 있지만 `LeRobotDataset.save_episode()`는 이 인자를 받지 않고, 안에서 만드는 값(파일 위치, 영상 구간)만 채운다. 에피소드 단위 값이 필요하면 프레임 단위 feature로 반복해서 넣는 것이 지금 확인된 방법이다.
  - **확인하지 않은 것:** `lerobot-train`이 이런 추가 컬럼이 있는 데이터셋으로 학습되는지(ACT는 `observation.state`, `action`, 이미지만 쓰므로 무시할 것으로 보이나 실행해 보지 않았다), 데이터셋을 나중에 합치거나 컬럼을 추가하는 방법, 문자열 feature(`source`).
  - 3-3 데이터셋에는 넣지 않았다(결정 보류). `seed`, 에피소드 번호 매핑은 `data/frames/manifest.json`의 `episodes`(raw 에피소드 번호 ↔ 시드)에 있고 데이터셋 에피소드 번호는 raw 번호 순서의 0부터다(`docs/09-dataset.md`).

### 이미지 해상도·카메라 (3-3에서 확정: 비스듬한 `main` 1개, 240×320)

> **3-2에서 `top`이 아니라 비스듬한 학습 카메라(`RECORD_CAMERA_*`, 방위각 110°·고도 -35°·거리 0.8 m)로 정했다.** 키는 `observation.images.main`이다.
> 이 카메라에서 블록은 약 13~15 × 15~19 px로 보인다(`docs/08-recording.md`). 아래 `top` 추천 문단은 3-1 때의 **추정이라 그대로 두되 현재 값이 아니다.**
> 아래 추천은 **직접 학습해서 확인한 값이 아니라 계산과 짐작**이다.

- (3-1 때 추정, 현재 아님) 카메라: **top 1개**를 임시 추천. 이유는 학습 비용(이미지 수에 비례)과 블록·목표·집게가 모두 보이기 때문이다. 위에서 보면 집게 높이는 알기 어렵다.
- 해상도: **240×320(세로×가로)** 임시 추천. `top` 카메라 기준 3cm 블록은 640×480에서 약 22px, 240×320에서 약 11px로 추정한다(카메라 높이 0.7m, fovy 50°로 계산한 값, 화면에서 직접 재지 않음). ACT의 ResNet18은 해상도를 정해 두지 않는다(`SP/policies/act/configuration_act.py`에 해상도 항목 없음).
- 서버 화면 설정은 `CAMERA = "free"`(마우스로 돌리는 카메라)다. 학습 이미지는 반드시 **고정 카메라**로 그려야 한다.
- 이미지 크기는 `shape`에 박히므로 데이터셋을 만든 뒤 바꾸려면 다시 만들어야 한다.

## 5. 녹화 렌더링이 30fps로 되는가 (직접 측정)

`top` 카메라, 그림자·반사 끔, `MUJOCO_GL=egl`(CPU), home 자세, 40장 평균 (스크립트: 이 세션 scratchpad의 `render_bench.py`, 저장소에는 넣지 않음):

| 해상도 | 한 장 시간(벽시계) | CPU 시간 | 조건 |
|---|---|---|---|
| 640×480 | 59.4 ms | 93.5 ms | `LP_NUM_THREADS=2` |
| 240×320 | 53.8 ms | 81.4 ms | `LP_NUM_THREADS=2` |
| 640×480 | 53.7 ms | 108.5 ms | 스레드 제한 없음 |
| 240×320 | 53.9 ms | 88.1 ms | 스레드 제한 없음 |

- 30fps는 한 틱 33ms인데 렌더링만 약 54ms라서 **실시간 30fps 녹화는 지금 방식으로는 안 된다**(렌더링만으로 약 18fps).
- 해상도를 640×480 → 240×320으로 줄여도 거의 빨라지지 않는다. 원인은 확인하지 않았다(해상도가 아닌 고정 비용으로 보이나 모른다).
- 대안(미검증): 녹화 중에는 물리(30Hz)와 관절 값(qpos, ctrl)만 저장하고 이미지는 에피소드가 끝난 뒤 저장한 qpos로 다시 그린다. Handoff.md 메모와 같은 방향이다. 블록은 자유관절이라 **블록 qpos도 같이 저장**해야 같은 장면을 다시 그릴 수 있다.

## 6. ACT 설정

`SP/policies/act/configuration_act.py` 기본값(24~151행):

| 항목 | 기본값 | 뜻 |
|---|---|---|
| `chunk_size` | 100 | 한 번에 예측하는 행동 개수. 30fps에서 약 3.3초 분량 |
| `n_action_steps` | 100 | 예측 하나를 몇 스텝 그대로 실행할지(`chunk_size` 이하) |
| `n_obs_steps` | 1 | 1만 지원 |
| `vision_backbone` | resnet18 | torchvision ResNet만 가능 |
| `pretrained_backbone_weights` | `ResNet18_Weights.IMAGENET1K_V1` | **다운로드 필요**(`None`이면 안 받음) |
| `use_vae` / `latent_dim` | True / 32 | |
| `dim_model`, `n_heads`, `n_encoder_layers`, `n_decoder_layers` | 512, 8, 4, 1 | |
| `kl_weight` | 10.0 | |
| `temporal_ensemble_coeff` | None | 쓰려면 `n_action_steps=1` 필수(소스가 강제) |
| `optimizer_lr` | 1e-5 | |
| 정규화 | MEAN_STD | 이미지·state·action 모두. 통계는 `meta/stats.json`에서 옴 |

- 이미지 키가 `observation.images.`로 시작하면 카메라로 인식한다. 카메라가 여러 개면 **모두 같은 shape**여야 한다.
- 학습 가능한 파라미터 수: 51,597,190(52M, 실제 로그).
- 사전학습 가중치 다운로드: `https://download.pytorch.org/models/resnet18-f37072fd.pth`, 44.7MB, 이 서버에서 받아짐(0.2초 안팎).

## 7. 학습 명령 (`SP/scripts/lerobot_train.py`, `SP/configs/train.py`)

**이 서버, CPU, 확인용(실제로 돌린 명령):**

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
train/.venv/bin/lerobot-train \
  --dataset.repo_id=local/handover_smoke --dataset.root=<데이터셋 폴더> \
  --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --batch_size=2 --num_workers=0 --steps=5 --log_freq=1 --save_freq=5 \
  --wandb.enable=false --output_dir=<출력 폴더>
```

- **`--policy.push_to_hub=false` 필수.** 기본값이 `True`다(`SP/configs/policies.py:70`). 안 끄면 학습이 Hub에 올리려고 한다.
- `--dataset.root`: 로컬 데이터셋 폴더. 주석(`SP/configs/default.py:34-37`): `root`가 없으면 `$HF_LEROBOT_HOME/{repo_id}`를 찾고, Hub에서 받은 것은 `$HF_LEROBOT_HOME/hub` 아래에 캐시된다.
- Hub에서 받아 학습: `--dataset.repo_id=<사용자>/<이름>`(root 없이). **이 방식은 실행해 보지 않았다.**
- 기본 학습 길이는 `steps=100000`, `batch_size=8`, `save_freq=20000`(`SP/configs/train.py:101-123`). 값은 이름이 같은 인자로 바꾼다.
- 체크포인트는 `<output_dir>/checkpoints/<스텝>/pretrained_model/`(config.json, model.safetensors, 전·후처리 json)에 저장되고 `checkpoints/last`가 최신을 가리킨다(실제 출력 확인).

## 8. 평가(추론) API — `select_action`

`SP/policies/act/modeling_act.py:93-135`, `SP/common/control_utils.py:43`, `SP/policies/utils.py:97`. 실제로 돌려 확인한 코드(`train/smoke_test.py infer`):

```python
policy = ACTPolicy.from_pretrained(ckpt_dir); policy.to("cpu"); policy.reset()   # 에피소드 시작마다 reset()
pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt_dir)
obs = {"observation.state": np.float32 (6,), "observation.images.main": np.uint8 (H, W, 3)}   # smoke_test.py는 키가 top인 가짜 데이터셋이라 top을 쓴다
action = predict_action(obs, policy, torch.device("cpu"), pre, post, use_amp=False, task=TASK)
# → torch.Tensor (1, 6) float32, 실제 관절 단위(post가 역정규화). 서버 ctrl에 넣을 값
```

- `predict_action`이 이미지를 `(H,W,3) uint8 → (3,H,W) float32 [0,1]`로 바꾸고, 배치 차원을 붙이고, 정규화하고, `policy.select_action`을 부르고, 역정규화한다.
- `select_action`은 한 번에 행동 1개를 돌려주는 **대기열 방식**이다. 대기열이 비면(처음과 `n_action_steps`마다) 정책이 청크를 예측하고, 그 사이 호출은 대기열에서 꺼내기만 한다. 측정: 첫 호출 57.7ms(64×64, CPU 2스레드), 이후 0.4~0.8ms.
- 에피소드(블록 리셋)마다 **`policy.reset()`을 불러 대기열을 비워야** 한다.
- `lerobot-eval`은 gym 환경용이라 우리는 쓰지 않고 시뮬레이션에서 직접 위 루프를 돌린다(`lerobot-eval`은 실행해 보지 않았다).

## 9. CPU 동작 확인 결과 (실측)

| 항목 | 값 |
|---|---|
| 데이터 | 2에피소드 × 30프레임, 64×64, 합성(회색 배경 + 빨간 사각형) |
| 설정 | ACT 기본(chunk 100), CPU, 스레드 2, batch 2, 5스텝 |
| 스텝당 시간(`updt_s`) | 첫 스텝 0.403초, 이후 0.313~0.318초 |
| 전체 | 벽시계 5.75초(시작·모델 생성 포함), 최대 메모리 1,472,812KB(약 1.4GB) |
| 결과 | 종료 코드 0, 체크포인트 저장, `select_action` 호출 성공 |

- **이 숫자는 64×64, batch 2의 값이라 실제 학습 시간의 근거가 되지 않는다.** 240×320에서는 이미지 연산이 크게 늘어난다(측정하지 않음).
- 5스텝의 loss(71.7 → 50.8)는 의미가 없다.
- torchcodec은 이 서버에 시스템 FFmpeg 라이브러리(`libavutil.so.59/60`)가 없어 불러올 수 없었고, lerobot이 **자동으로 pyav로 바꿔서** 학습은 정상 동작했다(경고로만 나옴). sudo 없이는 시스템 FFmpeg를 설치하지 않았다.

## 10. Colab에서 하려면

> 아래는 소스에서 읽은 내용으로 짠 **절차이며 Colab에서 실행해 보지 않았다.**

**올릴 것**
- 데이터셋 폴더 통째로(`meta/`, `data/`, `videos/`). 3-2 이후 실제 데이터는 용량이 작다(영상 압축). 방법은 Google Drive 마운트 또는 압축 파일 업로드. Hub에 올리면 공개 여부와 얼굴 이미지가 없다는 점을 확인한 뒤 **Private**으로 한다(올리는 것은 사용자가 직접 결정).
- 웹캠 영상은 만들지도 올리지도 않는다(시뮬레이션 화면과 숫자만).

**설치(Colab 셀)**

```bash
pip install "lerobot[training]==0.6.1"     # Colab 기본 CUDA torch를 그대로 쓴다. lerobot 요구 범위는 torch >=2.7,<2.12, numpy <2.3
```

- Colab의 Python이 3.12 이상이어야 한다(lerobot 요구). 아니면 설치가 안 된다.
- Colab 기본 numpy가 2.3 이상이면 pip이 낮출 수 있어 런타임 재시작이 필요할 수 있다(확인 못 함).

**학습(GPU)**

```bash
lerobot-train --dataset.repo_id=local/handover_v1 --dataset.root=/content/handover_v1 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=/content/out --steps=<정함> --batch_size=<정함> --wandb.enable=false
```

**내려받을 것**: `<output_dir>/checkpoints/last/pretrained_model/` 폴더 전체(config.json, model.safetensors, 전·후처리 json과 safetensors). 이 폴더만 있으면 서버 CPU에서 `from_pretrained`로 추론할 수 있다.

## 11. 아직 모르는 것

1. **GPU/RAM 요구 사항과 예상 학습 시간**: 설치된 패키지에 문서가 없어 확인하지 못했다. 위의 CPU 실측(64×64)은 근거가 되지 않는다. 웹의 LeRobot 문서나 Colab에서 짧게 돌려 보면 알 수 있다.
2. ~~이미지 해상도와 카메라 개수~~ → **3-3에서 확정: 비스듬한 `main` 카메라 1개, 240×320.** 학습이 잘 되는지는 아직 모른다.
3. **옆 카메라(`side`)를 추가할지는 2-6(보조 카메라)에서 따로 정한다.** 아래는 3-1 때의 메모다. 위에서 보면 집게 높이를 알기 어렵다. `scene.xml`은 수정 금지라 카메라를 추가하려면 코드(`MjvCamera` 프리셋)로 해야 한다. 카메라가 2개면 학습 비용도 약 2배가 된다(이미지 처리량 기준 추정). 결정: 사용자.
4. **30fps 녹화**: 렌더링 한 장이 약 54ms라 실시간으로는 안 된다(5절). 녹화 후 다시 그리는 방식이 되는지, 그 이미지가 실시간 화면과 같은지는 3-2에서 확인한다.
5. ~~`observation.state`를 qpos로 할지 ctrl로 할지, `action`이 같은 시점인지~~ → **확정: state = qpos[k][:6], action = 같은 k의 ctrl[k][:6]** (4절).
6. ~~추가 칼럼을 이 버전에서 어떻게 넣는지~~ → **4절에 확인 결과가 있다.** 프레임 단위 feature(`seed`, `is_intervention`)는 된다. 에피소드 단위 메타 API는 못 찾았다. `lerobot-train`이 추가 컬럼 있는 데이터셋으로 도는지와 `source`(문자열)는 확인하지 않았다.
7. **환경 하나로 합치기**: 서버(MuJoCo, numpy 2.5.4)와 lerobot(numpy `<2.3`)이 충돌한다. 시뮬레이션 평가를 어디서 돌릴지(별도 프로세스 + 통신, 또는 numpy를 맞춘 환경) 정해야 한다. mujoco 3.16.0이 numpy 2.2.6에서 동작하는지는 확인하지 않았다.
8. **Hub에서 받아 학습**, `lerobot-eval`, Colab 설치는 실행해 보지 않았다.
9. **실제 학습 양**: 시연 20개(06-experiment.md)로 ACT가 학습되는지, 필요한 스텝 수는 모른다. 에피소드 길이도 아직 모른다(`chunk_size` 100은 3.3초 분량이라 에피소드 길이를 본 뒤 조정).
10. `Handoff.md`의 "LeRobot 현재 버전에서 ACT 학습 명령과 데이터셋 형식 확인" 항목은 이 문서로 대부분 해결됐다(체크는 사용자가 한다).
