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
7. ~~환경 하나로 합치기~~ → **3-4에서 해결: `train/.venv`에 mujoco 3.16.0을 설치했다**(12절). mujoco 3.16.0은 numpy 2.2.6에서 동작하고, 서버 환경과 결과가 비트 단위로 같다.
8. **Hub에서 받아 학습**, `lerobot-eval`, Colab 설치는 실행해 보지 않았다.
9. **실제 학습 양**: 시연 20개(06-experiment.md)로 ACT가 학습되는지, 필요한 스텝 수는 모른다. 에피소드 길이도 아직 모른다(`chunk_size` 100은 3.3초 분량이라 에피소드 길이를 본 뒤 조정).
10. `Handoff.md`의 "LeRobot 현재 버전에서 ACT 학습 명령과 데이터셋 형식 확인" 항목은 이 문서로 대부분 해결됐다(체크는 사용자가 한다).

## 12. 평가 (3-4)

학습된 ACT 정책을 이 서버(CPU) 시뮬레이션에서 돌려 성공률을 잰다. Colab 학습은 `docs/10-colab.md`, 결과 폴더 형식은 `experiments/README.md`.
**아래 숫자는 이 서버에서 직접 실행해 얻은 값이다. 학습된 정책으로는 아직 평가하지 않았다(체크포인트가 없다). 기저선(아무것도 안 하기, 5스텝 학습 ACT)만 돌려 봤다.**

### 환경 결정: train/.venv에 mujoco 설치 (a안)

| | 값 |
|---|---|
| 설치 | `train/.venv`에 `mujoco==3.16.0`(서버와 같은 버전)과 의존성(PyOpenGL, glfw, absl-py, etils, websockets, zipp), 테스트용 `pytest==9.2.0`(pluggy, iniconfig, Pygments) |
| 크기 | `train/.venv` 1,765 MB → 1,897 MB (**+132 MB**) |
| numpy | **2.2.6 그대로**(`pip freeze` 전후 비교: 추가만 있고 변경·삭제 없음), `pip check` 이상 없음 |
| 서버 환경과의 일치 | 같은 시뮬레이션(400스텝)과 렌더링을 server/.venv(numpy 2.5.4)와 train/.venv(numpy 2.2.6)에서 돌려 **qpos 해시와 이미지 해시가 같다**. 정답 재생 20개 결과(JSON 전체)도 두 환경에서 같다 |

(b안(두 프로세스로 분리)은 쓰지 않았다. 동기화 코드가 늘고 한 칸 어긋날 위험이 있어서다.)

### 파일과 사용

| 파일 | 하는 일 |
|---|---|
| `server/eval_config.py` | 평가 설정값(최대 시간 45초, 시작 자세, 변화량 제한 기본 끔, 실패 유형 임계값). `server/config.py`는 건드리지 않았다 |
| `server/eval_env.py` | 환경과 롤아웃(mujoco와 server 모듈만 씀, 두 venv 모두 동작) + 아무것도 안 하는/정답 재생 정책 |
| `server/eval_metrics.py` | Wilson 구간, 실패 유형 판정, 요약, CSV/JSON 쓰기 |
| `server/eval_replay.py` | 정답 재생 검사(아래) |
| `train/evaluate.py` | ACT 어댑터와 명령줄 (`--policy hold|act`) |

```bash
cd ~/handover
train/.venv/bin/python train/evaluate.py --policy act --checkpoint checkpoints/dexter_act_v1/pretrained_model --name act_v1_demo20   # 50개 시드
train/.venv/bin/python train/evaluate.py --policy hold --n 5 --name baseline_hold_1000-1004                                         # 기저선
server/.venv/bin/python -m server.eval_replay                                                                                         # 정답 재생 검사(`-I` 없이)
```

### 평가 루프 (녹화와 같은 규칙)

샘플 k마다(시뮬레이션 시간 30 Hz 격자, `server/grid.py`) ① 관측: `state = qpos[:6]`(float32), 이미지 = `StateRenderer.render(qpos 13개)` → ② 정책 → ③ 행동 6개를 `ctrl`에 그대로(녹화 때 `action = ctrl`) → ④ 격자가 정한 6 또는 7 물리 스텝 진행.

- 성공 판정은 `server/success.py`의 `is_success` + `SuccessHold`(0.5초 유지). 유지가 채워진 샘플에서 끝낸다.
- **ACT는 `n_action_steps`(기본 100)마다만 관측을 읽는다**(`select_action`은 행동 대기열이 비었을 때만 `predict_action_chunk`를 부른다, 소스로 확인). 그래서 `k % n_action_steps == 0`일 때만 렌더링·추론한다. 매 샘플 렌더링했을 때와 **행동이 비트까지 같다는 것을 테스트로 확인했다**(`train/tests/test_evaluate_act.py`, 정책 실행을 두 방식으로 195샘플 돌려 `ctrl` 전체를 비교). 시간 앙상블(`n_action_steps=1`)을 쓰면 매 샘플 읽으므로 이 최적화가 안 된다.
- 에피소드마다 `policy.reset()`, 블록은 평가 시드 1000~1049(`EVAL_SEED_RANGE`, 50개)의 위치에 회전 없이 놓고 목표는 고정.
- **최대 시간 45초**: 시연 20개를 찍은 **뒤에** 정한 값이다(06 문서는 "기준은 녹화 전에 고정"). 근거(성공 시연): 길이 중앙값 19.7초, 최대 36.7초, 첫 성공까지 중앙값 17.8초, 최대 34.0초. 45초 = 가장 느린 성공의 1.3배. "중앙값의 3배"는 59.0초로 녹화 최대 60초와 사실상 같아서 쓰지 않았다.
- **시작 자세 home**: 시연 20개 중 16개가 home에서 시작했다(첫 샘플이 평가 초기 상태와 팔 1e-3 rad, 블록 0.5 mm 이내로 일치, 테스트로 확인). **나머지 4개(raw 4, 5, 7, 15번)는 작업 상자 쪽 다른 자세에서 시작했다**(한계: 학습 데이터의 20%는 평가와 시작 자세가 다르다).
- **변화량 제한 기본 끔**(`--limit-step`으로 켬): 성공 시연 20개에서 샘플 사이 `ctrl` 변화가 팔 최대 0.248 rad, 집게 최대 0.715 rad이고, 에피소드별 최대 8.2%의 샘플이 서버의 제한 0.08 rad/틱을 넘었다(제어 스레드가 손 메시지마다 추가로 돌아서로 보이나 원인은 소스로만 추정). 샘플당 0.08 rad로 제한하면 시연보다 빡빡해진다. 켜면 시연에서 관찰한 최대값을 올림한 느슨한 값(팔 0.25, 집게 0.72 rad/샘플)으로 자르고 제한이 걸린 샘플 비율을 결과에 기록한다. `limit_step`은 `server/ik.py`의 `limit_joint_step`과 같은 계산이다(ik.py는 mink가 필요해 train/.venv에서 못 쓴다. 무작위 입력으로 동일함을 테스트로 확인).
- 알려진 차이: 학습 이미지는 AV1을 거친 것(블록 픽셀 오차 평균 2.1/255)이고 평가 이미지는 무손실 렌더링이다.
- 렌더러 인스턴스의 첫 렌더링만 몇 픽셀이 1단계 달라서(3-3 확인) 첫 에피소드의 첫 정책 출력이 1e-5~1e-2 달라지고 실행 순서에 따라 결과가 바뀌었다(건너뛰기 비교 테스트가 처음에 실패하며 발견). 그래서 `EvalEnv`가 만들어질 때 버리는 렌더링을 한 번 한다.

### 실패 유형 판정 (위에서부터 먼저 맞는 것, 임계값은 임시)

| 유형 | 규칙 |
|---|---|
| 성공 | 최대 시간 안에 성공 조건 0.5초 유지 |
| 집기 실패 | 블록 높이가 끝까지 0.03 m 미만. 2 cm 넘게 밀렸으면 "밀침" 표시 |
| 시간 초과 | 끝날 때 블록이 공중이거나, 목표 안 바닥인데 판정이 안 됨(집게 안 열림 등) |
| 목표 빗나감 | 한 번 들렸고, 끝날 때 바닥이며 목표 중심에서 0.06 m 이내(성공 영역 밖) |
| 옮기다 떨어뜨림 | 한 번 들렸고, 끝날 때 바닥이며 목표 중심에서 0.06 m 넘게 떨어짐 |
| 기타 | 위에 안 걸림 |

### CPU 속도 (실측, 5스텝 학습 ACT 체크포인트 = 학습된 모델과 같은 구조 52M, 스레드 2, nice 19)

| 항목 | 값 |
|---|---|
| 관측 렌더링(240×320, 그림자·반사 끔) | **약 54 ms/회** (53.9~54.5) |
| ACT 추론(전처리 + 청크 100개 예측 + 후처리) | **약 84~87 ms/회** (첫 호출 포함 평균) |
| 물리(6~7 mj_step) | 샘플당 약 0.08 ms |
| 에피소드 하나(45초 = 1,351샘플, 렌더링·추론 14회) | **벽시계 약 2.0~2.2초** |
| **50개 에피소드** | **102.3초(약 1.7분)** + 준비 2.7초 (실제로 50개를 돌렸다) |
| 비교: 매 샘플 렌더링(`--render-every-sample`) | 에피소드 하나 74.6초 → 50개는 **약 62분**(1개를 재서 계산한 환산값) |

한 샘플당 평균 1.51 ms(= (렌더 + 추론 + 물리) ÷ 샘플)다. 학습이 잘 되면 성공 시점에 일찍 끝나서 더 빠르다.
측정은 모두 실패(5스텝 학습이라 로봇이 안 움직임)하는 에피소드라 45초를 다 돈 경우다. 정책이 실제로 움직이면 접촉 계산 때문에 물리 시간이 조금 늘 수 있다(미측정).

### 기저선 (학습 전 평가 스크립트 확인)

| 정책 | 시드 | 결과 |
|---|---|---|
| 아무것도 안 함(home 자세 유지) | 1000~1004 | **0/5 = 0%** (Wilson 95% 0.0~43.5%), 5개 모두 집기 실패 |
| 5스텝 학습 ACT(거의 무작위) | 1000~1004 | **0/5 = 0%** (Wilson 0.0~43.5%), 5개 모두 집기 실패. 이 체크포인트로 50개 시드 전체도 돌려 **0/50 = 0%**(Wilson 0.0~7.1%)로 속도를 쟀다 |

결과 파일: `experiments/baseline_hold_1000-1004/`, `experiments/baseline_act5step_1000-1004/`.

### 정답 재생 검사: 평가 루프가 녹화와 어긋나 있는지

시연의 `ctrl[k]`를 평가 샘플 k에 그대로 넣고(시연과 같은 시드의 블록 위치, 시연 길이만큼) 결과를 비교한다(`server/eval_replay.py`, 성공 시연 20개).
**결과: 처음 정한 참고 기준(성공 20개 중 15개 이상, 블록 최종 위치 차이 1 cm 이하)에 못 미쳤다. 이 기준은 이후 합격선으로 쓰지 않기로 했다(아래 "결정").**

| 항목 | 값 |
|---|---|
| 재생에서도 성공 | **12/20** (기준 15개 미만) |
| 블록 최종 위치 차이 (최소/25%/중앙/75%/최대) | 0.3 / 3.1 / 16.4 / 68.9 / 194.3 mm, 10 mm 이내 8/20 |
| 시연이 처음 성공한 샘플에서의 블록 차이 | 0.3 / 3.1 / 14.4 / 42.0 / 194.3 mm (보조 지표) |
| 재생 성공 12개의 성공 시각 차이 | 최대 0.2초(대부분 0.1초 이내) |
| 재생 성공 12개의 최종 블록 차이 | 0.3 / 4.8 / 85.9 mm, 10 mm 초과 4개(raw 1, 7, 11, 22) |
| 재생 실패 8개 | raw 5, 6, 8, 10, 12, 16, 20, 23 |
| 팔 관절 궤적 RMS 차이(중앙값) | 0.028 rad (home 시작), 0.018 rad (시연의 첫 자세에서 시작) |

**원인 후보를 하나씩 바꿔 가며 확인한 결과** (진단 스크립트는 저장소에 넣지 않았다. 같은 입력이면 같은 결과: 두 번 돌려 동일, 두 venv에서 동일):

| 후보 | 확인 | 결과 |
|---|---|---|
| 초기 자세(home 시작 vs 시연의 첫 자세에서 시작) | 둘 다 돌림 | 성공 12/12 동일. raw 4, 7, 15는 home에서 시작해도 재생 성공 → **원인 아님** |
| ctrl이 바뀌는 시점(격자 5 ms 어긋남): 샘플 격자 `grid` vs 틱 경계 `floor` vs `ceil` | 세 가지 | 성공 12, 13, 13개로 거의 같고 실패하는 집합이 겹침 → 일부 영향은 있으나 **주된 원인 아님** |
| ctrl을 샘플 시각보다 −7~+2 스텝(5 ms 단위) 옮기기, 에피소드별 최적값 | 에피소드마다 팔 RMS가 최소인 이동을 탐색 | 최적 이동은 **−2~−5 스텝(중앙 −3, 약 15 ms)**: 시연에서 샘플 k의 ctrl이 샘플 시각보다 앞선 틱부터 적용됐다는 가설과 맞음. 팔 RMS는 20~40% 줄지만 **성공 12/20, 블록 최종 차이는 그대로** → 팔 궤적 오차의 한 원인이지만 실패의 원인은 아님 |
| 접촉 물리의 민감성: ctrl에 작은 노이즈 | 1e-9 / 1e-6 / 1e-4 / 1e-3 rad | 1e-9에서는 완전히 같고, **1e-6 rad만으로 raw 20의 결과가 뒤바뀜**(성공 12→13). 접촉 물리는 아주 민감하다. 다만 raw 5, 6, 10, 12, 23은 노이즈 1e-9~1e-3 rad의 모든 변형에서 계속 실패(raw 8, 16은 1e-3에서만 성공으로 바뀜) |
| 집기 직전 팔 위치 오차 | 실패 8개에서 집게를 처음 닫으라는 명령(ctrl 집게 < 0.3)이 나오는 샘플의 집게 끝 위치 차이 | **0.0~1.3 mm** → 실패는 접근 위치 오차 때문으로 보이지 않음(집게를 닫고 들어 올리는 접촉 구간에서 갈라지는 것으로 보임). 주의: 일부 에피소드는 처음에 주먹을 쥐었다 펴서 이 시점이 실제 집기 시점이 아닐 수 있다 |
| 성공 뒤 후퇴 동작 | raw 1: 블록 위치 시간순 비교 | 집어서 옮기는 약 27초 동안 재생과 시연이 **0.2~0.6 mm로 일치**하다가, 집게를 연 뒤 팔이 물러나며 블록을 건드리는 구간에서만 갈라짐(15 mm → 86 mm). **성공 시점이 같아도 "최종" 블록 위치 차이는 크게 나온다**(성공 뒤의 후퇴 동작에 좌우됨) |

**해석(확정이 아니라 증거에서 좁힌 후보):** (1) 평가 루프의 기본 규칙(격자, 관측, ctrl 적용, 물리)은 어긋나 있지 않은 것으로 보인다. 근거: 시작 상태가 시연과 1e-3 rad/0.5 mm 이내로 일치하고, 재생 성공 12개는 성공 시각이 시연과 0.2초 이내이며, raw 1은 27초 동안 블록이 0.6 mm 이내로 일치한다. (2) 남는 차이는 **시연이 30 Hz 샘플로만 남아 있어서 ctrl이 바뀌는 정확한 시각을 모르고(약 15 ms 앞섬이 에피소드마다 −2~−5 스텝으로 달라짐), 집게가 블록을 쥐는 접촉이 민감해서** 열린 고리(재생)에서는 일부 시연이 다른 결과가 나오는 것으로 보인다. 이것이 모든 실패를 설명하는지는 **확인하지 못했다**(raw 5, 6, 10, 12, 23은 어떤 변형에서도 계속 실패하는 이유를 풀지 못했다. 예를 들어 raw 6, 10은 시연에서 집게가 열린 채(ctrl 0.9) 블록이 올라가는 구간이 있는데 그 원인은 확인하지 못했다).
닫힌 고리(정책이 매 청크마다 관측을 다시 보는) 평가에서 이 민감성이 같은 크기로 나타나는지는 학습된 정책이 나온 뒤에야 알 수 있다.

**재생 실패 8개의 유형 분류** (평가와 같은 분류기, `experiments/replay_baseline/results.csv`):

| raw | 시드 | 분류 | 블록이 3 cm 이상 들렸나 | 최대 높이 | 마지막 블록의 목표 중심 거리 |
|---|---|---|---|---|---|
| 5 | 6 | 집기 실패 | 아니오 | 20.1 mm | 156.5 mm |
| 6 | 7 | 집기 실패 | 아니오 | 18.5 mm | 104.7 mm |
| 8 | 9 | 집기 실패 | 아니오 | 21.3 mm | 166.4 mm |
| 10 | 11 | 집기 실패 | 아니오 | 17.0 mm | 188.6 mm |
| 12 | 14 | 집기 실패 | 아니오 | 22.9 mm | 32.6 mm |
| 16 | 18 | 집기 실패 | 아니오 | 24.1 mm | 59.6 mm |
| 20 | 23 | 집기 실패 | 아니오 | 20.8 mm | 30.9 mm |
| 23 | 27 | 집기 실패 | 아니오 | 20.9 mm | 70.2 mm |

**주의(분류기의 한계, 중요):** 같은 열을 **재생에 성공한 12개**로 보면 블록 최대 높이가 16.6~30.6 mm(중앙 약 21 mm)이고 **3 cm 이상 들린 것은 raw 1(30.6 mm) 하나뿐**이다. 즉 성공한 시연들도 블록을 거의 들지 않고 바닥에서 밀거나 아주 낮게 들어 옮겼다. 실패 8개의 최대 높이(17.0~24.1 mm)는 성공 12개의 범위 안에 있어서, **"3 cm 이상 들렸나"로는 실패 8개를 성공과 구분하지 못한다.** 아래 "집기 실패" 분류는 규칙대로의 결과(3 cm 미만)일 뿐 "집게로 집는 데 실패했다"는 증거가 아니다.
성공 조건이 "블록이 목표 안 바닥 + 집게 열림"이라 밀어서도 성공할 수 있다(과제 정의 그대로). 그래서 학습 정책 평가에서도 `grasp_fail`(3 cm 미만)에는 "밀었는데 목표를 빗나감"이 섞인다(`pushed` 표시가 일부만 가른다). 임계값(`LIFT_Z_MIN_M`) 조정이나 유형 정의 변경은 사용자가 정한다(바꾸지 않았다).

8개 모두 규칙상 **"집기 실패(3 cm 미만)"**로 분류됐다. "3 cm 이상 들렸다가 놓는 단계에서 빗나감"은 하나도 없다(raw 1만 3 cm를 넘었고 성공했다). 최대 높이 17~24 mm는 바닥 위 블록 중심(15 mm)보다 2~9 mm 높은 정도라 건드려서 기울거나 밀린 수준이다(3 cm 임계값 미만). 목표에서 멀리 떨어진 것은 블록이 시작 위치 근처에 남았기 때문으로 보인다(위치 이동은 따로 확인하지 않았다. 예외로 raw 12, 20은 목표 31~33 mm로 가깝지만 들리지 않았고 성공 영역 밖이다).

### 결정: 정답 재생은 합격선으로 쓰지 않는다 (사용자 결정)

> "재생 15/20 기준은 사전에 정한 휴리스틱이었고, 열린 고리 재생은 접촉에 민감(1e-6 rad 노이즈로 결과가 바뀜)해서 평가 신뢰도의 합격선으로 쓰지 않는다.
> 신뢰 근거는 **결정성(두 환경 동일), 관측 정확 일치, 집기 직전 위치 0~1.3 mm, raw 1의 27초 궤적 0.2~0.6 mm 일치**다.
> 재생 12/20은 **열린 고리 기준선**으로 학습 정책 성공률과 나란히 보고한다."

- 평가 스크립트(`train/evaluate.py`)는 이 결정에 따라 그대로 쓴다. 위 "미달"은 사실로 남기되 합격/불합격 판정으로 쓰지 않는다(이전 문장 "신뢰할 수 없다"는 철회됐다).
- 기준선은 `experiments/replay_baseline/`(`python -m server.eval_replay --start home --out experiments/replay_baseline`)에 저장했다. **시연 시드 0~99의 열린 고리 재생이며 평가 시드(1000~1049)와 다르다.** 학습 정책의 성공률(닫힌 고리, 평가 시드)과 같은 표에 나란히 적되 조건이 다르다는 점을 같이 쓴다.
- `server/eval_replay.py`는 종료 코드와 출력에서 15/20, 1 cm를 "합격선"이 아니라 "참고 기준"으로만 표시한다.

### 06-experiment.md와 달라지는 점 (06은 수정하지 않았다)

1. 06은 성공 기준의 "일정 시간"을 "녹화 전에 고정"한다고 했는데, **최대 시간(45초)을 시연 20개를 찍은 뒤에 정했다**.
2. 평가 시드는 06이 "50~100개", `EVAL_SEED_RANGE`는 50개(1000~1049)다. 100개로 하려면 `server/config.py`를 고쳐야 한다. 평가 시드가 시연 시드와 겹치지 않는 것은 이미 테스트로 확인된다.
3. 성공 판정에 **0.5초 유지**가 들어 있다(06에는 없음, `docs/08-recording.md` 기준).
4. 실패 유형이 06의 3가지에서 **"시간 초과"가 더해져 4가지(+기타)**다.
5. "학습 시드 3개 평균"은 조건 3개(기본, A, B) × 3 = **Colab 학습 9회**(각 약 1시간 추정)다.
6. 시연의 20%(4개)는 평가와 **다른 자세에서 시작**했다.
7. 평가 이미지(무손실)와 학습 이미지(AV1 4:4:4)에 작은 차이가 있다.
