# 09. LeRobot 데이터셋 만들기 (3-3)

작성일: 2026-11-03. 이 문서의 숫자는 이 서버(`dev-server-01`, GPU 없음)에서 **직접 실행해서 얻은 값**이다. 추정은 "추정"이라고 표시했다.

`data/raw/`의 시연 에피소드(숫자만 저장, 이미지 없음)를 LeRobot 데이터셋(v3.0)으로 바꿔서 Colab에서 학습할 수 있게 만든다.
이미지는 저장된 `qpos`(블록 포함 13개)로 시뮬레이션을 **다시 그려서** 만든다. 웹캠 영상과 사람 얼굴은 어디에도 없다.

## 순서와 명령

환경이 둘로 나뉜다: `server/.venv`(mujoco, numpy 2.5)와 `train/.venv`(lerobot, numpy 2.2). 그래서 2단계다. 둘 다 `data/raw`는 읽기만 한다.

```bash
cd ~/handover

# 0) 선택·품질 점검 (보고만 한다. 아무것도 버리지 않는다)
train/.venv/bin/python -I train/episode_select.py

# 1) 프레임 렌더링 (server/.venv). nice 19, 렌더 스레드 2개로 시작한다. 끊겨도 다시 실행하면 끝난 에피소드는 건너뛴다
server/.venv/bin/python -m server.render_frames            # 성공 20개 → data/frames/ (PNG 13,667장 + manifest.json)

# 2) 데이터셋 생성 (train/.venv). data/lerobot/dexter_pick_place_v1/ 이 이미 있으면 덮어쓰지 않고 멈춘다
train/.venv/bin/python -I train/build_dataset.py            # 추가로 뺄 에피소드: --exclude 5,12

# 3) 검증
train/.venv/bin/python -I train/verify_dataset.py

# 4) 눈으로 보기: 아무 에피소드나 mp4 + contact sheet로 (/tmp에 저장)
server/.venv/bin/python -m server.replay_episode --episode 7                      # raw 번호. data/raw에서 다시 그림
train/.venv/bin/python -I train/encode_mp4.py dataset --root data/lerobot/dexter_pick_place_v1 --episode 6 \
    --out /tmp/ds_ep6.mp4 --sheet /tmp/ds_ep6.png                                 # 데이터셋 번호(0부터). 데이터셋 안의 영상을 디코딩
```

`-m server...`는 `-I` 없이 실행한다(`-I`는 현재 폴더를 import 경로에서 빼서 `server`를 못 찾는다).
설정값은 `train/dataset_config.py`(데이터셋 이름, task, 인코더, 점검 기준)와 `server/config.py`(`RECORD_*`)에 있다.

## 에피소드 선택 규칙

**학습에 쓰는 것: json의 `success`(내가 누른 값)가 true인 에피소드.** 실패는 제외. 품질 점검에서 문제가 있어도 자동으로 빼지 않고, 뺄 번호는 내가 `--exclude`로 정한다.

| raw 번호 | 시드 | 포함 | 이유 | 틱 |
|---|---|---|---|---|
| 0 | 0 | X | 실패(max_length, 60초 초과 자동 종료) | 1800 |
| 1, 2, 3, 4 | 1–4 | O | 성공 | 1019, 693, 1008, 912 |
| 5 | 6 | O | 성공 | 942 |
| 6, 7, 8 | 7–9 | O | 성공 | 1102, 591, 858 |
| 9 | 10 | X | 실패(max_length) | 1800 |
| 10, 11, 12 | 11, 12, 14 | O | 성공 | 1014, 850, 502 |
| 13 | 15 | X | 실패(stop_fail) | 1216 |
| 14 | 16 | X | 실패(stop_fail) | 959 |
| 15–23 | 17–27 | O | 성공(15, 16, 17, 18, 19, 20, 21, 22, 23번) | 466, 555, 458, 460, 531, 378, 332, 404, 592 |

포함 20개, 제외 4개(0, 9, 13, 14). 총 **13,667 프레임**. 데이터셋 에피소드 번호는 포함된 raw 번호 순서의 0부터다(데이터셋 0 = raw 1, … 데이터셋 6 = raw 7, 데이터셋 19 = raw 23). 번호 ↔ 시드 대응은 `data/frames/manifest.json`의 `episodes`에 있다.

## 품질 점검 (포함된 20개, 보고용)

| 점검 | 결과 |
|---|---|
| ctrl, qpos, sim_time_s의 NaN/inf | 0개 |
| shape (틱 수, qpos 13, ctrl 6) | 이상 없음 |
| 틱 간격(시뮬레이션 시간) | 모두 30/35 ms 격자 안(±1 ms) |
| 수동 결과와 자동 판정이 다른 에피소드 | 0개(20개 모두 자동 판정도 성공) |
| 장면·로봇 모델 해시 | 20개 모두 같음 |
| 길이 | 11.0~36.7초 (ACT 기본 chunk 100 = 3.3초보다 모두 김) |

분포 (최소 / 25% / 중앙 / 75% / 최대):

| 항목 | 최소 | 25% | 중앙 | 75% | 최대 |
|---|---|---|---|---|---|
| 틱 수 | 332 | 464.5 | 591.5 | 919.5 | 1102 |
| 길이(s) | 11.0 | 15.5 | 19.7 | 30.6 | 36.7 |
| 정지 비율 | 0.370 | 0.551 | 0.596 | 0.620 | 0.676 |
| 가장 긴 정지(s) | 0.67 | 0.95 | 1.10 | 1.41 | 3.67 |

3-2의 임시 경고 기준(정지 50%, 3초)을 넘는 것: 정지 비율 18개(16번, 19번만 아래), 가장 긴 정지 2개(1번 3.43초, 6번 3.67초). **기준이 너무 빡빡하다고 판단되어 기본은 전부 포함으로 두었다. 제외 여부는 사람이 정한다.** 정지 비율이 높은 것이 학습에 얼마나 나쁜지는 모른다(학습해 보기 전에는).

## 프레임 대응 (한 칸도 어긋나면 안 된다)

```
샘플 k:  observation.state = qpos[k][:6]   (행동을 적용하기 전 팔 관절 6개, float32)
         action            = ctrl[k][:6]   (같은 순간의 모터 목표, float32)
         이미지            = qpos[k] 13개로 다시 그린 그림 (블록 위치·회전 포함)
```

- fps 30, task 한 문장 고정: `"pick the red block and place it on the green target"`.
- feature: `observation.state`(6), `action`(6), `observation.images.main`(240×320×3, video). 카메라가 위에서 내려다보는 `top`이 아니라 비스듬한 학습 카메라(방위각 110°, 고도 -35°, 거리 0.8 m)라서 키를 `main`으로 했다.
- `timestamp = frame_index / 30`으로 기록된다. 실제 샘플 간격은 30/35 ms가 번갈아 나와서 최대 1.67 ms 어긋난다(`docs/08-recording.md` "시간 격자").
- 테스트: `train/tests/test_dataset_correspondence.py`(가짜 데이터로 build와 검사를 돌리고, 한 칸·세 칸 어긋난 데이터를 검사가 잡는지도 확인), `server/tests/test_render_frames.py`(프레임 k = qpos[k] 렌더), `server/tests/test_episode_select.py`.

## 이미지 저장 방식과 용량

**video(AV1)**로 정했다. 이 서버의 `av`(pyav)에서 `libsvtav1`, `libaom-av1`, `libx264`, `libx265`의 인코더가 모두 열린다. 디코딩은 torchcodec이 시스템 FFmpeg가 없어 못 불러오고 lerobot이 자동으로 pyav로 바꾼다(경고만 나오고 학습은 된다).

**프레임 중간 형식은 PNG(무손실).** 블록이 13~15 px로 작아서 JPEG → AV1 이중 손실 압축을 피했다. 에피소드 7(591장) 실측:

| 형식 | 합계 | 장당 |
|---|---|---|
| PNG | 23.21 MB | 39.3 KB |
| JPEG q95 | 13.61 MB | 23.0 KB |

전체 `data/frames`는 PNG 13,667장 **536 MB**(실측). JPEG였다면 약 314 MB(추정: 장당 23 KB × 13,667).

**AV1 설정은 기본값(libsvtav1, yuv420p, crf 30)에서 바꿨다.** 이유(실측):

1. 블록 픽셀(빨강)에서 영상으로 디코딩한 이미지와 PNG의 평균 절대 오차(0~255)를 쟀다. 처음 만든 libsvtav1 + yuv420p는 crf를 30 → 18 → 10으로 낮춰도 9.6 → 8.8 → 8.4로 거의 안 줄었다.
2. 코덱 없이 RGB → 4:2:0(색을 2×2 픽셀로 줄임) → RGB만 거쳐도 블록 픽셀 오차가 평균 7.5(최대 10.9, 9프레임)였다. 즉 **오차의 대부분은 압축이 아니라 yuv420p 색 축소**다.
3. `libsvtav1`은 yuv444p를 열지 못한다(`avcodec_open2` 실패). 에피소드 21(332프레임, crf 18) 비교:

| 방식 | 용량 | 블록 오차 평균/최대 | 인코딩 |
|---|---|---|---|
| libsvtav1 yuv420p | 2.45 MB | 9.05 / 24.8 | 0.5 s |
| libsvtav1 yuv444p | 실패 | - | - |
| **libaom-av1 yuv444p (선택)** | **1.74 MB** | **2.07 / 6.2** | 6.6 s |
| libx264 yuv444p | 1.95 MB | 4.36 / 13.4 | 0.3 s |
| libx265 yuv444p | 1.46 MB | 4.10 / 15.3 | 1.9 s |

선택: **libaom-av1, yuv444p, crf 18, g 2, cpu-used 8, 스레드 2.** 전체 데이터셋을 이 설정으로 만든 결과(아래 "검증 결과")의 블록 오차는 평균 2.10(최대 3.93), 전체 평균 0.86이다.
이 4:4:4 AV1 영상이 **Colab(torchcodec 또는 pyav)에서 잘 디코딩되는지는 확인하지 못했다.** 이 서버에서는 pyav로 학습까지 돌았다. 안 되면 `use_videos=False`(PNG 이미지 저장, 데이터셋이 수백 MB)가 대안이다.

## 디스크 사용량 (실측)

| 항목 | 크기 | 비고 |
|---|---|---|
| `data/raw` | 2.0 MB | 원본. 지우지 않는다 |
| `data/frames` | **536 MB** | PNG 13,667장. 데이터셋을 만든 뒤에는 학습에 필요 없다 |
| `data/lerobot/dexter_pick_place_v1` | **71.1 MB** | videos 70.11 + data 0.88 + meta 0.11 |
| 데이터셋 tar | 71,106,560 byte | |
| 데이터셋 tar.gz | 61,526,362 byte | 영상이 이미 압축돼 있어 gzip 이득 13% |

**`data/frames`를 지워도 되는가:** 학습과 Colab 이동에는 필요 없다. 다시 필요한 경우는 인코더 설정을 바꿔 데이터셋을 다시 만들 때뿐이고, 그때는 `server.render_frames`로 약 14분(831초) 걸려 다시 그린다. 또 `data/frames`는 영상의 화질을 검증할 때 쓰는 무손실 기준이다(`verify_dataset.py`). 4단계에서 시연을 더 찍어 합칠 계획이면 인코더 설정을 확정한 뒤 지우는 쪽을 권한다. 지울지는 내가 정한다.

## 걸린 시간 (실측)

| 단계 | 시간 |
|---|---|
| 프레임 렌더링 13,667장 | 831초(13.9분), 장당 약 60 ms(PNG 저장 포함. 화면 렌더링만은 약 54 ms), nice 19, CPU 약 1.4코어 |
| 데이터셋 생성(libaom-av1 4:4:4) | 353초 |
| (참고) libsvtav1 4:2:0으로 만든 첫 시도 | 92초, 100.8 MB(crf 18) |
| 검증(`verify_dataset.py`) | 시간은 재지 않았다(수 분 이내) |

## 검증 결과 (실제 출력)

`train/.venv/bin/python -I train/verify_dataset.py` (요약, 학습 카메라 경고 제외):

```
에피소드 수: 20
총 프레임 수: 13667
fps: 30, robot_type: so101_sim, tasks: ['pick the red block and place it on the green target']
features: observation.state float32 (6,), action float32 (6,), observation.images.main video (240, 320, 3), + timestamp/frame_index/episode_index/index/task_index
첫 프레임:
  observation.state = [0.0, -1.74016, 1.20027, -1.20003, -0.0, 0.9]
  action = [0.0, -1.74, 1.2, -1.2, 0.0, 0.9]
  observation.images.main: shape=(3, 240, 320) dtype=torch.float32 min=0.000 max=1.000
  timestamp=0.0000 frame_index=0 episode_index=0 task='pick the red block and place it on the green target'
데이터셋 폴더 크기: 71.09 MB (data/ 0.88, videos/ 70.11, meta/ 0.11)

[상태/행동 정확 일치] 에피소드 20개, 프레임 13667개 -> 통과
[이미지 대응] 샘플 160프레임 중 판정 121개 (이웃 PNG와 바뀐 픽셀이 20개 미만인 정지 장면 39개는 판정에서 뺌): 바뀐 영역에서 같은 k의 PNG가 이웃(k±3)보다 더 가까움 121/121, 어긋남 의심 0개
  디코딩 vs PNG 평균 절대 오차(0~255): 전체 평균 0.861 (최대 1.161), 블록 픽셀 평균 2.095 (최대 3.926, 160프레임)
결과: 통과
```

- 상태/행동은 **13,667프레임 모두 npz의 k번째와 `array_equal`로 정확히 같다**(float32 변환 후).
- 이미지는 손실 압축이라 정확 일치가 아니라 "같은 k의 PNG가 k±3의 PNG보다 가깝다"로 본다. 영상은 한 칸이라도 밀리면 걸린다(테스트에서 어긋난 데이터를 일부러 만들어 확인).
- stats (정규화 통계), `observation.state`: mean `[-0.0683, -0.1818, 0.4435, 1.0969, -0.0, 0.619]`, std `[0.4665, 0.4234, 0.3347, 0.5782, 0.0001, 0.3717]`. `action` std `[0.4789, 0.4201, 0.3359, 0.5736, 0.0, 0.4763]`.
- **주의: `wrist_roll`(5번째)은 거의 움직이지 않는다.** action은 항상 0(std 0.0), state는 ±0.0008 rad 안에서 흔들린다(std 0.0001). MEAN_STD 정규화에서 state는 이 작은 흔들림이 크게 증폭된다(±0.0008 / 0.0001 = 최대 8). 5스텝 학습은 NaN 없이 돌았지만 장기 학습에서 영향이 있는지는 모른다. 이 관절은 손 조종에 안 쓰이는 것으로 보인다(확인은 하지 않았다).
- 검증 스크립트 규칙을 한 번 고쳤다(기록): 처음엔 "이웃과 전체 평균 차이가 2.0 미만이면 제외"로 했더니 160개 중 157개가 빠져서 검사가 비었다(로봇이 화면에서 작아 전체 평균에 움직임이 묻힘). 지금은 "바뀐 픽셀 영역에서만 비교"한다.

눈으로 확인: 데이터셋 에피소드 6(raw 7번, 591프레임)의 contact sheet(`/tmp/dexter_dataset_ep6_sheet.png`, 16장)를 열어 봤다. 집게가 블록으로 내려가 집고(k≈157~197), 초록 목표 쪽으로 옮겨(k≈236~275) 놓은 뒤, 집게를 열고 팔이 올라가는 흐름(k≈551~590)이 보인다. 16장 샘플이라 들어 올린 높이 같은 세부는 못 봤다. mp4는 이 환경에서 재생해 볼 수 없어 contact sheet로만 확인했고, mp4 파일(`/tmp/dexter_dataset_ep6.mp4`, `/tmp/dexter_episode_000007.mp4`)은 만들어졌다(사용자가 직접 재생해서 확인).

## CPU 5스텝 학습 (실측)

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
/usr/bin/time -v nice -n 19 train/.venv/bin/lerobot-train \
  --dataset.repo_id=local/dexter_pick_place_v1 --dataset.root=data/lerobot/dexter_pick_place_v1 \
  --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --batch_size=8 --num_workers=0 --steps=5 --log_freq=1 --save_freq=5 --wandb.enable=false --output_dir=<저장소 밖 폴더>
```

| 항목 | 값 |
|---|---|
| 결과 | 종료 코드 0, 5스텝 완료, 체크포인트 저장, NaN 없음 |
| 스텝당 시간(`updt_s`) | 1.99(첫 스텝) → 1.69 → 1.60 → 1.59 → 1.59 초 (정상 상태 약 1.6초) |
| 데이터 읽기(`data_s`) | 0.05~0.11 초 (영상 디코딩 포함, num_workers 0) |
| 벽시계 전체 | 13.9 초(시작·모델 생성 포함) |
| **최대 메모리(RSS, 호스트 RAM)** | **1,881,864 KB ≈ 1.88 GB (1.79 GiB)** |
| loss | 85.7 → 74.2 → 58.7 → 43.4 → 38.0 (5스텝이라 의미 없음) |

**메모리 주의:** 위 1.88 GB는 이 서버 CPU 프로세스의 **호스트 RAM 최대값**이다. **Colab GPU의 VRAM이 아니다.** GPU 메모리는 측정하지 못했다(GPU 없음). 파라미터 52M개 기준으로 파라미터+기울기+AdamW 상태만 약 0.8 GB(계산: 51.6M × 4 byte × 4)이고, 활성값(batch 8, 240×320)이 더해진다. **추정: VRAM 약 2~4 GB**(근거 없는 짐작에 가깝다. Colab에서 `nvidia-smi`로 재야 한다). Colab 무료 GPU(T4 16 GB급)에는 충분할 것으로 *보이나* 확인하지 않았다.

**학습 시간 추정(추정이다. GPU에서 재지 않았다):**

- 1 epoch = 13,667 / 8 = 1,708 스텝. lerobot 기본 100,000 스텝은 약 58 epoch.
- CPU 2스레드 1.6 s/스텝에 GPU가 10~30배 빠르다고 **가정**하면 0.05~0.16 s/스텝. 여기에 Colab CPU의 영상 디코딩(이 서버에서는 0.05~0.11 s/스텝, 스레드 2개 기준)이 병목일 수 있어 **0.1~0.2 s/스텝**으로 잡는다.
- 20,000 스텝 ≈ 0.6~1.1 시간, 100,000 스텝 ≈ 2.8~5.6 시간(**추정**). 무료 Colab은 세션 시간 제한이 있어 체크포인트를 자주 저장해야 한다(`--save_freq`).
- 필요한 스텝 수는 모른다(시연 20개로 학습되는지도 미확인).

## Colab으로 옮기기

> 아래는 소스와 도움말로 짠 **절차이며 Colab에서 실행해 보지 않았다.** Hub에는 올리지 않았다(토큰 없음, 명령만 적는다).

**1) 압축본 만들기** (저장소 밖 폴더에):

```bash
cd ~/handover
tar cf /tmp/dexter_pick_place_v1.tar -C data/lerobot dexter_pick_place_v1     # 71.1 MB. 영상이 이미 압축돼 gzip(61.5 MB)과 큰 차이 없음
```

**2) 올리는 방법 비교**

| | Google Drive | Hugging Face Hub (private) |
|---|---|---|
| 준비 | 노트북 브라우저로 tar 업로드(서버 → 노트북 → Drive) | 계정과 **쓰기 토큰**이 필요(내가 직접 만든다. 이 저장소/Claude는 토큰을 다루지 않는다) |
| Colab에서 | Drive 마운트 후 tar 풀기 | `--dataset.repo_id=<사용자>/<이름>`로 바로 받기(**이 방식은 실행해 보지 않았다**) 또는 다운로드 후 root 지정 |
| 장점 | 토큰 없음, 단순. 71 MB라 빠르다 | 버전 관리, 여러 번 재사용, 데이터셋 갱신이 쉬움 |
| 단점 | 매번 수동 업로드 | 공개 여부 실수 위험(반드시 private). 시뮬레이션 화면과 숫자뿐이라 얼굴 이미지는 없다. 한 번 올린 것은 캐시/색인될 수 있다 |
| 권장 | **지금은 이쪽**(71 MB, 한 번) | 4단계에서 데이터셋이 자주 바뀔 때 |

**Drive 방식 (Colab 셀)**

```python
from google.colab import drive
drive.mount('/content/drive')
```
```bash
mkdir -p /content/data && tar xf /content/drive/MyDrive/dexter_pick_place_v1.tar -C /content/data
pip install "lerobot[training]==0.6.1"      # 07 문서 10절. Python 3.12 이상 필요
lerobot-train --dataset.repo_id=local/dexter_pick_place_v1 --dataset.root=/content/data/dexter_pick_place_v1 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=8 --steps=20000 --save_freq=5000 --output_dir=/content/out --wandb.enable=false
```

**Hub 방식 (명령만. 실행 안 함)** — `hf` CLI(huggingface_hub 1.33.0의 `hf upload --help`에서 확인한 옵션):

```bash
train/.venv/bin/hf auth login                      # 토큰은 내가 직접 입력
train/.venv/bin/hf upload <사용자>/dexter_pick_place_v1 data/lerobot/dexter_pick_place_v1 . --repo-type dataset --private
# Colab: --dataset.repo_id=<사용자>/dexter_pick_place_v1 (root 없이). 비공개이므로 Colab에도 로그인이 필요하다
```

학습 명령에는 **`--policy.push_to_hub=false`를 반드시 넣는다**(기본값이 true라서 안 끄면 학습 결과를 Hub에 올리려 한다).

**내려받을 것:** `/content/out/checkpoints/last/pretrained_model/` 폴더 전체(`docs/07-lerobot-notes.md` 10절). 같은 `local/dexter_pick_place_v1` 이름과 `stats.json`이 정규화에 쓰이므로 데이터셋을 다시 만들면 체크포인트와 짝이 달라진다.

## 알려진 한계

- 4단계용 컬럼(seed, is_intervention, source)은 **이번 데이터셋에 넣지 않았다.** 넣는 방법의 확인 결과는 `docs/07-lerobot-notes.md` 4절(프레임 단위 feature는 가능, 에피소드 메타 API는 못 찾음). 에피소드 ↔ 시드 대응은 `data/frames/manifest.json`에 있다.
- 렌더러 인스턴스의 **첫 렌더링만** 이후 렌더링과 몇 픽셀(7/76,800)이 1단계(0~255) 다르다(실측, 깊은 원인 미확인). 렌더링을 한 번에 돌렸으므로 `data/frames`에서는 첫 프레임(raw 1번 k=0)만 해당할 것으로 보인다(다른 프레임과 비교해 보지는 않았다). 테스트는 최대 오차 1까지 허용한다.
- 4:4:4 AV1이 Colab 디코더에서 되는지 미확인(위 참고).
- 시연 20개로 ACT가 학습되는지, 해상도 240×320과 카메라 1개로 충분한지, 정지 구간이 학습을 해치는지 모른다.
- `train/.venv`에 pytest가 없어 `train/tests`는 pytest로 돌리지 못했다. 같은 함수를 `tmp_path` 대신 임시 폴더로 직접 호출해서 8개 모두 통과를 확인했다. pytest 설치 여부는 사용자가 정한다.
- 기존 테스트 2개가 실패한다(이번 작업과 무관, 고치지 않음): `server/tests/test_control_relative.py::test_app_default_mapper_follows_the_config_mode`, `server/tests/test_mapping_relative.py::test_defaults_modes`. 둘 다 `config.MAPPING_MODE == "relative"`를 기대하는데 작업 트리의 `server/config.py`는 `"absolute"`다(HEAD는 `"relative"`, git diff로 확인).
