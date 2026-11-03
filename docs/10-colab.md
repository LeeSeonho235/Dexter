# 10. Colab에서 ACT 학습하기 (3-4)

이 서버에는 GPU가 없어서 학습은 Google Colab(무료 T4 GPU)에서 하고, 학습된 정책의 평가는 이 서버(CPU)에서 한다.
노트북: [train/colab/dexter_train.ipynb](../train/colab/dexter_train.ipynb) (학습만 한다. mujoco는 설치하지 않는다)

> **먼저 읽기:** 이 노트북과 아래 절차는 **Colab에서 실제로 실행해 보지 못한 상태**로 만들었다(이 서버에는 Colab이 없다).
> 노트북의 코드 문법, 설정 기본값, 한 줄 요약 함수, 체크포인트 복사·복원 함수, 학습 명령은 이 서버에서 테스트했다(`train/tests/test_colab_notebook.py`).
> Colab에서 실제로 되는지(AV1 4:4:4 영상 디코딩, 설치, GPU 속도)는 아래 5, 6번 셀의 출력으로 처음 확인된다. 막히면 출력을 그대로 복사해서 알려 달라.

## 준비물

- Google 계정, Drive 여유 공간 약 2 GB(체크포인트 하나가 약 592 MB: 모델 198 MB + 이어하기용 상태 394 MB)
- 이 PC(노트북)의 브라우저. 서버에서 파일을 내려받고 올리는 데 VS Code를 쓴다.
- 데이터셋: `data/lerobot/dexter_pick_place_v1/` (71.1 MB). 만드는 방법은 `docs/09-dataset.md`.

## 내가 직접 해야 하는 단계 (체크리스트)

### A. 서버에서 파일 두 개 준비 (VS Code 터미널)

1. 데이터셋을 tar 한 파일로 묶는다. 저장소 안 `tmp/` 폴더에 만들면 VS Code 탐색기에 보인다(`tmp/`는 git에 안 올라간다).
   ```bash
   cd ~/handover && mkdir -p tmp && tar cf tmp/dexter_dataset_v1.tar -C data/lerobot dexter_pick_place_v1
   ls -l tmp/dexter_dataset_v1.tar          # 약 71 MB (71,106,560 byte)
   ```
2. VS Code 왼쪽 탐색기에서 다음 두 파일을 **마우스 오른쪽 버튼 → Download**로 내 PC에 받는다.
   - `tmp/dexter_dataset_v1.tar`
   - `train/colab/dexter_train.ipynb`

### B. Google Drive에 올리기 (브라우저)

3. [drive.google.com](https://drive.google.com) → **내 드라이브**에 `dexter`라는 새 폴더를 만든다(이름 그대로. 노트북이 `내 드라이브/dexter`를 쓴다).
4. 3번에서 만든 `dexter` 폴더 안으로 `dexter_dataset_v1.tar`를 끌어다 놓아 올린다(71 MB, 1분 안팎).

### C. Colab 열기와 GPU 선택

5. [colab.research.google.com](https://colab.research.google.com) → **파일 → 노트북 업로드** → 2번에서 받은 `dexter_train.ipynb`를 고른다.
6. 메뉴 **런타임 → 런타임 유형 변경** → 하드웨어 가속기에서 **T4 GPU**를 고르고 저장한다. (GPU 선택이 안 보이거나 무료 한도가 찼으면 몇 시간 뒤에 다시 시도한다.)

### D. 셀을 위에서부터 하나씩 실행 (셀 왼쪽 ▶ 버튼 또는 Shift+Enter)

7. **1번 설정 셀:** 그냥 실행한다(바꿀 값은 `STEPS`, `BATCH_SIZE` 등. 기본값은 아래 "학습 설정" 참고).
8. **2번 GPU 확인 셀:** `nvidia-smi` 표에 `Tesla T4`가 보이고 Python 버전이 3.12 이상이어야 한다. "GPU가 없습니다"가 나오면 6번으로 돌아간다. Python이 3.12 미만이라는 오류가 나오면 그 메시지를 그대로 알려 달라(lerobot 0.6.1은 3.12 이상이 필요하다).
9. **3번 설치 셀**(`pip install`): 몇 분 걸린다. 끝나고 "세션을 다시 시작하라"는 메시지가 나오면 **런타임 → 세션 다시 시작**을 누른 뒤 1번 설정 셀부터 3번까지 다시 실행한다(numpy 버전 때문이다. 이런 일이 생기는지는 확인하지 못했다).
10. **버전 확인 셀:** `lerobot 0.6.1`, `CUDA 사용 가능: True`가 보여야 한다.
11. **4번 Drive 셀:** "Google Drive에 연결" 창이 뜨면 허용한다. `['data', 'meta', 'videos']`가 출력되어야 한다.
12. **도우미 함수 셀:** 실행만 한다("도우미 함수 준비 완료").
13. **5번 디코딩 확인 셀:** `에피소드 20 프레임 13667`과 `디코딩 확인 통과`가 나와야 한다. **실패하면** 아래 "디코딩이 안 될 때"를 본다.
14. **6번 100스텝 시험 셀:** 몇 분 안에 끝난다. 마지막에 이런 줄이 나온다.
    ```
    [요약] 스텝당 0.xx초, 20,000스텝 예상 y.y시간, GPU 메모리 최대 z.z GB (측정 스텝 11~100번 평균)
    ```
    **이 줄을 그대로 복사해서 나에게 보내 달라.** 예상 시간과 GPU 메모리를 보고 본 학습을 할지 정한다(오류가 나면 오류 출력을 복사해 보낸다).
15. **7번 본 학습 셀:** 20,000스텝 기준 약 0.7~1.1시간(추정)이다. 화면에 `step:` 줄이 100스텝마다 나온다. **학습 중에는 브라우저 탭을 닫지 말고 컴퓨터가 잠들지 않게 한다.**
    체크포인트는 2,000스텝마다 저장되고 2분마다 Drive의 `dexter/checkpoints/last`로 복사된다.
16. **8번 결과 셀:** 체크포인트 크기가 **약 198 MB**로 나오고 파일 목록이 출력된 뒤, `dexter_act_v1_pretrained_model.tar`가 브라우저로 내려받아진다(이 PC의 다운로드 폴더). 파일이 6~7개이고 합쳐서 198 MB 근처여야 한다.

### E. 서버로 옮기기

17. 이 PC에서 내려받은 `dexter_act_v1_pretrained_model.tar`의 압축을 푼다(탐색기에서 오른쪽 버튼 → 압축 풀기). `pretrained_model`이라는 폴더가 나온다.
18. VS Code 탐색기에서 저장소 루트(`handover`)에 `checkpoints` 폴더를, 그 안에 `dexter_act_v1` 폴더를 만든다(폴더 위에서 오른쪽 버튼 → New Folder). `checkpoints/`는 git에 안 올라간다.
19. 17번의 `pretrained_model` 폴더를 **VS Code 탐색기의 `checkpoints/dexter_act_v1` 위로 끌어다 놓는다**(드래그 앤 드롭). 올리겠냐고 물으면 확인한다.
    - 드래그가 안 되면 tar 파일 자체를 `checkpoints/dexter_act_v1` 위로 끌어다 놓고, 서버 터미널에서 풀어도 된다.
      ```bash
      cd ~/handover/checkpoints/dexter_act_v1 && tar xf dexter_act_v1_pretrained_model.tar && rm dexter_act_v1_pretrained_model.tar
      ```
20. 서버 터미널에서 확인한다.
    ```bash
    cd ~/handover && ls checkpoints/dexter_act_v1/pretrained_model
    du -sh checkpoints/dexter_act_v1/pretrained_model      # 약 198M
    ```
    아래 파일이 있어야 한다: `config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_preprocessor_step_3_normalizer_processor.safetensors`, `policy_postprocessor.json`, `policy_postprocessor_step_0_unnormalizer_processor.safetensors`, `train_config.json`.
21. 평가를 돌린다(서버, `nice`와 스레드 제한이 스크립트 안에 들어 있다).
    ```bash
    cd ~/handover && train/.venv/bin/python train/evaluate.py --policy act --checkpoint checkpoints/dexter_act_v1/pretrained_model --name act_v1_demo20
    ```
    자세한 설명과 결과 읽는 법은 `docs/07-lerobot-notes.md`의 "평가 (3-4)"와 `experiments/README.md`.
    **여러 스텝의 체크포인트를 한 번에 비교**하려면(Drive의 `dexter/checkpoints/step_010000/pretrained_model` 등을 같은 방법으로 `checkpoints/` 아래에 옮긴 뒤)
    `--checkpoints a,b,c`로 부른다(예시와 결과 구조는 `experiments/README.md`). 50개 시드에 체크포인트당 약 2분이다.

## 학습 설정 (노트북 1번 셀 기본값)

| 변수 | 값 | 근거 |
|---|---|---|
| `STEPS` | 20,000 (시험은 `TRIAL_STEPS` 100) | 13,667프레임 ÷ 배치 8 = 1,708스텝/epoch → 약 11.7 epoch. lerobot 기본은 100,000스텝(약 58 epoch)이다. 20,000스텝으로 되는지 보고 늘린다 |
| `BATCH_SIZE` | 8 | lerobot 기본값. 서버 CPU에서 재 본 값(3-3)과 같아 비교하기 쉽다 |
| `SAVE_FREQ` | 2,000 | 약 4~7분마다(추정). 세션이 끊겨도 최대 그만큼만 잃는다 |
| `LOG_FREQ` | 100 | |
| `NUM_WORKERS` | 2 | 무료 Colab은 CPU가 2개 안팎으로 알려져 있다(미확인) |

학습 명령(노트북이 만드는 것. 07 문서 7절과 같고 **`--policy.push_to_hub=false`와 `--wandb.enable=false`를 반드시 넣는다**):

```bash
lerobot-train --dataset.repo_id=local/dexter_pick_place_v1 --dataset.root=/content/data/dexter_pick_place_v1 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=8 --num_workers=2 --steps=20000 --log_freq=100 --save_freq=2000 \
  --wandb.enable=false --output_dir=/content/out
```

**T4 시간 추정(추정이고 Colab에서 재지 않았다):** ACT 한 스텝(배치 8, 240×320)을 계산으로 약 270 GFLOP으로 잡고(ResNet18 약 67 + 인코더 약 68 + 디코더 약 48 + VAE 인코더 약 85), T4 FP32 실효 2.5~3 TFLOPS로 나누면 약 0.1초이다. 데이터 읽기(AV1 디코딩, Colab CPU 2개)를 더해 **0.12~0.2초/스텝**으로 본다. 20,000스텝 ≈ 0.7~1.1시간, 100,000스텝 ≈ 3.3~5.6시간. 이 서버 CPU(스레드 2개)에서 잰 값은 스텝당 약 1.6초(데이터 읽기 0.05~0.11초 별도)다. **6번 셀의 [요약] 줄이 진짜 값이다.**

**GPU 메모리:** 3-3에서 재 둔 1.88 GB는 이 서버의 호스트 RAM이고 GPU 메모리가 아니다. VRAM은 6번 셀이 처음 잰다(추정: 2~4 GB, 근거가 약함). 무료 T4는 16 GB급이다.

## 세션이 끊겼을 때: 이어서 학습하기

무료 Colab은 오래 쓰면 세션이 끊길 수 있다(시간 한도는 계정과 시점마다 다르다). 체크포인트는 Drive에 2분마다 복사되므로 마지막 저장 지점부터 이어서 할 수 있다.

1. 노트북을 다시 열고 **1~4번 셀과 도우미 함수 셀을 다시 실행**한다(설치, Drive 연결, 데이터셋 풀기).
2. 1번 설정 셀에서 `RESUME = True`로 바꾸고 다시 실행한다.
3. **7번 본 학습 셀**을 실행한다. 맨 위에 `이어서 학습: /content/out/checkpoints/<스텝>/pretrained_model/train_config.json`이 나온다.
   - Drive의 `dexter/checkpoints/last`를 `/content/out/checkpoints/<스텝>/`으로 되돌리고 `lerobot-train --resume=true --config_path=...`를 실행한다(소스에서 확인한 방식: `SP/configs/train.py`의 `resume`, `config_path`). **실제로 이어지는지는 Colab에서 해 보지 못했다.** 총 스텝(`STEPS`)은 처음 값이 유지된다.
4. 이어서 한 뒤 8번 셀로 내려받는다.

Drive 폴더 구성: `dexter/dexter_dataset_v1.tar`, `dexter/checkpoints/last/`(가장 최근, `last.step`에 스텝 번호), `dexter/checkpoints/step_010000/`(10,000의 배수만 따로 보관), `dexter/train.log`. 20,000스텝이면 약 1.8 GB다.

## 평가 때 불러오는 경로 (정확히)

학습이 만드는 구조(3-1/3-3에서 직접 돌려 확인한 구조):

```
<output_dir>/checkpoints/<스텝>/pretrained_model/   ← 평가에는 이 폴더만 필요(198 MB)
    config.json
    model.safetensors                                   (약 198 MB)
    policy_preprocessor.json
    policy_preprocessor_step_3_normalizer_processor.safetensors      ← 학습 데이터의 stats(정규화 통계)
    policy_postprocessor.json
    policy_postprocessor_step_0_unnormalizer_processor.safetensors
    train_config.json
<output_dir>/checkpoints/<스텝>/training_state/      ← 이어서 학습할 때만 필요(394 MB)
<output_dir>/checkpoints/last -> <스텝>               ← 가장 최근을 가리키는 링크
```

서버에서는 `checkpoints/dexter_act_v1/pretrained_model`을 쓴다: `--checkpoint checkpoints/dexter_act_v1/pretrained_model`.
정규화 통계가 체크포인트 안에 들어 있어서 평가는 데이터셋 없이 돈다. 단, 데이터셋을 다시 만들면(stats가 달라짐) 이전 체크포인트와 짝이 맞지 않는다.

## 디코딩이 안 될 때 (13번 셀 실패)

13번 셀이 실패하면(예: AV1 4:4:4 영상을 못 연다) 서버에서 데이터셋을 다시 만들어 올린다. `data/frames`가 있어야 한다(없으면 `docs/09`의 1단계로 약 14분).

| 대안 | 명령(서버) | 용량 | 비고 |
|---|---|---|---|
| **PNG 이미지 모드** (가장 안전) | `train/.venv/bin/python -I train/build_dataset.py --image-mode --out data/lerobot/dexter_pick_place_png` | **505.9 MB** (실측, 영상 파일 없이 parquet 안에 이미지) | 영상 디코더가 필요 없고 무손실(오차 0). 올리고 받는 데 시간이 더 걸린다 |
| H.264 4:4:4 | `... build_dataset.py --codec h264 --out data/lerobot/dexter_pick_place_h264` | 약 79 MB(추정: 에피소드 21 실측 비율로 계산) | 널리 디코딩된다. 블록 색 오차 평균 4.4(AV1 4:4:4는 2.1). 이 서버에서 pyav로 만들고 읽는 것은 테스트로 확인했다 |

새 폴더에 만든 뒤(`--out`은 아직 없는 폴더여야 한다) 같은 방법으로 tar를 만들어 올리고, 노트북 1번 셀의 `DATASET_TAR`와 `DATASET_ROOT`를 새 이름으로 바꾼다(`REPO_ID`는 `local/dexter_pick_place_v1` 그대로 쓴다. `meta/info.json`에는 이름이 아니라 구조가 들어 있다).
이미지 모드는 에피소드 20개 전체를 만들어 `verify_dataset.py`로 검증했다(상태/행동 정확 일치, 이미지 오차 0). 이 서버에서 만들 때 127초 걸렸다.

## 자주 막히는 곳

| 증상 | 원인 후보와 할 일 |
|---|---|
| "GPU가 없습니다" | 6번(런타임 유형)을 다시 한다. 무료 GPU는 한도가 있어서 시간을 두고 시도한다 |
| 3번 설치 뒤 numpy 오류 | **세션 다시 시작** 뒤 1번 셀부터 다시 실행한다 |
| `dexter_dataset_v1.tar 가 없습니다` | Drive의 `내 드라이브/dexter/` 안에 파일이 있는지, 폴더 이름이 `dexter`인지 확인한다(1번 셀의 `DRIVE_DIR`) |
| 7번에서 `/content/out 이 이미 있습니다` | 같은 세션에서 이미 한 번 돌렸다. 새로 시작하려면 새 셀에서 `!rm -rf /content/out`, 이어서 하려면 `RESUME = True` |
| GPU 메모리 부족(`CUDA out of memory`) | `BATCH_SIZE`를 4로 줄이고 6번 셀부터 다시 한다(다시 학습해야 한다) |
| 학습이 너무 느림 | 6번 [요약]의 `data_s`가 큰지 본다(AV1 디코딩이 병목일 수 있다). `NUM_WORKERS`를 늘려 보거나 위 "PNG 이미지 모드"로 바꾼다 |
| Drive 용량 부족 | `dexter/checkpoints/step_*` 폴더를 지운다(`last`만 있으면 된다) |

## 하지 않는 것

- 이 노트북은 Hugging Face Hub에 올리지 않고(`--policy.push_to_hub=false`), wandb를 쓰지 않는다(`--wandb.enable=false`). 로그는 노트북 출력과 `train.log` 파일뿐이다.
- 평가용 mujoco는 Colab에 설치하지 않는다. 평가는 서버에서 한다.
