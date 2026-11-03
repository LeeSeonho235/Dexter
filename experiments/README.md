# experiments/

평가(3-4) 결과 폴더. 숫자(CSV, JSON)와 실패 원인 분석용 시뮬레이션 화면 PNG(옵션)만 둔다. **웹캠 영상이나 사람 얼굴은 저장하지 않는다.**
평가는 `train/evaluate.py`로 돌린다(자세한 설명: `docs/07-lerobot-notes.md`의 "평가 (3-4)", Colab 학습: `docs/10-colab.md`).

```bash
cd ~/handover
train/.venv/bin/python train/evaluate.py --policy act --checkpoint checkpoints/dexter_act_v1/pretrained_model --name act_v1_demo20
train/.venv/bin/python train/evaluate.py --policy act --checkpoint <...> --save-fail-frames 4        # 실패 에피소드 마지막 4장을 PNG로
train/.venv/bin/python train/evaluate.py --policy hold --n 5 --name baseline_hold_1000-1004          # 아무것도 안 하는 기저선
```

## 폴더 구성

```
experiments/<이름>/
    results.csv        에피소드마다 한 줄(시드, 성공, 실패 유형, 성공까지 시간, 블록 최종 위치, 제한 비율, 단계별 시간 …)
    summary.json       정책·체크포인트(model.safetensors의 SHA-256)·평가 설정·버전, 성공률, Wilson 95% 구간, 시간 분포, 실패 유형 개수, 속도
    fail_frames/       --save-fail-frames K 일 때만. seed_<시드>_<실패유형>_<번호>.png (시뮬레이션 화면)
```

이미 있는 폴더에는 덮어쓰지 않는다(`--name`을 바꾼다).

### 체크포인트 여러 개 한 번에 비교 (`--checkpoints a,b,c`)

Colab은 2,000스텝마다 저장하므로 4,000/10,000/20,000스텝 같은 체크포인트를 한 번에 평가해 나란히 볼 수 있다(50개 시드에 체크포인트당 약 2분, 체크포인트를 하나씩 불러서 같은 환경으로 돌린다).

```bash
train/.venv/bin/python train/evaluate.py --policy act --name compare_steps \
  --checkpoints checkpoints/step_004000/pretrained_model,checkpoints/step_010000/pretrained_model,checkpoints/step_020000/pretrained_model
```

```
experiments/compare_steps/
    summary.json            comparison: 체크포인트별 한 줄(이름, 경로, model.safetensors SHA-256, 성공률, Wilson 95% 구간, 성공까지 중앙값, 실패 유형 개수),
                            comparison_lines: 사람이 읽는 한 줄 비교표
    step_004000/            체크포인트별 하위 폴더(이름은 .../step_004000/pretrained_model 이면 step_004000, 겹치면 _2). 아래는 위와 같은 results.csv, summary.json
    step_010000/
    step_020000/
```

`--checkpoint`(하나)로 부르면 위의 단일 구조(하위 폴더 없음) 그대로다. `--checkpoint`와 `--checkpoints`는 같이 쓸 수 없고, `--checkpoints`에는 2개 이상 적는다.

## 평가 조건 (이 값들은 결과와 함께 summary.json에 기록된다)

| 항목 | 값 | 비고 |
|---|---|---|
| 평가 시드 | 1000~1049 (50개) | `server/config.py`의 `EVAL_SEED_RANGE`. 시연 시드(0~99)와 겹치지 않는다. 블록 위치는 시드로, 목표는 고정 |
| 최대 시간 | **45초** (`EVAL_MAX_S`) | **시연 20개를 찍은 뒤에 정한 값이다.** 06 문서는 "기준은 녹화 전에 고정"이라고 했는데 시연 길이를 본 뒤 정했다. 근거: 성공 시연 길이 중앙값 19.7초, 최대 36.7초(첫 성공까지 중앙값 17.8초, 최대 34.0초). 45초 = 가장 느린 성공의 1.3배. 결과를 발표할 때 이 점을 그대로 적는다 |
| 성공 | `server/success.py`의 `is_success` + 0.5초 유지 | 블록이 목표 안(경계 포함 0.03 m) + 바닥(z < 0.02 m) + 집게 열림(≥ 0.5 rad)이 0.5초 이상. 시간 안에 못 채우면 실패 |
| 시작 자세 | home(집게 열림) | 시연 20개 중 4개(raw 4, 5, 7, 15번)는 다른 자세에서 시작했다. 그 4개의 시작 자세는 평가와 다르다(한계) |
| 시간 격자 | 녹화와 같은 30 Hz 격자(`server/grid.py`) | 샘플 사이 6 또는 7 물리 스텝(30/35 ms 교대) |
| 관측 | state = `qpos[:6]`(float32), 이미지 = 녹화 카메라 240×320 렌더링 | 학습 데이터셋을 만든 `StateRenderer`와 같다 |
| 행동 | 정책 출력 6개를 `ctrl`에 그대로 | 변화량 제한은 기본 끔(`--limit-step`으로 켬). 시연의 ctrl이 샘플 사이에 0.08 rad를 자주 넘었기 때문 |

## 실패 유형 판정 (위에서부터 먼저 맞는 것)

| 키 | 이름 | 규칙 |
|---|---|---|
| `success` | 성공 | 시간 안에 성공 판정 유지 |
| `grasp_fail` | 집기 실패 | 블록 높이가 끝까지 0.03 m 미만(안 들림). 블록이 2 cm 넘게 밀렸으면 `pushed` 표시 |
| `timeout` | 시간 초과 | 끝날 때 블록이 공중이거나, 목표 안 바닥에 있는데 판정이 안 됨(집게 안 열림, 유지 부족) |
| `target_miss` | 목표 빗나감 | 한 번 들렸고, 끝날 때 바닥에 있으며 목표 중심에서 0.06 m 이내(성공 영역 밖) |
| `dropped` | 옮기다 떨어뜨림 | 한 번 들렸고, 끝날 때 바닥에 있으며 목표 중심에서 0.06 m 넘게 떨어져 있음 |
| `other` | 기타 | 위에 안 걸림 |

임계값(0.03, 0.02, 0.06 m)은 `server/eval_config.py`에 있고 **임시 값**이다. 실제 실패 사례의 마지막 프레임(`--save-fail-frames`)을 눈으로 보며 맞춘다.
이 유형은 06 문서의 3가지(집기 실패 / 옮기다 떨어뜨림 / 목표 빗나감)에 "시간 초과"를 더한 것이다.

## 읽는 법과 주의

- 성공률은 성공 수 / 시도 수이고 **Wilson 95% 신뢰구간**을 같이 적는다. 50개에서도 구간이 약 ±13%p라서 작은 차이는 우연일 수 있다.
- ACT 추론은 결정적이라(VAE 잠재변수 0) 같은 체크포인트와 시드의 결과는 같다. 결과가 달라지는 것은 **학습 시드**(Colab 학습을 다시 하는 경우) 때문이다. 06 문서의 "학습 시드 3개 평균"은 조건마다 Colab 학습을 3번 하는 것이다.
- 평가 이미지는 무손실 렌더링이고 학습 이미지는 AV1 영상을 거친 것이라 아주 작은 차이(블록 픽셀 오차 평균 2.1/255)가 있다.
- **재생 기준선(`replay_baseline/`)**: 성공 시연 20개의 `ctrl`을 그대로 재생한 **열린 고리 재생이며 시연 시드(0~99)의 블록 위치에서 돌린 것이라 평가 시드(1000~1049)와 다르다.** 재생 성공 12/20이다. 접촉에 민감해서(ctrl에 1e-6 rad 노이즈로 결과가 바뀜) 합격선이 아니라 **열린 고리 기준선**으로, 학습 정책의 성공률(닫힌 고리, 평가 시드)과 나란히 보고한다. 성공하지 못한 8개는 규칙상 모두 "집기 실패(블록 최대 높이 3 cm 미만)"로 분류되지만, 성공한 12개도 최대 높이가 16.6~30.6 mm라 이 분류만으로는 성공과 구분되지 않는다(밀어서 성공한 경우가 많다. `docs/07` 12절). 파일: `results.csv`(에피소드별 시드, 성공, 성공 시각, 블록 최종 차이 mm, 실패 유형, 3 cm 이상 들렸는지, 최대 높이, 목표까지 거리), `summary.json`. 다시 만들려면 `server/.venv/bin/python -m server.eval_replay --start home --out experiments/replay_baseline`(이미 있으면 덮어쓰지 않는다).
- 기저선(`baseline_*`): 아무것도 안 하는 정책과 학습 전 거의 무작위인 ACT(5스텝 학습)로 5개 시드를 돌린 결과. 평가 스크립트가 끝까지 돌고 성공률이 0%에 가깝다는 것을 보여 준다.
