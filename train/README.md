# train/

LeRobot(ACT) 학습·확인용 폴더. 이 서버에는 GPU가 없어서 여기서는 데이터셋 만들기와 CPU 동작 확인만 한다. 실제 학습은 Colab 같은 외부 GPU에서 한다.

- 가상환경: `train/.venv` (서버용 `server/.venv`와 분리. git에는 올리지 않음)
- 자세한 내용(버전, feature 설계, 명령): [docs/07-lerobot-notes.md](../docs/07-lerobot-notes.md)

## 동작 확인 (CPU, 합성 데이터)

```bash
cd ~/handover
SP=<저장소 밖 임시 폴더>   # root는 아직 없는 폴더여야 한다
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
train/.venv/bin/python -I train/smoke_test.py make $SP/ds
train/.venv/bin/lerobot-train --dataset.repo_id=local/handover_smoke --dataset.root=$SP/ds \
  --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --batch_size=2 --num_workers=0 --steps=5 --log_freq=1 --save_freq=5 --wandb.enable=false --output_dir=$SP/out
train/.venv/bin/python -I train/smoke_test.py infer $SP/ds $SP/out/checkpoints/last/pretrained_model
```

`--policy.push_to_hub=false`를 빼면 학습이 Hugging Face Hub에 올리려고 한다(기본값 true).
