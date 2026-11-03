"""Colab 노트북(train/colab/dexter_train.ipynb) 검증. Colab에서 실제로 실행하는 대신, 구조·코드 문법·설정 기본값·버전 일치와
도우미 함수(요약 한 줄, 체크포인트 복사/복원, 학습 명령 구성)를 이 서버에서 실행해 확인한다.
"""

import ast
import json
import re
from pathlib import Path

import pytest

TRAIN_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = TRAIN_DIR / "colab" / "dexter_train.ipynb"
REQUIREMENTS = TRAIN_DIR / "requirements.txt"

# 실제 5스텝 CPU 학습 로그(3-3)의 형식. updt_s, data_s 값은 그 실행의 값이다.
REAL_LOG = """INFO 2026-11-03 20:19:43 ot_train.py:641 step:1 smpl:8 ep:0 epch:0.00 loss:85.660 grdn:1120.402 lr:1.0e-05 updt_s:1.987 data_s:0.106 smp/s:4 l1_loss:0.823 kld_loss:8.484
INFO 2026-11-03 20:19:44 ot_train.py:641 step:2 smpl:16 ep:0 epch:0.00 loss:74.245 grdn:1146.212 lr:1.0e-05 updt_s:1.691 data_s:0.051 smp/s:5 l1_loss:0.654 kld_loss:7.359
INFO 2026-11-03 20:19:46 ot_train.py:641 step:3 smpl:24 ep:0 epch:0.00 loss:58.743 grdn:869.929 lr:1.0e-05 updt_s:1.601 data_s:0.086 smp/s:5 l1_loss:0.721 kld_loss:5.802
INFO 2026-11-03 20:19:48 ot_train.py:641 step:4 smpl:32 ep:0 epch:0.00 loss:43.364 grdn:785.019 lr:1.0e-05 updt_s:1.588 data_s:0.054 smp/s:5 l1_loss:0.647 kld_loss:4.272
INFO 2026-11-03 20:19:49 ot_train.py:641 step:5 smpl:40 ep:0 epch:0.00 loss:38.034 grdn:637.355 lr:1.0e-05 updt_s:1.592 data_s:0.050 smp/s:5 l1_loss:0.583 kld_loss:3.745
"""
GPU_CSV = "412 MiB\n2650 MiB\n3200 MiB\n3198 MiB\n"


@pytest.fixture(scope="module")
def nb():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells(nb) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def runnable_source(src: str) -> str:
    """Colab 전용 줄(! 명령, % 매직)을 빼고 파이썬 문법만 검사할 수 있게 한다."""
    return "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%")))


def test_notebook_is_valid_json_with_expected_structure(nb):
    assert nb["nbformat"] == 4 and len(nb["cells"]) >= 12
    assert nb["metadata"]["accelerator"] == "GPU"
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown") and isinstance(c["source"], list)
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None  # 실행 결과를 남기지 않는다


def test_every_code_cell_is_valid_python(nb):
    for i, src in enumerate(code_cells(nb)):
        ast.parse(runnable_source(src), filename=f"cell{i}")


def test_lerobot_version_is_pinned_and_matches_requirements(nb):
    wanted = re.search(r"^lerobot\[training\]==([0-9.]+)", REQUIREMENTS.read_text(encoding="utf-8"), flags=re.M).group(1)
    pip_lines = [line for src in code_cells(nb) for line in src.splitlines() if line.lstrip().startswith("!pip install")]
    assert len(pip_lines) == 1 and f"lerobot[training]=={wanted}" in pip_lines[0]
    assert f"== '{wanted}'" in "\n".join(code_cells(nb))  # 설치 뒤 버전 확인


def test_defaults_match_the_agreed_values(nb):
    settings = next(src for src in code_cells(nb) if "STEPS =" in src and "DRIVE_DIR" in src)
    values = {t.targets[0].id: ast.literal_eval(t.value) for t in ast.parse(settings).body if isinstance(t, ast.Assign) and isinstance(t.value, ast.Constant)}
    assert (values["STEPS"], values["BATCH_SIZE"], values["SAVE_FREQ"], values["LOG_FREQ"], values["NUM_WORKERS"]) == (20_000, 8, 2_000, 100, 2)
    assert values["RESUME"] is False and values["TRIAL_STEPS"] == 100


def test_notebook_never_installs_mujoco_or_uses_wandb_and_hub_push(nb):
    code = "\n".join(code_cells(nb))
    assert "mujoco" not in code.lower()
    assert "--policy.push_to_hub=false" in code and "--wandb.enable=false" in code
    assert "push_to_hub=true" not in code.lower() and "huggingface-cli" not in code and "hf upload" not in code


@pytest.fixture()
def helpers(nb, tmp_path):
    """도우미 함수 셀과 설정 셀을 실행한 네임스페이스(경로는 tmp_path로 바꾼다)."""
    ns: dict = {}
    for src in code_cells(nb):
        if "def summarize_trial" in src or ("STEPS =" in src and "DRIVE_DIR" in src):
            exec(compile(runnable_source(src), "cell", "exec"), ns)
    ns.update(DRIVE_DIR=str(tmp_path / "drive"), OUT_DIR=str(tmp_path / "out"))
    return ns


def test_summary_line_matches_the_requested_format(helpers):
    line = helpers["summarize_trial"](REAL_LOG, GPU_CSV, 20_000)
    # 스텝당 (1.987+0.106 ... )/5 평균 = 1.76초, 20,000스텝 = 9.8시간, GPU 3200 MiB = 3.1 GB
    assert "스텝당 1.76초, 20,000스텝 예상 9.8시간, GPU 메모리 최대 3.1 GB" in line
    assert line.startswith("[요약]")


def test_summary_skips_warmup_steps_when_there_are_enough_steps(helpers):
    warm = "".join(f"step:{i} updt_s:9.0 data_s:1.0\n" for i in range(1, 11))
    steady = "".join(f"step:{i} updt_s:0.10 data_s:0.02\n" for i in range(11, 101))
    assert "스텝당 0.12초" in helpers["summarize_trial"](warm + steady, "", 20_000)


def test_summary_handles_missing_data(helpers):
    assert "찾지 못했습니다" in helpers["summarize_trial"]("no steps here", GPU_CSV, 20_000)
    assert "측정 못 함" in helpers["summarize_trial"](REAL_LOG, "", 20_000)


def make_checkpoint(out: Path, step: str, size: int = 10) -> None:
    d = out / "checkpoints" / step
    (d / "pretrained_model").mkdir(parents=True)
    (d / "training_state").mkdir()
    (d / "pretrained_model" / "model.safetensors").write_bytes(b"m" * size)
    (d / "pretrained_model" / "train_config.json").write_text("{}")
    (d / "training_state" / "optimizer.bin").write_bytes(b"o" * size)


def test_sync_copies_last_and_ten_thousand_multiples_and_skips_unchanged(helpers, tmp_path):
    out, drive = Path(helpers["OUT_DIR"]), tmp_path / "drive" / "checkpoints"
    assert helpers["sync_checkpoints"](out, drive) is None  # 체크포인트가 아직 없다
    for step in ("002000", "010000", "012000"):
        make_checkpoint(out, step)
    (out / "checkpoints" / "last").symlink_to("012000")
    assert helpers["sync_checkpoints"](out, drive) == "012000"
    assert (drive / "last" / "pretrained_model" / "model.safetensors").exists()
    assert (drive / "last.step").read_text() == "012000"
    assert (drive / "step_010000" / "pretrained_model").is_dir()
    assert not (drive / "step_002000").exists()  # 10,000의 배수가 아닌 중간 체크포인트는 last만 남긴다
    stamp = (drive / "last" / "pretrained_model" / "model.safetensors").stat().st_mtime_ns
    helpers["sync_checkpoints"](out, drive)
    assert (drive / "last" / "pretrained_model" / "model.safetensors").stat().st_mtime_ns == stamp  # 다시 복사하지 않는다
    make_checkpoint(out, "014000")
    (out / "checkpoints" / "last").unlink()
    (out / "checkpoints" / "last").symlink_to("014000")
    assert helpers["sync_checkpoints"](out, drive) == "014000" and (drive / "last.step").read_text() == "014000"
    assert not list(drive.glob("*.tmp"))


def test_restore_rebuilds_out_dir_and_returns_the_resume_config(helpers, tmp_path):
    drive_ck = Path(helpers["DRIVE_DIR"]) / "checkpoints"
    make_checkpoint(drive_ck.parent / "src", "006000")
    (drive_ck).mkdir(parents=True)
    import shutil

    shutil.copytree(drive_ck.parent / "src" / "checkpoints" / "006000", drive_ck / "last")
    (drive_ck / "last.step").write_text("006000")
    config_path = helpers["restore_from_drive"]()
    out = Path(helpers["OUT_DIR"])
    assert config_path == str(out / "checkpoints" / "006000" / "pretrained_model" / "train_config.json") and Path(config_path).exists()
    assert (out / "checkpoints" / "last").is_symlink() and (out / "checkpoints" / "last").resolve().name == "006000"


class FakePopen:
    last_cmd: list[str] = []

    def __init__(self, cmd, **kwargs):
        FakePopen.last_cmd = cmd
        self.stdout = iter(["INFO step:1 updt_s:0.1 data_s:0.0\n", "noise\n"])

    def wait(self):
        return 0


def test_training_command_has_required_flags_and_resume_variant(helpers, tmp_path):
    helpers["subprocess"] = type("S", (), {"Popen": FakePopen, "PIPE": -1, "STDOUT": -2})
    helpers["run_train"].__globals__["subprocess"] = helpers["subprocess"]
    helpers["run_train"]("/content/out", 20_000, str(tmp_path / "t.log"), 100, 2_000)
    cmd = FakePopen.last_cmd
    for flag in ("--policy.type=act", "--policy.device=cuda", "--policy.push_to_hub=false", "--wandb.enable=false", "--batch_size=8", "--num_workers=2",
                 "--steps=20000", "--log_freq=100", "--save_freq=2000", "--output_dir=/content/out",
                 "--dataset.repo_id=local/dexter_pick_place_v1", "--dataset.root=/content/data/dexter_pick_place_v1"):
        assert flag in cmd, flag
    helpers["run_train"]("/content/out", 20_000, str(tmp_path / "t.log"), 100, 2_000, resume_config="/content/out/checkpoints/006000/pretrained_model/train_config.json")
    assert FakePopen.last_cmd == ["lerobot-train", "--resume=true", "--config_path=/content/out/checkpoints/006000/pretrained_model/train_config.json"]
    assert "step:1" in (tmp_path / "t.log").read_text()
