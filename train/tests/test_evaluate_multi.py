"""체크포인트 여러 개 비교(--checkpoints)의 파일 구조: 가짜 체크포인트 경로 두 개와 가짜 정책으로 확인한다(학습된 모델 불필요)."""

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev  # noqa: E402

HOME_CTRL = np.array([0.0, -1.74, 1.2, -1.2, 0.0, 0.9])


class FakeAct:
    """체크포인트를 읽지 않는 가짜 ACT: 항상 home 자세를 낸다(이미지 불필요, 빠름)."""

    name = "act"
    n_action_steps = 100
    temporal_ensemble = False
    loaded: list = []

    def __init__(self, checkpoint, device="cpu", skip_render=True):
        FakeAct.loaded.append(str(checkpoint))

    def reset(self):
        pass

    def needs_image(self, k):
        return False

    def act(self, k, obs):
        return HOME_CTRL.copy()


@pytest.fixture()
def fake_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "ActPolicy", FakeAct)
    FakeAct.loaded = []
    paths = []
    for step, payload in (("step_004000", b"aaa"), ("step_010000", b"bbbb")):
        d = tmp_path / "ckpts" / step / "pretrained_model"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(payload)
        paths.append(d)
    return paths


def run(tmp_path, *extra):
    return ev.main(["--policy", "act", "--n", "2", "--max-s", "1.0", "--out-root", str(tmp_path / "exp"), *extra])


def test_two_checkpoints_make_subfolders_and_a_comparison_table(tmp_path, fake_checkpoints):
    a, b = fake_checkpoints
    assert run(tmp_path, "--name", "cmp", "--checkpoints", f"{a},{b}") == 0
    root = tmp_path / "exp" / "cmp"
    assert FakeAct.loaded == [str(a), str(b)]  # 순서대로 하나씩 불러서 평가한다
    for label in ("step_004000", "step_010000"):
        assert (root / label / "results.csv").is_file() and (root / label / "summary.json").is_file()
        rows = list(csv.DictReader((root / label / "results.csv").open(encoding="utf-8")))
        assert [r["seed"] for r in rows] == ["1000", "1001"]
        sub = json.loads((root / label / "summary.json").read_text(encoding="utf-8"))
        assert sub["checkpoint"]["path"] in (str(a), str(b)) and sub["summary"]["n"] == 2
    top = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert top["kind"] == "checkpoint_comparison" and [r["name"] for r in top["comparison"]] == ["step_004000", "step_010000"]
    for r in top["comparison"]:
        assert set(r) >= {"name", "checkpoint", "n", "successes", "success_rate", "wilson95", "median_time_s", "failure_counts", "model_safetensors_sha256", "dir"}
        assert r["n"] == 2 and r["successes"] == 0 and r["median_time_s"] is None and r["failure_counts"]["grasp_fail"] == 2
        assert len(r["wilson95"]) == 2
    assert top["comparison"][0]["model_safetensors_sha256"] != top["comparison"][1]["model_safetensors_sha256"]  # 파일 내용이 다르면 해시도 다르다
    assert len(top["comparison_lines"]) == 2 and "step_004000" in top["comparison_lines"][0] and "성공률 0/2" in top["comparison_lines"][0]
    assert not (root / "results.csv").exists()  # 여러 개일 때 상위 폴더에는 결과 표가 없고 하위 폴더에만 있다


def test_single_checkpoint_keeps_the_flat_layout(tmp_path, fake_checkpoints):
    assert run(tmp_path, "--name", "one", "--checkpoint", str(fake_checkpoints[0])) == 0
    root = tmp_path / "exp" / "one"
    assert (root / "results.csv").is_file() and (root / "summary.json").is_file() and not (root / "step_004000").exists()
    assert json.loads((root / "summary.json").read_text(encoding="utf-8"))["summary"]["n"] == 2


def test_argument_errors(tmp_path, fake_checkpoints):
    a, b = fake_checkpoints
    with pytest.raises(SystemExit):  # 둘을 같이 쓸 수 없다
        run(tmp_path, "--checkpoint", str(a), "--checkpoints", f"{a},{b}")
    with pytest.raises(SystemExit):  # 하나만 적은 --checkpoints
        run(tmp_path, "--checkpoints", str(a))
    with pytest.raises(SystemExit):  # 체크포인트 없음
        run(tmp_path)
    with pytest.raises(SystemExit):  # model.safetensors가 없는 폴더
        run(tmp_path, "--checkpoints", f"{a},{tmp_path}")
    assert not (tmp_path / "exp").exists()  # 오류면 아무것도 만들지 않는다


def test_existing_output_folder_is_not_overwritten(tmp_path, fake_checkpoints):
    a, b = fake_checkpoints
    run(tmp_path, "--name", "dup", "--checkpoints", f"{a},{b}")
    with pytest.raises(SystemExit):
        run(tmp_path, "--name", "dup", "--checkpoints", f"{a},{b}")


def test_checkpoint_labels_are_unique_and_readable():
    used: set[str] = set()
    assert ev.checkpoint_label(Path("x/step_004000/pretrained_model"), used) == "step_004000"
    assert ev.checkpoint_label(Path("y/step_004000/pretrained_model"), used) == "step_004000_2"  # 이름이 겹치면 번호를 붙인다
    assert ev.checkpoint_label(Path("z/my_model"), used) == "my_model"
