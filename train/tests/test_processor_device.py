"""Colab(GPU)에서 학습한 체크포인트는 policy_preprocessor.json의 device_processor가 'cuda'로 저장된다. GPU 없는 서버에서도 CPU로 열려야 한다.

가짜 체크포인트는 test_evaluate_act.py와 같은 작은 무작위 ACT를 /tmp(tmp_path)에 만들고 processor json의 device만 'cuda'로 바꿔 쓴다.
저장된 json이 평가 중에 수정되지 않는 것도 확인한다.
"""

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev  # noqa: E402
from test_evaluate_act import tiny_checkpoint  # noqa: E402,F401  (pytest fixture 재사용)

from server.eval_env import Observation  # noqa: E402


def set_device(folder: Path, device: str) -> None:
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = folder / name
        data = json.loads(path.read_text(encoding="utf-8"))
        for step in data["steps"]:
            if step.get("registry_name") == "device_processor":
                step["config"]["device"] = device
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.fixture()
def cuda_checkpoint(tiny_checkpoint, tmp_path) -> Path:
    """tiny_checkpoint의 복사본에서 두 processor json의 device를 'cuda'로 바꾼 것(Colab에서 학습한 체크포인트와 같은 상태)."""
    dst = tmp_path / "colab_ckpt" / "pretrained_model"
    shutil.copytree(tiny_checkpoint, dst)
    set_device(dst, "cuda")
    return dst


def device_steps(pipeline) -> list[str]:
    return [step.device for step in pipeline.steps if type(step).__name__ == "DeviceProcessorStep"]


def test_override_dict_targets_only_existing_device_steps(cuda_checkpoint, tmp_path):
    pre, post = ev.device_overrides(cuda_checkpoint, "cpu")
    assert pre == {"device_processor": {"device": "cpu"}} and post == {"device_processor": {"device": "cpu"}}
    # device_processor가 없으면 빈 override(쓰이지 않은 키는 lerobot이 오류로 처리하므로 넣지 않는다)
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "policy_preprocessor.json").write_text(json.dumps({"steps": [{"registry_name": "normalizer_processor", "config": {}}]}), encoding="utf-8")
    assert ev.device_overrides(bare, "cpu") == ({}, {})  # 후처리 json이 없는 폴더도 깨지지 않는다


@pytest.mark.skipif(__import__("torch").cuda.is_available(), reason="이 컴퓨터에는 CUDA가 있어서 원래 오류가 재현되지 않는다")
def test_without_overrides_a_cuda_saved_processor_fails_on_a_cpu_machine(cuda_checkpoint):
    """원래 오류의 재현: override 없이 열면 'Requested device cuda but CUDA is not available'."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(cuda_checkpoint))
    with pytest.raises(Exception, match="(?i)cuda"):
        make_pre_post_processors(policy.config, pretrained_path=str(cuda_checkpoint))


def test_cuda_saved_checkpoint_opens_on_cpu_and_runs_select_action(cuda_checkpoint):
    policy = ev.ActPolicy(cuda_checkpoint, "cpu")
    assert device_steps(policy.pre) == ["cpu"] and device_steps(policy.post) == ["cpu"]
    assert next(policy.policy.parameters()).device.type == "cpu"
    obs = Observation(state=np.zeros(6, dtype=np.float32), image=np.full((240, 320, 3), 120, dtype=np.uint8), qpos=np.zeros(13))
    policy.reset()
    action = policy.act(0, obs)
    assert action.shape == (6,) and np.isfinite(action).all()
    # 후처리가 정규화를 되돌린다: 가짜 체크포인트의 action 통계(평균 [0, -1, 1, -1, 0, 0.5], 표준편차 0.3) 근처의 값
    assert np.abs(action - np.array([0.0, -1.0, 1.0, -1.0, 0.0, 0.5])).max() < 3.0


def test_normalizer_statistics_are_restored_from_safetensors(cuda_checkpoint):
    policy = ev.ActPolicy(cuda_checkpoint, "cpu")
    norm = next(s for s in policy.pre.steps if type(s).__name__.startswith("Normalizer"))
    unnorm = next(s for s in policy.post.steps if type(s).__name__.startswith("Unnormalizer"))
    # test_evaluate_act.py의 fixture가 저장한 값: action 평균 [0, -1, 1, -1, 0, 0.5], 표준편차 0.3 / state 평균 0, 표준편차 1
    np.testing.assert_allclose(norm._tensor_stats["action"]["mean"].numpy(), [0.0, -1.0, 1.0, -1.0, 0.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(norm._tensor_stats["action"]["std"].numpy(), np.full(6, 0.3), atol=1e-6)
    np.testing.assert_allclose(norm._tensor_stats["observation.state"]["std"].numpy(), np.ones(6), atol=1e-6)
    np.testing.assert_allclose(unnorm._tensor_stats["action"]["mean"].numpy(), norm._tensor_stats["action"]["mean"].numpy())


def test_saved_json_files_are_not_modified_by_loading(cuda_checkpoint):
    before = {n: (cuda_checkpoint / n).read_bytes() for n in ("policy_preprocessor.json", "policy_postprocessor.json", "config.json")}
    ev.ActPolicy(cuda_checkpoint, "cpu")
    for name, content in before.items():
        assert (cuda_checkpoint / name).read_bytes() == content
    data = json.loads((cuda_checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
    assert [s["config"]["device"] for s in data["steps"] if s["registry_name"] == "device_processor"] == ["cuda"]  # 여전히 'cuda'로 저장돼 있다
