"""평가용 ACT 어댑터(train/evaluate.py): 렌더링 건너뛰기가 행동에 영향이 없는지, reset, 학습 데이터셋과 같은 관측인지.

작은 무작위 ACT 체크포인트를 테스트 안에서 만들어 쓴다(큰 체크포인트에 의존하지 않는다). 시뮬레이션은 실제 EvalEnv와 렌더러를 쓴다.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))
import dataset_config as dcfg  # noqa: E402
import evaluate as ev  # noqa: E402
from build_dataset import state_action_for  # noqa: E402
from episode_select import load_arrays, load_metas, select  # noqa: E402

from server import config  # noqa: E402
from server.eval_env import EvalEnv, make_observation, rollout  # noqa: E402

N_ACTION_STEPS = 10


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    import torch

    torch.manual_seed(0)
    cfg = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            dcfg.IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        pretrained_backbone_weights=None, chunk_size=N_ACTION_STEPS, n_action_steps=N_ACTION_STEPS, dim_model=32, n_heads=2,
        dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1, latent_dim=8, n_vae_encoder_layers=1, device="cpu", push_to_hub=False,
    )
    policy = ACTPolicy(cfg)
    stats = {
        "observation.state": {"mean": np.zeros(6), "std": np.ones(6), "min": -np.ones(6), "max": np.ones(6)},
        "action": {"mean": np.array([0.0, -1.0, 1.0, -1.0, 0.0, 0.5]), "std": np.full(6, 0.3), "min": -np.ones(6), "max": np.ones(6)},
        dcfg.IMAGE_KEY: {"mean": np.full((3, 1, 1), 0.5), "std": np.full((3, 1, 1), 0.25), "min": np.zeros((3, 1, 1)), "max": np.ones((3, 1, 1))},
    }
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    out = tmp_path_factory.mktemp("tiny_act")
    policy.save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)
    return out


@pytest.fixture(scope="module")
def env():
    with EvalEnv() as e:
        yield e


def test_needs_image_schedule(tiny_checkpoint):
    skip = ev.ActPolicy(tiny_checkpoint, skip_render=True)
    every = ev.ActPolicy(tiny_checkpoint, skip_render=False)
    assert skip.n_action_steps == N_ACTION_STEPS and not skip.temporal_ensemble
    assert [k for k in range(35) if skip.needs_image(k)] == [0, 10, 20, 30]
    assert all(every.needs_image(k) for k in range(35))


def test_skipping_renders_does_not_change_actions(tiny_checkpoint, env):
    """k % n_action_steps == 0일 때만 렌더링·추론해도 행동이 매 샘플 렌더링했을 때와 완전히 같다(select_action은 큐가 빌 때만 관측을 읽는다)."""
    n_samples = 195
    skip = ev.ActPolicy(tiny_checkpoint, skip_render=True)
    every = ev.ActPolicy(tiny_checkpoint, skip_render=False)
    a = rollout(env, skip, 1000, max_samples=n_samples, record_trace=True, stop_on_success=False)
    b = rollout(env, every, 1000, max_samples=n_samples, record_trace=True, stop_on_success=False)
    assert a.policy_calls == 20 and b.policy_calls == n_samples  # 렌더링 횟수: 20번 대 매 샘플
    np.testing.assert_array_equal(a.ctrl_trace, b.ctrl_trace)  # 행동이 비트까지 같다
    np.testing.assert_array_equal(a.trace, b.trace)  # 시뮬레이션 결과도 같다
    assert a.ctrl_trace.std(axis=0).max() > 0 or np.ptp(a.ctrl_trace, axis=0).max() > 0  # 행동이 상수가 아니라서 의미 있는 비교다
    assert a.render_s < b.render_s / 5  # 건너뛴 쪽이 훨씬 빠르다


def test_reset_makes_episodes_independent(tiny_checkpoint, env):
    policy = ev.ActPolicy(tiny_checkpoint, skip_render=True)
    first = rollout(env, policy, 1001, max_samples=25, record_trace=True, stop_on_success=False)
    rollout(env, policy, 1002, max_samples=13, stop_on_success=False)  # 큐가 중간에 남은 채로 끝나는 에피소드
    again = rollout(env, policy, 1001, max_samples=25, record_trace=True, stop_on_success=False)
    np.testing.assert_array_equal(first.ctrl_trace, again.ctrl_trace)  # policy.reset()이 큐를 비웠다


def test_action_is_float_six_numbers_in_ctrl_units(tiny_checkpoint, env):
    policy = ev.ActPolicy(tiny_checkpoint)
    env.reset(1000)
    policy.reset()
    action = policy.act(0, env.observation())
    assert action.shape == (6,) and action.dtype == np.float64 and np.isfinite(action).all()
    # 후처리가 정규화를 되돌린다: 통계 평균 [0, -1, 1, -1, 0, 0.5], 표준편차 0.3 근처의 값
    assert np.abs(action - np.array([0.0, -1.0, 1.0, -1.0, 0.0, 0.5])).max() < 3.0


def test_first_sample_requires_an_image(tiny_checkpoint, env):
    policy = ev.ActPolicy(tiny_checkpoint)
    env.reset(1000)
    policy.reset()
    with pytest.raises(RuntimeError):
        policy.act(0, env.observation(with_image=False))


# --- 학습 데이터셋과 같은 관측 ---

DATASET_OK = (dcfg.DATASET_ROOT / "meta" / "info.json").exists()
RAW_OK = (config.RECORD_DIR / "episode_000019.npz").exists()


@pytest.mark.skipif(not (DATASET_OK and RAW_OK), reason="data/lerobot 또는 data/raw가 이 컴퓨터에 없음")
def test_eval_observation_equals_the_lerobot_dataset_frame_for_the_same_qpos():
    """같은 qpos에서 평가 관측(state, image)이 LeRobotDataset k번째 프레임과 같다: state는 정확히, image는 영상 압축 오차 안(평균 3/255 이하)."""
    from verify_dataset import decoded_array, load_dataset

    ds = load_dataset(dcfg.DATASET_ROOT)
    indices = [r["index"] for r in select(load_metas(config.RECORD_DIR)) if r["included"]]
    e = indices.index(19)
    table = ds.hf_dataset.with_format("numpy")
    ep = np.asarray(table["episode_index"]).reshape(-1)
    start = int(np.argmax(ep == e))
    arrays = load_arrays(config.RECORD_DIR, 19)
    exp_state, _ = state_action_for(arrays)
    with EvalEnv() as env:
        env.renderer.render(arrays["qpos"][0])  # 렌더러 첫 호출 특성 제거
        for k in (0, 150, 300, 450):
            obs = make_observation(env.renderer, arrays["qpos"][k])
            item = ds[start + k]
            np.testing.assert_array_equal(obs.state, item["observation.state"].numpy())
            np.testing.assert_array_equal(obs.state, exp_state[k])
            mae = float(np.abs(obs.image.astype(np.int16) - decoded_array(ds, start + k).astype(np.int16)).mean())
            assert mae < 3.0, f"k={k}: 이미지 평균 오차 {mae:.2f}"
