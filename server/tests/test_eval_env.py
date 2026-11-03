"""평가 환경과 롤아웃: 녹화와 같은 격자·관측 규칙, 재현성, 시작 상태, 성공 종료, 시간 제한, 변화량 제한."""

import mujoco
import numpy as np
import pytest
from PIL import Image

from server import config
from server import eval_config as ec
from server.episode_store import EpisodeStore
from server.eval_env import EvalEnv, HoldPolicy, Observation, ReplayPolicy, limit_step, make_observation, rollout
from server.grid import grid_step_index
from server.ik import limit_joint_step
from server.render import StateRenderer
from server.render_frames import frame_path

HAVE_RAW = (config.RECORD_DIR / "episode_000019.npz").exists()
HAVE_FRAMES = frame_path(config.REPO_ROOT / "data" / "frames", 19, 0).exists()


@pytest.fixture(scope="module")
def env():
    with EvalEnv() as e:
        yield e


# --- 변화량 제한: server/ik.py의 함수와 같은 계산(ik.py는 mink가 필요해서 train/.venv에서는 못 쓴다) ---


def test_limit_step_equals_server_ik_limit_joint_step():
    rng = np.random.default_rng(0)
    for _ in range(200):
        cur, goal = rng.uniform(-2, 2, 6), rng.uniform(-2, 2, 6)
        step = float(rng.uniform(0.01, 0.5))
        np.testing.assert_array_equal(limit_step(cur, goal, step), limit_joint_step(cur, goal, step))


# --- 관측: 학습 데이터와 같은 함수·값 ---


def test_observation_state_is_float32_qpos_first_six_and_image_is_state_renderer_output(env):
    env.reset(1000)
    obs = env.observation()
    assert obs.state.dtype == np.float32 and obs.state.shape == (6,)
    np.testing.assert_array_equal(obs.state, env.qpos[:6].astype(np.float32))
    assert obs.image.shape == (config.RECORD_IMAGE_HEIGHT_PX, config.RECORD_IMAGE_WIDTH_PX, 3) and obs.image.dtype == np.uint8
    # 같은 클래스·같은 입력 → 같은 이미지. 렌더러 인스턴스의 첫 렌더링만 몇 픽셀이 1단계 다를 수 있어서(3-3 확인) 최대 오차 1까지 허용한다.
    again = env.renderer.render(env.qpos)
    assert int(np.abs(obs.image.astype(np.int16) - again.astype(np.int16)).max()) <= 1
    np.testing.assert_array_equal(again, env.renderer.render(env.qpos))  # 두 번째부터는 정확히 같다
    assert obs.qpos.shape == (13,)


def test_first_observation_of_a_new_env_equals_later_ones():
    """새 EvalEnv의 첫 관측 이미지가 이후 관측과 정확히 같다(렌더러 워밍업. 안 하면 첫 렌더링만 몇 픽셀 1단계 달라 정책 출력이 달라진다)."""
    with EvalEnv() as fresh:
        fresh.reset(1000)
        first = fresh.observation().image
        np.testing.assert_array_equal(first, fresh.observation().image)


def test_observation_without_image_has_no_image(env):
    env.reset(1000)
    assert env.observation(with_image=False).image is None


@pytest.mark.skipif(not (HAVE_RAW and HAVE_FRAMES), reason="data/raw 또는 data/frames가 이 컴퓨터에 없음")
def test_observation_matches_training_frames_for_the_same_qpos(env):
    """학습 데이터셋 프레임을 만든 PNG(server/render_frames.py)와 평가 관측 이미지가 같은 qpos에서 같다(렌더러 첫 호출 특성으로 최대 오차 1)."""
    arrays = EpisodeStore().load_arrays(19)
    with StateRenderer() as warm:
        warm.render(arrays["qpos"][0])
        for k in (0, 200, 400):
            obs = make_observation(warm, arrays["qpos"][k])
            png = np.asarray(Image.open(frame_path(config.REPO_ROOT / "data" / "frames", 19, k)))
            assert int(np.abs(obs.image.astype(np.int16) - png.astype(np.int16)).max()) <= 1
            np.testing.assert_array_equal(obs.state, arrays["qpos"][k, :6].astype(np.float32))  # train/build_dataset.state_action_for와 같은 변환


# --- 시작 상태 ---


@pytest.mark.skipif(not HAVE_RAW, reason="data/raw가 이 컴퓨터에 없음")
def test_start_state_matches_demos_that_started_at_home(env):
    """home에서 시작한 시연(16개)의 첫 샘플과 평가 초기 상태가 같다: 팔 1e-3 rad, 블록 0.5 mm 이내."""
    store = EpisodeStore()
    checked = 0
    for meta in store.scan():
        if not meta["success"]:
            continue
        q0 = store.load_arrays(meta["episode_index"])["qpos"][0]
        if np.abs(q0[:6] - env.park_ctrl()).max() > 0.01:
            continue  # home이 아닌 자세에서 시작한 시연(raw 4, 5, 7, 15)
        env.reset(meta["seed"])
        assert np.abs(env.qpos[:6] - q0[:6]).max() < 1e-3
        assert np.abs(env.qpos[6:9] - q0[6:9]).max() < 5e-4
        checked += 1
    assert checked == 16


def test_park_ctrl_is_home_with_open_gripper(env):
    park = env.park_ctrl()
    np.testing.assert_allclose(park[:5], env.model.key(config.START_KEYFRAME).ctrl[:5])
    assert park[5] == config.GRIPPER_OPEN_RAD


def test_reset_is_independent_of_earlier_episodes(env):
    env.reset(1003)
    first = env.qpos.copy()
    env.reset(1004)
    for _ in range(50):
        env.step(env.park_ctrl() + 0.3)
    env.reset(1003)
    np.testing.assert_array_equal(env.qpos, first)
    assert env.k == 0 and env.sim_time_s == 0.0


# --- 시간 격자: 샘플 k는 녹화와 같은 물리 스텝에서 찍힌다 ---


def test_sample_times_follow_the_recording_grid(env):
    env.reset(1000)
    dt = env.timestep_s
    for k in range(40):
        assert env.sim_time_s == pytest.approx(grid_step_index(k, config.RECORD_FPS, dt) * dt, abs=1e-9)
        env.step(env.park_ctrl())
    # 간격은 30 ms / 35 ms가 번갈아 나온다(평균 33.3 ms)
    times = []
    env.reset(1000)
    for _ in range(31):
        times.append(env.sim_time_s)
        env.step(env.park_ctrl())
    gaps_ms = np.round(np.diff(times) * 1000).astype(int)
    assert set(gaps_ms) == {30, 35} and times[-1] == pytest.approx(1.0, abs=0.0026)


def test_step_applies_ctrl_to_the_next_interval_only(env):
    env.reset(1000)
    target = env.park_ctrl() + np.array([0.3, 0, 0, 0, 0, 0])
    env.step(target)
    np.testing.assert_array_equal(env.data.ctrl[:6], target)
    with pytest.raises(ValueError):
        env.step(np.array([np.nan] * 6))
    with pytest.raises(ValueError):
        env.step(np.zeros(5))


# --- 롤아웃 ---


def test_hold_policy_fails_with_grasp_fail_and_runs_to_the_time_limit(env):
    res = rollout(env, HoldPolicy(env.park_ctrl()), 1000, max_s=3.0, keep_tail=3, record_trace=True)
    assert not res.success and res.failure == "grasp_fail" and not res.pushed
    assert res.n_samples == 91 and res.trace.shape == (91, 13) and len(res.tail_qpos) == 3
    assert res.policy_calls == 0 and res.render_s < 0.05  # 이미지가 필요 없는 정책은 렌더링하지 않는다
    np.testing.assert_allclose(res.final_block_m, res.block_start_m, atol=1e-4)


def test_rollout_is_deterministic(env):
    a = rollout(env, HoldPolicy(env.park_ctrl()), 1001, max_s=2.0, record_trace=True)
    rollout(env, HoldPolicy(env.park_ctrl()), 1002, max_s=2.0)
    b = rollout(env, HoldPolicy(env.park_ctrl()), 1001, max_s=2.0, record_trace=True)
    np.testing.assert_array_equal(a.trace, b.trace)


class _BlockInTargetEnv(EvalEnv):
    """리셋 뒤 블록을 목표 안(집게는 열림)에 둔다: 성공 종료 경로를 시험하려고."""

    def reset(self, seed, init_arm=None):
        super().reset(seed, init_arm)
        self.data.qpos[6:9] = (*config.TARGET_CENTER_M, config.BLOCK_HALF_M)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)


def test_success_ends_the_episode_when_the_hold_time_is_reached():
    with _BlockInTargetEnv(with_renderer=False) as e:
        res = rollout(e, HoldPolicy(e.park_ctrl()), 1000, max_s=10.0)
    assert res.success and res.failure == "success"
    assert res.time_to_success_s == pytest.approx(config.SUCCESS_HOLD_S, abs=0.04)  # 0.5초 유지
    assert res.n_samples == 16  # 샘플 0..15: 15번째 샘플이 0.5초(100 스텝)이고 거기서 끝난다


def test_time_limit_counts_samples(env):
    res = rollout(env, HoldPolicy(env.park_ctrl()), 1000, max_s=ec.EVAL_MAX_S)
    assert res.n_samples == int(round(ec.EVAL_MAX_S * ec.EVAL_FPS)) + 1 == 1351


class _JumpPolicy:
    name = "jump"

    def reset(self):
        pass

    def needs_image(self, k):
        return False

    def act(self, k, obs: Observation):
        return np.array([0.0, -1.7, 1.2, -1.2, 0.0, 0.9]) + (1.0 if k % 2 else -1.0) * np.array([1, 0, 0, 0, 0, 0])


def test_limit_step_option_clips_and_counts(env):
    free = rollout(env, _JumpPolicy(), 1000, max_s=1.0, record_trace=True)
    lim = rollout(env, _JumpPolicy(), 1000, max_s=1.0, limit=True, record_trace=True)
    assert free.limited_samples == 0 and lim.limited_samples > 0
    steps = np.abs(np.diff(lim.ctrl_trace[:, 0]))
    assert steps.max() <= ec.LIMIT_STEP_ARM_RAD + 1e-9
    assert np.abs(np.diff(free.ctrl_trace[:, 0])).max() > 1.9  # 제한 없으면 그대로 큰 변화


def test_replay_policy_returns_recorded_ctrl_then_holds_the_last():
    ctrl = np.arange(18, dtype=float).reshape(3, 6)
    p = ReplayPolicy(ctrl)
    np.testing.assert_array_equal(p.act(1, None), ctrl[1])
    np.testing.assert_array_equal(p.act(10, None), ctrl[2])
