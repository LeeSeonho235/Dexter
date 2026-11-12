"""AUTO 루프(server/auto.py + SimWorker): 실제 MuJoCo, 실제 소켓(가짜 정책 프로세스).

핵심: 정답 재생 같은 가짜 정책으로 라이브 AUTO 루프의 궤적이 평가(server/eval_env.py rollout)와 비트까지 같아야 한다(exact_start).
"""

import threading
import time

import numpy as np
import pytest
from fake_policy import FakePolicyServer, InProcessPolicy, park_chunk

from server import auto_config, config, policy_ipc as ipc
from server.auto import AutoEpisode, apply_eval_start
from server.eval_env import EvalEnv, ReplayPolicy, rollout
from server.owner import ControlOwner
from server.sim_worker import SimWorker

SEED = 1002


def started_policy() -> InProcessPolicy:
    """이미 떠 있는 가짜 정책(AutoEpisode만 시험할 때)."""
    policy = InProcessPolicy()
    policy.started = True
    return policy


@pytest.fixture
def worker():
    w = SimWorker(
        config.SCENE_PATH, keyframe=config.START_KEYFRAME, camera=config.CAMERA, width_px=config.RENDER_WIDTH_PX,
        height_px=config.RENDER_HEIGHT_PX, frame_fps=config.FRAME_FPS, jpeg_quality=config.JPEG_QUALITY,
        max_catch_up_s=config.MAX_CATCH_UP_S, shadow=config.RENDER_SHADOW, reflection=config.RENDER_REFLECTION, tick_hz=config.SIM_TICK_HZ,
    )
    w.start()
    yield w
    w.stop()


def moving_ctrl(n: int) -> np.ndarray:
    """팔이 천천히 움직이는 정해진 행동열(n, 6): 평가와 AUTO가 같은 입력을 받았을 때 궤적이 같은지 보려는 것."""
    park = park_chunk(1)[0].astype(np.float64)
    t = np.arange(n)[:, None] / 30.0
    return park + 0.3 * np.sin(2 * np.pi * t * np.array([0.3, 0.2, 0.25, 0.15, 0.1, 0.0]))


def chunked_predict(ctrl: np.ndarray, n: int):
    """j번째 요청에 ctrl[j*n:(j+1)*n]를 돌려준다(청크 방식 흉내). 끝나면 마지막 값을 채운다."""

    def predict(state, image, reset, i):
        rows = ctrl[i * n : (i + 1) * n]
        if len(rows) < n:
            rows = np.concatenate([rows, np.tile(ctrl[-1], (n - len(rows), 1))])
        return rows.astype(np.float32)

    return predict


def run_episode(worker, episode: AutoEpisode, timeout_s=60.0):
    assert worker.start_auto(episode) is None
    assert episode.done.wait(timeout_s), "episode did not finish"
    return episode.result


def test_live_auto_matches_eval_rollout_bit_for_bit(tmp_path):
    n, max_s = 20, 4.0
    ctrl = moving_ctrl(int(max_s * 30) + 5).astype(np.float32).astype(np.float64)  # 소켓은 float32로 보낸다
    ref_policy = ReplayPolicy(ctrl)
    with EvalEnv() as env:
        ref = rollout(env, ref_policy, SEED, max_s=max_s, record_trace=True)
        env.reset(SEED)
        k_images = {k: env.observation(with_image=True) for k in (0,)}  # 첫 관측(참고)
    fake = FakePolicyServer(predict=chunked_predict(ctrl, n), n_action_steps=n)
    client = ipc.PolicyClient(fake.launcher, request_timeout_s=5.0)
    client.ensure_started()
    w = SimWorker(
        config.SCENE_PATH, keyframe=config.START_KEYFRAME, camera=config.CAMERA, width_px=config.RENDER_WIDTH_PX,
        height_px=config.RENDER_HEIGHT_PX, frame_fps=config.FRAME_FPS, jpeg_quality=config.JPEG_QUALITY,
        max_catch_up_s=config.MAX_CATCH_UP_S, shadow=config.RENDER_SHADOW, reflection=config.RENDER_REFLECTION, tick_hz=config.SIM_TICK_HZ,
    )
    w.start()
    try:
        w.owner.try_acquire_auto()
        episode = AutoEpisode(SEED, client, max_s=max_s, exact_start=True, record_trace=True)
        result = run_episode(w, episode)
    finally:
        w.stop()
        client.shutdown()
    assert result.outcome == "timeout" and ref.failure != "success"
    assert result.n_samples == ref.n_samples
    assert np.array_equal(result.qpos_trace, ref.trace), "AUTO 라이브 궤적이 평가와 다르다"
    assert np.array_equal(result.ctrl_trace, ref.ctrl_trace)
    # 첫 요청의 관측이 평가의 첫 관측(state float32, 렌더 이미지)과 같다
    first = fake.requests[0]
    assert first["reset"] is True and np.array_equal(first["state"], k_images[0].state) and np.array_equal(first["image"], k_images[0].image)
    assert all(not r["reset"] for r in fake.requests[1:])
    # 행동을 낸 샘플은 n_samples - 1개(마지막 샘플은 판정만), n개마다 새 청크 요청 한 번
    assert result.policy_calls == len(fake.requests) == -(-(ref.n_samples - 1) // n)


def test_exact_start_equals_eval_reset():
    import mujoco

    with EvalEnv() as env:
        env.reset(SEED)
        model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key(config.START_KEYFRAME).id)
        apply_eval_start(model, data, SEED)
        assert np.array_equal(data.qpos, env.data.qpos) and np.array_equal(data.qvel, env.data.qvel)
        assert np.array_equal(data.ctrl, env.data.ctrl) and data.time == env.data.time


def test_timeout_and_sample_count(worker):
    worker.owner.try_acquire_auto()
    result = run_episode(worker, AutoEpisode(SEED, started_policy(), max_s=1.0, exact_start=True))
    assert result.outcome == "timeout" and result.n_samples == 31 and result.time_s == pytest.approx(1.0, abs=0.01)


def test_success_is_detected_with_hold_time(worker, monkeypatch):
    """성공 조건이 1.0초부터 계속 맞으면 0.5초 유지 뒤인 1.5초에 성공으로 끝난다(시각은 시작 기준)."""
    from server import auto as auto_module

    state = {"t": None}
    real_is_success = auto_module.is_success

    def fake_is_success(block_m, gripper_rad, **kw):
        return True

    # 시간 조건은 SuccessHold 앞에서 걸 수 없으니, 첫 판정부터 계속 True로 둔다: 0.5초 유지 뒤(15번째 샘플 근처)에 끝나야 한다.
    monkeypatch.setattr(auto_module, "is_success", fake_is_success)
    worker.owner.try_acquire_auto()
    result = run_episode(worker, AutoEpisode(SEED, started_policy(), max_s=10.0, exact_start=True))
    assert result.outcome == "success"
    assert result.time_to_success_s == pytest.approx(0.5, abs=0.02) and result.time_s == result.time_to_success_s
    assert result.n_samples == 16
    assert real_is_success is not fake_is_success


def test_user_stop_ends_episode_and_holds_last_ctrl(worker):
    worker.owner.try_acquire_auto()
    episode = AutoEpisode(SEED, started_policy(), max_s=30.0, exact_start=True)
    assert worker.start_auto(episode) is None
    deadline = time.monotonic() + 10
    while episode.progress()["sample"] < 20 and time.monotonic() < deadline:
        time.sleep(0.02)
    episode.request_stop()
    assert episode.done.wait(5)
    assert episode.result.outcome == "stopped" and episode.result.time_s < 5.0
    assert np.array_equal(episode.result.final_ctrl, park_chunk(1)[0].astype(np.float64))


def test_policy_death_stops_with_error_and_sim_keeps_running(worker):
    policy = started_policy()
    policy.fail_request_after = 1
    policy.started = True
    worker.owner.try_acquire_auto()
    result = run_episode(worker, AutoEpisode(SEED, policy, max_s=30.0, exact_start=True))
    assert result.outcome == "error" and "끊겼" in result.error
    t0 = worker.sim_time_s
    time.sleep(0.3)
    assert worker.is_alive and worker.sim_time_s > t0  # 시뮬레이션 스레드는 계속 돈다


def test_real_socket_process_death_mid_episode(worker):
    fake = FakePolicyServer(die_after=1, n_action_steps=10)
    client = ipc.PolicyClient(fake.launcher, request_timeout_s=2.0)
    client.ensure_started()
    worker.owner.try_acquire_auto()
    result = run_episode(worker, AutoEpisode(SEED, client, max_s=30.0, exact_start=True))
    assert result.outcome == "error" and result.policy_calls == 1
    assert client.status()["state"] == "off"
    client.shutdown()


def test_slow_policy_is_an_error_not_a_hang(worker):
    fake = FakePolicyServer(delay_s=1.5, n_action_steps=10)
    client = ipc.PolicyClient(fake.launcher, request_timeout_s=0.3)
    client.ensure_started()
    worker.owner.try_acquire_auto()
    t0 = time.monotonic()
    result = run_episode(worker, AutoEpisode(SEED, client, max_s=30.0, exact_start=True))
    assert result.outcome == "error" and "안에 오지 않았습니다" in result.error
    assert time.monotonic() - t0 < 5.0
    client.shutdown()


def test_waiting_for_the_policy_does_not_burst_physics(worker):
    """정책을 기다린 시간(0.25초)을 따라잡느라 한 틱에 시뮬레이션이 크게 뛰면 안 된다."""
    fake = FakePolicyServer(delay_s=0.25, n_action_steps=20)
    client = ipc.PolicyClient(fake.launcher, request_timeout_s=3.0)
    client.ensure_started()
    worker.owner.try_acquire_auto()
    episode = AutoEpisode(SEED, client, max_s=2.0, exact_start=True)
    assert worker.start_auto(episode) is None
    # exact_start는 시작할 때 시뮬레이션 시각을 한 번 옮긴다(가라앉히기 1초). 그 효과를 빼려고 첫 요청이 끝난 뒤부터 잰다.
    while episode.progress()["last_latency_ms"] is None and not episode.done.is_set():
        time.sleep(0.002)
    last = worker.sim_time_s
    max_jump_s = 0.0
    while not episode.done.is_set():
        now = worker.sim_time_s
        max_jump_s = max(max_jump_s, abs(now - last)) if now > last else max_jump_s
        last = now
        time.sleep(0.002)
    assert episode.result.outcome == "timeout" and len(episode.result.requests) == 3
    assert all(r["stall_ms"] >= 250 for r in episode.result.requests)
    assert max_jump_s < 0.12, f"sim time jumped {max_jump_s:.3f}s in one tick (burst)"
    client.shutdown()


def test_manual_ctrl_is_not_applied_while_auto_owns_the_robot(worker):
    """AUTO 중에 ControlWorker가 set_ctrl을 불러도 ctrl이 덮어써지지 않는다(ctrl을 쓰는 주체는 하나)."""
    worker.owner.try_acquire_auto()
    episode = AutoEpisode(SEED, started_policy(), max_s=30.0, exact_start=True)
    assert worker.start_auto(episode) is None
    deadline = time.monotonic() + 10
    while episode.progress()["sample"] < 5 and time.monotonic() < deadline:
        time.sleep(0.01)
    other = np.array([0.5, -0.5, 0.5, -0.5, 0.5, 0.0])
    worker.set_ctrl(other)
    time.sleep(0.3)  # 샘플 사이(33ms)에 여러 틱이 지나도 other가 data.ctrl에 들어가면 안 된다
    episode.request_stop()
    assert episode.done.wait(5)
    assert np.array_equal(episode.result.final_ctrl, park_chunk(1)[0].astype(np.float64))


def test_second_episode_is_refused_while_one_runs(worker):
    worker.owner.try_acquire_auto()
    first = AutoEpisode(SEED, started_policy(), max_s=30.0, exact_start=True)
    assert worker.start_auto(first) is None
    assert worker.start_auto(AutoEpisode(SEED, started_policy())) is not None
    first.request_stop()
    assert first.done.wait(5)


def test_state_renderer_is_created_once_and_in_sim_thread(worker):
    worker.owner.try_acquire_auto()
    run_episode(worker, AutoEpisode(SEED, started_policy(), max_s=0.5, exact_start=True))
    renderer = worker._state_renderer
    assert renderer is not None
    worker.owner.release_auto()
    worker.owner.try_acquire_auto()
    run_episode(worker, AutoEpisode(SEED, started_policy(), max_s=0.5, exact_start=True))
    assert worker._state_renderer is renderer
