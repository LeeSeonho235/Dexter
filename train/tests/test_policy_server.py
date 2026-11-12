"""정책 프로세스(train/policy_server.py): 청크가 evaluate.ActPolicy(select_action을 하나씩)와 비트까지 같은지, 소켓 프로토콜, 종료 조건.

작은 무작위 체크포인트(테스트 안에서 만듦)와 실제 체크포인트(있을 때)로 확인한다.
"""

import os
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

TRAIN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAIN_DIR.parent
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(REPO_ROOT))
import dataset_config as dcfg  # noqa: E402
import evaluate as ev  # noqa: E402
import policy_server as ps  # noqa: E402

from server import policy_ipc as ipc  # noqa: E402
from server.eval_env import EvalEnv, make_observation  # noqa: E402

N_ACTION_STEPS = 10
REAL_CHECKPOINT = REPO_ROOT / "checkpoints" / "dexter_act_v1" / "pretrained_model"


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

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
def observations():
    """서로 다른 관측 2개(home / 블록과 팔이 다른 자세)."""
    with EvalEnv() as env:
        env.reset(1002)
        first = env.observation()
        qpos = env.qpos.copy()
        qpos[:5] += np.array([0.1, -0.2, 0.15, 0.1, 0.05])
        qpos[6:8] = [0.23, -0.05]
        second = make_observation(env.renderer, qpos)
    return first, second


def reference_actions(checkpoint: Path, obs0, obs1, n: int) -> np.ndarray:
    """evaluate.ActPolicy.act를 샘플마다 부른 결과: 샘플 0(obs0)에서 청크를 예측하고 샘플 n(obs1)에서 다시 예측한다."""
    ref = ev.ActPolicy(checkpoint, "cpu")
    ref.reset()
    rows = []
    for k in range(2 * n):
        obs = obs0 if k < n else obs1
        rows.append(ref.act(k, obs if k % n == 0 else make_stub(obs)))
    return np.array(rows).astype(np.float32)


def make_stub(obs):
    """관측이 필요 없는 샘플: 이미지 없이(None) 넘긴다(ActPolicy가 직전 이미지를 대신 쓴다)."""
    from server.eval_env import Observation

    return Observation(state=obs.state, image=None, qpos=obs.qpos)


@pytest.mark.parametrize("mode", ["predict", "loop"])
def test_chunk_equals_select_action_one_by_one_tiny(tiny_checkpoint, observations, mode):
    obs0, obs1 = observations
    expected = reference_actions(tiny_checkpoint, obs0, obs1, N_ACTION_STEPS)
    policy = ps.load_act_policy_class()(tiny_checkpoint, "cpu", mode)
    assert policy.n_action_steps == N_ACTION_STEPS
    got0 = policy.predict_chunk(obs0.state, obs0.image, reset=True)
    got1 = policy.predict_chunk(obs1.state, obs1.image, reset=False)
    assert got0.shape == (N_ACTION_STEPS, 6) and got0.dtype == np.float32
    assert np.array_equal(np.concatenate([got0, got1]), expected)


@pytest.mark.skipif(not (REAL_CHECKPOINT / "model.safetensors").is_file(), reason="실제 체크포인트 없음")
@pytest.mark.parametrize("mode", ["predict", "loop"])
def test_chunk_equals_select_action_one_by_one_real(observations, mode):
    obs0, obs1 = observations
    ref = ev.ActPolicy(REAL_CHECKPOINT, "cpu")
    n = ref.n_action_steps
    expected = reference_actions(REAL_CHECKPOINT, obs0, obs1, n)
    policy = ps.load_act_policy_class()(REAL_CHECKPOINT, "cpu", mode)
    got = np.concatenate([policy.predict_chunk(obs0.state, obs0.image, reset=True), policy.predict_chunk(obs1.state, obs1.image)])
    assert got.shape == (2 * n, 6)
    assert np.array_equal(got, expected), f"max abs diff {np.abs(got - expected).max():.3e}"


def test_reset_flag_does_not_change_the_chunk(tiny_checkpoint, observations):
    obs0, _ = observations
    policy = ps.load_act_policy_class()(tiny_checkpoint, "cpu", "predict")
    a = policy.predict_chunk(obs0.state, obs0.image, reset=True)
    b = policy.predict_chunk(obs0.state, obs0.image, reset=False)
    assert np.array_equal(a, b)


# --- serve(): 소켓 프로토콜 (가짜 predict) ---


@pytest.fixture
def pair():
    server_sock, client_sock = socket.socketpair()
    yield server_sock, client_sock
    server_sock.close()
    client_sock.close()


def run_serve(server_sock, predict, idle_timeout_s=30.0):
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault("reason", ps.serve(server_sock, predict, idle_timeout_s, {"n_action_steps": 3})), daemon=True)
    thread.start()
    return thread, result


def fake_predict(state, image, reset):
    return np.tile(state, (3, 1)) + np.arange(3, dtype=np.float32)[:, None]


def request(client_sock, seq, state, image, reset=False):
    ipc.send_message(client_sock, ipc.T_REQUEST, ipc.encode_request(seq, reset, state, image))
    return ipc.recv_message(client_sock, 5.0)


def test_serve_ready_request_ping_shutdown(pair):
    server_sock, client_sock = pair
    thread, result = run_serve(server_sock, fake_predict)
    msg_type, payload = ipc.recv_message(client_sock, 5.0)
    assert msg_type == ipc.T_READY and b"n_action_steps" in payload
    state = np.arange(6, dtype=np.float32)
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    msg_type, payload = request(client_sock, 7, state, image, reset=True)
    seq, actions, infer_ms = ipc.decode_chunk(payload)
    assert msg_type == ipc.T_CHUNK and seq == 7 and infer_ms >= 0
    assert np.array_equal(actions, np.tile(state, (3, 1)) + np.arange(3)[:, None])
    ipc.send_message(client_sock, ipc.T_PING)
    assert ipc.recv_message(client_sock, 5.0)[0] == ipc.T_PONG
    ipc.send_message(client_sock, ipc.T_SHUTDOWN)
    assert ipc.recv_message(client_sock, 5.0)[0] == ipc.T_BYE
    thread.join(5)
    assert result["reason"] == "shutdown"


def test_serve_bad_request_returns_error_and_keeps_serving(pair):
    server_sock, client_sock = pair
    thread, result = run_serve(server_sock, fake_predict)
    ipc.recv_message(client_sock, 5.0)
    ipc.send_message(client_sock, ipc.T_REQUEST, b"short")
    assert ipc.recv_message(client_sock, 5.0)[0] == ipc.T_ERROR
    # 이미지 없는 요청도 오류
    assert request(client_sock, 1, np.zeros(6, np.float32), None)[0] == ipc.T_ERROR
    ipc.send_message(client_sock, ipc.T_PING)
    assert ipc.recv_message(client_sock, 5.0)[0] == ipc.T_PONG
    client_sock.close()
    thread.join(5)
    assert result["reason"] == "client_closed"


def test_serve_predict_exception_is_reported_not_fatal(pair):
    server_sock, client_sock = pair

    def boom(state, image, reset):
        raise RuntimeError("boom")

    thread, result = run_serve(server_sock, boom)
    ipc.recv_message(client_sock, 5.0)
    msg_type, payload = request(client_sock, 1, np.zeros(6, np.float32), np.zeros((4, 5, 3), np.uint8))
    assert msg_type == ipc.T_ERROR and b"boom" in payload
    ipc.send_message(client_sock, ipc.T_PING)
    assert ipc.recv_message(client_sock, 5.0)[0] == ipc.T_PONG
    client_sock.close()
    thread.join(5)


def test_serve_ends_when_idle(pair):
    server_sock, client_sock = pair
    t0 = time.monotonic()
    thread, result = run_serve(server_sock, fake_predict, idle_timeout_s=0.6)
    thread.join(5)
    assert result["reason"] == "idle" and 0.5 < time.monotonic() - t0 < 3.0


def test_serve_ping_resets_idle_timer(pair):
    server_sock, client_sock = pair
    thread, result = run_serve(server_sock, fake_predict, idle_timeout_s=1.0)
    ipc.recv_message(client_sock, 5.0)
    for _ in range(3):
        time.sleep(0.5)
        ipc.send_message(client_sock, ipc.T_PING)
        assert ipc.recv_message(client_sock, 5.0)[0] == ipc.T_PONG
    assert thread.is_alive()  # 1.5초가 지났지만 PING이 타이머를 되돌렸다
    client_sock.close()
    thread.join(5)


def test_serve_bad_magic_is_protocol_error(pair):
    server_sock, client_sock = pair
    thread, result = run_serve(server_sock, fake_predict)
    ipc.recv_message(client_sock, 5.0)
    client_sock.sendall(b"XXXX" + b"\x00" * 8)
    thread.join(5)
    assert result["reason"] == "protocol_error"


# --- 실제 프로세스: 소켓 권한, 종료 조건, 정리 ---


def launcher_for(checkpoint, idle_s=600.0, parent_pid=None):
    def launch(socket_path, log_path):
        cmd = [
            sys.executable, str(TRAIN_DIR / "policy_server.py"), "--socket", socket_path, "--checkpoint", str(checkpoint),
            "--idle-timeout-s", str(idle_s), "--nice", "19", "--cleanup-dir", os.path.dirname(socket_path),
        ]
        if parent_pid:
            cmd += ["--parent-pid", str(parent_pid)]
        with open(log_path, "ab") as log_file:
            return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file)

    return launch


def test_real_process_request_matches_in_process_and_cleans_up(tiny_checkpoint, observations):
    obs0, _ = observations
    client = ipc.PolicyClient(launcher_for(tiny_checkpoint, parent_pid=os.getpid()), start_timeout_s=120.0, request_timeout_s=30.0)
    client.ensure_started()
    try:
        sock_dir = client._dir
        mode = stat.S_IMODE(os.stat(os.path.join(sock_dir, "policy.sock")).st_mode)
        assert mode == 0o600 and stat.S_IMODE(os.stat(sock_dir).st_mode) == 0o700
        assert client.ready_info["n_action_steps"] == N_ACTION_STEPS
        result = client.request(obs0.state, obs0.image, reset=True)
        direct = ps.load_act_policy_class()(tiny_checkpoint, "cpu", "predict").predict_chunk(obs0.state, obs0.image, True)
        assert np.array_equal(result.actions.astype(np.float32), direct)
        assert result.infer_ms > 0 and result.total_ms >= result.infer_ms
        proc = client._proc
    finally:
        client.shutdown()
    assert proc.poll() == 0
    assert not os.path.exists(sock_dir)


def test_real_process_exits_by_itself_when_idle(tiny_checkpoint):
    client = ipc.PolicyClient(launcher_for(tiny_checkpoint, idle_s=2.0), request_timeout_s=30.0, idle_timeout_s=2.0)
    client.ensure_started()
    proc, sock_dir = client._proc, client._dir
    assert client.status()["state"] == "on"
    assert proc.wait(15) == 0
    assert client.status()["state"] == "off"
    assert not os.path.exists(sock_dir)
    client.shutdown()


def test_real_process_exits_when_server_connection_drops(tiny_checkpoint):
    """서버 쪽 소켓이 닫히면(서버가 죽으면 커널이 닫는다) 정책 프로세스가 몇 초 안에 끝나고 폴더를 지운다."""
    client = ipc.PolicyClient(launcher_for(tiny_checkpoint), request_timeout_s=30.0)
    client.ensure_started()
    proc, sock_dir = client._proc, client._dir
    t0 = time.monotonic()
    client._sock.close()  # 서버가 죽은 것과 같다
    assert proc.wait(10) == 0
    assert time.monotonic() - t0 < 5.0
    assert not os.path.exists(sock_dir)
    client._sock = None
    client.shutdown()


def test_real_process_refuses_temporal_ensemble(tmp_path, tiny_checkpoint):
    import json
    import shutil

    bad = tmp_path / "te"
    shutil.copytree(tiny_checkpoint, bad)
    cfg = json.loads((bad / "config.json").read_text())
    cfg["temporal_ensemble_coeff"] = 0.01
    cfg["n_action_steps"] = 1
    (bad / "config.json").write_text(json.dumps(cfg))
    client = ipc.PolicyClient(launcher_for(bad), start_timeout_s=60.0)
    with pytest.raises(ipc.PolicyError, match="시간 앙상블"):
        client.ensure_started()
