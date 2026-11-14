"""state.scene(2-6 옆모습 보조 화면용): 값과 직렬화, 스냅샷이 없을 때, 10Hz 유지, 그리고 녹화 파일 바이트가 그대로인지."""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_app import receive_until
from test_aux_isolation import TS, fake_tick_episode

from server import config
from server.app import _scene_message, _state_message, create_app
from server.episode_store import EpisodeStore
from server.view import default_view


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def fake_worker(snapshot):
    return SimpleNamespace(snapshot=snapshot, sim_time_s=1.0, frames_rendered=3, view=default_view())


def test_scene_message_is_none_before_first_tick():
    assert _scene_message(fake_worker(None)) is None


def test_scene_message_values_and_json_serializable():
    snapshot = {"tip_m": np.zeros(3), "joints_rad": np.zeros(6), "block_m": np.array([0.2123456, -0.05, 0.015])}
    scene = _scene_message(fake_worker(snapshot))
    assert scene == {
        "block_m": [0.2123, -0.05, 0.015],
        "block_half_m": config.BLOCK_HALF_M,
        "target_center_m": list(config.TARGET_CENTER_M),
        "target_half_m": config.TARGET_HALF_M,
    }
    assert all(type(v) is float for v in scene["block_m"])  # numpy 값이 아니라 순수 float
    assert json.loads(json.dumps(scene)) == scene


def test_scene_message_does_not_modify_the_snapshot():
    snapshot = {"block_m": np.array([0.2, 0.0, 0.015])}
    before = snapshot["block_m"].copy()
    _scene_message(fake_worker(snapshot))
    assert np.array_equal(snapshot["block_m"], before) and list(snapshot) == ["block_m"]


def test_state_over_websocket_carries_scene_from_the_real_sim(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        item, _ = receive_until(
            ws, lambda m: isinstance(m, dict) and m["type"] == "state" and m["scene"] is not None, max_messages=600
        )
    scene = item["scene"]
    assert len(scene["block_m"]) == 3 and all(isinstance(v, float) for v in scene["block_m"])
    assert scene["block_half_m"] == config.BLOCK_HALF_M
    assert scene["target_center_m"] == list(config.TARGET_CENTER_M)
    assert scene["target_half_m"] == config.TARGET_HALF_M
    assert abs(scene["block_m"][2] - config.BLOCK_HALF_M) < 0.01  # 블록이 바닥 위에 있다
    assert "control" in item and "tip_m" in item["control"]  # 집게 끝은 기존 control에 이미 있다


def test_state_interval_is_still_10hz_with_scene(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        t0 = time.monotonic()
        count = 0
        while time.monotonic() - t0 < 1.0:
            receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "state")
            count += 1
    assert 6 <= count <= 12


class _StateEveryTick:
    """fake_tick_episode의 aux 자리에 들어가 틱마다 실제 _state_message를 만든다(서버가 state를 보내는 것과 같은 코드)."""

    def __init__(self) -> None:
        self.count = 0
        self.last = None

    def offer_qpos(self, qpos) -> None:
        snapshot = {"tip_m": np.zeros(3), "joints_rad": np.array(qpos[:6]), "block_m": np.array(qpos[6:9])}
        conn = SimpleNamespace(hand_count=0, invalid_count=0, last_hand=None)
        control = SimpleNamespace(state=lambda: {})
        self.last = _state_message(conn, fake_worker(snapshot), control)
        self.count += 1


def test_recorded_episode_and_npz_bytes_are_identical_when_state_with_scene_is_built_every_tick(tmp_path):
    ep_off = fake_tick_episode(None)
    state_maker = _StateEveryTick()
    ep_on = fake_tick_episode(state_maker)
    assert state_maker.count >= 1000 and state_maker.last["scene"] is not None  # 실제로 만들었다(의미 있는 비교)

    EpisodeStore(root=tmp_path / "off").save(ep_off, TS)
    EpisodeStore(root=tmp_path / "on").save(ep_on, TS)
    assert (tmp_path / "off" / "episode_000000.npz").read_bytes() == (tmp_path / "on" / "episode_000000.npz").read_bytes()
