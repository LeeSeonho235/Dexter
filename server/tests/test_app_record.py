"""WebSocket으로 녹화 흐름: 리셋 → 시작 → 끝 → 저장, 거절 규칙, 최대 길이 자동 종료, 삭제, 쓰기 실패, 쓰기가 막혀도 틱이 도는지."""

import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import config
from server.app import create_app
from server.episode_store import EpisodeStore
from server.seeds import block_xy_for_seed


def record(action, **extra):
    return json.dumps({"type": "record", "action": action, **extra})


RESET = json.dumps({"type": "reset"})


def receive_until(ws, predicate, max_messages=400):
    seen = []
    for _ in range(max_messages):
        message = ws.receive()
        item = message["bytes"] if message.get("bytes") is not None else json.loads(message["text"])
        seen.append(item)
        if predicate(item):
            return item, seen
    raise AssertionError(f"not received in {max_messages} messages; last: {seen[-3:]!r}")


def state_where(ws, predicate, **kwargs):
    item, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "state" and predicate(m["record"]), **kwargs)
    return item["record"]


def last_state(ws):
    item, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "state")
    return item


def reset_and_wait(ws):
    ws.send_text(RESET)
    return state_where(ws, lambda r: r["block_ready"])


def start_and_wait(ws, ticks=5):
    ws.send_text(record("start"))
    return state_where(ws, lambda r: r["recording"] and r["ticks"] >= ticks)


def make_client(tmp_path, **kwargs):
    return TestClient(create_app(record_dir=tmp_path / "raw", **kwargs))


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def test_state_has_record_fields_and_no_folder_is_created(client, tmp_path):
    with client.websocket_connect(config.WS_PATH) as ws:
        rec = last_state(ws)["record"]
    assert rec["recording"] is False and rec["ticks"] == 0 and rec["elapsed_s"] == 0
    assert rec["next_seed"] == 0 and rec["current_seed"] is None and rec["seeds_left"] == 100
    assert rec["block_ready"] is False and rec["resetting"] is False
    assert (rec["saved_success"], rec["saved_fail"], rec["saved_total"]) == (0, 0, 0)
    assert rec["pending_writes"] == 0 and rec["save_error_count"] == 0 and rec["last_episode"] is None
    assert rec["success_condition_met"] is False and rec["still_ratio"] is None
    assert rec["max_s"] == 60.0
    assert not (tmp_path / "raw").exists()  # 첫 저장 때 만든다


def test_start_without_reset_is_refused_and_connection_survives(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        ws.send_text(record("start"))
        err, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
        assert "리셋" in err["reason"]
        ws.send_text(json.dumps({"type": "ping", "id": 5}))
        receive_until(ws, lambda m: m == {"type": "pong", "id": 5})
        st = last_state(ws)
        assert st["invalid_count"] == 0  # 형식은 맞아서 잘못된 메시지로 세지 않는다
        assert st["record"]["recording"] is False
        assert st["record"]["last_event"]["level"] == "warn"


def test_full_flow_reset_record_stop_success_saves_files(client, tmp_path):
    with client.websocket_connect(config.WS_PATH) as ws:
        rec = reset_and_wait(ws)
        assert rec["current_seed"] == 0 and rec["next_seed"] == 1 and rec["resetting"] is False

        rec = start_and_wait(ws, ticks=15)
        assert rec["block_ready"] is False  # 시작하면 "리셋 직후" 표시가 꺼진다
        assert rec["elapsed_s"] > 0.3 and rec["still_ratio"] is not None

        ws.send_text(record("stop_success"))
        rec = state_where(ws, lambda r: r["saved_total"] == 1)
        assert (rec["saved_success"], rec["saved_fail"]) == (1, 0)
        assert rec["last_episode"]["seed"] == 0 and rec["last_episode"]["success"] is True
        assert rec["pending_writes"] == 0 and rec["last_event"]["level"] == "info"

    raw = tmp_path / "raw"
    assert sorted(p.name for p in raw.iterdir()) == ["episode_000000.json", "episode_000000.npz"]
    meta = json.loads((raw / "episode_000000.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 0 and meta["source"] == "demo" and meta["success"] is True
    assert meta["end_reason"] == "stop_success" and meta["is_intervention_any"] is False
    x, y = block_xy_for_seed(0)
    assert meta["block_start_m"][0] == pytest.approx(x, abs=0.005) and meta["block_start_m"][1] == pytest.approx(y, abs=0.005)
    arrays = EpisodeStore(raw).load_arrays(0)
    n = meta["n_ticks"]
    assert arrays["qpos"].shape == (n, 13) and arrays["ctrl"].shape == (n, 6) and n >= 15
    np.testing.assert_allclose(arrays["qpos"][0, 6:9], meta["block_start_m"], atol=0.003)  # 첫 샘플의 블록 = 시작 위치
    spacing_ms = set(np.round(np.diff(arrays["sim_time_s"]) * 1000).astype(int).tolist())
    assert spacing_ms <= {30, 35}


def test_second_reset_clears_ready_while_running_and_advances_the_seed(client):
    """녹화 없이 리셋을 두 번 하면 시드가 하나 건너뛰고, 두 번째 리셋이 진행되는 동안에는 '리셋 직후'가 꺼져 있다."""
    with client.websocket_connect(config.WS_PATH) as ws:
        rec = reset_and_wait(ws)
        assert rec["current_seed"] == 0 and rec["next_seed"] == 1
        ws.send_text(RESET)
        rec = state_where(ws, lambda r: r["resetting"])
        assert rec["block_ready"] is False  # 진행 중에는 시작할 수 없다
        ws.send_text(record("start"))
        err, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
        assert "리셋" in err["reason"]
        rec = state_where(ws, lambda r: r["block_ready"])
        assert rec["current_seed"] == 1 and rec["next_seed"] == 2 and rec["resetting"] is False


def test_reset_is_refused_while_recording(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        reset_and_wait(ws)
        start_and_wait(ws)
        ws.send_text(RESET)
        err, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
        assert "녹화 중" in err["reason"]
        rec = last_state(ws)["record"]
        assert rec["recording"] is True and rec["next_seed"] == 1 and rec["resetting"] is False  # 아무 변화 없음
        ws.send_text(record("start"))  # 두 번 시작도 거절
        err, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
        assert "이미 녹화 중" in err["reason"]
        ws.send_text(record("discard"))


def test_discard_saves_nothing_and_does_not_count(client, tmp_path):
    with client.websocket_connect(config.WS_PATH) as ws:
        reset_and_wait(ws)
        start_and_wait(ws)
        ws.send_text(record("discard"))
        rec = state_where(ws, lambda r: not r["recording"])
        assert rec["saved_total"] == 0 and rec["ticks"] == 0
        time.sleep(0.3)
        assert rec["last_event"]["level"] in ("info", "warn")
        # 취소한 뒤에는 다시 리셋해야 시작할 수 있다
        ws.send_text(record("start"))
        err, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
        assert "리셋" in err["reason"]
    assert not (tmp_path / "raw").exists() or list((tmp_path / "raw").iterdir()) == []


def test_stop_and_discard_when_not_recording_are_refused(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        for action in ("stop_success", "stop_fail", "discard"):
            ws.send_text(record(action))
            err, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
            assert "녹화 중이 아닙니다" in err["reason"]


def test_max_length_ends_recording_as_failure_and_notifies(tmp_path):
    with make_client(tmp_path, record_max_s=1.0) as client, client.websocket_connect(config.WS_PATH) as ws:
        reset_and_wait(ws)
        ws.send_text(record("start"))
        rec = state_where(ws, lambda r: r["saved_total"] == 1, max_messages=800)
        assert (rec["saved_success"], rec["saved_fail"]) == (0, 1)
        assert rec["recording"] is False
        events = []
        rec2 = state_where(ws, lambda r: r["last_event"] is not None and "저장됨" in r["last_event"]["text"])
        events.append(rec2["last_event"])
    meta = json.loads((tmp_path / "raw" / "episode_000000.json").read_text(encoding="utf-8"))
    assert meta["end_reason"] == "max_length" and meta["success"] is False and meta["n_ticks"] == 30


def test_delete_last_checks_episode_number(client, tmp_path):
    with client.websocket_connect(config.WS_PATH) as ws:
        for _ in range(2):
            reset_and_wait(ws)
            start_and_wait(ws)
            ws.send_text(record("stop_fail"))
            state_where(ws, lambda r: not r["recording"] and r["pending_writes"] == 0 and r["saved_total"] >= 1)
        rec = state_where(ws, lambda r: r["saved_total"] == 2 and r["pending_writes"] == 0)
        assert rec["last_episode"]["index"] == 1 and rec["next_seed"] == 2

        ws.send_text(record("delete_last", episode=0))  # 최근이 아닌 번호: 거절
        err, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
        assert "가장 최근" in err["reason"]
        assert last_state(ws)["record"]["saved_total"] == 2

        ws.send_text(record("delete_last", episode=1))
        rec = state_where(ws, lambda r: r["saved_total"] == 1 and r["pending_writes"] == 0)
        assert rec["last_episode"]["index"] == 0 and rec["next_seed"] == 1  # 지운 번호의 시드를 다시 쓴다
        assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == ["episode_000000.json", "episode_000000.npz"]

        ws.send_text(record("delete_last", episode=0))
        state_where(ws, lambda r: r["saved_total"] == 0 and r["pending_writes"] == 0)
        ws.send_text(record("delete_last", episode=0))
        err, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
        assert "삭제할 에피소드가 없습니다" in err["reason"]


def test_delete_is_refused_while_recording(client):
    with client.websocket_connect(config.WS_PATH) as ws:
        reset_and_wait(ws)
        start_and_wait(ws)
        ws.send_text(record("delete_last", episode=0))
        err, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
        assert "녹화 중" in err["reason"] or "삭제할 에피소드" in err["reason"]
        ws.send_text(record("discard"))


def test_invalid_messages_are_rejected_and_connection_stays(client):
    bad = [
        record("explode"),
        json.dumps({"type": "record"}),
        json.dumps({"type": "record", "action": "start", "episode": 1}),
        record("delete_last"),
        record("delete_last", episode=-3),
        record("delete_last", episode="1"),
        json.dumps({"type": "reset", "seed": 1000}),
        json.dumps({"type": "reset", "extra": 1}),
    ]
    with client.websocket_connect(config.WS_PATH) as ws:
        for text in bad:
            ws.send_text(text)
        ws.send_text(json.dumps({"type": "ping", "id": 9}))
        _, seen = receive_until(ws, lambda m: m == {"type": "pong", "id": 9})
        st, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "state" and m["invalid_count"] == len(bad))
        assert st["record"]["recording"] is False and st["record"]["saved_total"] == 0
        frame, _ = receive_until(ws, lambda m: isinstance(m, bytes))  # 화면도 계속 온다
        assert frame.startswith(b"\xff\xd8")
    assert client.app.state.hub.worker.is_alive


def test_save_failure_is_shown_and_not_counted(tmp_path, monkeypatch):
    def boom(self, ep, timestep_s):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(EpisodeStore, "save", boom)
    with make_client(tmp_path) as client, client.websocket_connect(config.WS_PATH) as ws:
        reset_and_wait(ws)
        start_and_wait(ws)
        ws.send_text(record("stop_success"))  # 성공으로 눌렀지만 저장이 실패한다
        rec = state_where(ws, lambda r: r["save_error_count"] == 1)
        assert rec["saved_success"] == 0 and rec["saved_total"] == 0  # 성공 수에 넣지 않았다
        assert rec["pending_writes"] == 0
        assert rec["last_event"]["level"] == "error" and "저장 실패" in rec["last_event"]["text"]
        assert "No space left" in rec["last_event"]["text"]
        assert rec["last_episode"] is None


def test_blocked_disk_does_not_slow_the_tick_loop(tmp_path, monkeypatch):
    """디스크 쓰기가 멈춰도(쓰기 스레드가 막힘) stop 호출은 바로 끝나고 시뮬레이션 틱은 계속 돈다."""
    release = threading.Event()
    original = EpisodeStore.save

    def slow(self, ep, timestep_s):
        release.wait(10.0)
        return original(self, ep, timestep_s)

    monkeypatch.setattr(EpisodeStore, "save", slow)
    with make_client(tmp_path) as client, client.websocket_connect(config.WS_PATH) as ws:
        worker = client.app.state.hub.worker
        reset_and_wait(ws)
        start_and_wait(ws, ticks=10)
        worker.tick_stats.reset()
        sim_before = worker.sim_time_s
        ws.send_text(record("stop_success"))
        rec = state_where(ws, lambda r: r["pending_writes"] == 1 and not r["recording"])
        assert rec["saved_total"] == 0  # 아직 안 써졌으니 세지 않는다
        time.sleep(1.0)
        assert worker.sim_time_s - sim_before > 0.8  # 시뮬레이션은 계속 진행
        stats = worker.tick_stats.summary()
        assert stats["n"] > 15 and stats["max_ms"] < 250  # 틱이 쓰기 때문에 늘어지지 않는다(렌더링 약 55 ms 포함)
        release.set()
        rec = state_where(ws, lambda r: r["saved_total"] == 1 and r["pending_writes"] == 0)
        assert rec["saved_success"] == 1


def test_next_seed_and_counts_survive_a_restart(tmp_path):
    with make_client(tmp_path) as client, client.websocket_connect(config.WS_PATH) as ws:
        reset_and_wait(ws)
        start_and_wait(ws)
        ws.send_text(record("stop_success"))
        state_where(ws, lambda r: r["saved_total"] == 1)
    with make_client(tmp_path) as client, client.websocket_connect(config.WS_PATH) as ws:
        rec = last_state(ws)["record"]
        assert rec["next_seed"] == 1 and rec["saved_success"] == 1 and rec["current_seed"] is None
        assert rec["episode_index"] == 1  # 다음 에피소드 번호
        ws.send_text(record("start"))  # 재시작 직후에도 리셋부터
        err, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
        assert "리셋" in err["reason"]
