"""WebSocket으로 교정 흐름(4-2): 리셋 → AUTO → 넘겨받기 → 가짜 손으로 교정 → 돌려주기 → 성공 → 저장/버리기, 거절 규칙, 저장 실패, 순수 AUTO 한 줄, 시드 안내.
시뮬레이션, 제어, 녹화 서비스, 제어 주체, 교정 저장소는 모두 진짜고, 정책은 가짜(ScriptedPolicy: 블록 위치를 아는 되먹임 제어기)다.
브라우저 대신 테스트가 실제와 같은 JSON 메시지(hand 30Hz, height, auto)를 보낸다."""

import json
import time

import numpy as np
import pytest
from fake_policy import InProcessPolicy, park_chunk
from fastapi.testclient import TestClient

from server import config, correction_config as cc
from server.app import create_app
from server.correction_headless import CLOSED_PINCH, OPEN_PINCH, FakeHuman, ScriptedPolicy, wait_for
from server.correction_store import CorrectionStore

AUTO = lambda action: json.dumps({"type": "auto", "action": action})  # noqa: E731
RESET = json.dumps({"type": "reset"})


def receive_until(ws, predicate, max_messages=600):
    seen = []
    for _ in range(max_messages):
        message = ws.receive()
        item = message["bytes"] if message.get("bytes") is not None else json.loads(message["text"])
        seen.append(item)
        if predicate(item):
            return item, seen
    raise AssertionError(f"not received in {max_messages} messages; last: {seen[-3:]!r}")


def state_where(ws, predicate, **kwargs):
    item, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "state" and predicate(m), **kwargs)
    return item


def error_after(ws, send_text):
    ws.send_text(send_text)
    item, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
    return item["reason"]


def reset_and_wait(ws):
    ws.send_text(RESET)
    return state_where(ws, lambda m: m["record"]["block_ready"])


class Env:
    def __init__(self, client, hub, ws, policy, corrections):
        self.client, self.hub, self.ws, self.policy, self.corrections = client, hub, ws, policy, corrections

    def auto(self):
        return self.hub.auto.state()

    def human(self):
        return FakeHuman(
            self.hub.control, self.hub.worker,
            send_hand=lambda m: self.ws.send_text(json.dumps(m)),
            send_height=lambda d, s: self.ws.send_text(json.dumps({"type": "height", "dir": d, "slow": s})),
        )

    def send(self, action):
        self.ws.send_text(AUTO(action))


def build(tmp_path, policy=None, **app_kwargs):
    holder: dict = {}
    scripted = policy is None
    if scripted:
        policy = ScriptedPolicy(lambda: holder["hub"].worker.snapshot["block_m"])
    app = create_app(record_dir=tmp_path / "raw", correction_dir=tmp_path / "corrections", policy_client=policy, **app_kwargs)
    return app, holder, policy


@pytest.fixture
def env(tmp_path):
    app, holder, policy = build(tmp_path)
    with TestClient(app) as client:
        holder["hub"] = client.app.state.hub
        with client.websocket_connect("/ws") as ws:
            yield Env(client, client.app.state.hub, ws, policy, tmp_path / "corrections")


def start_auto_and_take_over(env, *, phase=4):
    reset_and_wait(env.ws)
    env.send("start")
    assert wait_for(lambda: env.auto()["status"] == "running" and env.policy.phase >= phase, 60)
    env.send("takeover")
    assert wait_for(lambda: env.auto()["owner"] == "HUMAN_WAIT", 3.0)


def test_full_path_takeover_correct_handback_success_then_save(env):
    hub = env.hub
    start_auto_and_take_over(env)
    state = env.auto()
    assert state["status"] == "running" and state["owner"] == "HUMAN_WAIT"
    assert hub.control.state()["gripper_sync"]["waiting"] is True  # 대기 중에도 "집게 맞추기" 조건이 켜져 있다
    human = env.human()
    human.hold(0.8, OPEN_PINCH)  # 로봇은 블록을 쥔 채, 손은 펴져 있다
    assert env.auto()["owner"] == "HUMAN"
    assert hub.control.state()["gripper_sync"] == {"active": True, "waiting": True, "robot_closed": True}
    z = [f["block_z"] for f in human.frame_log if f["owner"] == "HUMAN"]
    assert min(z) > 0.05  # 블록이 떨어지지 않았다(들린 높이 유지)
    human.hold(0.4, CLOSED_PINCH)
    assert hub.control.state()["gripper_sync"]["waiting"] is False
    tip = np.array(hub.worker.snapshot["tip_m"])
    human.move_target_to([tip[0] - 0.01, tip[1], tip[2]], pinch=CLOSED_PINCH)
    assert env.auto()["human_s"] > 1.0 and env.auto()["interventions"] == 1
    env.send("handback")
    assert wait_for(lambda: env.auto()["status"] == "success", 60), env.auto()
    state = env.auto()
    assert state["owner"] == "IDLE" and state["session_success"] == 1
    pending = state["correction"]["pending"]
    assert pending["interventions"] == 1 and pending["outcome"] == "success" and pending["human_time_s"] > 1.0
    json.dumps(state)  # 상태 메시지로 보낼 수 있다
    # 고르기 전에는 저장하지 않고, 새 AUTO와 블록 리셋을 받지 않는다
    assert not env.corrections.exists() or list(env.corrections.glob("correction_*")) == []
    assert "고르세요" in env.hub.auto.handle("start")
    assert "고르세요" in env.hub.recording.handle_reset()
    assert env.auto()["status"] == "success"
    # 저장
    env.send("save_correction")
    assert wait_for(lambda: env.auto()["correction"]["saved_count"] == 1, 10)
    assert env.auto()["correction"]["pending"] is None and env.auto()["correction"]["last_saved"]["interventions"] == 1
    store = CorrectionStore(env.corrections)
    meta = store.scan()[0]
    arrays = store.load_arrays(0)
    owner = arrays["owner"]
    assert meta["seed"] == env.auto()["seed"] and meta["intervention_count"] == 1 and meta["auto_result"]["outcome"] == "success"
    assert np.array_equal(arrays["is_intervention"], owner == cc.OWNER_HUMAN) and (owner == cc.OWNER_HUMAN_WAIT).any()
    assert meta["human_time_s"] == pytest.approx(float((owner == cc.OWNER_HUMAN).sum()) / 30, abs=1e-3)
    seg = meta["interventions"][0]
    assert np.all(owner[seg["start_sample"] : seg["end_sample"] + 1] == cc.OWNER_HUMAN)
    rows = store.csv_rows()
    assert len(rows) == 1 and rows[0]["kind"] == "correction_saved" and rows[0]["episode_index"] == "0" and int(rows[0]["seed"]) == meta["seed"]
    assert env.hub.recording.handle_reset() is None  # 선택을 마쳤으니 다시 리셋할 수 있다


def test_discard_after_a_stopped_correction_leaves_only_a_ledger_line(env):
    start_auto_and_take_over(env, phase=1)
    human = env.human()
    human.hold(0.6, OPEN_PINCH)
    env.send("stop")  # HUMAN 중 중지
    assert wait_for(lambda: env.auto()["status"] == "stopped", 5)
    assert env.auto()["correction"]["pending"]["end_reason"] == "user_stop"
    env.send("discard_correction")
    assert wait_for(lambda: len(CorrectionStore(env.corrections).csv_rows()) == 1, 5)
    rows = CorrectionStore(env.corrections).csv_rows()
    assert rows[0]["kind"] == "correction_discarded" and rows[0]["outcome"] == "stopped" and float(rows[0]["human_time_s"]) > 0.3
    assert list(env.corrections.glob("correction_*")) == [] and env.auto()["correction"]["pending"] is None
    assert env.auto()["session_attempts"] == 0  # 중지는 시도에 세지 않는다


def test_stop_while_waiting_for_the_hand_has_no_intervention_so_only_a_ledger_line(env):
    start_auto_and_take_over(env, phase=1)
    env.send("stop")  # HUMAN_WAIT 중 중지
    assert wait_for(lambda: env.auto()["status"] == "stopped", 5)
    state = env.auto()
    assert state["owner"] == "IDLE" and state["correction"]["pending"] is None  # 손을 한 번도 안 보였으니 사람 구간이 없다
    assert wait_for(lambda: len(CorrectionStore(env.corrections).csv_rows()) == 1, 5)
    assert CorrectionStore(env.corrections).csv_rows()[0]["kind"] == "auto_only"
    assert env.hub.worker.owner.current.value == "IDLE"


def test_invalid_order_requests_are_refused_without_invalid_count(env):
    ws = env.ws
    for action in ("takeover", "handback", "save_correction", "discard_correction"):
        reason = error_after(ws, AUTO(action))
        assert reason, action  # IDLE에서는 모두 이유가 돌아온다
    assert "실행 중이 아닙니다" in error_after(ws, AUTO("takeover"))
    assert "없습니다" in error_after(ws, AUTO("save_correction"))
    state = state_where(ws, lambda m: True)
    assert state["invalid_count"] == 0 and state["auto"]["owner"] == "IDLE"
    ws.send_text(json.dumps({"type": "auto", "action": "take_over"}))  # 모르는 값은 잘못된 메시지
    assert state_where(ws, lambda m: m["invalid_count"] == 1)
    ws.send_text(json.dumps({"type": "ping", "id": 5}))
    receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "pong")  # 연결은 유지된다


def test_handback_and_double_takeover_orders_during_a_run(env):
    start_auto_and_take_over(env, phase=1)
    assert "이미 손을 기다리는 중" in env.hub.auto.handle("takeover")
    assert "아직 손을 기다리는 중" in env.hub.auto.handle("handback")  # 대기 중 돌려주기는 안 된다
    assert env.auto()["owner"] == "HUMAN_WAIT"
    env.send("stop")
    assert wait_for(lambda: env.auto()["status"] == "stopped", 5)
    assert "실행 중이 아닙니다" in env.hub.auto.handle("handback")


def test_pure_auto_run_writes_one_csv_line_and_no_pending(tmp_path):
    policy = InProcessPolicy(lambda s, i, r, n: park_chunk(100))
    policy.started = True
    app, holder, _ = build(tmp_path, policy=policy, auto_max_s=1.5)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hub = client.app.state.hub
        reset_and_wait(ws)
        ws.send_text(AUTO("start"))
        state = state_where(ws, lambda m: m["auto"]["status"] == "timeout")
        assert state["auto"]["correction"]["pending"] is None and state["auto"]["human_s"] == 0.0 and state["auto"]["interventions"] == 0
        store = CorrectionStore(tmp_path / "corrections")
        assert wait_for(lambda: len(store.csv_rows()) == 1, 5)
        row = store.csv_rows()[0]
        assert row["kind"] == "auto_only" and row["outcome"] == "timeout" and row["end_reason"] == "auto_limit" and row["human_time_s"] == "0.0"
        assert int(row["seed"]) == state["auto"]["seed"] and len(row["created_at"]) > 10
        assert list((tmp_path / "corrections").glob("correction_*")) == []  # 개입이 없으면 에피소드 파일은 만들지 않는다
        assert hub.recording.handle_reset() is None  # 선택할 것이 없으니 바로 다시 리셋


def test_save_failure_is_shown_and_not_counted(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("x")  # 이 아래에는 폴더를 만들 수 없다
    holder: dict = {}
    policy = ScriptedPolicy(lambda: holder["hub"].worker.snapshot["block_m"])
    app = create_app(record_dir=tmp_path / "raw", correction_store=CorrectionStore(blocker / "corrections"), policy_client=policy)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        holder["hub"] = hub = client.app.state.hub
        e = Env(client, hub, ws, policy, blocker / "corrections")
        start_auto_and_take_over(e, phase=1)
        human = e.human()
        human.hold(0.6, OPEN_PINCH)
        e.send("stop")
        assert wait_for(lambda: e.auto()["correction"]["pending"] is not None, 5)
        e.send("save_correction")
        assert wait_for(lambda: e.auto()["correction"]["save_error_count"] == 1, 10)
        state = e.auto()
        assert state["correction"]["saved_count"] == 0 and state["correction"]["pending"] is None  # 성공으로 세지 않는다
        assert state["last_event"]["level"] == "error" and "저장 실패" in state["last_event"]["text"] and "디스크" in state["last_event"]["text"]


def test_policy_death_after_handback_still_asks_what_to_do_with_the_human_part(tmp_path):
    policy = InProcessPolicy(lambda s, i, r, n: park_chunk(20), n_action_steps=20)
    policy.started = True
    policy.fail_request_after = 1  # 돌려준 직후 요청에서 죽는다
    app, holder, _ = build(tmp_path, policy=policy)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hub = client.app.state.hub
        e = Env(client, hub, ws, policy, tmp_path / "corrections")
        reset_and_wait(ws)
        e.send("start")
        assert wait_for(lambda: e.auto()["status"] == "running" and (e.auto()["sample"] or 0) >= 3, 15)
        e.send("takeover")
        assert wait_for(lambda: e.auto()["owner"] == "HUMAN_WAIT", 3.0)
        human = e.human()
        human.hold(0.5, OPEN_PINCH)
        e.send("handback")
        assert wait_for(lambda: e.auto()["status"] == "error", 10)
        state = e.auto()
        assert "끊겼" in state["error"] and state["owner"] == "IDLE" and state["session_attempts"] == 0
        assert state["correction"]["pending"]["end_reason"] == "error" and state["correction"]["pending"]["human_time_s"] > 0.2
        e.send("save_correction")
        assert wait_for(lambda: e.auto()["correction"]["saved_count"] == 1, 10)
        assert CorrectionStore(tmp_path / "corrections").scan()[0]["auto_result"]["outcome"] == "error"


def test_next_seed_guide_matches_the_blocks_and_skips_seeds_already_used_by_corrections(tmp_path):
    """서버를 다시 켜도 교정 폴더에서 쓴 시드는 다음 시드에서 빠진다. 안내 용도는 배분표를 따르고 평가 시드 규칙은 그대로다."""
    corrections = tmp_path / "corrections"
    store = CorrectionStore(corrections)
    from test_correction_store import make_recording

    for seed in (28, 29, 40):
        store.append_csv(make_recording(seed=seed, human=(0, 0), wait=(0, 0)), "auto_only")
    app, holder, policy = build(tmp_path)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        state = state_where(ws, lambda m: True)
        assert state["auto"]["next_seed"] == 41 and state["auto"]["next_seed_purpose"] == cc.PURPOSE_EXTRA_DEMO  # 40~51: 추가 시연용
        assert state["auto"]["ready_seed"] is None
        ready = reset_and_wait(ws)
        assert ready["record"]["current_seed"] == 41
        auto = state_where(ws, lambda m: m["auto"]["ready_seed"] == 41)["auto"]
        assert auto["ready_seed_purpose"] == cc.PURPOSE_EXTRA_DEMO and auto["next_seed"] == 42
        # 평가 시드는 AUTO에서 여전히 막힌다
        client.app.state.hub.recording._current_seed = config.EVAL_SEED_RANGE[0]
        assert "평가 시드" in error_after(ws, AUTO("start"))
