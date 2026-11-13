"""보조 카메라(2-6)를 SimWorker와 서버(app)에 연결한 뒤의 격리 확인. 학습·녹화 경로는 바이트 단위로 그대로여야 한다.

라이브 서버는 시계에 의존해서 틱마다 몇 스텝을 도는지가 달라진다. 그래도 "스텝 번호 k에서의 qpos"는 ctrl이 같으면 결정적이다.
그래서 실제 SimWorker에 실제 두 시점 aux(렌더 중)를 붙이고, 물리 스텝마다의 qpos 바이트를 aux가 없을 때와 겹치는 스텝에서 비교한다.
  1 라이브 물리 궤적: aux 없음 = 실제 aux(렌더 중) = 값을 망가뜨리는 가짜 aux
  2 값을 망가뜨리거나 예외를 던지는 aux가 시뮬레이션을 못 죽인다(사본만 넘기고, 예외면 aux를 뗀다)
  3 코드 구조: sim_worker.py는 aux를 import하지 않고, 호출은 물리 루프 뒤에서 qpos 사본으로만 한다
  4 라이브 시뮬레이션이 aux를 돌리는 동안에도 학습용 StateRenderer 이미지가 같다
  5 서버 통합: aux_frame 형식, 메인 프레임(바이너리) 형식 그대로, aux 메시지가 로봇·녹화 상태를 안 바꿈, aux 꺼짐·시작 실패·연결 끊김
(녹화 npz 바이트는 test_aux_isolation.py 4와 test_app_scene.py가 가짜 틱으로 확인한다.)
"""

import ast
import base64
import io
import json
import time

import mujoco
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_app import receive_until
from test_aux_isolation import SERVER_DIR, imported_and_used_names, qpos_set, sha

from server import config
from server.app import _state_message, create_app
from server.aux_view import AuxViewWorker
from server.render import StateRenderer
from server.sim_worker import SimWorker

SIM_SECONDS = 1.2
MIN_COMMON_STEPS = 150  # 1.2 s / 0.005 s = 240 스텝 안팎 중 겹쳐야 하는 최소 수


def make_sim() -> SimWorker:
    return SimWorker(
        config.SCENE_PATH, keyframe=config.START_KEYFRAME, camera=config.CAMERA, width_px=config.RENDER_WIDTH_PX,
        height_px=config.RENDER_HEIGHT_PX, frame_fps=config.FRAME_FPS, jpeg_quality=config.JPEG_QUALITY,
        max_catch_up_s=config.MAX_CATCH_UP_S, shadow=config.RENDER_SHADOW, reflection=config.RENDER_REFLECTION,
        tick_hz=config.SIM_TICK_HZ,
    )


class StepLog:
    """녹화기 자리(attach_recorder): 물리 스텝 직전마다 (스텝 번호 → qpos 바이트)를 쌓는다."""

    def __init__(self, timestep_s: float) -> None:
        self.timestep_s = timestep_s
        self.steps: dict[int, bytes] = {}

    def on_step(self, sim_time_s: float, qpos: np.ndarray, ctrl: np.ndarray) -> None:
        self.steps[round(sim_time_s / self.timestep_s)] = qpos.tobytes()


def run_sim(aux, seconds: float = SIM_SECONDS) -> dict[int, bytes]:
    sim = make_sim()
    sim.start()
    log = StepLog(sim.timestep_s)
    sim.attach_recorder(log)
    if aux is not None:
        sim.attach_aux(aux)
    sim.set_viewer_count(1)  # 메인 화면도 같이 렌더링하는 실제 조건
    try:
        time.sleep(seconds)
    finally:
        sim.stop()
    return log.steps


def start_real_aux() -> AuxViewWorker:
    aux = AuxViewWorker(enabled=True, fps=30.0)
    aux.start()
    aux.set_viewer_count(1)
    return aux


@pytest.fixture(scope="module")
def baseline_steps():
    steps = run_sim(None)
    assert len(steps) >= MIN_COMMON_STEPS
    return steps


def assert_same_trajectory(baseline: dict[int, bytes], other: dict[int, bytes]) -> int:
    common = sorted(set(baseline) & set(other))
    assert len(common) >= MIN_COMMON_STEPS and common[0] == 0, len(common)
    assert common == list(range(common[0], common[-1] + 1))  # 빠진 스텝이 없다
    mismatched = [k for k in common if baseline[k] != other[k]]
    assert not mismatched, f"스텝 {mismatched[:5]}...에서 qpos 바이트가 다르다"
    return len(common)


# --- 1. 라이브 물리 궤적 ---


def test_1_live_physics_trajectory_is_byte_identical_with_a_real_rendering_aux(baseline_steps):
    aux = start_real_aux()
    try:
        steps = run_sim(aux)
        state = aux.state()
    finally:
        aux.stop()
    assert state["frames"] >= 2 and state["running"] and state["error"] is None, state  # aux가 실제로 렌더했다(의미 있는 비교)
    assert assert_same_trajectory(baseline_steps, steps) >= MIN_COMMON_STEPS


class HostileAux:
    """받은 qpos 사본을 NaN으로 망가뜨린다(사본이 아니라면 시뮬레이션이 깨진다)."""

    def __init__(self) -> None:
        self.offers: list[np.ndarray] = []

    def offer_qpos(self, qpos: np.ndarray, auto_running: bool = False) -> None:
        self.offers.append(qpos.copy())
        qpos[:] = np.nan


def test_1_live_physics_trajectory_is_byte_identical_with_a_hostile_aux(baseline_steps):
    hostile = HostileAux()
    steps = run_sim(hostile)
    assert len(hostile.offers) >= 30
    assert all(np.isfinite(offer).all() for offer in hostile.offers)  # 틱마다 새 사본이 간다(앞 틱의 NaN이 안 돌아온다)
    assert assert_same_trajectory(baseline_steps, steps) >= MIN_COMMON_STEPS


def test_1_trajectory_comparison_is_not_vacuous(baseline_steps):
    """qpos가 스텝마다 실제로 변하고(블록이 떨어지고 팔이 자리를 잡는다), 한 스텝이라도 다르면 비교가 잡아낸다."""
    assert len(set(baseline_steps.values())) > 50
    tampered = dict(baseline_steps)
    tampered[10] = bytes(len(tampered[10]))
    with pytest.raises(AssertionError):
        assert_same_trajectory(baseline_steps, tampered)


# --- 2. 죽지 않는다 ---


def test_2_raising_aux_is_detached_and_the_simulation_keeps_running():
    class RaisingAux:
        def __init__(self) -> None:
            self.calls = 0

        def offer_qpos(self, qpos, auto_running=False):
            self.calls += 1
            raise RuntimeError("aux가 고장 났다(테스트용)")

    aux = RaisingAux()
    sim = make_sim()
    sim.start()
    sim.attach_aux(aux)
    try:
        time.sleep(0.5)
        t0 = sim.sim_time_s
        time.sleep(0.4)
        assert sim.is_alive and sim.sim_time_s > t0  # 시뮬레이션은 계속 돈다
        assert aux.calls == 1 and sim._aux is None  # 첫 실패 뒤에 aux를 뗐다(틱마다 로그를 쏟지 않는다)
        assert np.isfinite(sim.snapshot["tip_m"]).all()
    finally:
        sim.stop()


def test_2_aux_sees_the_auto_flag_and_none_detaches():
    class SpyAux:
        def __init__(self) -> None:
            self.flags: list[bool] = []

        def offer_qpos(self, qpos, auto_running=False):
            assert qpos.shape == (13,)
            self.flags.append(auto_running)

    spy = SpyAux()
    sim = make_sim()
    sim.start()
    sim.attach_aux(spy)
    try:
        time.sleep(0.4)
        sim.attach_aux(None)
        count = len(spy.flags)
        time.sleep(0.2)
        assert len(spy.flags) in (count, count + 1)  # 떼는 순간 진행 중이던 틱 하나는 올 수 있다
    finally:
        sim.stop()
    assert count > 3 and not any(spy.flags)  # AUTO가 아니면 False


# --- 3. 코드 구조 ---


def test_3_sim_worker_never_imports_aux_and_hands_over_a_copy_after_the_physics_loop():
    path = SERVER_DIR / "sim_worker.py"
    modules, _ = imported_and_used_names(path)
    assert not {m for m in modules if "aux" in m}, modules
    tree = ast.parse(path.read_text(encoding="utf-8"))
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_loop")
    physics = [n for n in ast.walk(loop) if isinstance(n, ast.While) and "target_sim_s" in ast.unparse(n.test)]
    assert len(physics) == 1
    assert [n for n in ast.walk(physics[0]) if isinstance(n, ast.Attribute) and n.attr in ("_aux", "offer_qpos")] == []  # 물리 루프 안에서는 aux를 안 부른다
    offers = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "offer_qpos"]
    assert len(offers) == 1 and offers[0].lineno > physics[0].end_lineno
    assert "data.qpos.copy()" in ast.unparse(offers[0])  # 사본만 넘긴다


def test_3_only_app_imports_aux_view_among_the_runtime_modules():
    importers = []
    for path in sorted(SERVER_DIR.glob("*.py")):
        modules, _ = imported_and_used_names(path)
        if any(m == "server.aux_view" or m.startswith("server.aux_view.") for m in modules):
            importers.append(path.name)
    assert importers == ["app.py", "aux_view_bench.py", "aux_view_probe.py"] or set(importers) <= {"app.py", "aux_view_bench.py", "aux_view_probe.py"}, importers
    assert "app.py" in importers


# --- 4. 라이브 시뮬레이션이 aux를 돌리는 동안의 학습용 이미지 ---


def test_4_state_renderer_images_are_identical_while_a_live_sim_feeds_a_rendering_aux():
    home = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH)).key("home").qpos.copy()
    poses = qpos_set(home)
    with StateRenderer() as main:
        main.render(home)  # 첫 렌더링은 버린다(docs/07)
        before = [sha(main.render(q)) for q in poses]
        aux = start_real_aux()
        sim = make_sim()
        sim.start()
        sim.attach_aux(aux)
        sim.set_viewer_count(1)
        try:
            deadline = time.monotonic() + 15
            while aux.state()["frames"] < 2:
                assert time.monotonic() < deadline, "aux가 렌더하지 않았다"
                time.sleep(0.02)
            during = [sha(main.render(q)) for q in poses]
            during += [sha(main.render(q)) for q in poses]
        finally:
            sim.stop()
            aux.stop()
        after = [sha(main.render(q)) for q in poses]
    assert len(set(before)) == len(poses)
    assert during == before + before and after == before


# --- 5. 서버 통합 ---


def aux_app(**aux_kwargs):
    kwargs = {"enabled": True, "fps": 20.0, **aux_kwargs}
    return create_app(aux_factory=lambda on_frame: AuxViewWorker(on_frame=on_frame, **kwargs))


def collect(ws, seconds: float) -> list:
    """seconds 동안 서버가 보내는 메시지를 모은다(바이너리는 bytes, 텍스트는 dict)."""
    out = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        message = ws.receive()
        out.append(message["bytes"] if message.get("bytes") is not None else json.loads(message["text"]))
    return out


def last_state(items):
    states = [m for m in items if isinstance(m, dict) and m["type"] == "state"]
    return states[-1] if states else None


def aux_frames(items):
    return [m for m in items if isinstance(m, dict) and m["type"] == "aux_frame"]


def test_5_aux_frame_format_and_main_frame_format_are_as_specified():
    with TestClient(aux_app()) as client, client.websocket_connect(config.WS_PATH) as ws:
        idle = collect(ws, 1.0)
        assert aux_frames(idle) == []  # 보조 화면을 켜지 않으면 보내지 않는다
        assert last_state(idle)["aux"]["viewers"] == 0 and last_state(idle)["aux"]["enabled"] is True

        ws.send_text(json.dumps({"type": "aux", "enabled": True}))
        item, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "aux_frame", max_messages=800)
        second, seen = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "aux_frame", max_messages=800)
        for frame in (item, second):
            assert set(frame) == {"type", "seq", "w", "h", "views", "jpeg"}
            assert (frame["w"], frame["h"], frame["views"]) == (320, 120, 2)
            jpeg = base64.b64decode(frame["jpeg"], validate=True)
            assert jpeg[:2] == b"\xff\xd8" and Image.open(io.BytesIO(jpeg)).size == (320, 120)
        assert second["seq"] > item["seq"]

        more = collect(ws, 0.7)
        assert last_state(more)["aux"]["viewers"] == 1
        # 메인 화면: 바이너리 JPEG 그대로(크기, 형식), 텍스트 메시지 종류는 알려진 것뿐
        binaries = [m for m in seen + more if isinstance(m, bytes)]
        assert binaries and all(b[:2] == b"\xff\xd8" for b in binaries)
        assert Image.open(io.BytesIO(binaries[-1])).size == (config.RENDER_WIDTH_PX, config.RENDER_HEIGHT_PX)
        assert {m["type"] for m in seen + more if isinstance(m, dict)} <= {"state", "aux_frame", "pong", "error"}
        assert json.dumps(last_state(more)["aux"])  # state.aux는 JSON으로 보낼 수 있다


def test_5_aux_message_does_not_change_robot_or_recording_state_and_off_stops_frames():
    with TestClient(aux_app()) as client, client.websocket_connect(config.WS_PATH) as ws:
        before = last_state(collect(ws, 0.6))
        ws.send_text(json.dumps({"type": "aux", "enabled": True}))
        receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "aux_frame", max_messages=800)
        during = last_state(collect(ws, 0.6))
        for key in ("controlling", "mapping_mode", "gripper_closed"):
            assert during["control"][key] == before["control"][key], key
        for key in ("saved_total", "episode_index", "reset_phase", "resetting", "block_ready"):
            assert during["record"][key] == before["record"][key], key
        for key in ("status", "owner"):
            assert during["auto"][key] == before["auto"][key], key

        ws.send_text(json.dumps({"type": "aux", "enabled": False}))
        receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "state" and m["aux"]["viewers"] == 0, max_messages=800)
        assert aux_frames(collect(ws, 0.8)) == []  # 끄면 더 안 온다
        assert client.app.state.hub.aux.state()["viewers"] == 0


def test_5_invalid_aux_message_is_rejected_like_other_invalid_messages():
    with TestClient(aux_app()) as client, client.websocket_connect(config.WS_PATH) as ws:
        ws.send_text(json.dumps({"type": "aux", "enabled": "yes"}))
        item, _ = receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "state" and m["invalid_count"] == 1, max_messages=800)
        assert item["aux"]["viewers"] == 0


def test_5_when_aux_is_disabled_the_message_is_ignored_and_nothing_is_created():
    app = create_app(aux_factory=lambda on_frame: AuxViewWorker(enabled=False, on_frame=on_frame))
    with TestClient(app) as client, client.websocket_connect(config.WS_PATH) as ws:
        ws.send_text(json.dumps({"type": "aux", "enabled": True}))
        items = collect(ws, 1.0)
        state = last_state(items)
        assert state["aux"]["enabled"] is False and state["aux"]["running"] is False and state["aux"]["viewers"] == 0
        assert aux_frames(items) == [] and not [m for m in items if isinstance(m, dict) and m["type"] == "error"]
        hub = client.app.state.hub
        assert hub.aux._thread is None and hub.aux._renderer is None and hub.worker._aux is None  # 스레드, 렌더러, 연결 모두 없다


def test_5_aux_start_failure_keeps_the_server_running_and_is_shown_in_state(tmp_path):
    app = create_app(aux_factory=lambda on_frame: AuxViewWorker(enabled=True, scene_path=tmp_path / "없는.xml", on_frame=on_frame))
    with TestClient(app) as client, client.websocket_connect(config.WS_PATH) as ws:
        ws.send_text(json.dumps({"type": "aux", "enabled": True}))
        items = collect(ws, 1.0)
        state = last_state(items)
        assert state["aux"]["error"] is not None and state["aux"]["running"] is False
        assert aux_frames(items) == []
        assert [m for m in items if isinstance(m, bytes)]  # 메인 화면은 계속 온다
        assert client.app.state.hub.worker._aux is None  # 죽은 aux는 시뮬레이션에 연결하지 않는다


def test_5_closing_the_connection_releases_the_aux_viewer():
    with TestClient(aux_app()) as client:
        with client.websocket_connect(config.WS_PATH) as ws:
            ws.send_text(json.dumps({"type": "aux", "enabled": True}))
            receive_until(ws, lambda m: isinstance(m, dict) and m["type"] == "aux_frame", max_messages=800)
            assert client.app.state.hub.aux.state()["viewers"] == 1
        deadline = time.monotonic() + 5
        while client.app.state.hub.aux.state()["viewers"] != 0:
            assert time.monotonic() < deadline, "연결이 끊겼는데 aux 보는 연결이 남아 있다"
            time.sleep(0.05)


def test_5_state_message_has_aux_only_when_an_aux_is_given():
    from types import SimpleNamespace

    from server.view import default_view

    worker = SimpleNamespace(snapshot=None, sim_time_s=0.0, frames_rendered=0, view=default_view())
    conn = SimpleNamespace(hand_count=0, invalid_count=0, last_hand=None)
    control = SimpleNamespace(state=lambda: {})
    assert "aux" not in _state_message(conn, worker, control)
    off = AuxViewWorker(enabled=False)
    assert _state_message(conn, worker, control, aux=off)["aux"] == off.state()
