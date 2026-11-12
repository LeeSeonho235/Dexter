"""보조 카메라 격리 증명(2-6 격리 테스트 1~4): 보조 화면이 켜져 있어도 학습 이미지, 평가 결과, 설정 해시, 녹화 파일이 바뀌지 않는다.

이 중 하나라도 실패하면 구현을 멈추고 원인부터 보고한다(테스트를 고쳐서 통과시키지 않는다).
1 main 바이트 동일: StateRenderer 이미지가 aux 꺼짐 → 켜짐(보조 스레드가 실제로 렌더 중) → 꺼짐에서 같다.
2 결정적 롤아웃 동일: EvalEnv 롤아웃 결과가 aux를 켜고 끄며 돌려도 같다(라이브 루프는 시계에 의존해 비트 비교가 안 되므로 이 방식).
3 코드 경로 분리: 학습·녹화·AUTO·평가 코드가 aux_view를 import하지 않고, aux_view는 학습용 렌더러·카메라를 쓰지 않는다. RECORD_* 값과 config_sha256이 그대로다.
4 녹화 불변: EpisodeRecorder가 만든 에피소드와 저장한 npz가 aux 켬/끔에서 같다.
"""

import ast
import hashlib
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from server import config
from server.aux_view import AuxViewWorker
from server.episode_store import EpisodeStore, config_sha256, config_snapshot
from server.eval_env import EvalEnv, ReplayPolicy, rollout
from server.recorder import EpisodeRecorder
from server.render import StateRenderer
from server.seeds import block_qpos_for_seed

SERVER_DIR = Path(config.REPO_ROOT) / "server"
TS = 0.005
AUX_TEST_FPS = 50.0  # 테스트에서는 보조 스레드가 쉬지 않고 그리게 한다(설정 파일의 기본 5와 별개)


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def make_aux() -> AuxViewWorker:
    aux = AuxViewWorker(enabled=True, fps=AUX_TEST_FPS)
    aux.start()
    aux.set_viewer_count(1)
    return aux


def wait_frames(aux: AuxViewWorker, n: int, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    start = aux.state()["frames"]
    while aux.state()["frames"] < start + n:
        assert time.monotonic() < deadline, "aux worker did not render frames in time"
        time.sleep(0.01)


@pytest.fixture(scope="module")
def home_qpos():
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    return model.key("home").qpos.copy()


# --- 1. main 바이트 동일 ---


def qpos_set(home_qpos):
    """고정 qpos 묶음: home, 시드 몇 개의 블록 위치, 팔을 움직인 자세."""
    out = [home_qpos.copy()]
    for seed in (0, 1, 7, 1000):
        q = home_qpos.copy()
        q[6:13] = block_qpos_for_seed(seed)
        out.append(q)
    for pan, lift in ((0.4, -1.2), (-0.5, -1.0)):
        q = home_qpos.copy()
        q[0], q[1] = pan, lift
        out.append(q)
    return out


def test_1_main_image_bytes_are_identical_with_aux_off_on_off(home_qpos):
    qposes = qpos_set(home_qpos)
    with StateRenderer() as main:
        main.render(home_qpos)  # 렌더러 인스턴스의 첫 렌더링은 버린다(docs/07: 첫 렌더링만 다르다)
        off_before = [sha(main.render(q)) for q in qposes]

        aux = make_aux()
        try:
            frames_before = aux.state()["frames"]
            on_hashes = []
            for i in range(4):  # 보조 스레드가 그리는 동안 여러 번 돈다
                aux.offer_qpos(qposes[i % len(qposes)])
                on_hashes.append([sha(main.render(q)) for q in qposes])
            wait_frames(aux, 2)
            assert aux.state()["frames"] > frames_before, "aux가 실제로 렌더하지 않았다: 이 테스트는 의미가 없다"
        finally:
            aux.stop()

        off_after = [sha(main.render(q)) for q in qposes]
    assert len(set(off_before)) == len(qposes)  # 자세마다 이미지가 다르다(빈 비교가 아니다)
    for hashes in on_hashes:
        assert hashes == off_before
    assert off_after == off_before


def aux_frame_jpeg(qpos: np.ndarray) -> bytes:
    aux = make_aux()
    try:
        aux.offer_qpos(qpos)
        wait_frames(aux, 2)
        return aux.latest_frame[1]
    finally:
        aux.stop()


def test_1_render_flags_differing_between_aux_and_main_do_not_leak(home_qpos, monkeypatch):
    """set_render_flags는 Renderer마다 따로 있는 scene에 쓴다. 두 값(그림자·반사)이 aux와 main에서 서로 달라도 main 이미지가 같다."""
    qposes = qpos_set(home_qpos)

    # (a) main은 그림자·반사 켬, aux는 설정 그대로(꺼짐)
    with StateRenderer(shadow=True, reflection=True) as main_on:
        main_on.render(home_qpos)
        before = [sha(main_on.render(q)) for q in qposes]
        aux = make_aux()
        try:
            aux.offer_qpos(home_qpos)
            during = [sha(main_on.render(q)) for q in qposes]
            wait_frames(aux, 2)
            during += [sha(main_on.render(q)) for q in qposes]
        finally:
            aux.stop()
        after = [sha(main_on.render(q)) for q in qposes]
    assert during == before + before and after == before

    # (b) main은 꺼짐(설정 그대로), aux는 켬
    with StateRenderer() as main_off:
        main_off.render(home_qpos)
        before_off = [sha(main_off.render(q)) for q in qposes]
        monkeypatch.setattr(config, "RENDER_SHADOW", True)  # aux는 렌더할 때마다 config 값을 읽는다
        monkeypatch.setattr(config, "RENDER_REFLECTION", True)
        aux = make_aux()
        try:
            aux.offer_qpos(home_qpos)
            wait_frames(aux, 2)
            aux_flags_on = aux.latest_frame[1]
            during_off = [sha(main_off.render(q)) for q in qposes]
        finally:
            aux.stop()
        after_off = [sha(main_off.render(q)) for q in qposes]
    assert during_off == before_off and after_off == before_off

    # 이 비교가 빈 것이 아님: aux에 켠 플래그가 실제로 aux 그림을 바꿨다
    monkeypatch.setattr(config, "RENDER_SHADOW", False)
    monkeypatch.setattr(config, "RENDER_REFLECTION", False)
    assert aux_frame_jpeg(home_qpos) != aux_flags_on


# --- 2. 결정적 롤아웃 동일 ---


class _AuxFeedingReplay(ReplayPolicy):
    """ReplayPolicy와 같은 ctrl을 내면서 샘플마다 관측 qpos를 aux에 넘긴다(aux가 있으면)."""

    def __init__(self, ctrl: np.ndarray, aux: AuxViewWorker | None) -> None:
        super().__init__(ctrl)
        self._aux = aux

    def act(self, k, obs):
        if self._aux is not None:
            self._aux.offer_qpos(obs.qpos)
            time.sleep(0.002)  # 보조 스레드가 실제로 겹쳐 돌 시간을 준다
        return super().act(k, obs)


def scripted_ctrl(env: EvalEnv, n: int) -> np.ndarray:
    park = env.park_ctrl()
    ctrl = np.tile(park, (n, 1))
    ctrl[:, 0] += 0.5 * np.sin(np.linspace(0.0, 3.0, n))
    ctrl[:, 1] += 0.2 * np.sin(np.linspace(0.0, 2.0, n))
    return ctrl


def run_rollout(env: EvalEnv, ctrl: np.ndarray, seed: int, aux: AuxViewWorker | None) -> dict:
    res = rollout(env, _AuxFeedingReplay(ctrl, aux), seed, stop_on_success=False, max_samples=len(ctrl) - 1, record_trace=True)
    row = res.row()
    for timing in ("render_s", "policy_s", "physics_s"):  # 벽시계 시간은 결정적이지 않다
        row.pop(timing)
    return {"row": row, "trace": sha(res.trace), "ctrl_trace": sha(res.ctrl_trace), "tail": [sha(q) for q in res.tail_qpos]}


def test_2_deterministic_rollout_is_identical_with_aux_off_on_off():
    with EvalEnv(with_renderer=False) as env:
        ctrl = scripted_ctrl(env, 120)
        off_before = run_rollout(env, ctrl, 3, None)
        aux = make_aux()
        try:
            frames_before = aux.state()["frames"]
            on = run_rollout(env, ctrl, 3, aux)
            assert aux.state()["frames"] > frames_before, "aux가 실제로 렌더하지 않았다: 이 테스트는 의미가 없다"
        finally:
            aux.stop()
        off_after = run_rollout(env, ctrl, 3, None)
    assert on == off_before
    assert off_after == off_before


HAVE_RAW = (config.RECORD_DIR / "episode_000019.npz").exists()


@pytest.mark.skipif(not HAVE_RAW, reason="data/raw가 이 컴퓨터에 없음")
def test_2b_demo_replay_result_is_identical_with_aux_on():
    """실제 시연 정답 재생(eval_replay.replay_episode)의 결과 전체(dict)가 aux가 돌 때도 같다. data/raw는 읽기만 한다."""
    from server.eval_replay import replay_episode

    store = EpisodeStore()
    meta = next(m for m in store.scan() if m["episode_index"] == 19)
    arrays = store.load_arrays(19)
    with EvalEnv(with_renderer=False) as env:
        before = replay_episode(env, arrays, meta, "home")
        aux = make_aux()
        try:
            qpos = arrays["qpos"]
            frames_before = aux.state()["frames"]
            for k in range(0, len(qpos), 5):  # 재생 동안 시연 자세를 aux에 계속 넘긴다
                aux.offer_qpos(qpos[k])
            during = replay_episode(env, arrays, meta, "home")
            wait_frames(aux, 1)
            assert aux.state()["frames"] > frames_before
        finally:
            aux.stop()
    assert during == before


# --- 3. 코드 경로 분리 ---

NOT_IMPORTING_AUX = [
    "render.py", "recorder.py", "auto.py", "auto_config.py", "auto_headless.py", "eval_env.py", "eval_config.py", "eval_metrics.py",
    "eval_replay.py", "episode_store.py", "grid.py", "seeds.py", "success.py", "policy_ipc.py", "record_camera_check.py", "record_bench.py",
    "render_frames.py", "replay_episode.py", "correction.py", "correction_config.py", "correction_store.py", "correction_headless.py",
]
TRAINING_RENDER_NAMES = {"StateRenderer", "make_record_camera", "record_view"}


def imported_and_used_names(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return modules, names


@pytest.mark.parametrize("filename", NOT_IMPORTING_AUX)
def test_3_training_recording_auto_eval_code_does_not_import_aux(filename):
    path = SERVER_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} 없음")
    modules, _ = imported_and_used_names(path)
    offenders = {m for m in modules if "aux_view" in m}
    assert not offenders, f"{filename}이 보조 화면을 import한다: {offenders}"


@pytest.mark.parametrize("filename", ["aux_view.py", "aux_view_config.py"])
def test_3_aux_does_not_use_the_training_renderer_or_camera(filename):
    _, names = imported_and_used_names(SERVER_DIR / filename)
    assert not (names & TRAINING_RENDER_NAMES), names & TRAINING_RENDER_NAMES


def test_3_importing_and_running_aux_leaves_record_config_and_hash_unchanged():
    before_snapshot = config_snapshot()
    before_hash = config_sha256(before_snapshot)
    # 학습·재생 카메라와 이미지 크기는 이 값으로 고정돼 있다(docs/07, server/render.py). 보조 화면 때문에 바뀌면 안 된다.
    assert (config.RECORD_CAMERA_AZIMUTH_DEG, config.RECORD_CAMERA_ELEVATION_DEG, config.RECORD_CAMERA_DISTANCE_M) == (110.0, -35.0, 0.8)
    assert (config.RECORD_IMAGE_HEIGHT_PX, config.RECORD_IMAGE_WIDTH_PX) == (240, 320)

    aux = make_aux()
    try:
        aux.offer_qpos(np.zeros(13))
        wait_frames(aux, 1)
    finally:
        aux.stop()

    assert config_snapshot() == before_snapshot
    assert config_sha256(config_snapshot()) == before_hash
    assert not any(name.startswith("AUX_") for name in before_snapshot)  # aux 값은 config.py 쪽 해시에 들어가지 않는다


def test_3_aux_config_imports_shared_values_instead_of_duplicating_them():
    from server import aux_view_config as ac

    assert ac.AUX_CAMERA_VIEW is config.VIEW_PRESETS["side"]
    assert ac.AUX_CAMERA_LOOKAT_M is config.VIEW_LOOKAT_M


# --- 4. 녹화 불변 ---


def fake_tick_episode(aux: AuxViewWorker | None):
    """SimWorker._loop와 같은 순서(ctrl → on_step → mj_step)로 가짜 틱을 돌려 에피소드 하나를 녹화한다."""
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    done = []
    rec = EpisodeRecorder(sink=done.append, timestep_s=TS)
    rec.begin(0, 5, (0.2, 0.0, 0.015), config.TARGET_CENTER_M)
    for step in range(1200):
        data.ctrl[:6] = np.array([0.4 * np.sin(step / 150.0), -1.74, 1.2, -1.2, 0.0, 0.0])
        rec.on_step(data.time, data.qpos, data.ctrl)
        if aux is not None:
            aux.offer_qpos(data.qpos)
            if step % 50 == 0:
                time.sleep(0.005)
        mujoco.mj_step(model, data)
    ep = rec.stop(False)
    assert ep is not None
    return ep


def test_4_recorded_episode_and_npz_bytes_are_identical_with_aux_on(tmp_path):
    ep_off = fake_tick_episode(None)
    aux = make_aux()
    try:
        frames_before = aux.state()["frames"]
        ep_on = fake_tick_episode(aux)
        wait_frames(aux, 1)
        assert aux.state()["frames"] > frames_before, "aux가 실제로 렌더하지 않았다: 이 테스트는 의미가 없다"
    finally:
        aux.stop()

    for name in ("sim_time_s", "qpos", "ctrl", "is_intervention"):
        assert sha(getattr(ep_on, name)) == sha(getattr(ep_off, name)), name
    assert (ep_on.still_ratio, ep_on.still_longest_s, ep_on.end_reason) == (ep_off.still_ratio, ep_off.still_longest_s, ep_off.end_reason)

    # 같은 에피소드 번호로 각각 저장해서 npz 파일의 바이트를 비교한다. created_at은 json에만 들어간다.
    EpisodeStore(root=tmp_path / "off").save(ep_off, TS)
    EpisodeStore(root=tmp_path / "on").save(ep_on, TS)
    assert (tmp_path / "off" / "episode_000000.npz").read_bytes() == (tmp_path / "on" / "episode_000000.npz").read_bytes()
