"""교정(4-2) 에피소드: AUTO → HUMAN_WAIT → HUMAN → AUTO 전환, 청크 큐 버리기, policy.reset, 기록의 짝(owner/ctrl/is_intervention), 시간 제한, 성공/중지.
실제 MuJoCo(SimWorker)와 AutoEpisode를 쓰고, 손 쪽(ControlWorker)은 가짜(FakeControl)로 두어 사람 역할을 테스트가 한다."""

import threading
import time

import numpy as np
import pytest
from fake_policy import InProcessPolicy, park_chunk

from server import auto as auto_module
from server import config, correction_config as cc
from server.auto import AutoEpisode
from server.correction import check_alignment, handback_jumps_rad, intervention_segments
from server.owner import Owner
from server.sim_worker import SimWorker

SEED = 1002


class FakeControl:
    def __init__(self):
        self.holds = []

    def request_hold(self, ctrl):
        self.holds.append(np.array(ctrl))


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


def wait_for(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def make_episode(worker, policy, control=None, **kwargs):
    assert worker.owner.try_acquire_auto()
    kwargs.setdefault("max_s", 30.0)
    return AutoEpisode(
        SEED, policy, exact_start=True, owner=worker.owner, control=control or FakeControl(), hold_ctrl=worker.set_ctrl, record_samples=True, **kwargs
    )


def recording_policy():
    resets: list[bool] = []

    def predict(state, image, reset, i):
        resets.append(bool(reset))
        return park_chunk(100)

    policy = InProcessPolicy(predict)
    policy.started = True
    return policy, resets


def test_takeover_handback_cycle_records_matching_owner_and_ctrl(worker):
    policy, resets = recording_policy()
    control = FakeControl()
    episode = make_episode(worker, policy, control)
    assert worker.start_auto(episode) is None
    assert wait_for(lambda: episode.progress()["sample"] >= 10)
    park = park_chunk(1)[0].astype(np.float64)
    assert episode.request_takeover() is None
    assert wait_for(lambda: worker.owner.current is Owner.HUMAN_WAIT)
    # 대기 중: 정책을 더 부르지 않고, ControlWorker에는 AUTO가 멈춘 ctrl이 넘어갔다
    calls = policy.calls
    time.sleep(0.25)
    assert policy.calls == calls and len(control.holds) == 1 and np.array_equal(control.holds[0], park)
    # 사람이 넘겨받아(전환은 ControlWorker가 한다) ctrl을 쓴다
    human = park + np.array([0.1, -0.05, 0.05, 0.0, 0.02, 0.0])
    assert worker.owner.transition(Owner.HUMAN_WAIT, Owner.HUMAN)
    worker.set_ctrl(human)
    time.sleep(0.4)
    assert policy.calls == calls  # HUMAN 중에도 정책을 부르지 않는다
    assert episode.request_handback() is None
    assert wait_for(lambda: worker.owner.current is Owner.AUTO)
    assert wait_for(lambda: policy.calls == calls + 1)  # 다음 샘플에서 즉시 새 청크를 요청
    time.sleep(0.2)
    episode.request_stop()
    assert episode.done.wait(5)
    result = episode.result
    rec = result.recording
    assert result.outcome == "stopped" and check_alignment(rec) == []
    assert resets == [True, True]  # 처음과, 돌려준 직후(policy.reset)
    owner = rec.owner
    assert set(owner.tolist()) == {cc.OWNER_AUTO, cc.OWNER_HUMAN_WAIT, cc.OWNER_HUMAN}
    # 주체 순서: AUTO... WAIT... HUMAN... AUTO...
    order = [int(owner[0])] + [int(owner[k]) for k in range(1, len(owner)) if owner[k] != owner[k - 1]]
    assert order == [cc.OWNER_AUTO, cc.OWNER_HUMAN_WAIT, cc.OWNER_HUMAN, cc.OWNER_AUTO]
    assert np.array_equal(rec.is_intervention, owner == cc.OWNER_HUMAN)
    segments = intervention_segments(owner)
    assert len(segments) == 1 and rec.intervention_count == 1
    start, end = segments[0]
    assert np.all(owner[start : end + 1] == cc.OWNER_HUMAN) and (start == 0 or owner[start - 1] != cc.OWNER_HUMAN)
    # 짝: 대기 중 ctrl은 AUTO가 멈춘 값 그대로, AUTO 샘플은 정책 행동(park), HUMAN 샘플은 park 또는 사람이 낸 ctrl(적용은 다음 틱 시작부터)
    for k in range(len(owner)):
        if owner[k] in (cc.OWNER_AUTO, cc.OWNER_HUMAN_WAIT):
            assert np.array_equal(rec.ctrl[k], park), k
        else:
            assert np.array_equal(rec.ctrl[k], human) or np.array_equal(rec.ctrl[k], park), k
    human_rows = rec.ctrl[owner == cc.OWNER_HUMAN]
    assert np.array_equal(human_rows[-1], human)
    first_human = int(np.argmax(np.all(human_rows == human, axis=1)))
    assert np.all(np.all(human_rows[first_human:] == human, axis=1))  # 사람 ctrl이 적용된 뒤에는 계속 그 값
    # 돌려준 순간의 점프 = 직전 사람 ctrl과 정책 첫 행동의 최대 관절 차이
    assert handback_jumps_rad(owner, rec.ctrl) == [pytest.approx(float(np.max(np.abs(human - park))))]
    assert result.latencies["takeover_request_to_wait_ms"] and result.latencies["handback_request_to_auto_ms"]
    assert rec.qpos.shape == (len(owner), 13) and np.all(np.diff(rec.sim_time_s) > 0)


def test_takeover_during_a_policy_call_discards_the_chunk_and_holds_ctrl(worker):
    in_call = threading.Event()
    calls = []

    def slow_predict(state, image, reset, i):
        calls.append(i)
        in_call.set()
        time.sleep(0.4)
        return park_chunk(100) + 0.2  # 적용되면 ctrl이 park와 달라진다

    policy = InProcessPolicy(slow_predict)
    policy.started = True
    episode = make_episode(worker, policy)
    assert worker.start_auto(episode) is None
    assert in_call.wait(5)
    assert episode.request_takeover() is None  # 정책이 답하는 동안 눌렀다
    assert wait_for(lambda: worker.owner.current is Owner.HUMAN_WAIT)
    time.sleep(0.3)
    episode.request_stop()
    assert episode.done.wait(5)
    rec = episode.result.recording
    assert calls == [0]  # 요청은 한 번뿐이고 받아 온 청크는 버려졌다
    assert np.all(rec.owner == cc.OWNER_HUMAN_WAIT)  # 첫 샘플부터 대기: AUTO 행동이 한 번도 적용되지 않았다
    assert np.all(rec.ctrl == rec.ctrl[0]) and not np.allclose(rec.ctrl[0], park_chunk(1)[0] + 0.2)


def test_requests_are_refused_in_the_wrong_state(worker):
    policy, _ = recording_policy()
    episode = make_episode(worker, policy)
    assert "실행 중이 아닙니다" in episode.request_takeover()  # 아직 시작 전
    assert worker.start_auto(episode) is None
    assert wait_for(lambda: episode.progress()["sample"] >= 3)
    assert "조종 중이 아닙니다" in episode.request_handback()  # AUTO에서 돌려주기
    assert episode.request_takeover() is None
    assert "이미" in (episode.request_takeover() or "") or worker.owner.current is not Owner.AUTO
    assert wait_for(lambda: worker.owner.current is Owner.HUMAN_WAIT)
    assert "이미 손을 기다리는 중" in episode.request_takeover()
    assert "아직 손을 기다리는 중" in episode.request_handback()  # 대기 중 돌려주기는 안 된다
    assert worker.owner.current is Owner.HUMAN_WAIT
    episode.request_stop()  # HUMAN_WAIT 중 중지
    assert episode.done.wait(5)
    assert episode.result.outcome == "stopped" and episode.result.end_reason == "user_stop"
    assert "실행 중이 아닙니다" in episode.request_takeover()  # 끝난 뒤
    plain = AutoEpisode(SEED, policy)  # owner 없이 만든 에피소드는 전환을 지원하지 않는다
    assert "지원하지 않습니다" in plain.request_takeover()


def test_auto_limit_counts_only_auto_samples_and_total_limit_ends_the_episode(worker):
    policy, _ = recording_policy()
    episode = make_episode(worker, policy, max_s=1.0)
    assert worker.start_auto(episode) is None
    assert wait_for(lambda: episode.progress()["sample"] >= 15)  # AUTO 0.5초
    assert episode.request_takeover() is None
    assert wait_for(lambda: worker.owner.current is Owner.HUMAN_WAIT)
    time.sleep(1.2)  # 손을 기다린 시간은 AUTO 시간에 들어가지 않는다
    assert worker.owner.transition(Owner.HUMAN_WAIT, Owner.HUMAN)
    time.sleep(0.3)
    assert episode.request_handback() is None
    assert episode.done.wait(10)
    result = episode.result
    times = auto_module.owner_time_s(result.recording.owner)
    assert result.outcome == "timeout" and result.end_reason == "auto_limit"
    assert times["auto_time_s"] == pytest.approx(1.0, abs=0.1)  # AUTO 샘플만 1초(AUTO_MAX_S 규칙)
    assert times["total_s"] > 2.2 and times["human_wait_time_s"] > 1.0 and times["human_time_s"] > 0.2


def test_total_limit_ends_even_if_the_human_never_shows_a_hand(worker):
    policy, _ = recording_policy()
    episode = make_episode(worker, policy, max_s=30.0, total_max_s=1.5)
    assert worker.start_auto(episode) is None
    assert wait_for(lambda: episode.progress()["sample"] >= 5)
    assert episode.request_takeover() is None
    assert episode.done.wait(10)
    result = episode.result
    assert result.outcome == "timeout" and result.end_reason == "total_limit"
    assert result.recording.n_ticks == 1.5 * 30 + 1  # 총 상한 45샘플 + 상한에 닿은 마지막 판정 샘플
    assert result.recording.human_samples == 0  # 손을 한 번도 안 보였으니 사람 구간은 없다


def test_success_is_judged_while_human_controls(worker, monkeypatch):
    policy, _ = recording_policy()
    monkeypatch.setattr(auto_module, "is_success", lambda block_m, gripper_rad, **kw: worker.owner.current is Owner.HUMAN)
    episode = make_episode(worker, policy)
    assert worker.start_auto(episode) is None
    assert wait_for(lambda: episode.progress()["sample"] >= 5)
    assert episode.request_takeover() is None
    assert wait_for(lambda: worker.owner.current is Owner.HUMAN_WAIT)
    assert worker.owner.transition(Owner.HUMAN_WAIT, Owner.HUMAN)
    assert episode.done.wait(10)
    result = episode.result
    assert result.outcome == "success" and result.end_reason == "success"
    assert result.recording.owner[-1] == cc.OWNER_HUMAN and result.time_to_success_s is not None  # HUMAN에서 성공으로 끝났다
    assert check_alignment(result.recording) == []


def test_policy_death_after_handback_keeps_the_recording(worker):
    policy = InProcessPolicy(lambda s, i, r, n: park_chunk(100))
    policy.started = True
    policy.fail_request_after = 1  # 두 번째 요청(돌려준 직후)에서 죽는다
    episode = make_episode(worker, policy)
    assert worker.start_auto(episode) is None
    assert wait_for(lambda: episode.progress()["sample"] >= 5)
    assert episode.request_takeover() is None
    assert wait_for(lambda: worker.owner.current is Owner.HUMAN_WAIT)
    assert worker.owner.transition(Owner.HUMAN_WAIT, Owner.HUMAN)
    time.sleep(0.3)
    assert episode.request_handback() is None
    assert episode.done.wait(10)
    result = episode.result
    assert result.outcome == "error" and "끊겼" in result.error and result.end_reason == "error"
    assert result.recording.human_samples > 0 and check_alignment(result.recording) == []  # 사람 구간이 있으니 저장 선택을 물을 수 있다
