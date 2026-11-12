"""제어 주체(server/owner.py): 한 순간에 하나, 허용된 전환만. 그리고 ControlWorker가 AUTO 중에 입력을 버리고 끝나면 이어받는지."""

import threading

import numpy as np

from server import config
from server.control import ControlWorker
from server.owner import ALLOWED_TRANSITIONS, REFUSED_TRANSITIONS, ControlOwner, Owner


def test_starts_idle_and_idle_auto_transitions():
    owner = ControlOwner()
    assert owner.current is Owner.IDLE and not owner.is_auto
    assert ALLOWED_TRANSITIONS.isdisjoint(REFUSED_TRANSITIONS)
    assert owner.try_acquire_auto() and owner.is_auto and owner.current is Owner.AUTO
    assert not owner.try_acquire_auto()  # 이미 AUTO
    assert owner.release_auto() and owner.current is Owner.IDLE
    assert not owner.release_auto()


def test_human_transitions_follow_the_4_2_rules():
    """4-2: AUTO → HUMAN_WAIT → HUMAN → AUTO(돌려주기)만 허용하고, 건너뛰기와 거꾸로 가기는 모두 거부한다(이전에는 HUMAN 전환 전부 거부)."""
    owner = ControlOwner()
    for expected, new in REFUSED_TRANSITIONS:
        assert not owner.transition(expected, new)
    assert owner.current is Owner.IDLE
    assert not owner.transition(Owner.IDLE, Owner.HUMAN_WAIT)  # IDLE에서 바로 넘겨받기는 안 된다
    assert owner.try_acquire_auto()
    assert not owner.transition(Owner.AUTO, Owner.HUMAN)  # 손을 기다리는 단계를 건너뛸 수 없다
    assert owner.current is Owner.AUTO
    assert owner.transition(Owner.AUTO, Owner.HUMAN_WAIT) and owner.is_busy and not owner.control_writes_ctrl
    assert not owner.transition(Owner.HUMAN_WAIT, Owner.AUTO)  # 대기에서 AUTO로 곧장 돌아갈 수 없다
    assert owner.transition(Owner.HUMAN_WAIT, Owner.HUMAN) and owner.control_writes_ctrl
    assert not owner.transition(Owner.HUMAN, Owner.HUMAN_WAIT)
    assert owner.transition(Owner.HUMAN, Owner.AUTO) and owner.is_auto
    assert owner.transition(Owner.AUTO, Owner.HUMAN_WAIT)
    assert owner.release_to_idle() and owner.current is Owner.IDLE and not owner.is_busy  # 끝나면 어디서든 IDLE
    assert not owner.release_to_idle()


def test_only_one_thread_wins_the_acquire():
    owner = ControlOwner()
    wins = []
    barrier = threading.Barrier(8)

    def run():
        barrier.wait()
        wins.append(owner.try_acquire_auto())

    threads = [threading.Thread(target=run) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert wins.count(True) == 1


class FakeSim:
    """ControlWorker가 쓰는 부분만 있는 가짜 시뮬레이션(실제 mink IK는 ControlWorker가 만든다)."""

    def __init__(self, owner):
        self.owner = owner
        self.ctrls = []
        self.snapshot = None
        self.view = None
        self.uses_free_camera = False

    def set_ctrl(self, ctrl):
        self.ctrls.append(np.array(ctrl))


def hand_msg(x=0.5, y=0.5):
    return {
        "type": "hand", "t": 1.0, "detected": True, "wrist": [x, y, 0.0], "index_tip": [x, y, 0.0], "thumb_tip": [x, y, 0.0],
        "pinch": 0.1, "pinch_ratio": 0.9, "palm_open": True, "hand_scale": 0.12,
    }


def test_control_worker_drops_hand_and_height_while_auto_then_adopts_ctrl():
    owner = ControlOwner()
    sim = FakeSim(owner)
    clock = [0.0]
    control = ControlWorker(sim, clock=lambda: clock[0])
    control.prepare()
    park = control.park_ctrl()
    assert park[5] == config.GRIPPER_OPEN_RAD

    # AUTO 동안: 손·높이 메시지는 쌓이지 않고 ctrl도 안 쓴다
    assert owner.try_acquire_auto()
    for i in range(10):
        clock[0] += 0.033
        control.submit_hand(hand_msg(), clock[0])
        control.submit_height(1, False, clock[0])
        control.tick()
    assert sim.ctrls == []
    assert control._pending is None and not control.state()["controlling"]

    # AUTO가 끝나며 마지막 ctrl을 넘긴다: 다음 틱에 이어받는다(튀지 않게 ControlWorker의 ctrl이 그 값이 된다)
    final = park + np.array([0.1, -0.1, 0.1, -0.1, 0.1, 0.0])
    owner.release_auto()
    control.request_adopt(final)
    control.tick()
    assert np.array_equal(control._ctrl, final)
    assert sim.ctrls == []  # 이어받기만 하고 시뮬레이션에 새로 쓰지는 않는다(SimWorker는 AutoRunner가 같은 값을 이미 갖게 했다)

    # 이후 손이 보이면 이 자세에서 출발해 한 틱에 최대 변화량만큼만 움직인다
    for i in range(6):
        clock[0] += 0.033
        control.submit_hand(hand_msg(0.4, 0.4), clock[0])
        control.tick()
    assert sim.ctrls
    first_step = np.abs(sim.ctrls[0][:5] - final[:5]).max()
    assert first_step <= config.MAX_JOINT_STEP_RAD + 1e-9
