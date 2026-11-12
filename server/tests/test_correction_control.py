"""넘겨받기(4-2) ControlWorker 쪽: HUMAN_WAIT → HUMAN(손 3프레임), 항상 상대 이동 + 현재 집게 끝 기준, 집게 일치 판정과 안내 조건, IK 시작 자세.
가짜 시뮬레이션(FakeSim)과 가짜 시계, 가짜 손 메시지로 한 틱씩 돌린다(실제 mink IK는 ControlWorker가 만든다)."""

import numpy as np
import pytest

from server import config, correction_config as cc
from server.control import ControlWorker
from server.correction import HandoffGripper
from server.owner import ControlOwner, Owner

TIP_M = np.array([0.22, 0.02, 0.06])  # 작업 상자 안(블록 위에서 정책이 멈칫하는 높이)


class FakeSim:
    def __init__(self, owner: ControlOwner) -> None:
        self.owner = owner
        self.ctrls: list[np.ndarray] = []
        self.snapshot = None
        self.view = None
        self.uses_free_camera = False

    def set_ctrl(self, ctrl) -> None:
        self.ctrls.append(np.array(ctrl))


def hand_msg(x=0.5, y=0.5, pinch_ratio=0.9):
    return {
        "type": "hand", "t": 1.0, "detected": True, "wrist": [x, y, 0.0], "index_tip": [x, y, 0.0], "thumb_tip": [x, y, 0.0],
        "pinch": 0.1, "pinch_ratio": pinch_ratio, "palm_open": True, "hand_scale": 0.12,
    }


class Rig:
    """AUTO가 tip에서 hold ctrl로 멈춘 상태에서 넘겨받기를 요청한 직후까지 만든다."""

    def __init__(self, tip=TIP_M, gripper_ctrl=-0.1):
        self.owner = ControlOwner()
        self.sim = FakeSim(self.owner)
        self.clock = [0.0]
        self.control = ControlWorker(self.sim, clock=lambda: self.clock[0])
        self.control.prepare()
        solved = self.control._ik.solve(tip)  # 이 집게 끝 위치에 해당하는 관절(실제 정책 자세 대신)
        assert solved.ok
        self.joints = np.r_[solved.q_rad, 0.17]
        self.sim.snapshot = {"tip_m": np.array(tip, dtype=float), "joints_rad": self.joints, "block_m": np.array([0.2, 0.0, 0.015])}
        self.hold = np.r_[solved.q_rad + 0.01, gripper_ctrl]  # ctrl은 관절과 조금 다르다(추종 오차)
        assert self.owner.try_acquire_auto()
        self.control.request_hold(self.hold)
        assert self.owner.transition(Owner.AUTO, Owner.HUMAN_WAIT)

    def tick(self, msg=None, dt=0.033):
        self.clock[0] += dt
        if msg is not None:
            self.control.submit_hand(msg, self.clock[0])
        self.control.tick()

    def lost_hand(self, dt=0.033):
        """손이 안 보이는 프레임(detected false)이 왔다."""
        self.clock[0] += dt
        self.control.submit_hand(None, self.clock[0])
        self.control.tick()

    def wait_state(self):
        return self.control.state()["gripper_sync"]


def test_wait_does_not_write_ctrl_until_three_consecutive_frames():
    rig = Rig()
    rig.tick()  # 요청 처리
    assert rig.control.state()["human_session"] and rig.owner.current is Owner.HUMAN_WAIT
    rig.tick(hand_msg())
    rig.tick(hand_msg())
    assert rig.owner.current is Owner.HUMAN_WAIT and rig.sim.ctrls == []  # 2프레임: 아직 아무도 ctrl을 쓰지 않는다
    rig.lost_hand()  # 손이 사라지면 처음부터 다시 센다
    rig.tick(hand_msg())
    rig.tick(hand_msg())
    assert rig.owner.current is Owner.HUMAN_WAIT
    rig.tick(hand_msg())  # 3프레임 연속
    assert rig.owner.current is Owner.HUMAN
    assert rig.control.human_started_monotonic is not None


def test_always_starts_relative_from_the_real_tip_even_if_config_says_absolute(monkeypatch):
    monkeypatch.setattr(config, "MAPPING_MODE", "absolute")
    rig = Rig()
    for _ in range(3):
        rig.tick(hand_msg(0.3, 0.7))  # 절대 모드였다면 손 위치로 로봇이 튀었을 자리
    state = rig.control.state()
    assert state["mapping_mode"] == "relative" and state["controlling"]
    assert state["reference_hand"] is not None
    assert np.allclose(state["target_m"], TIP_M, atol=1e-3)  # 기준은 작업 상자 중심이 아니라 지금 집게 끝
    assert not np.allclose(TIP_M, config.RELATIVE_START_POSITION_M, atol=1e-3)
    # 손이 가만히 있으면 첫 1초 동안 ctrl이 변화량 제한 안에서만 움직인다
    for _ in range(30):
        rig.tick(hand_msg(0.3, 0.7))
    steps = np.abs(np.diff(np.vstack([rig.hold, *rig.sim.ctrls]), axis=0)).max()
    assert steps <= config.MAX_JOINT_STEP_RAD + 1e-9


def test_hand_leaving_and_returning_makes_a_new_reference():
    rig = Rig()
    for _ in range(3):
        rig.tick(hand_msg(0.5, 0.5))
    assert rig.control.state()["controlling"]
    rig.lost_hand()
    assert not rig.control.state()["controlling"] and rig.owner.current is Owner.HUMAN
    first_ref = rig.control.state()["reference_hand"]
    for _ in range(3):
        rig.tick(hand_msg(0.4, 0.4))
    state = rig.control.state()
    assert state["controlling"] and state["reference_hand"] != first_ref  # 기존 클러치와 같은 동작


def test_gripper_notice_when_holding_the_block_and_hand_is_open():
    """블록을 쥔 채(로봇 닫힘) 넘겨받았는데 손은 펴져 있다: 안내가 뜨고 집게는 그대로. 손을 쥐면 안내가 사라지고 그때부터 따라간다."""
    rig = Rig(gripper_ctrl=0.0)  # 닫힘 쪽(0.4보다 작음). 목표(-0.1)와 조금 다르다
    state = rig.wait_state()
    assert state["active"] is False  # 아직 control이 요청을 처리하기 전
    rig.tick()
    state = rig.wait_state()
    assert state == {"active": True, "waiting": True, "robot_closed": True}  # 대기 시작부터 안내 조건이 켜진다
    for _ in range(3):
        rig.tick(hand_msg(pinch_ratio=0.9))  # 손 폄
    assert rig.owner.current is Owner.HUMAN
    assert rig.wait_state()["waiting"] is True
    for _ in range(20):
        rig.tick(hand_msg(pinch_ratio=0.9))
    assert rig.wait_state()["waiting"] is True
    assert all(abs(c[5] - 0.0) < 1e-12 for c in rig.sim.ctrls)  # 집게 ctrl은 AUTO가 낸 값 그대로
    for _ in range(5):
        rig.tick(hand_msg(pinch_ratio=0.4))  # 애매한 구간(0.3~0.5): 아직 모른다
    assert rig.wait_state()["waiting"] is True
    rig.tick(hand_msg(pinch_ratio=0.1))  # 쥠: 로봇(닫힘)과 일치
    assert rig.wait_state() == {"active": True, "waiting": False, "robot_closed": True}  # 안내가 사라진다
    for _ in range(3):
        rig.tick(hand_msg(pinch_ratio=0.1))
    assert abs(rig.sim.ctrls[-1][5] - config.GRIPPER_CLOSED_RAD) < 1e-9  # 이제 손을 따라간다(닫힘 목표)
    gripper = np.array([c[5] for c in rig.sim.ctrls])
    assert np.abs(np.diff(np.r_[0.0, gripper])).max() <= config.MAX_GRIPPER_STEP_RAD + 1e-9
    rig.tick(hand_msg(pinch_ratio=0.9))  # 일치한 뒤에는 손을 펴면 열린다
    assert rig.sim.ctrls[-1][5] > gripper[-1]


def test_gripper_notice_disappears_when_hand_already_matches():
    rig = Rig(gripper_ctrl=config.GRIPPER_OPEN_RAD)  # 로봇 열림
    rig.tick()
    assert rig.wait_state()["waiting"] is True
    for _ in range(3):
        rig.tick(hand_msg(pinch_ratio=0.9))  # 손도 열림
    assert rig.wait_state() == {"active": True, "waiting": False, "robot_closed": False}


def test_gripper_notice_clears_when_the_session_ends():
    rig = Rig()
    rig.tick()
    for _ in range(3):
        rig.tick(hand_msg(pinch_ratio=0.9))
    assert rig.wait_state()["waiting"] is True
    assert rig.owner.transition(Owner.HUMAN, Owner.AUTO)  # 돌려줌
    rig.tick(hand_msg())
    assert rig.wait_state() == {"active": False, "waiting": False, "robot_closed": None}
    assert rig.control.state()["mapping_mode"] == config.MAPPING_MODE  # 매퍼도 설정값으로 돌아온다


def test_handoff_gripper_unit():
    g = HandoffGripper(-0.1, config.PINCH_CLOSE_BELOW, config.PINCH_OPEN_ABOVE)
    assert g.robot_closed and g.waiting and g.hold_rad == -0.1 and g.closed
    for ratio in (None, 0.9, 0.5, 0.4, 0.3, 0.31):
        g.update(ratio)
        if ratio in (None, 0.4, 0.3, 0.31):
            pass
    assert g.waiting  # 0.9(폄)만 있었고 쥔 적이 없다
    g.update(0.29)
    assert not g.waiting and g.hold_rad is None and g.closed
    g.update(0.6)
    assert not g.closed  # 일치한 뒤에는 일반 래치
    free = HandoffGripper(-0.1, config.PINCH_CLOSE_BELOW, config.PINCH_OPEN_ABOVE, require_match=False)
    assert not free.waiting and free.hold_rad is None


def test_ik_warm_start_flag_is_read_at_takeover(monkeypatch):
    for flag in (True, False):
        monkeypatch.setattr(cc, "HUMAN_IK_WARM_START", flag)
        rig = Rig()
        rig.tick()
        for _ in range(3):
            rig.tick(hand_msg())
        # warm start면 IK 내부 자세가 지금 관절과 같다. 아니면 기본 시작 자세(seed)에서 풀려 다른 관절 조합일 수 있다.
        solved = rig.control._ik.arm_q_rad
        if flag:
            assert np.abs(solved - rig.joints[:5]).max() < 0.02
        assert rig.owner.current is Owner.HUMAN


def test_height_keys_work_in_human_but_not_before_the_reference():
    rig = Rig()
    rig.tick()
    rig.control.submit_height(-1, False, rig.clock[0])
    rig.tick()
    assert rig.control.state()["target_z_m"] is None  # 아직 기준점이 없으면 무시
    for _ in range(3):
        rig.tick(hand_msg())
    z0 = rig.control.state()["target_z_m"]
    for _ in range(6):
        rig.control.submit_height(-1, False, rig.clock[0])
        rig.tick(hand_msg())
    assert rig.control.state()["target_z_m"] < z0  # 키보드 높이는 HUMAN 중 기존대로


@pytest.mark.parametrize("tip", [(0.03, 0.0, 0.35)])
def test_tip_outside_the_workspace_is_clamped_and_reported_by_target(tip):
    """집게 끝이 작업 상자 밖(home 자세)이면 첫 목표가 상자 안으로 잘린다 — 한계로 문서에 적고 측정은 correction_headless가 한다."""
    rig = Rig.__new__(Rig)
    rig.owner = ControlOwner()
    rig.sim = FakeSim(rig.owner)
    rig.clock = [0.0]
    rig.control = ControlWorker(rig.sim, clock=lambda: rig.clock[0])
    rig.control.prepare()
    rig.sim.snapshot = {"tip_m": np.array(tip), "joints_rad": np.r_[rig.control._home_ctrl()[:5], 0.9], "block_m": np.zeros(3)}
    rig.hold = rig.control._home_ctrl()
    rig.hold[5] = config.GRIPPER_OPEN_RAD
    assert rig.owner.try_acquire_auto()
    rig.control.request_hold(rig.hold)
    assert rig.owner.transition(Owner.AUTO, Owner.HUMAN_WAIT)
    rig.tick()
    for _ in range(3):
        rig.tick(hand_msg(pinch_ratio=0.9))
    target = np.array(rig.control.state()["target_m"])
    lows = [config.WORKSPACE_X_M[0], config.WORKSPACE_Y_M[0], config.WORKSPACE_Z_M[0]]
    highs = [config.WORKSPACE_X_M[1], config.WORKSPACE_Y_M[1], config.WORKSPACE_Z_M[1]]
    assert np.all(target >= np.array(lows) - 1e-9) and np.all(target <= np.array(highs) + 1e-9)
    assert abs(target[2] - tip[2]) > 0.1  # 높이가 상자 안으로 크게 잘렸다
