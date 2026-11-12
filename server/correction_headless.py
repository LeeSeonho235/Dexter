"""교정(4-2) 헤드리스 검증과 측정. 브라우저 없이 서버 객체(SimWorker, ControlWorker, AutoEpisode)를 직접 돌린다. 서버 환경(server/.venv)에서 실행.

  server/.venv/bin/python -m server.correction_headless --scenario handback         # 가짜 정책 → 넘겨받기 → 교정 → 돌려주기 → 성공
  server/.venv/bin/python -m server.correction_headless --scenario human_finishes   # 사람이 끝까지 해서 HUMAN에서 성공
  server/.venv/bin/python -m server.correction_headless --scenario gripper          # 블록을 쥔 채 넘겨받기: 손을 편 채 시작해도 블록이 떨어지지 않는가
  server/.venv/bin/python -m server.correction_headless --scenario ik --repeat 3    # 넘겨받은 직후 1초 관절 변화: ik.set_arm vs ik.reset (가짜 정책)
  server/.venv/bin/python -m server.correction_headless --scenario ik_real --seed 1000 --repeat 2   # 같은 비교를 실제 체크포인트 자세에서
  server/.venv/bin/python -m server.correction_headless --scenario outside          # 집게 끝이 작업 상자 밖일 때 넘겨받기
  server/.venv/bin/python -m server.correction_headless --scenario real --seed 1000 # 실제 체크포인트(정책 프로세스) + 스크립트된 사람

가짜 정책(ScriptedPolicy)은 블록 위치를 아는 상태 되먹임 제어기(IK로 접근 → 내림 → 쥠 → 듦 → 목표 위로 → 내려놓음)라서
사람이 중간에 넘겨받았다가 돌려줘도 지금 자세에서 이어간다. 가짜 사람(FakeHuman)은 실제 경로와 같은 손 메시지(30Hz)와 키보드 높이 명령만 쓴다.
평가 시드(1000~1049)도 여기서는 내부 함수로 쓸 수 있다(화면의 AUTO는 막혀 있다). 저장하는 파일은 임시 폴더의 숫자 파일뿐이다.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from server import config, correction_config as cc  # noqa: E402
from server import policy_ipc as ipc  # noqa: E402
from server.auto import AutoEpisode  # noqa: E402
from server.auto_headless import make_worker  # noqa: E402
from server.control import ControlWorker  # noqa: E402
from server.correction import (  # noqa: E402
    check_alignment,
    handback_jumps_rad,
    intervention_segments,
    owner_time_s,
    takeover_first_second_steps_rad,
    typical_step_rad,
)
from server.correction_store import CorrectionStore  # noqa: E402
from server.ik import ArmIK, limit_joint_step  # noqa: E402
from server.mapping import screen_step_to_world, view_basis  # noqa: E402
from server.owner import Owner  # noqa: E402
from server.seeds import block_xy_for_seed  # noqa: E402

FPS = float(cc.CORRECTION_FPS)
OPEN_PINCH, CLOSED_PINCH = 0.9, 0.1  # 손을 편 / 쥔 pinch_ratio(펴면 0.5 위, 쥐면 0.3 아래: server/config.py)
VIEWERS = 0  # --viewers: 1이면 화면 보는 브라우저가 있는 것처럼 화면 렌더링(15fps, JPEG)도 켠다(틱 간격 측정용)


def wait_for(condition, timeout_s: float, poll_s: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(poll_s)
    return condition()


# --- 가짜 정책 ---


class ScriptedPolicy:
    """PolicyClient처럼 request/status/touch/shutdown/ensure_started를 가진 가짜. 관측(qpos 6개)과 블록 위치로 되먹임 제어한다.
    reset=True(돌려준 직후 policy.reset())를 받아도 진행 단계는 그대로 둔다: 실제 ACT도 reset은 큐만 비운다."""

    STEP_RAD = 0.05  # 행동 한 개(30Hz)에서 바뀌는 최대 관절 각도

    def __init__(self, block_fn, target_xy=config.TARGET_CENTER_M, n_action_steps: int = 10) -> None:
        self._block_fn = block_fn
        self._target_xy = np.array(target_xy, dtype=float)
        self._n = n_action_steps
        self._ik = ArmIK()
        self.phase = 0
        self._close_samples = 0
        self.started = True
        self.calls = 0
        self.resets = 0
        self.log: list[tuple[int, int, float]] = []  # (요청 번호, 단계, 집게 끝 높이)

    # PolicyClient 모양
    def ensure_started(self, cancel=None) -> None:
        self.started = True

    def status(self) -> dict:
        return {"state": "on", "idle_remaining_s": 600.0, "load_s": 0.1, "pid": 1}

    def touch(self) -> None:
        pass

    def shutdown(self, timeout_s: float = 5.0) -> None:
        pass

    def _goal(self, tip: np.ndarray, block: np.ndarray) -> tuple[np.ndarray, float]:
        z_floor = config.WORKSPACE_Z_M[0]
        open_rad, closed_rad = config.GRIPPER_OPEN_RAD, config.GRIPPER_CLOSED_RAD
        xy_err_block = float(np.linalg.norm(tip[:2] - block[:2]))
        xy_err_target = float(np.linalg.norm(tip[:2] - self._target_xy))
        if self.phase == 0:
            goal, grip = np.array([block[0], block[1], 0.10]), open_rad
            if xy_err_block < 0.008 and abs(tip[2] - 0.10) < 0.012:
                self.phase = 1
        elif self.phase == 1:
            goal, grip = np.array([block[0], block[1], z_floor]), open_rad
            if tip[2] < z_floor + 0.007:
                self.phase = 2
        elif self.phase == 2:
            goal, grip = np.array([tip[0], tip[1], z_floor]), closed_rad
            self._close_samples += self._n
            if self._close_samples >= 30:
                self.phase = 3
        elif self.phase == 3:
            goal, grip = np.array([tip[0], tip[1], 0.08]), closed_rad
            if tip[2] > 0.07:
                self.phase = 4
        elif self.phase == 4:
            goal, grip = np.array([self._target_xy[0], self._target_xy[1], 0.08]), closed_rad
            if xy_err_target < 0.008:
                self.phase = 5
        elif self.phase == 5:
            goal, grip = np.array([self._target_xy[0], self._target_xy[1], 0.03]), closed_rad
            if tip[2] < 0.04:
                self.phase = 6
        else:
            goal, grip = np.array([self._target_xy[0], self._target_xy[1], 0.03]), open_rad
        return goal, grip

    def request(self, state: np.ndarray, image, reset: bool) -> ipc.ChunkResult:
        t0 = time.perf_counter()
        self.calls += 1
        self.resets += int(bool(reset))
        q = np.asarray(state, dtype=float)
        self._ik.set_arm(q[:5])
        tip = self._ik.tip_position_m()
        block = np.asarray(self._block_fn(), dtype=float)
        goal, grip = self._goal(tip, block)
        solved = self._ik.solve(np.clip(goal, [0.18, -0.18, 0.015], [0.28, 0.18, 0.12]))
        arm_goal = solved.q_rad if solved.ok else q[:5]
        ctrl = q[:6].copy()
        rows = []
        for _ in range(self._n):
            ctrl[:5] = limit_joint_step(ctrl[:5], arm_goal, self.STEP_RAD)
            ctrl[5] = limit_joint_step(np.array([ctrl[5]]), np.array([grip]), config.MAX_GRIPPER_STEP_RAD)[0]
            rows.append(ctrl.copy())
        self.log.append((self.calls, self.phase, float(tip[2])))
        return ipc.ChunkResult(actions=np.array(rows, dtype=np.float64), infer_ms=1.0, total_ms=(time.perf_counter() - t0) * 1000.0)


# --- 가짜 사람 ---


class FakeHuman:
    """실제 경로와 같은 입력만 쓴다: 손 메시지(wrist 위치, pinch_ratio)와 키보드 높이(dir). 30Hz로 보낸다.
    send_hand(msg)와 send_height(dir, slow)는 바꿔 끼울 수 있다(기본은 ControlWorker에 직접, 앱 시험에서는 WebSocket)."""

    def __init__(self, control: ControlWorker, worker, *, send_hand=None, send_height=None, hz: float = FPS) -> None:
        self.control, self.worker = control, worker
        self._send_hand = send_hand or (lambda msg: control.submit_hand(msg, time.monotonic()))
        self._send_height = send_height or (lambda direction, slow: control.submit_height(direction, slow, time.monotonic()))
        self._dt = 1.0 / hz
        self._next = time.monotonic()
        self.hx, self.hy = 0.5, 0.5
        self.pinch_ratio = OPEN_PINCH
        self.frames = 0
        self.frame_log: list[dict] = []  # 프레임마다 (시각, 블록 z, 집게 안내 대기, 소유자)
        self.third_frame_sent_s: float | None = None
        self._consecutive = 0

    def _msg(self, detected: bool = True) -> dict:
        if not detected:
            return {"type": "hand", "t": time.monotonic(), "detected": False, "wrist": None, "index_tip": None, "thumb_tip": None,
                    "pinch": None, "pinch_ratio": None, "palm_open": None, "hand_scale": None}
        p = [self.hx, self.hy, 0.0]
        return {"type": "hand", "t": time.monotonic(), "detected": True, "wrist": p, "index_tip": p, "thumb_tip": p,
                "pinch": 0.1, "pinch_ratio": self.pinch_ratio, "palm_open": True, "hand_scale": 0.12}

    def frame(self, detected: bool = True) -> None:
        """한 프레임 보내고 다음 30Hz 시각까지 기다린다. 이 순간의 블록 높이와 안내 상태를 기록한다."""
        self._next += self._dt
        time.sleep(max(0.0, self._next - time.monotonic()))
        self._send_hand(self._msg(detected))
        self.frames += 1
        self._consecutive = self._consecutive + 1 if detected else 0
        if self._consecutive == config.START_CONSECUTIVE_FRAMES and self.third_frame_sent_s is None:
            self.third_frame_sent_s = time.monotonic()
        snapshot = self.worker.snapshot
        sync = self.control.state()["gripper_sync"]
        self.frame_log.append(
            {"t": time.monotonic(), "block_z": None if snapshot is None else float(snapshot["block_m"][2]),
             "tip": None if snapshot is None else [float(v) for v in snapshot["tip_m"]], "waiting": sync["waiting"],
             "owner": self.worker.owner.current.value}
        )

    def hold(self, seconds: float, pinch: float | None = None, detected: bool = True) -> None:
        if pinch is not None:
            self.pinch_ratio = pinch
        for _ in range(max(1, int(round(seconds / self._dt)))):
            self.frame(detected)

    def move_target_to(self, goal_m, *, pinch: float | None = None, speed_m_s: float = 0.12, tol_m: float = 0.004, timeout_s: float = 20.0,
                       stop_if=None) -> bool:
        """명령 목표(control.state()["target_m"])가 goal_m에 올 때까지: 앞뒤·좌우는 손 이동으로, 높이는 키보드로. 도착하면 True."""
        if pinch is not None:
            self.pinch_ratio = pinch
        goal = np.asarray(goal_m, dtype=float)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if stop_if is not None and stop_if():
                self._send_height(0, False)
                return False
            state = self.control.state()
            target = state["target_m"]
            if target is None or not state["controlling"]:
                self.frame()  # 기준점이 잡힐 때까지 손만 보인다
                continue
            err = goal - np.asarray(target)
            if float(np.linalg.norm(err[:2])) < tol_m and abs(err[2]) < tol_m:
                self._send_height(0, False)
                return True
            step = err[:2]
            norm = float(np.linalg.norm(step))
            if norm > speed_m_s * self._dt:
                step = step / norm * speed_m_s * self._dt
            azimuth = self.control._view_azimuth_deg()
            forward, right = view_basis(azimuth)
            gain, aspect = config.RELATIVE_GAIN_XY_M_PER_WIDTH, config.IMAGE_ASPECT
            self.hx += -float(step @ right) / gain
            self.hy += -float(step @ forward) * aspect / gain
            self._send_height(0 if abs(err[2]) < tol_m else (1 if err[2] > 0 else -1), False)
            self.frame()
        self._send_height(0, False)
        return False

    def release_height(self) -> None:
        self._send_height(0, False)


def human_pick_and_place(human: FakeHuman, stack: "Stack", episode: AutoEpisode, *, drop_after_carry: bool = True, stop_if=None) -> dict:
    """사람이 지금 자세에서 블록을 집어 목표에 놓는다(손 입력만). 반환: 단계별 시각 등 요약."""
    steps: dict = {}
    block = lambda: np.array(stack.worker.snapshot["block_m"])  # noqa: E731
    robot_closed = lambda: stack.control.state()["gripper_sync"]["robot_closed"]  # noqa: E731
    stop = lambda: episode.done.is_set() or (stop_if is not None and stop_if())  # noqa: E731
    # 1) 손을 보이고 집게를 로봇에 맞춘다
    wait_started = time.monotonic()
    match_pinch = CLOSED_PINCH if robot_closed() else OPEN_PINCH
    human.hold(0.4, match_pinch)
    steps["synced_after_s"] = time.monotonic() - wait_started
    if stop():
        return steps
    # 2) 집게를 열고 블록 위로(높이 0.10) → 내리기 → 쥐기 → 들기 → 목표 위로 → 내리기 → 놓기
    b = block()
    human.move_target_to([float(b[0]), float(b[1]), 0.10], pinch=OPEN_PINCH, stop_if=stop)
    b = block()
    human.move_target_to([float(b[0]), float(b[1]), 0.10], pinch=OPEN_PINCH, stop_if=stop)
    human.move_target_to([float(b[0]), float(b[1]), config.WORKSPACE_Z_M[0]], pinch=OPEN_PINCH, stop_if=stop)
    human.hold(0.5, OPEN_PINCH)
    steps["grasp_tip_z"] = float(stack.worker.snapshot["tip_m"][2])
    human.hold(1.2, CLOSED_PINCH)
    human.move_target_to([float(b[0]), float(b[1]), 0.08], pinch=CLOSED_PINCH, stop_if=stop)
    steps["lifted_block_z"] = float(stack.worker.snapshot["block_m"][2])
    tx, ty = config.TARGET_CENTER_M
    human.move_target_to([tx, ty, 0.08], pinch=CLOSED_PINCH, stop_if=stop)
    human.move_target_to([tx, ty, 0.03], pinch=CLOSED_PINCH, stop_if=stop)
    if drop_after_carry:
        human.hold(1.2, OPEN_PINCH)
        human.hold(0.8, OPEN_PINCH)
    return steps


# --- 시험 장치 ---


class Stack:
    """SimWorker + ControlWorker(+ 틱 간격을 주체별로 기록). 앱과 같은 순서로 에피소드를 맡긴다."""

    def __init__(self, viewers: int | None = None) -> None:
        self.worker = make_worker(VIEWERS if viewers is None else viewers)
        self.control = ControlWorker(self.worker)
        self.control.start()
        self.tick_log: list[tuple[str, float]] = []
        original = self.worker.tick_stats.add

        def add(interval_s: float) -> None:
            self.tick_log.append((self.worker.owner.current.value, interval_s * 1000.0))
            original(interval_s)

        self.worker.tick_stats.add = add

    def close(self) -> None:
        self.control.stop()
        self.worker.stop()

    def reset_block(self, seed: int) -> None:
        done = threading.Event()
        outcome: dict = {}

        def on_reset(_seed, ok, reason):
            outcome.update(ok=ok, reason=reason)
            done.set()

        reason = self.control.request_reset(seed, block_xy_for_seed(seed), on_reset)
        assert reason is None, reason
        assert done.wait(30) and outcome["ok"], outcome
        time.sleep(0.1)

    def start(self, seed: int, policy, *, max_s: float = 45.0, total_max_s: float = cc.TOTAL_MAX_S) -> AutoEpisode:
        worker, control = self.worker, self.control

        def on_finish(result) -> None:  # AutoRunner._on_finished와 같은 마무리
            worker.set_ctrl(result.final_ctrl)
            control.request_adopt(result.final_ctrl)
            worker.owner.release_to_idle()

        assert worker.owner.try_acquire_auto()
        episode = AutoEpisode(
            seed, policy, max_s=max_s, on_finish=on_finish, owner=worker.owner, control=control, hold_ctrl=worker.set_ctrl,
            record_samples=True, total_max_s=total_max_s,
        )
        assert worker.start_auto(episode) is None
        return episode

    def ticks_for(self, owner: str) -> dict:
        values = [ms for who, ms in self.tick_log if who == owner]
        if not values:
            return {"n": 0}
        return {"n": len(values), "mean_ms": round(float(np.mean(values)), 2), "max_ms": round(float(np.max(values)), 2),
                "over_40ms": int(sum(v > 40 for v in values)), "over_50ms": int(sum(v > 50 for v in values))}


# --- 분석 ---


def analyze(result, store_dir: Path | None = None) -> dict:
    rec = result.recording
    out: dict = {"outcome": result.outcome, "end_reason": result.end_reason, "n_samples": result.n_samples, "error": result.error}
    if rec is None:
        return out
    times = owner_time_s(rec.owner, rec.fps)
    out.update(
        {k: round(v, 2) for k, v in times.items()},
        interventions=[list(s) for s in intervention_segments(rec.owner)],
        alignment_problems=check_alignment(rec),
        handback_jump_rad=[round(v, 4) for v in handback_jumps_rad(rec.owner, rec.ctrl)],
        takeover_first_second_max_step_rad=[round(v, 4) for v in takeover_first_second_steps_rad(rec.owner, rec.ctrl, rec.fps)],
        typical_step_rad={k: round(v, 4) for k, v in typical_step_rad(rec.ctrl).items()},
        latencies_ms={k: [round(x, 1) for x in v] for k, v in result.latencies.items()},
    )
    if store_dir is not None and rec.human_samples > 0:
        store = CorrectionStore(store_dir)
        meta = store.save(rec)
        files = sorted(Path(store_dir).glob(f"correction_{meta['episode_index']:06d}.*"))
        out["saved_files"] = {f.name: f.stat().st_size for f in files}
    return out


def arm_steps_after_human_start(rec, horizon_s: float = 1.0) -> float:
    """HUMAN 구간이 시작된 직후 1초 동안 샘플 사이 ctrl의 팔 관절(앞 5개) 최대 변화(rad/틱)."""
    segments = intervention_segments(rec.owner)
    if not segments:
        return float("nan")
    start = segments[0][0]
    stop = min(len(rec.owner), start + int(round(horizon_s * rec.fps)) + 1)
    diffs = np.abs(np.diff(rec.ctrl[max(0, start - 1) : stop, :5], axis=0))
    return float(diffs.max())


# --- 시나리오 ---


def takeover_when_carrying(stack: Stack, policy: ScriptedPolicy, episode: AutoEpisode, timeout_s: float = 60.0) -> float:
    """정책이 블록을 들고 옮기는 단계(phase 4)가 되면 넘겨받기를 요청한다. 요청한 시각(monotonic)."""
    assert wait_for(lambda: policy.phase >= 4 or episode.done.is_set(), timeout_s), "scripted policy never reached carrying"
    assert not episode.done.is_set(), episode.result
    reason = episode.request_takeover()
    assert reason is None, reason
    return time.monotonic()


def scenario_handback(seed: int, store_dir: Path | None, hand_open_s: float = 0.8) -> dict:
    """가짜 정책이 블록을 들고 가다가 → 넘겨받기(손을 편 채 시작: 집게가 안 맞음) → 집게 맞추기 → 조금 교정 → 돌려주기 → 정책이 이어서 성공."""
    stack = Stack()
    try:
        stack.reset_block(seed)
        policy = ScriptedPolicy(lambda: stack.worker.snapshot["block_m"])
        episode = stack.start(seed, policy)
        t_req = takeover_when_carrying(stack, policy, episode)
        assert wait_for(lambda: stack.worker.owner.current is Owner.HUMAN_WAIT, 2.0)
        wait_ms = (time.monotonic() - t_req) * 1000.0
        human = FakeHuman(stack.control, stack.worker)
        human.hold(hand_open_s, OPEN_PINCH)  # 로봇은 쥔 채인데 손은 펴져 있다
        human_started = stack.control.human_started_monotonic
        sync_notice_during_open = [f["waiting"] for f in human.frame_log]
        z_open = [f["block_z"] for f in human.frame_log if f["owner"] == "HUMAN"]
        human.hold(0.4, CLOSED_PINCH)  # 쥠 → 일치
        z_all = [f["block_z"] for f in human.frame_log]
        notice_off_index = next((i for i, f in enumerate(human.frame_log) if f["owner"] == "HUMAN" and not f["waiting"]), None)
        tip = np.array(stack.worker.snapshot["tip_m"])
        human.move_target_to([tip[0] - 0.015, tip[1], tip[2]], pinch=CLOSED_PINCH)  # 교정: 조금 옮긴다
        human.hold(0.3, CLOSED_PINCH)
        t_handback = time.monotonic()
        reason = episode.request_handback()
        assert reason is None, reason
        assert wait_for(lambda: stack.worker.owner.current is Owner.AUTO or episode.done.is_set(), 2.0)
        handback_ms = (time.monotonic() - t_handback) * 1000.0
        assert episode.done.wait(90), "episode did not finish"
        out = analyze(episode.result, store_dir)
        out.update(
            takeover_request_to_wait_ms=round(wait_ms, 1), handback_request_to_auto_ms=round(handback_ms, 1),
            third_frame_to_human_ms=None if human_started is None or human.third_frame_sent_s is None
            else round((human_started - human.third_frame_sent_s) * 1000.0, 1),
            block_z_min_while_hand_open_m=None if not z_open else round(min(z_open), 4),
            block_z_max_while_hand_open_m=None if not z_open else round(max(z_open), 4),
            block_z_min_overall_m=round(min(z for z in z_all if z is not None), 4),
            notice_on_frames=int(sum(f["waiting"] for f in human.frame_log if f["owner"] == "HUMAN")),
            notice_off_at_frame=notice_off_index, policy_resets=policy.resets, human_tick_ms=stack.ticks_for("HUMAN"),
            auto_tick_ms=stack.ticks_for("AUTO"), wait_tick_ms=stack.ticks_for("HUMAN_WAIT"), sync_notice_first_frames=sync_notice_during_open[:3],
        )
        return out
    finally:
        stack.close()


def scenario_human_finishes(seed: int, store_dir: Path | None) -> dict:
    """넘겨받은 사람이 끝까지 해서 HUMAN 상태에서 성공(= HUMAN에서도 성공 판정이 돈다)."""
    stack = Stack()
    try:
        stack.reset_block(seed)
        policy = ScriptedPolicy(lambda: stack.worker.snapshot["block_m"])
        episode = stack.start(seed, policy)
        assert wait_for(lambda: policy.phase >= 1, 60)  # 블록 위에 도착(내리기 직전)
        assert episode.request_takeover() is None
        assert wait_for(lambda: stack.worker.owner.current is Owner.HUMAN_WAIT, 2.0)
        human = FakeHuman(stack.control, stack.worker)
        steps = human_pick_and_place(human, stack, episode)
        assert episode.done.wait(40), "episode did not finish"
        out = analyze(episode.result, store_dir)
        out.update(human_steps={k: round(v, 3) for k, v in steps.items()}, human_tick_ms=stack.ticks_for("HUMAN"),
                   final_owner_before_end="HUMAN" if episode.result.recording.owner[-1] == cc.OWNER_HUMAN else "other")
        return out
    finally:
        stack.close()


def scenario_ik(seed: int, warm: bool, hold_s: float = 1.5) -> dict:
    """블록을 든 채 넘겨받고 손은 가만히 둔다(집게는 일치시킴): 첫 1초 팔 관절 ctrl 최대 변화(rad/틱)."""
    original = cc.HUMAN_IK_WARM_START
    cc.HUMAN_IK_WARM_START = warm
    stack = Stack()
    try:
        stack.reset_block(seed)
        policy = ScriptedPolicy(lambda: stack.worker.snapshot["block_m"])
        episode = stack.start(seed, policy)
        takeover_when_carrying(stack, policy, episode)
        assert wait_for(lambda: stack.worker.owner.current is Owner.HUMAN_WAIT, 2.0)
        human = FakeHuman(stack.control, stack.worker)
        human.hold(hold_s, CLOSED_PINCH)  # 로봇이 쥔 상태(닫힘)와 일치하는 손, 가만히
        episode.request_stop()
        assert episode.done.wait(10)
        rec = episode.result.recording
        return {"warm_start": warm, "arm_max_step_rad_per_tick_first_1s": round(arm_steps_after_human_start(rec), 4),
                "typical": {k: round(v, 4) for k, v in typical_step_rad(rec.ctrl[rec.owner == cc.OWNER_AUTO]).items()}}
    finally:
        cc.HUMAN_IK_WARM_START = original
        stack.close()


def scenario_ik_real(seed: int, takeover_at_s: float, repeat: int) -> dict:
    """실제 체크포인트 정책이 만든 자세(IK가 만든 자세가 아니다)에서 넘겨받고 손은 가만히 둔다: ik.set_arm vs ik.reset.
    집게는 로봇 상태에 맞춰 둔다(집게 변화가 팔 관절 숫자에 섞이지 않게 팔 관절 앞 5개만 본다)."""
    client = ipc.PolicyClient()
    client.ensure_started()
    out: dict = {"warm": [], "reset": []}
    original = cc.HUMAN_IK_WARM_START
    try:
        for _ in range(repeat):
            for warm in (True, False):
                cc.HUMAN_IK_WARM_START = warm
                stack = Stack()
                try:
                    stack.reset_block(seed)
                    episode = stack.start(seed, client)
                    assert wait_for(lambda: episode.progress()["elapsed_s"] >= takeover_at_s or episode.done.is_set(), 90)
                    assert episode.request_takeover() is None
                    assert wait_for(lambda: stack.worker.owner.current is Owner.HUMAN_WAIT, 3.0)
                    human = FakeHuman(stack.control, stack.worker)
                    robot_closed = stack.control.state()["gripper_sync"]["robot_closed"]
                    human.hold(1.6, CLOSED_PINCH if robot_closed else OPEN_PINCH)
                    snap = stack.worker.snapshot
                    episode.request_stop()
                    assert episode.done.wait(10)
                    rec = episode.result.recording
                    ctrl_before = rec.ctrl[intervention_segments(rec.owner)[0][0] - 1]
                    out["warm" if warm else "reset"].append(
                        {"arm_max_step_rad_per_tick_first_1s": round(arm_steps_after_human_start(rec), 4), "tip_at_end_m": [round(float(v), 3) for v in snap["tip_m"]],
                         "human_samples": rec.human_samples, "robot_closed": robot_closed, "ctrl_at_takeover": [round(float(v), 3) for v in ctrl_before]}
                    )
                finally:
                    stack.close()
    finally:
        cc.HUMAN_IK_WARM_START = original
        client.shutdown()
    return out


def scenario_outside(seed: int, warm: bool = True, hold_s: float = 2.0) -> dict:
    """AUTO를 시작하자마자(집게 끝이 home 자세: 작업 상자 밖) 넘겨받는다. 첫 목표가 상자 안으로 잘릴 때 최대 이동 속도."""
    original = cc.HUMAN_IK_WARM_START
    cc.HUMAN_IK_WARM_START = warm
    stack = Stack()
    try:
        stack.reset_block(seed)
        policy = ScriptedPolicy(lambda: stack.worker.snapshot["block_m"])
        policy._goal = lambda tip, block: (np.array(tip), config.GRIPPER_OPEN_RAD)  # 가만히 서 있는 정책(home 유지)
        episode = stack.start(seed, policy)
        assert wait_for(lambda: episode.progress()["sample"] >= 3, 10)
        tip0 = np.array(stack.worker.snapshot["tip_m"])
        assert episode.request_takeover() is None
        assert wait_for(lambda: stack.worker.owner.current is Owner.HUMAN_WAIT, 2.0)
        human = FakeHuman(stack.control, stack.worker)
        human.hold(hold_s, OPEN_PINCH)  # 로봇 열림 + 손 펴짐 → 바로 일치
        tips = np.array([f["tip"] for f in human.frame_log if f["tip"] is not None])
        owners = [f["owner"] for f in human.frame_log if f["tip"] is not None]
        first_human = owners.index("HUMAN") if "HUMAN" in owners else 0
        seg = tips[first_human:]
        speed = np.linalg.norm(np.diff(seg, axis=0), axis=1) * FPS if len(seg) > 1 else np.array([0.0])
        target = stack.control.state()["target_m"]  # 에피소드를 끝내기 전에 읽는다(끝나면 상태가 지워진다)
        episode.request_stop()
        assert episode.done.wait(10)
        rec = episode.result.recording
        return {
            "tip_at_takeover_m": [round(float(v), 3) for v in tip0], "first_target_m": target, "warm_start": warm,
            "max_tip_speed_m_per_s": round(float(speed.max()), 3), "mean_tip_speed_first_1s_m_per_s": round(float(speed[:30].mean()), 3),
            "tip_distance_travelled_m": round(float(np.linalg.norm(seg[-1] - seg[0])), 3),
            "arm_max_step_rad_per_tick_first_1s": round(arm_steps_after_human_start(rec), 4),
            "max_step_limit_rad_per_tick": config.MAX_JOINT_STEP_RAD,
        }
    finally:
        cc.HUMAN_IK_WARM_START = original
        stack.close()


def scenario_real(seed: int, store_dir: Path | None, takeover_at_s: float = 8.0, mode: str = "pick") -> dict:
    """실제 체크포인트(정책 프로세스)로 AUTO를 시작하고 takeover_at_s 뒤에 스크립트된 사람이 넘겨받아 집어서 놓는다. 성공 여부는 보고만 한다.
    mode="handback": 사람이 집게를 맞추고 가만히 있다가 곧바로 정책에게 돌려준다(돌려주는 순간의 ctrl 점프를 실제 정책으로 재려고). 정책이 끝까지 가거나 시간 초과까지 본다."""
    client = ipc.PolicyClient()
    t0 = time.perf_counter()
    client.ensure_started()
    ready_s = time.perf_counter() - t0
    stack = Stack()
    try:
        stack.reset_block(seed)
        episode = stack.start(seed, client)
        assert wait_for(lambda: episode.progress()["elapsed_s"] >= takeover_at_s or episode.done.is_set(), 90)
        info: dict = {"policy_ready_s": round(ready_s, 2), "seed": seed, "takeover_at_s": takeover_at_s}
        if episode.done.is_set():
            info["note"] = "정책이 넘겨받기 전에 이미 끝났다"
            info.update(analyze(episode.result, store_dir))
            return info
        t_req = time.monotonic()
        assert episode.request_takeover() is None
        assert wait_for(lambda: stack.worker.owner.current is Owner.HUMAN_WAIT, 3.0)
        info["takeover_request_to_wait_ms"] = round((time.monotonic() - t_req) * 1000.0, 1)
        human = FakeHuman(stack.control, stack.worker)
        if mode == "handback":
            robot_closed = stack.control.state()["gripper_sync"]["robot_closed"]
            human.hold(1.0, CLOSED_PINCH if robot_closed else OPEN_PINCH)  # 집게를 맞추고 가만히
            steps = {}
            info["handback_accepted"] = episode.request_handback() is None
            assert wait_for(lambda: stack.worker.owner.current is Owner.AUTO or episode.done.is_set(), 3.0)
            episode.done.wait(60)
        else:
            steps = human_pick_and_place(human, stack, episode)
            # 사람이 다 놓은 뒤에도 안 끝났으면(성공 대기) 정책에게 돌려주고 끝까지 본다
            if not episode.done.wait(3.0):
                reason = episode.request_handback()
                info["handback_after_human"] = reason is None
            episode.done.wait(60)
        if not episode.done.is_set():
            episode.request_stop()
            episode.done.wait(10)
        info.update(analyze(episode.result, store_dir))
        info.update(human_steps={k: round(v, 3) for k, v in steps.items()}, human_tick_ms=stack.ticks_for("HUMAN"),
                    policy_calls=episode.result.policy_calls)
        return info
    finally:
        stack.close()
        client.shutdown()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", required=True, choices=["handback", "human_finishes", "gripper", "ik", "ik_real", "outside", "real"])
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--viewers", type=int, default=0, help="1이면 화면 렌더링도 켠다(실제 서버처럼). 틱 간격 측정용")
    parser.add_argument("--takeover-at-s", type=float, default=8.0)
    parser.add_argument("--real-mode", choices=["pick", "handback"], default="pick", help="real: 사람이 집어서 놓는다 / 곧바로 정책에게 돌려준다")
    parser.add_argument("--out-dir", help="교정 에피소드를 저장해 볼 폴더(기본: 임시 폴더, 끝나면 지운다)")
    args = parser.parse_args(argv)
    global VIEWERS
    VIEWERS = args.viewers
    try:
        os.nice(5)
    except OSError:
        pass
    with tempfile.TemporaryDirectory(prefix="dexter-correction-") as tmp:
        store_dir = Path(args.out_dir) if args.out_dir else Path(tmp)
        if args.scenario in ("handback", "gripper"):
            result = scenario_handback(args.seed, store_dir)
        elif args.scenario == "human_finishes":
            result = scenario_human_finishes(args.seed, store_dir)
        elif args.scenario == "ik":
            result = {"warm": [], "reset": []}
            for _ in range(args.repeat):
                result["warm"].append(scenario_ik(args.seed, True))
                result["reset"].append(scenario_ik(args.seed, False))
        elif args.scenario == "ik_real":
            result = scenario_ik_real(args.seed, args.takeover_at_s, args.repeat)
        elif args.scenario == "outside":
            result = {"warm": scenario_outside(args.seed, True), "reset": scenario_outside(args.seed, False)}
        else:
            result = scenario_real(args.seed, store_dir, args.takeover_at_s, args.real_mode)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
