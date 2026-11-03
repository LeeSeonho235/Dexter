"""평가(3-4) 환경과 롤아웃: 정책을 시뮬레이션에서 돌려 성공 여부를 잰다. mujoco와 server 모듈만 쓴다(lerobot 없음).
server/.venv와 train/.venv(mujoco 3.16.0 설치) 어디서나 같은 결과가 나온다(qpos·이미지 해시 일치 확인).

녹화(server/recorder.py, docs/08-recording.md)와 같은 규칙으로 한 샘플을 돈다:
    샘플 k (시뮬레이션 시간 격자 server/grid.py, 물리 스텝 직전):
      관측: state = qpos[:6] (float32), 이미지 = StateRenderer.render(qpos 13개)  ← 데이터셋 프레임을 만든 것과 같은 클래스·카메라·설정
      정책 → action(6) → data.ctrl[:6]에 그대로 넣는다(녹화 때 action = ctrl)
      grid가 정한 개수(6 또는 7)의 mj_step을 진행 → 다음 샘플
- 초기 상태: home 키프레임 + park ctrl(home 자세, 집게 열림 0.9) → 블록을 시드 위치에 회전 없이 놓고 EVAL_SETTLE_S 동안 가라앉힌다.
- 성공 판정은 server/success.py의 is_success + SuccessHold(0.5초 유지)를 그대로 쓴다. 유지가 채워진 샘플에서 끝낸다.
- 이미지 렌더링과 정책 추론은 policy.needs_image(k)가 True인 샘플에서만 한다. ACT는 n_action_steps마다만 관측을 읽기 때문이다.
단위: 길이 m, 각도 rad, 시간 s. EGL은 만든 스레드에서만 쓰고 close()를 불러야 한다(with 문 가능). MUJOCO_GL은 import 전에 정해야 한다.
"""

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import mujoco
import numpy as np

from server import config
from server import eval_config as ec
from server.eval_metrics import classify_failure
from server.grid import grid_step_index
from server.render import StateRenderer
from server.seeds import block_qpos_for_seed
from server.success import QPOS_GRIPPER, SuccessHold, is_success

ARM_DOF = 6  # ctrl, state 개수(팔 5 + 집게 1)


def limit_step(current: np.ndarray, goal: np.ndarray, max_step: np.ndarray | float) -> np.ndarray:
    """server/ik.py의 limit_joint_step과 같은 계산(ik.py는 mink를 import해서 train/.venv에서는 못 쓴다. 같은 결과인지 테스트가 확인)."""
    current = np.asarray(current, dtype=float)
    goal = np.asarray(goal, dtype=float)
    return current + np.clip(goal - current, -max_step, max_step)


@dataclass
class Observation:
    state: np.ndarray  # (6,) float32 = qpos[:6]
    image: np.ndarray | None  # (H, W, 3) uint8. 이번 샘플에 필요 없으면 None
    qpos: np.ndarray  # (13,) float64 복사본(블록 포함, 정책에는 넘기지 않는다)


def make_observation(renderer: StateRenderer | None, qpos: np.ndarray) -> Observation:
    """학습 데이터와 같은 관측: state는 qpos[:6]을 float32로(train/build_dataset.py state_action_for와 같다), 이미지는 StateRenderer.render."""
    qpos = np.asarray(qpos, dtype=np.float64)
    image = renderer.render(qpos) if renderer is not None else None
    return Observation(state=qpos[:ARM_DOF].astype(np.float32), image=image, qpos=qpos.copy())


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def needs_image(self, k: int) -> bool: ...

    def act(self, k: int, obs: Observation) -> np.ndarray: ...


class HoldPolicy:
    """아무것도 안 하는 정책: park 자세(home + 집게 열림) ctrl을 계속 낸다(기저선)."""

    name = "hold"

    def __init__(self, park_ctrl: np.ndarray) -> None:
        self._ctrl = np.asarray(park_ctrl, dtype=float).copy()

    def reset(self) -> None:
        pass

    def needs_image(self, k: int) -> bool:
        return False

    def act(self, k: int, obs: Observation) -> np.ndarray:
        return self._ctrl.copy()


class ReplayPolicy:
    """정답 재생 정책: 시연의 ctrl[k]를 샘플 k에서 그대로 낸다(끝나면 마지막 값 유지). 평가 루프가 녹화와 어긋나 있는지 보는 검사용."""

    name = "replay"

    def __init__(self, ctrl: np.ndarray) -> None:
        self._ctrl = np.asarray(ctrl, dtype=float)[:, :ARM_DOF]

    def reset(self) -> None:
        pass

    def needs_image(self, k: int) -> bool:
        return False

    def act(self, k: int, obs: Observation) -> np.ndarray:
        return self._ctrl[min(k, len(self._ctrl) - 1)].copy()


BoundaryFn = Callable[[int], int]


class EvalEnv:
    def __init__(self, with_renderer: bool = True, boundary: str = "grid") -> None:
        self.model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.timestep_s = float(self.model.opt.timestep)
        self.renderer = StateRenderer() if with_renderer else None
        if self.renderer is not None:
            # 렌더러 인스턴스의 첫 렌더링만 이후 렌더링과 몇 픽셀이 1단계 다르다(3-3에서 확인, 원인 미확인). 첫 에피소드의 첫 관측만 다르면
            # 정책 출력이 1e-5~1e-2 달라져 실행 순서에 따라 결과가 바뀌므로, 버리는 렌더링을 한 번 해 둔다(건너뛰기 비교 테스트에서 확인).
            self.renderer.render(self.model.key(config.START_KEYFRAME).qpos)
        self._boundary_name = boundary
        self._boundary = self._make_boundary(boundary)
        self._t0_s = 0.0
        self.k = 0

    def _make_boundary(self, name: str) -> BoundaryFn:
        fps, dt = ec.EVAL_FPS, self.timestep_s
        if name == "grid":  # 녹화와 같은 격자(server/grid.py). 기본값
            return lambda k: grid_step_index(k, fps, dt)
        # 아래는 진단용: ctrl이 바뀌는 시점을 격자와 다르게 해 본다(원본 시연의 틱 경계는 floor에 가깝다)
        if name == "floor":
            return lambda k: math.floor(k / (fps * dt) + 1e-9)
        if name == "ceil":
            return lambda k: math.ceil(k / (fps * dt) - 1e-9)
        raise ValueError(f"unknown boundary {name!r}")

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()

    def __enter__(self) -> "EvalEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- 상태 ---

    def park_ctrl(self) -> np.ndarray:
        """home 자세 + 집게 열림(server/control.py의 _park_ctrl과 같다)."""
        park = np.array(self.model.key(config.START_KEYFRAME).ctrl[:ARM_DOF], dtype=float)
        park[QPOS_GRIPPER] = config.GRIPPER_OPEN_RAD
        return park

    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos[:13]

    @property
    def sim_time_s(self) -> float:
        return float(self.data.time - self._t0_s)

    def reset(self, seed: int, init_arm: np.ndarray | None = None) -> None:
        """시드의 블록 위치로 새 에피소드를 시작한다. init_arm(6개, rad)을 주면 팔을 그 자세로 시작한다(정답 재생 검사용)."""
        mujoco.mj_resetData(self.model, self.data)  # 이전 에피소드의 접촉·warm start가 남지 않게
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(config.START_KEYFRAME).id)
        block = block_qpos_for_seed(seed)
        if init_arm is None:
            self.data.ctrl[:ARM_DOF] = self.park_ctrl()
            self.data.qpos[ARM_DOF:13] = block
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            for _ in range(round(ec.EVAL_SETTLE_S / self.timestep_s)):
                mujoco.mj_step(self.model, self.data)
        else:
            init_arm = np.asarray(init_arm, dtype=float)
            self.data.qpos[:ARM_DOF] = init_arm
            self.data.qpos[ARM_DOF:13] = block
            self.data.ctrl[:ARM_DOF] = init_arm
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
        self._t0_s = float(self.data.time)
        self.k = 0

    def observation(self, with_image: bool = True) -> Observation:
        return make_observation(self.renderer if with_image else None, self.qpos)

    def step(self, ctrl: np.ndarray) -> None:
        """ctrl(6개)을 넣고 다음 샘플까지(6 또는 7 스텝) 물리를 진행한다."""
        ctrl = np.asarray(ctrl, dtype=float)
        if ctrl.shape != (ARM_DOF,) or not np.all(np.isfinite(ctrl)):
            raise ValueError(f"ctrl must be {ARM_DOF} finite numbers, got {ctrl!r}")
        self.data.ctrl[:ARM_DOF] = ctrl
        for _ in range(self._boundary(self.k + 1) - self._boundary(self.k)):
            mujoco.mj_step(self.model, self.data)
        self.k += 1


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    failure: str  # "success" 또는 eval_config.FAILURE_KEYS
    pushed: bool
    time_to_success_s: float | None
    sim_s: float  # 끝난 시각(샘플 시각, s)
    n_samples: int
    block_start_m: tuple[float, float, float]
    final_block_m: tuple[float, float, float]
    final_gripper_rad: float
    max_block_z_m: float
    max_block_move_m: float
    limited_samples: int = 0
    policy_calls: int = 0  # 관측(이미지)을 만들고 정책에 넘긴 횟수. ACT는 n_action_steps마다 한 번
    render_s: float = 0.0
    policy_s: float = 0.0
    physics_s: float = 0.0
    tail_qpos: list = field(default_factory=list)  # 마지막 샘플들의 qpos(실패 프레임 저장용)
    trace: np.ndarray | None = None  # (n_samples, 13) 샘플마다의 qpos(record_trace일 때)
    ctrl_trace: np.ndarray | None = None  # (n_samples - 1, 6) 샘플마다 낸 ctrl

    def row(self) -> dict:
        """eval_metrics.CSV_COLUMNS에 맞는 dict."""
        return {
            "seed": self.seed, "success": self.success, "failure": self.failure, "pushed": self.pushed,
            "time_to_success_s": self.time_to_success_s, "sim_s": self.sim_s, "n_samples": self.n_samples,
            "block_start_x_m": self.block_start_m[0], "block_start_y_m": self.block_start_m[1],
            "final_block_x_m": self.final_block_m[0], "final_block_y_m": self.final_block_m[1],
            "final_block_z_m": self.final_block_m[2], "final_gripper_rad": self.final_gripper_rad,
            "max_block_z_m": self.max_block_z_m, "max_block_move_m": self.max_block_move_m,
            "limited_samples": self.limited_samples, "policy_calls": self.policy_calls,
            "render_s": self.render_s, "policy_s": self.policy_s, "physics_s": self.physics_s,
        }


def rollout(
    env: EvalEnv,
    policy: Policy,
    seed: int,
    *,
    max_s: float = ec.EVAL_MAX_S,
    limit: bool = ec.EVAL_LIMIT_STEP_DEFAULT,
    init_arm: np.ndarray | None = None,
    stop_on_success: bool = True,
    max_samples: int | None = None,
    keep_tail: int = 0,
    record_trace: bool = False,
) -> EpisodeResult:
    """한 에피소드. 샘플 0..n_max의 상태를 보고(성공 판정), 마지막 샘플 전까지 정책을 불러 ctrl을 낸다."""
    n_max = max_samples if max_samples is not None else int(round(max_s * ec.EVAL_FPS))
    env.reset(seed, init_arm=init_arm)
    policy.reset()
    hold = SuccessHold()
    prev_ctrl = env.data.ctrl[:ARM_DOF].copy()
    start = env.qpos[ARM_DOF:ARM_DOF + 3].copy()
    max_z, max_move = float(start[2]), 0.0
    tail: deque = deque(maxlen=keep_tail)
    trace: list[np.ndarray] = []
    ctrls: list[np.ndarray] = []
    limited = calls = 0
    render_s = policy_s = physics_s = 0.0
    time_to_success: float | None = None
    arm_step = np.array([ec.LIMIT_STEP_ARM_RAD] * 5 + [ec.LIMIT_STEP_GRIPPER_RAD])

    k = 0
    for k in range(n_max + 1):
        qpos = env.qpos.copy()
        block = qpos[ARM_DOF:ARM_DOF + 3]
        max_z = max(max_z, float(block[2]))
        max_move = max(max_move, float(math.hypot(block[0] - start[0], block[1] - start[1])))
        if keep_tail:
            tail.append(qpos)
        if record_trace:
            trace.append(qpos)
        if hold.update(env.sim_time_s, is_success(block, float(qpos[QPOS_GRIPPER]))) and time_to_success is None:
            time_to_success = env.sim_time_s
            if stop_on_success:
                break
        if k == n_max:
            break
        need_image = policy.needs_image(k)
        t0 = time.perf_counter()
        obs = env.observation(with_image=need_image)
        render_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        action = np.asarray(policy.act(k, obs), dtype=float).reshape(-1)[:ARM_DOF]
        policy_s += time.perf_counter() - t0
        calls += int(need_image)
        if limit:
            limited_action = limit_step(prev_ctrl, action, arm_step)
            limited += int(not np.array_equal(limited_action, action))
            action = limited_action
        t0 = time.perf_counter()
        env.step(action)
        physics_s += time.perf_counter() - t0
        prev_ctrl = action
        if record_trace:
            ctrls.append(action)

    final = qpos  # 마지막으로 판정한 샘플(성공으로 끝났으면 그 샘플, 아니면 최대 시간의 샘플)
    final_block = tuple(float(v) for v in final[ARM_DOF:ARM_DOF + 3])
    success = time_to_success is not None
    failure, pushed = classify_failure(success, max_z, final_block, tuple(start), max_block_move_m=max_move)
    return EpisodeResult(
        seed=seed, success=success, failure=failure, pushed=pushed, time_to_success_s=time_to_success,
        sim_s=env.sim_time_s, n_samples=k + 1, block_start_m=tuple(float(v) for v in start), final_block_m=final_block,
        final_gripper_rad=float(final[QPOS_GRIPPER]), max_block_z_m=max_z, max_block_move_m=max_move, limited_samples=limited,
        policy_calls=calls, render_s=render_s, policy_s=policy_s, physics_s=physics_s, tail_qpos=list(tail),
        trace=np.array(trace) if record_trace else None, ctrl_trace=np.array(ctrls) if record_trace else None,
    )
