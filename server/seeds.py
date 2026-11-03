"""시드 → 블록 시작 위치. 시연 시드(0~99)와 평가 시드(1000~1049)는 겹치지 않는다(범위는 server/config.py).

같은 시드는 항상 같은 위치다(numpy PCG64 default_rng(seed)). 블록은 바닥 위에 회전 없이 놓고, 목표(초록)는 고정이다.
단위: m.
"""

import numpy as np

from server import config

DEMO = "demo"
EVAL = "eval"


def seed_kind(seed: int) -> str | None:
    """"demo", "eval", 또는 어느 범위에도 없으면 None."""
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        return None
    if config.DEMO_SEED_RANGE[0] <= seed < config.DEMO_SEED_RANGE[1]:
        return DEMO
    if config.EVAL_SEED_RANGE[0] <= seed < config.EVAL_SEED_RANGE[1]:
        return EVAL
    return None


def block_xy_for_seed(seed: int) -> tuple[float, float]:
    """시드의 블록 중심 (x, y)(m). 시연·평가 범위 밖의 시드는 ValueError."""
    if seed_kind(seed) is None:
        raise ValueError(f"seed {seed!r} is not in the demo or eval range")
    rng = np.random.default_rng(int(seed))
    x_m = float(rng.uniform(*config.BLOCK_SPAWN_X_M))
    y_m = float(rng.uniform(*config.BLOCK_SPAWN_Y_M))
    return x_m, y_m


def block_qpos_for_seed(seed: int) -> np.ndarray:
    """블록 자유 관절 qpos 7개: 위치 x, y, z(m) + 회전 쿼터니언 (w, x, y, z) = 회전 없음."""
    x_m, y_m = block_xy_for_seed(seed)
    return np.array([x_m, y_m, config.BLOCK_HALF_M, 1.0, 0.0, 0.0, 0.0])
