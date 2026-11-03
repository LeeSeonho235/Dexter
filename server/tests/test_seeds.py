"""시드: 시연/평가 범위가 겹치지 않고, 같은 시드는 항상 같은 위치이며, 블록 영역 안에 있다."""

import numpy as np
import pytest

from server import config
from server.seeds import DEMO, EVAL, block_qpos_for_seed, block_xy_for_seed, seed_kind


def demo_seeds():
    return range(*config.DEMO_SEED_RANGE)


def eval_seeds():
    return range(*config.EVAL_SEED_RANGE)


def test_demo_and_eval_ranges_are_the_documented_ones():
    assert config.DEMO_SEED_RANGE == (0, 100)
    assert config.EVAL_SEED_RANGE == (1000, 1050)
    assert len(demo_seeds()) == 100 and len(eval_seeds()) == 50


def test_ranges_do_not_overlap():
    assert set(demo_seeds()).isdisjoint(eval_seeds())
    assert config.DEMO_SEED_RANGE[1] <= config.EVAL_SEED_RANGE[0]


def test_seed_kind():
    assert seed_kind(0) == DEMO and seed_kind(99) == DEMO
    assert seed_kind(1000) == EVAL and seed_kind(1049) == EVAL
    for seed in (-1, 100, 999, 1050, 5000):
        assert seed_kind(seed) is None
    assert seed_kind(True) is None  # bool은 정수로 보지 않는다
    assert seed_kind(3.0) is None
    assert seed_kind(np.int64(7)) == DEMO


def test_same_seed_same_position_different_seeds_differ():
    assert block_xy_for_seed(5) == block_xy_for_seed(5)
    positions = {block_xy_for_seed(s) for s in [*demo_seeds(), *eval_seeds()]}
    assert len(positions) == 150  # 모두 다르다


def test_positions_stay_inside_the_spawn_region_and_workspace():
    (x0, x1), (y0, y1) = config.BLOCK_SPAWN_X_M, config.BLOCK_SPAWN_Y_M
    for seed in [*demo_seeds(), *eval_seeds()]:
        x, y = block_xy_for_seed(seed)
        assert x0 <= x <= x1 and y0 <= y <= y1
    assert config.WORKSPACE_X_M[0] <= x0 and x1 <= config.WORKSPACE_X_M[1]
    assert config.WORKSPACE_Y_M[0] <= y0 and y1 <= config.WORKSPACE_Y_M[1]


def test_spawn_region_is_clear_of_the_target():
    """블록(3cm)이 어느 시드에 놓여도 목표(5cm) 사각형과 겹치지 않는다."""
    cx, cy = config.TARGET_CENTER_M
    gap_m = abs(cy - config.TARGET_HALF_M - (config.BLOCK_SPAWN_Y_M[1] + config.BLOCK_HALF_M))
    assert config.BLOCK_SPAWN_Y_M[1] + config.BLOCK_HALF_M < cy - config.TARGET_HALF_M
    assert gap_m > 0.02


def test_seeds_outside_both_ranges_raise():
    for seed in (-1, 100, 999, 1050):
        with pytest.raises(ValueError):
            block_xy_for_seed(seed)


def test_block_qpos_is_on_the_floor_without_rotation():
    q = block_qpos_for_seed(3)
    assert q.shape == (7,)
    assert q[2] == config.BLOCK_HALF_M
    np.testing.assert_array_equal(q[3:], [1, 0, 0, 0])
