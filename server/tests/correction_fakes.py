"""4-3 테스트용 가짜 교정 파일(npz + json) 만들기. 서버·train 테스트가 같이 쓴다(import 전에 server/tests를 sys.path에 넣는다)."""

import json
from pathlib import Path

import numpy as np

from server import seeds

SCENE_SHA, ROBOT_SHA, CKPT_SHA = "a" * 64, "b" * 64, "c" * 64
OWNER_AUTO, OWNER_HUMAN, OWNER_WAIT = 0, 1, 2


def make_correction(
    corr_dir: Path,
    index: int,
    seed: int,
    runs: list[tuple[int, int]],
    n: int = 40,
    *,
    qpos: np.ndarray | None = None,
    scene: str = SCENE_SHA,
    robot: str = ROBOT_SHA,
    ckpt: str = CKPT_SHA,
    mujoco_version: str = "3.16.0",
    config_sha: str = "d" * 64,
    decision: str = "saved",
    block_xy: tuple[float, float] | None = None,
) -> dict:
    """runs = HUMAN 구간들 (start, end 포함). 각 구간 바로 앞 샘플 하나는 HUMAN_WAIT, 나머지는 AUTO."""
    corr_dir = Path(corr_dir)
    corr_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(index + 100 * seed)
    owner = np.zeros(n, dtype=np.int8)
    for s, e in runs:
        if s > 0:
            owner[s - 1] = OWNER_WAIT
        owner[s : e + 1] = OWNER_HUMAN
    if qpos is None:
        qpos = rng.uniform(-1, 1, size=(n, 13))
    arrays = {
        "sim_time_s": np.cumsum(np.where(np.arange(n) % 2 == 0, 0.030, 0.035)),
        "qpos": np.asarray(qpos, dtype=np.float64),
        "ctrl": rng.uniform(-1, 1, size=(n, 6)),
        "is_intervention": owner == OWNER_HUMAN,
        "owner": owner,
    }
    np.savez_compressed(corr_dir / f"correction_{index:06d}.npz", **arrays)
    x, y = block_xy if block_xy is not None else seeds.block_xy_for_seed(seed)
    meta = {
        "episode_index": index, "seed": seed, "block_start_m": [x, y, 0.0149], "source": "correction", "user_decision": "saved" if decision == "saved" else decision,
        "base_checkpoint": {"model_safetensors_sha256": ckpt}, "n_ticks": n,
        "interventions": [{"start_sample": s, "end_sample": e, "n_samples": e - s + 1} for s, e in runs],
        "scene_sha256": SCENE_SHA if scene is None else scene, "robot_model_sha256": robot, "mujoco_version": mujoco_version,
        "config_sha256": config_sha, "correction_config_sha256": "e" * 64,
    }
    (corr_dir / f"correction_{index:06d}.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta
