"""교정 데이터(data/corrections)에서 사람이 조종한(HUMAN) 구간을 잘라내고, 변환 전에 해야 할 검사를 모아 둔 순수 함수 모음(4-3).

numpy와 표준 라이브러리, server.seeds/config만 쓴다(mujoco 필요 없음). server/.venv와 train/.venv 어디서나 돌아간다.
교정 폴더는 읽기만 한다. 설명은 docs/13-correction.md("4-3에서 쓸 학습 구간 후보")와 이 단계의 계획.

규칙
- 학습에 쓰는 구간 = `is_intervention`이 True(= owner가 HUMAN)인 연속 구간. HUMAN_WAIT(로봇이 멈춘 구간)와 AUTO 구간은 쓰지 않는다.
- 구간 하나 = 데이터셋의 에피소드 하나(ACT가 예측하는 100스텝 행동이 AUTO 구간으로 넘어가지 않게).
- 샘플 k: state = qpos[k][:6], action = ctrl[k][:6] (3-2와 같은 규칙).
단위: 길이 m, 시간 s.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from server import config, seeds

OWNER_HUMAN = 1  # server/correction_config.py OWNER_HUMAN과 같은 값(npz의 owner 배열). 아래 테스트가 같은지 확인한다
POSITION_TOL_M = 1e-3  # 시드가 만드는 블록 위치와 json의 block_start_m이 이 안이면 "같은 위치"
# 환경 해시 중 사람이 허락해도 되는 것(나머지 scene/robot 해시와 체크포인트 해시는 허락 대상이 아니다)
ALLOWABLE_DIFF_FIELDS = ("mujoco_version", "config_sha256", "correction_config_sha256")


@dataclass(frozen=True)
class Segment:
    source_index: int  # 교정 에피소드 번호(그 컴퓨터의 correction_XXXXXX)
    seed: int
    start_sample: int
    end_sample: int  # 포함
    n_samples: int


def human_runs(is_intervention: np.ndarray) -> list[tuple[int, int]]:
    """True가 이어지는 구간 (start, end 포함) 목록."""
    flags = np.asarray(is_intervention, dtype=bool)
    runs, start = [], None
    for k, flag in enumerate(flags):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            runs.append((start, k - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def segments_for(meta: dict, arrays: dict[str, np.ndarray]) -> tuple[list[Segment], list[str]]:
    """한 교정 에피소드의 (HUMAN 구간들, 문제 목록). 문제가 있으면 변환하지 않고 멈추는 데 쓴다."""
    problems: list[str] = []
    i = meta["episode_index"]
    n = meta["n_ticks"]
    flags, owner = arrays["is_intervention"], arrays["owner"]
    if not (len(flags) == len(owner) == len(arrays["qpos"]) == len(arrays["ctrl"]) == n):
        return [], [f"교정 {i}: 배열 길이가 n_ticks {n}과 다르다"]
    if not np.array_equal(np.asarray(flags, dtype=bool), owner == OWNER_HUMAN):
        problems.append(f"교정 {i}: is_intervention이 owner==HUMAN과 다르다")
    runs = human_runs(flags)
    listed = [(r["start_sample"], r["end_sample"]) for r in meta.get("interventions", [])]
    if listed != runs:
        problems.append(f"교정 {i}: json의 interventions {listed} != 배열에서 센 구간 {runs}")
    if not np.isfinite(arrays["qpos"]).all() or not np.isfinite(arrays["ctrl"]).all():
        problems.append(f"교정 {i}: qpos/ctrl에 NaN/inf")
    segs = [Segment(i, int(meta["seed"]), s, e, e - s + 1) for s, e in runs]
    return segs, problems


def load_correction_metas(corr_dir: Path, exclude: set[int] | frozenset[int] = frozenset()) -> list[dict]:
    """저장된(user_decision == "saved", json과 npz가 둘 다 있는) 교정의 메타를 번호 순서로. 읽기만 한다."""
    metas = []
    for path in sorted(Path(corr_dir).glob("correction_*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("user_decision") != "saved" or meta["episode_index"] in exclude:
            continue
        if not path.with_suffix(".npz").is_file():
            raise FileNotFoundError(f"{path.with_suffix('.npz')} 가 없다(json만 있다)")
        metas.append(meta)
    metas.sort(key=lambda m: m["episode_index"])
    return metas


def load_correction_arrays(corr_dir: Path, index: int) -> dict[str, np.ndarray]:
    with np.load(Path(corr_dir) / f"correction_{index:06d}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def all_segments(corr_dir: Path, metas: list[dict]) -> tuple[list[Segment], list[str]]:
    segs: list[Segment] = []
    problems: list[str] = []
    for m in metas:
        s, p = segments_for(m, load_correction_arrays(corr_dir, m["episode_index"]))
        segs += s
        problems += p
    return segs, problems


def segment_qpos(corr_dir: Path, seg: Segment) -> np.ndarray:
    return load_correction_arrays(corr_dir, seg.source_index)["qpos"][seg.start_sample : seg.end_sample + 1]


# ---------------------------------------------------------------- 환경 해시 검사

def _checkpoint_sha(meta: dict) -> str | None:
    return (meta.get("base_checkpoint") or {}).get("model_safetensors_sha256")


def check_environment(
    metas: list[dict],
    *,
    expected_scene_sha256: str,
    expected_robot_sha256: str,
    reference: dict[str, str] | None = None,
    allow_diff: frozenset[str] | set[str] = frozenset(),
) -> list[str]:
    """교정 json들이 같은 환경에서 만들어졌는지. 문제 목록(비어 있으면 통과).

    - 장면(scene)과 로봇 모델 해시: 지금 프레임을 그릴 환경과 달라야 하면 항상 오류(허락 불가).
    - 체크포인트 해시: 교정들끼리 같아야 하고, reference["checkpoint_sha256"]가 있으면 그것과도 같아야 한다(허락 불가).
    - mujoco_version, config_sha256, correction_config_sha256: 교정들끼리 같아야 하고 reference에 있으면 그것과도 같아야 한다.
      다르면 오류. 사람이 --allow-diff로 이름을 적은 항목만 넘어간다(보고에 남긴다).
    """
    reference = reference or {}
    problems: list[str] = []
    for m in metas:
        i = m["episode_index"]
        if m.get("scene_sha256") != expected_scene_sha256:
            problems.append(f"교정 {i}: scene_sha256 {str(m.get('scene_sha256'))[:8]} != 지금 {expected_scene_sha256[:8]}")
        if m.get("robot_model_sha256") != expected_robot_sha256:
            problems.append(f"교정 {i}: robot_model_sha256 {str(m.get('robot_model_sha256'))[:8]} != 지금 {expected_robot_sha256[:8]}")
    ckpts = {_checkpoint_sha(m) for m in metas}
    if len(ckpts) > 1 or None in ckpts:
        problems.append(f"교정들의 base_checkpoint 해시가 하나로 같지 않다: {sorted(str(c)[:8] for c in ckpts)}")
    elif reference.get("checkpoint_sha256") and ckpts != {reference["checkpoint_sha256"]}:
        problems.append(f"base_checkpoint 해시 {next(iter(ckpts))[:8]} != 기대 {reference['checkpoint_sha256'][:8]}")
    for field in ALLOWABLE_DIFF_FIELDS:
        values = {m.get(field) for m in metas}
        wanted = reference.get(field)
        differs = len(values) > 1 or (wanted is not None and values != {wanted})
        if differs and field not in allow_diff:
            problems.append(f"{field}가 다르다: 교정 {sorted(str(v)[:12] for v in values)}" + (f", 기준 {str(wanted)[:12]}" if wanted else ""))
    return problems


def reference_from_demos(demo_metas: list[dict]) -> dict[str, str]:
    """시연 json들이 모두 같은 값을 가진 환경 항목만 기준으로 돌려준다(시연끼리도 다르면 그 항목은 기준이 없다)."""
    ref: dict[str, str] = {}
    for field in ALLOWABLE_DIFF_FIELDS:
        values = {m.get(field) for m in demo_metas}
        if len(values) == 1 and None not in values:
            ref[field] = next(iter(values))
    return ref


# ---------------------------------------------------------------- 평가 시드 분리 검사

def eval_block_positions_m() -> np.ndarray:
    """평가 시드 1000~1049의 블록 (x, y)(m), shape (50, 2)."""
    return np.array([seeds.block_xy_for_seed(s) for s in range(*config.EVAL_SEED_RANGE)])


def check_seed_separation(metas: list[dict], label: str) -> list[str]:
    """시드와 블록 위치가 평가 1000~1049와 섞이지 않았는지. 문제 목록(비어 있으면 통과).

    ① seed가 시연 범위(0~99) 안인가(평가 범위나 범위 밖이면 오류) ② json의 block_start_m이 그 시드가 만드는 위치와 같은가(시드 기록이 틀렸는지)
    ③ block_start_m이 평가 시드 50개의 위치 중 어느 것과도 1 mm 이내로 겹치지 않는가.
    """
    problems: list[str] = []
    eval_xy = eval_block_positions_m()
    for m in metas:
        i, seed = m["episode_index"], m["seed"]
        tag = f"{label} {i}(시드 {seed})"
        kind = seeds.seed_kind(seed)
        if kind == seeds.EVAL:
            problems.append(f"{tag}: 평가 시드 범위 {config.EVAL_SEED_RANGE}")
        elif kind != seeds.DEMO:
            problems.append(f"{tag}: 시연 시드 범위 {config.DEMO_SEED_RANGE} 밖")
        else:
            expected = np.array(seeds.block_xy_for_seed(seed))
            if np.abs(np.array(m["block_start_m"][:2]) - expected).max() > POSITION_TOL_M:
                problems.append(f"{tag}: block_start_m이 시드가 만드는 위치와 다르다")
        xy = np.array(m["block_start_m"][:2])
        near = np.abs(eval_xy - xy).max(axis=1) <= POSITION_TOL_M
        if near.any():
            hit = config.EVAL_SEED_RANGE[0] + int(np.argmax(near))
            problems.append(f"{tag}: 블록 시작 위치가 평가 시드 {hit}의 위치와 같다")
    return problems


def check_auto_runs_csv(path: Path) -> list[str]:
    """auto_runs.csv에 평가 시드 범위의 시드가 있으면 문제(없는 파일은 문제 없음으로 본다. 존재 여부는 호출한 쪽이 보고)."""
    path = Path(path)
    if not path.is_file():
        return []
    problems = []
    with path.open(encoding="utf-8", newline="") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            if seeds.seed_kind(int(row["seed"])) != seeds.DEMO:
                problems.append(f"{path.name} {n}번째 줄: 시드 {row['seed']}가 시연 범위 밖이다({row['kind']})")
    return problems
