"""에피소드 선택과 품질 점검(3-3). numpy와 json만 쓰므로 server/.venv, train/.venv 어디서나 돌아간다. data/raw는 읽기만 한다.

규칙
- 학습에 쓰는 것: json의 success(내가 누른 값)가 True인 에피소드. 실패는 제외.
- 품질 점검은 보고만 한다. 문제가 있어도 조용히 버리지 않고, 제외는 사람이 --exclude로 번호를 적을 때만 한다.

사용: python -I train/episode_select.py [--raw data/raw] [--exclude 5,12]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_config as cfg  # noqa: E402


def load_metas(raw_dir: Path) -> list[dict]:
    """json이 있는 에피소드 메타를 번호 순서로. json이 있어야 저장된 것이다(episode_store 규칙)."""
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(raw_dir).glob("episode_*.json"))]
    metas.sort(key=lambda m: m["episode_index"])
    return metas


def npz_path(raw_dir: Path, index: int) -> Path:
    return Path(raw_dir) / f"episode_{index:06d}.npz"


def load_arrays(raw_dir: Path, index: int) -> dict[str, np.ndarray]:
    with np.load(npz_path(raw_dir, index), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def select(metas: list[dict], exclude: set[int] | frozenset[int] = frozenset()) -> list[dict]:
    """모든 에피소드에 대해 {index, included, reason}. 포함 조건: success가 True이고 --exclude에 없음."""
    rows = []
    for m in metas:
        i = m["episode_index"]
        if m.get("success") is not True:
            included, reason = False, f"실패로 기록됨 ({m.get('end_reason')})"
        elif i in exclude:
            included, reason = False, "사용자가 제외 지정"
        else:
            included, reason = True, "성공"
        rows.append({"index": i, "seed": m.get("seed"), "included": included, "reason": reason})
    return rows


def check_arrays(meta: dict, arrays: dict[str, np.ndarray]) -> list[str]:
    """한 에피소드의 데이터 문제 목록(비어 있으면 문제 없음). shape, NaN/inf, 틱 간격."""
    problems: list[str] = []
    t, qpos, ctrl = arrays["sim_time_s"], arrays["qpos"], arrays["ctrl"]
    n = meta["n_ticks"]
    if qpos.shape != (n, cfg.QPOS_DIM) or ctrl.shape != (n, cfg.CTRL_DIM) or t.shape != (n,):
        problems.append(f"shape 불일치: t{t.shape} qpos{qpos.shape} ctrl{ctrl.shape} n_ticks={n}")
        return problems
    if not np.isfinite(qpos).all():
        problems.append("qpos에 NaN/inf")
    if not np.isfinite(ctrl).all():
        problems.append("ctrl에 NaN/inf")
    if not np.isfinite(t).all():
        problems.append("sim_time_s에 NaN/inf")
        return problems
    dt_ms = np.diff(t) * 1000.0
    if len(dt_ms):
        ok = np.zeros(len(dt_ms), dtype=bool)
        for expected in cfg.TICK_MS_EXPECTED:
            ok |= np.abs(dt_ms - expected) <= cfg.TICK_MS_TOLERANCE
        if not ok.all():
            problems.append(f"틱 간격 이상 {int((~ok).sum())}곳 (최소 {dt_ms.min():.1f}, 최대 {dt_ms.max():.1f} ms)")
    return problems


def quality_row(meta: dict, arrays: dict[str, np.ndarray]) -> dict:
    """한 에피소드의 점검 결과 한 줄. warnings는 보고용이다."""
    problems = check_arrays(meta, arrays)
    warnings = list(problems)
    duration_s = float(meta["duration_s"])
    if not cfg.LENGTH_S_WARN[0] <= duration_s <= cfg.LENGTH_S_WARN[1]:
        warnings.append(f"길이 {duration_s:.1f}초가 {cfg.LENGTH_S_WARN} 밖")
    if meta.get("success_differs_from_auto"):
        warnings.append("수동 결과와 자동 판정이 다름")
    if meta["still_ratio"] > cfg.STILL_RATIO_WARN:
        warnings.append(f"정지 비율 {meta['still_ratio']:.2f} > {cfg.STILL_RATIO_WARN}")
    if meta["still_longest_s"] > cfg.STILL_LONGEST_S_WARN:
        warnings.append(f"가장 긴 정지 {meta['still_longest_s']:.2f}초 > {cfg.STILL_LONGEST_S_WARN}")
    return {
        "index": meta["episode_index"],
        "seed": meta["seed"],
        "n_ticks": meta["n_ticks"],
        "duration_s": duration_s,
        "still_ratio": meta["still_ratio"],
        "still_longest_s": meta["still_longest_s"],
        "data_problems": problems,
        "warnings": warnings,
    }


def distribution(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    q = np.percentile(a, [0, 25, 50, 75, 100])
    return dict(zip(["min", "p25", "median", "p75", "max"], (float(x) for x in q), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=cfg.RAW_DIR)
    parser.add_argument("--exclude", default="", help="추가로 뺄 에피소드 번호(쉼표). 기본은 성공 전부 포함")
    args = parser.parse_args()
    exclude = {int(x) for x in args.exclude.split(",") if x.strip()}

    metas = load_metas(args.raw)
    rows = select(metas, exclude)
    by_index = {m["episode_index"]: m for m in metas}

    print(f"에피소드 {len(metas)}개 (포함 {sum(r['included'] for r in rows)}, 제외 {sum(not r['included'] for r in rows)})\n")
    print("| 번호 | 시드 | 포함 | 이유 | 틱 | 길이(s) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        m = by_index[r["index"]]
        print(f"| {r['index']} | {r['seed']} | {'O' if r['included'] else 'X'} | {r['reason']} | {m['n_ticks']} | {m['duration_s']:.1f} |")

    included = [r["index"] for r in rows if r["included"]]
    quality = [quality_row(by_index[i], load_arrays(args.raw, i)) for i in included]
    print(f"\n품질 점검(포함 {len(included)}개). 데이터 문제 = shape/NaN/틱 간격, 경고 = 위 기준 초과")
    print("| 번호 | 틱 | 정지비율 | 가장 긴 정지(s) | 데이터 문제 | 경고 |")
    print("|---|---|---|---|---|---|")
    for q in quality:
        print(f"| {q['index']} | {q['n_ticks']} | {q['still_ratio']:.3f} | {q['still_longest_s']:.2f} | {'; '.join(q['data_problems']) or '-'} | {'; '.join(q['warnings']) or '-'} |")

    print("\n분포 (최소 / 25% / 중앙 / 75% / 최대)")
    for name, key in (("틱 수", "n_ticks"), ("길이(s)", "duration_s"), ("정지 비율", "still_ratio"), ("가장 긴 정지(s)", "still_longest_s")):
        d = distribution([q[key] for q in quality])
        print(f"- {name}: {d['min']:.3f} / {d['p25']:.3f} / {d['median']:.3f} / {d['p75']:.3f} / {d['max']:.3f}")
    print(f"- 총 프레임 수(포함 에피소드의 틱 합): {sum(q['n_ticks'] for q in quality)}")
    print(f"- 정지 비율 {cfg.STILL_RATIO_WARN} 초과: {sum(q['still_ratio'] > cfg.STILL_RATIO_WARN for q in quality)}개, "
          f"가장 긴 정지 {cfg.STILL_LONGEST_S_WARN}초 초과: {sum(q['still_longest_s'] > cfg.STILL_LONGEST_S_WARN for q in quality)}개, "
          f"데이터 문제 있음: {sum(bool(q['data_problems']) for q in quality)}개")


if __name__ == "__main__":
    main()
