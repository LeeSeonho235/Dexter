"""3-4: 학습된 ACT 정책을 시뮬레이션에서 돌려 성공률을 잰다. train/.venv에서 실행한다(lerobot + mujoco 3.16.0, 서버와 같은 버전).

- 시뮬레이션 루프와 지표는 server/eval_env.py, server/eval_metrics.py(mujoco만 씀). 여기는 ACT를 부르는 어댑터와 명령줄이다.
- 평가 시드는 server/config.py의 EVAL_SEED_RANGE(1000~1049). 최대 시간 등은 server/eval_config.py(EVAL_MAX_S 45초, 시연 후에 정한 값).
- 관측: state = qpos[:6](float32), 이미지 = 녹화와 같은 StateRenderer 렌더링(240x320). 행동 6개를 ctrl에 그대로 넣는다.
  변화량 제한은 기본 끔(--limit-step으로 켬. 이유: server/eval_config.py).
- ACT는 n_action_steps(기본 100)마다만 관측을 읽으므로 그 샘플에서만 렌더링·추론한다(--render-every-sample로 끌 수 있음).
- 결과: experiments/<이름>/results.csv, summary.json (+ --save-fail-frames K: 실패 에피소드 마지막 K장 PNG, 시뮬레이션 화면뿐).
- 체크포인트 여러 개 비교: --checkpoints a,b,c (쉼표로 구분). 결과는 experiments/<이름>/<체크포인트별 하위 폴더>/(results.csv, summary.json)와
  experiments/<이름>/summary.json의 comparison(체크포인트, 성공률, Wilson 구간, 중앙 시간, 실패 유형 개수)에 남는다.

사용 (저장소 루트에서):
  train/.venv/bin/python train/evaluate.py --policy hold --n 5                       # 아무것도 안 하는 기저선
  train/.venv/bin/python train/evaluate.py --policy act --checkpoint checkpoints/dexter_act_v1/pretrained_model
  train/.venv/bin/python train/evaluate.py --policy act --name compare_steps \\
      --checkpoints checkpoints/step_004000/pretrained_model,checkpoints/step_010000/pretrained_model,checkpoints/step_020000/pretrained_model
"""

import os

# mujoco / torch를 import하기 전에 정한다.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LP_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 로컬 체크포인트만 쓴다
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # server 패키지
sys.path.insert(0, str(Path(__file__).resolve().parent))  # dataset_config

import dataset_config as dcfg  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

from server import config  # noqa: E402
from server import eval_config as ec  # noqa: E402
from server.episode_store import config_sha256, config_snapshot  # noqa: E402
from server.eval_env import EvalEnv, HoldPolicy, Observation, rollout  # noqa: E402
from server.eval_metrics import write_results  # noqa: E402


class ActPolicy:
    """lerobot ACTPolicy 어댑터. predict_action(전처리 → select_action → 후처리)을 부른다."""

    name = "act"

    def __init__(self, checkpoint: Path, device: str = "cpu", skip_render: bool = True) -> None:
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        torch.set_num_threads(ec.THREADS)
        self._torch = torch
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(str(checkpoint))
        self.policy.to(self.device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(checkpoint))
        cfg = self.policy.config
        if dcfg.IMAGE_KEY not in cfg.image_features:
            raise SystemExit(f"체크포인트의 이미지 입력 {list(cfg.image_features)} 에 {dcfg.IMAGE_KEY} 가 없습니다")
        self.n_action_steps = int(cfg.n_action_steps)
        self.temporal_ensemble = cfg.temporal_ensemble_coeff is not None
        self.skip_render = skip_render
        self._last_image: np.ndarray | None = None

    def reset(self) -> None:
        self.policy.reset()  # 에피소드마다 행동 대기열을 비운다
        self._last_image = None

    def needs_image(self, k: int) -> bool:
        """select_action은 대기열이 빌 때(k = 0, n, 2n, ...)만 관측을 읽는다. 시간 앙상블은 매 샘플 읽는다."""
        return (not self.skip_render) or self.temporal_ensemble or k % self.n_action_steps == 0

    def act(self, k: int, obs: Observation) -> np.ndarray:
        from lerobot.common.control_utils import predict_action

        if obs.image is not None:
            self._last_image = obs.image
        image = self._last_image  # 관측이 필요 없는 샘플에서는 직전 이미지를 넘긴다(대기열에서 꺼내기만 해서 쓰이지 않는다)
        if image is None:
            raise RuntimeError("첫 샘플에는 이미지가 필요합니다")
        observation = {"observation.state": obs.state, dcfg.IMAGE_KEY: image}
        action = predict_action(observation, self.policy, self.device, self.pre, self.post, use_amp=False, task=dcfg.TASK)
        return action.detach().cpu().numpy().astype(np.float64).reshape(-1)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def save_fail_frames(env: EvalEnv, result, out_dir: Path) -> None:
    folder = Path(out_dir) / "fail_frames"
    folder.mkdir(parents=True, exist_ok=True)
    for i, qpos in enumerate(result.tail_qpos):
        Image.fromarray(env.renderer.render(qpos)).save(folder / f"seed_{result.seed}_{result.failure}_{i}.png")


def checkpoint_label(path: Path, used: set[str]) -> str:
    """하위 폴더 이름: .../step_010000/pretrained_model 이면 step_010000, 아니면 폴더 이름. 겹치면 _2, _3을 붙인다."""
    path = Path(path)
    base = path.parent.name if path.name == "pretrained_model" and path.parent.name else path.name
    label, n = base, 1
    while label in used:
        n += 1
        label = f"{base}_{n}"
    used.add(label)
    return label


def comparison_row(label: str, ckpt_info: dict | None, summary: dict, out_dir: Path) -> dict:
    """체크포인트 하나의 한 줄 비교표 행."""
    s = summary["summary"]
    t = s["time_to_success_s"]
    return {
        "name": label,
        "checkpoint": ckpt_info["path"] if ckpt_info else None,
        "model_safetensors_sha256": ckpt_info["model_safetensors_sha256"] if ckpt_info else None,
        "n": s["n"], "successes": s["successes"], "success_rate": s["success_rate"], "wilson95": s["wilson95"],
        "median_time_s": t["median"] if t else None,
        "failure_counts": s["failure_counts"],
        "dir": str(out_dir),
    }


def comparison_line(row: dict) -> str:
    lo, hi = row["wilson95"]
    median = "-" if row["median_time_s"] is None else f"{row['median_time_s']:.1f}s"
    fails = ", ".join(f"{ec.FAILURE_LABELS_KO[k]} {v}" for k, v in row["failure_counts"].items() if v) or "없음"
    return f"{row['name']} | 성공률 {row['successes']}/{row['n']} = {row['success_rate']:.1%} | Wilson 95% {lo:.1%}~{hi:.1%} | 성공까지 중앙값 {median} | 실패: {fails}"


def run_policy(env: EvalEnv, policy, seeds: list[int], args, out_dir: Path, name: str, ckpt_info: dict | None) -> dict:
    """정책 하나를 시드 전체로 평가하고 out_dir에 results.csv, summary.json을 쓴다. 쓴 summary(dict)를 돌려준다."""
    print(f"정책 {policy.name}{'' if ckpt_info is None else f' ({ckpt_info['path']})'}, 시드 {seeds[0]}~{seeds[-1]} ({len(seeds)}개), 최대 {args.max_s}초, "
          f"변화량 제한 {'켬' if args.limit_step else '끔'}", flush=True)
    rows, start = [], time.perf_counter()
    for i, seed in enumerate(seeds, start=1):
        t0 = time.perf_counter()
        res = rollout(env, policy, seed, max_s=args.max_s, limit=args.limit_step, keep_tail=args.save_fail_frames)
        wall = time.perf_counter() - t0
        rows.append(res.row())
        if args.save_fail_frames and not res.success:
            out_dir.mkdir(parents=True, exist_ok=True)
            save_fail_frames(env, res, out_dir)
        elapsed = time.perf_counter() - start
        print(f"[{i}/{len(seeds)}] 시드 {seed}: {ec.FAILURE_LABELS_KO[res.failure]}"
              f"{'' if res.time_to_success_s is None else f' {res.time_to_success_s:.1f}s'} | 샘플 {res.n_samples}, "
              f"벽시계 {wall:.1f}s (물리 {res.physics_s:.1f}, 렌더 {res.render_s:.1f}, 정책 {res.policy_s:.1f}) | "
              f"남은 시간 약 {elapsed / i * (len(seeds) - i) / 60:.1f}분", flush=True)
    meta = {
        "run": name, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"), "policy": policy.name,
        "checkpoint": ckpt_info, "seeds": [seeds[0], seeds[-1]], "n_requested": len(seeds),
        "eval": {
            "max_s": args.max_s, "settle_s": ec.EVAL_SETTLE_S, "start_pose": ec.EVAL_START_POSE, "fps": ec.EVAL_FPS, "limit_step": args.limit_step,
            "limit_step_arm_rad": ec.LIMIT_STEP_ARM_RAD, "limit_step_gripper_rad": ec.LIMIT_STEP_GRIPPER_RAD,
            "render_only_when_needed": not args.render_every_sample, "max_s_chosen_after_demos": True,
            "failure_thresholds": {"lift_z_min_m": ec.LIFT_Z_MIN_M, "pushed_xy_m": ec.PUSHED_XY_M, "near_target_m": ec.NEAR_TARGET_M},
        },
        "act": None if policy.name != "act" else {"n_action_steps": policy.n_action_steps, "temporal_ensemble": policy.temporal_ensemble},
        "server_config_sha256": config_sha256(config_snapshot()),
        "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__, "python": platform.python_version(), "git_head": git_head()},
        "wall_s": round(time.perf_counter() - start, 1),
    }
    summary = write_results(out_dir, rows, meta)
    s = summary["summary"]
    print(f"\n성공률 {s['successes']}/{s['n']} = {s['success_rate']:.1%}  (Wilson 95% 구간 {s['wilson95'][0]:.1%}~{s['wilson95'][1]:.1%})")
    if s["time_to_success_s"]:
        t = s["time_to_success_s"]
        print(f"성공까지 걸린 시간(s) 최소/25%/중앙/75%/최대: {t['min']}/{t['p25']}/{t['median']}/{t['p75']}/{t['max']}")
    print("실패 유형:", {ec.FAILURE_LABELS_KO[k]: v for k, v in s["failure_counts"].items() if v})
    tm = s["timing_s"]
    calls, samples = max(1, tm["policy_calls"]), max(1, tm["samples"])
    print(f"속도: 샘플당 {1000 * (tm['render_total'] + tm['policy_total'] + tm['physics_total']) / samples:.2f} ms "
          f"(물리 {1000 * tm['physics_total'] / samples:.2f}, 렌더 {1000 * tm['render_total'] / calls:.1f} ms/회, 정책 {1000 * tm['policy_total'] / calls:.1f} ms/회 × {tm['policy_calls']}회), "
          f"에피소드당 벽시계 평균 {summary['wall_s'] / s['n']:.1f}s")
    print(f"결과: {out_dir}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", choices=["hold", "act"], required=True)
    parser.add_argument("--checkpoint", type=Path, help="pretrained_model 폴더 (--policy act)")
    parser.add_argument("--checkpoints", help="쉼표로 구분한 pretrained_model 폴더 여러 개(--policy act). 체크포인트별 하위 폴더와 비교표를 만든다")
    parser.add_argument("--seed-start", type=int, default=config.EVAL_SEED_RANGE[0])
    parser.add_argument("--n", type=int, default=config.EVAL_SEED_RANGE[1] - config.EVAL_SEED_RANGE[0], help="평가 시드 개수")
    parser.add_argument("--max-s", type=float, default=ec.EVAL_MAX_S)
    parser.add_argument("--limit-step", action="store_true", default=ec.EVAL_LIMIT_STEP_DEFAULT, help="변화량 제한을 켠다(기본 끔)")
    parser.add_argument("--render-every-sample", action="store_true", help="n_action_steps마다가 아니라 매 샘플 렌더링(느림, 검증용)")
    parser.add_argument("--save-fail-frames", type=int, default=0, metavar="K", help="실패 에피소드의 마지막 K장을 PNG로 저장")
    parser.add_argument("--name", help="결과 폴더 이름(기본: <정책>_<시각>)")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    ckpts: list[Path] = []
    if args.policy == "act":
        if args.checkpoint and args.checkpoints:
            parser.error("--checkpoint와 --checkpoints는 같이 쓸 수 없습니다")
        if args.checkpoints:
            ckpts = [Path(p.strip()) for p in args.checkpoints.split(",") if p.strip()]
            if len(ckpts) < 2:
                parser.error("--checkpoints에는 2개 이상 적으세요(하나면 --checkpoint)")
        elif args.checkpoint:
            ckpts = [args.checkpoint]
        else:
            parser.error("--policy act 에는 --checkpoint 또는 --checkpoints가 필요합니다")
        missing = [str(p) for p in ckpts if not (p / "model.safetensors").is_file()]
        if missing:
            parser.error(f"model.safetensors가 없는 체크포인트 폴더: {missing}")
    lo, hi = config.EVAL_SEED_RANGE
    seeds = list(range(args.seed_start, args.seed_start + args.n))
    if not seeds or seeds[0] < lo or seeds[-1] >= hi:
        parser.error(f"평가 시드는 {lo}~{hi - 1} 안이어야 합니다 (요청 {seeds[:1]}~{seeds[-1:]})")
    try:
        os.nice(ec.NICE_LEVEL)
    except OSError as err:
        print(f"nice 설정 실패(계속 진행): {err}", file=sys.stderr)

    name = args.name or f"{args.policy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out_root / name
    if out_dir.exists():
        parser.error(f"{out_dir} 가 이미 있습니다(덮어쓰지 않음)")

    multi = len(ckpts) > 1
    with EvalEnv() as env:
        if args.policy == "hold":
            run_policy(env, HoldPolicy(env.park_ctrl()), seeds, args, out_dir, name, None)
            return 0
        used: set[str] = set()
        comparison: list[dict] = []
        t_all = time.perf_counter()
        for ckpt in ckpts:
            label = checkpoint_label(ckpt, used)
            sub_dir = out_dir / label if multi else out_dir
            t_load = time.perf_counter()
            policy = ActPolicy(ckpt, args.device, skip_render=not args.render_every_sample)
            ckpt_info = {"path": str(ckpt), "model_safetensors_sha256": file_sha256(Path(ckpt) / "model.safetensors")}
            print(f"\n=== {label}: 준비 {time.perf_counter() - t_load:.1f}s ===", flush=True)
            summary = run_policy(env, policy, seeds, args, sub_dir, f"{name}/{label}" if multi else name, ckpt_info)
            comparison.append(comparison_row(label, ckpt_info, summary, sub_dir))
            del policy
            gc.collect()
        if multi:
            lines = [comparison_line(r) for r in comparison]
            top = {
                "run": name, "kind": "checkpoint_comparison", "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "seeds": [seeds[0], seeds[-1]], "n_requested": len(seeds), "max_s": args.max_s, "limit_step": args.limit_step,
                "comparison": comparison, "comparison_lines": lines, "wall_s": round(time.perf_counter() - t_all, 1),
            }
            (out_dir / "summary.json").write_text(json.dumps(top, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            print("\n=== 체크포인트 비교 ===")
            for line in lines:
                print(line)
            print(f"비교표: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
