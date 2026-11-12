"""교정 에피소드 저장소(4-2): data/corrections/correction_000012.npz + .json, 그리고 auto_runs.csv(AUTO를 한 번 돌릴 때마다 한 줄).

data/raw(사람 시연)와 섞이지 않는 새 폴더다. 학습 데이터셋 변환(train/build_dataset.py)은 data/raw만 읽으므로 실수로 합쳐지지 않는다.
웹캠 영상과 이미지는 저장하지 않는다. 저장하는 것은 숫자(관절, ctrl, 블록 위치, 주체)뿐이다.
파일 쓰기 방식은 3-2(server/episode_store.py)와 같다: 임시 이름에 쓰고 바꿔치기, npz → json 순서, json이 있어야 저장된 것.

npz: sim_time_s (T,), qpos (T,13), ctrl (T,6), is_intervention (T,) bool, owner (T,) int8 (0 AUTO, 1 HUMAN, 2 HUMAN_WAIT).
디스크 쓰기는 CorrectionWriter 스레드가 한다(시뮬레이션 틱을 막지 않는다). 실패해도 스레드는 죽지 않고 on_failed로 이유를 알린다.
"""

import csv
import hashlib
import json
import logging
import os
import queue
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from server import auto_config, config, correction_config as cc
from server.correction import (
    CorrectionRecording,
    handback_jumps_rad,
    intervention_segments,
    owner_time_s,
    takeover_first_second_steps_rad,
    typical_step_rad,
)
from server.episode_store import JOINT_NAMES, QPOS_LAYOUT, EpisodeStore, config_sha256

log = logging.getLogger(__name__)

CSV_FIELDS = ["seed", "kind", "outcome", "end_reason", "auto_time_s", "human_time_s", "total_s", "intervention_count", "episode_index", "checkpoint_sha256", "created_at"]
KIND_AUTO_ONLY = "auto_only"
KIND_SAVED = "correction_saved"
KIND_DISCARDED = "correction_discarded"
KIND_SAVE_FAILED = "correction_save_failed"  # 저장하려 했지만 디스크 오류. 시드는 이미 쓴 것이므로 장부에는 남긴다

_identity_cache: dict[tuple, dict[str, Any]] = {}


def checkpoint_identity(checkpoint_dir: Path | None = None) -> dict[str, Any]:
    """기준 체크포인트의 model.safetensors SHA-256과 경로. 파일이 크므로 (경로, 크기, 수정 시각)이 같으면 캐시를 쓴다(쓰기 스레드에서 부른다)."""
    folder = Path(checkpoint_dir) if checkpoint_dir is not None else Path(auto_config.POLICY_CHECKPOINT)
    path = folder / "model.safetensors"
    try:
        shown = str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        shown = str(path)
    try:
        stat = path.stat()
    except OSError:
        return {"model_safetensors_path": shown, "model_safetensors_sha256": None}
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _identity_cache:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        _identity_cache[key] = {"model_safetensors_path": shown, "model_safetensors_sha256": digest.hexdigest()}
    return dict(_identity_cache[key])


def correction_config_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name in sorted(vars(cc)):
        value = getattr(cc, name)
        if not name.isupper() or isinstance(value, Path):
            continue
        snapshot[name] = json.loads(json.dumps(value, default=str))
    return snapshot


def stem_of(index: int) -> str:
    return f"correction_{index:06d}"


class CorrectionStore:
    def __init__(self, root: Path | None = None, scene_path: Path | None = None) -> None:
        self.root = Path(root) if root is not None else cc.CORRECTION_DIR
        self.csv_path = self.root / cc.AUTO_RUNS_CSV_NAME
        self._episode_store = EpisodeStore(self.root, scene_path)  # 장면·설정 해시 계산만 빌려 쓴다(저장은 하지 않는다)

    # --- 읽기 ---

    def scan(self) -> list[dict[str, Any]]:
        metas: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return metas
        for path in sorted(self.root.glob("correction_*.json")):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                int(meta["episode_index"])
                metas.append(meta)
            except (OSError, ValueError, KeyError, TypeError):
                log.warning("skipping unreadable correction file: %s", path)
        metas.sort(key=lambda m: m["episode_index"])
        return metas

    def csv_rows(self) -> list[dict[str, str]]:
        try:
            with self.csv_path.open(encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
        except OSError:
            return []

    def used_seeds(self) -> list[int]:
        """교정 폴더에서 이미 쓴 시드(저장한 교정 + auto_runs.csv의 모든 줄). 다음 시드가 시연·교정·AUTO와 겹치지 않게 하려고 읽는다."""
        seeds: list[int] = []
        for meta in self.scan():
            try:
                seeds.append(int(meta["seed"]))
            except (KeyError, ValueError, TypeError):
                pass
        for row in self.csv_rows():
            try:
                seeds.append(int(row["seed"]))
            except (KeyError, ValueError, TypeError):
                pass
        return seeds

    def next_index(self) -> int:
        return max((m["episode_index"] for m in self.scan()), default=-1) + 1

    def load_arrays(self, index: int) -> dict[str, np.ndarray]:
        with np.load(self.root / f"{stem_of(index)}.npz", allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    # --- 쓰기(쓰기 스레드에서) ---

    def build_meta(self, rec: CorrectionRecording, index: int, decision: str = "saved") -> dict[str, Any]:
        ctx = self._episode_store.context()
        times = owner_time_s(rec.owner, rec.fps)
        segments = intervention_segments(rec.owner)
        identity = checkpoint_identity()
        corr_snapshot = correction_config_snapshot()
        return {
            "format_version": cc.CORRECTION_FORMAT_VERSION,
            "episode_index": index,
            "seed": rec.seed,
            "block_start_m": list(rec.block_start_m),
            "target_m": list(config.TARGET_CENTER_M),
            "source": cc.CORRECTION_SOURCE,
            "base_checkpoint": identity,
            "interventions": [{"start_sample": s, "end_sample": e, "n_samples": e - s + 1} for s, e in segments],
            "intervention_count": len(segments),
            "human_time_s": round(times["human_time_s"], 4),  # HUMAN 구간만(손을 기다린 시간은 빠진다): 06 문서의 "같은 시간 비교"에 쓰는 값
            "auto_time_s": round(times["auto_time_s"], 4),
            "human_wait_time_s": round(times["human_wait_time_s"], 4),
            "total_s": round(times["total_s"], 4),
            "auto_result": {
                "outcome": rec.outcome,
                "end_reason": rec.end_reason,
                "time_to_success_s": rec.time_to_success_s,
                "error": rec.error,
            },
            "user_decision": decision,
            "ctrl_jumps": {
                "handback_rad": handback_jumps_rad(rec.owner, rec.ctrl),
                "takeover_first_second_max_step_rad": takeover_first_second_steps_rad(rec.owner, rec.ctrl, rec.fps),
                "typical_step_rad": typical_step_rad(rec.ctrl),
                "handback_jump_warn_rad": cc.HANDBACK_JUMP_WARN_RAD,
            },
            "n_ticks": rec.n_ticks,
            "duration_s": round(float(rec.sim_time_s[-1] - rec.sim_time_s[0]), 4) if rec.n_ticks else 0.0,
            "fps": rec.fps,
            "owner_codes": {"AUTO": cc.OWNER_AUTO, "HUMAN": cc.OWNER_HUMAN, "HUMAN_WAIT": cc.OWNER_HUMAN_WAIT},
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "qpos_layout": QPOS_LAYOUT,
            "joint_names": JOINT_NAMES,
            "correction_config_sha256": config_sha256(corr_snapshot),
            "correction_config": corr_snapshot,
            **ctx,  # scene_sha256, robot_model_sha256, config_sha256, config, mujoco_version
            **rec.extras,
        }

    def save(self, rec: CorrectionRecording) -> dict[str, Any]:
        """npz → json 순서로 쓰고 메타를 돌려준다. 실패하면 임시 파일을 지우고 예외를 그대로 던진다."""
        self.root.mkdir(parents=True, exist_ok=True)
        index = self.next_index()
        stem = stem_of(index)
        meta = self.build_meta(rec, index)
        tmp_npz = self.root / f".{stem}.tmp.npz"
        tmp_json = self.root / f".{stem}.tmp.json"
        try:
            np.savez_compressed(
                tmp_npz,
                sim_time_s=rec.sim_time_s,
                qpos=rec.qpos,
                ctrl=rec.ctrl,
                is_intervention=rec.is_intervention,
                owner=rec.owner.astype(np.int8),
            )
            os.replace(tmp_npz, self.root / f"{stem}.npz")
            tmp_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp_json, self.root / f"{stem}.json")
        except BaseException:
            for tmp in (tmp_npz, tmp_json):
                tmp.unlink(missing_ok=True)
            if not (self.root / f"{stem}.json").exists():
                (self.root / f"{stem}.npz").unlink(missing_ok=True)
            raise
        return meta

    def append_csv(self, rec: CorrectionRecording, kind: str, episode_index: int | None = None) -> None:
        """AUTO 한 번에 한 줄. 헤더는 파일이 없을 때만 쓴다."""
        self.root.mkdir(parents=True, exist_ok=True)
        times = owner_time_s(rec.owner, rec.fps)
        new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        row = {
            "seed": rec.seed,
            "kind": kind,
            "outcome": rec.outcome,
            "end_reason": rec.end_reason,
            "auto_time_s": round(times["auto_time_s"], 3),
            "human_time_s": round(times["human_time_s"], 3),
            "total_s": round(times["total_s"], 3),
            "intervention_count": rec.intervention_count,
            "episode_index": "" if episode_index is None else episode_index,
            "checkpoint_sha256": checkpoint_identity()["model_safetensors_sha256"] or "",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)


class CorrectionWriter:
    """디스크 쓰기 스레드. submit_*는 기다리지 않고 바로 돌아온다."""

    def __init__(
        self,
        store: CorrectionStore,
        *,
        on_saved: Callable[[CorrectionRecording, dict[str, Any]], None],
        on_failed: Callable[[CorrectionRecording, str], None],
        on_csv_failed: Callable[[CorrectionRecording, str], None] | None = None,
    ) -> None:
        self._store = store
        self._on_saved = on_saved
        self._on_failed = on_failed
        self._on_csv_failed = on_csv_failed
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="correction-writer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit_save(self, rec: CorrectionRecording) -> None:
        self._queue.put(("save", rec))

    def submit_csv(self, rec: CorrectionRecording, kind: str) -> None:
        self._queue.put(("csv", (rec, kind)))

    def close(self, timeout_s: float = 10.0) -> None:
        self._queue.put(("stop", None))
        if self._thread.is_alive():
            self._thread.join(timeout_s)

    def _run(self) -> None:
        while True:
            kind, payload = self._queue.get()
            if kind == "stop":
                return
            try:
                if kind == "save":
                    self._handle_save(payload)
                else:
                    self._handle_csv(*payload)
            except Exception:  # noqa: BLE001 - 쓰기 스레드는 죽지 않는다
                log.exception("correction writer task failed")

    def _handle_save(self, rec: CorrectionRecording) -> None:
        try:
            meta = self._store.save(rec)
        except Exception as err:  # noqa: BLE001 - 디스크 가득, 권한 등 무엇이든 화면에 알린다
            log.error("saving correction (seed %s) failed: %r", rec.seed, err)
            self._on_failed(rec, f"{type(err).__name__}: {err}")
            # 시드는 이미 썼으니 장부에는 남기려 한다. 같은 원인으로 장부도 실패하면 화면에는 저장 실패 안내만 남기고(덮어쓰지 않는다) 로그에만 적는다.
            self._handle_csv(rec, KIND_SAVE_FAILED, notify=False)
            return
        self._on_saved(rec, meta)
        self._handle_csv(rec, KIND_SAVED, meta["episode_index"])

    def _handle_csv(self, rec: CorrectionRecording, kind: str, episode_index: int | None = None, notify: bool = True) -> None:
        try:
            self._store.append_csv(rec, kind, episode_index)
        except Exception as err:  # noqa: BLE001
            log.error("appending auto_runs.csv failed: %r", err)
            if notify and self._on_csv_failed is not None:
                self._on_csv_failed(rec, f"{type(err).__name__}: {err}")
