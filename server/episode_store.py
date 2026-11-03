"""에피소드 저장소: data/raw/episode_000012.npz(배열) + episode_000012.json(메타). 웹캠 영상이나 이미지는 저장하지 않는다.

형식을 npz + json으로 고른 이유: numpy만으로 읽히고(pickle 불필요), 학습 환경(train/.venv)에도 numpy가 있어 새 의존성이 없다.
json은 사람이 열어 보거나 목록만 훑을 때 쓴다. 쓰는 순서는 npz → json이고 json이 있어야 저장된 것으로 친다(쓰다 끊긴 파일 무시).
각 파일은 임시 이름으로 쓴 뒤 os.replace로 바꿔서 반쯤 쓰인 파일이 남지 않는다.

npz 배열(모두 같은 길이 T): sim_time_s (T,) float64, qpos (T,13) float64, ctrl (T,6) float64, is_intervention (T,) bool.
qpos 순서: [팔 6개(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper) rad,
            블록 위치 x, y, z (m), 블록 회전 쿼터니언 w, x, y, z]. 짝 맞추기 규칙은 server/recorder.py 맨 위 참고.

디스크 쓰기는 EpisodeWriter 스레드가 한다. 실패(디스크 가득 등)해도 스레드는 죽지 않고 on_failed로 이유를 알린다.
"""

import hashlib
import json
import logging
import os
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from server import config
from server.recorder import FinishedEpisode

log = logging.getLogger(__name__)

# 장면이 include하는 로봇 모델(해시에 같이 넣는다). 장면 xml이 바뀌지 않아도 로봇 모델이 바뀌면 재생 결과가 달라진다.
ROBOT_MODEL_PATH = config.REPO_ROOT / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
QPOS_LAYOUT = [*JOINT_NAMES, "block_x_m", "block_y_m", "block_z_m", "block_qw", "block_qx", "block_qy", "block_qz"]


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def config_snapshot() -> dict[str, Any]:
    """server/config.py의 대문자 설정값 전부(경로 값은 컴퓨터마다 달라서 뺀다). JSON으로 쓸 수 있게 바꾼다."""
    snapshot: dict[str, Any] = {}
    for name in sorted(vars(config)):
        value = getattr(config, name)
        if not name.isupper() or isinstance(value, Path):
            continue
        snapshot[name] = json.loads(json.dumps(value, default=str))
    return snapshot


def config_sha256(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()


def stem_of(index: int) -> str:
    return f"episode_{index:06d}"


class EpisodeStore:
    def __init__(self, root: Path | None = None, scene_path: Path | None = None) -> None:
        self.root = Path(root) if root is not None else config.RECORD_DIR
        self._scene_path = Path(scene_path) if scene_path is not None else config.SCENE_PATH
        self._context: dict[str, Any] | None = None

    def context(self) -> dict[str, Any]:
        """모든 에피소드에 같이 들어가는 값(장면·설정 해시 등). 처음 한 번만 계산한다."""
        if self._context is None:
            snapshot = config_snapshot()
            self._context = {
                "scene_sha256": file_sha256(self._scene_path),
                "robot_model_sha256": file_sha256(ROBOT_MODEL_PATH),
                "config_sha256": config_sha256(snapshot),
                "config": snapshot,
                "mujoco_version": mujoco.__version__,
            }
        return self._context

    def scan(self) -> list[dict[str, Any]]:
        """저장된 에피소드 메타(json)를 번호 순서로 읽는다. 읽을 수 없는 파일은 건너뛴다."""
        metas: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return metas
        for path in sorted(self.root.glob("episode_*.json")):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                int(meta["episode_index"])
                metas.append(meta)
            except (OSError, ValueError, KeyError, TypeError):
                log.warning("skipping unreadable episode file: %s", path)
        metas.sort(key=lambda m: m["episode_index"])
        return metas

    def build_meta(self, ep: FinishedEpisode, timestep_s: float) -> dict[str, Any]:
        ctx = self.context()
        return {
            "format_version": config.RECORD_FORMAT_VERSION,
            "episode_index": ep.episode_index,
            "seed": ep.seed,
            "source": ep.source,
            "success": ep.success,
            "end_reason": ep.end_reason,
            "auto_success": {
                "ever": ep.auto_success_ever,
                "first_sim_time_s": ep.auto_success_first_s,
                "at_end": ep.auto_success_at_end,
            },
            "success_differs_from_auto": ep.success != ep.auto_success_ever,
            "n_ticks": ep.n_ticks,
            "duration_s": round(float(ep.sim_time_s[-1] - ep.sim_time_s[0]), 4),
            "fps": config.RECORD_FPS,
            "sim_timestep_s": timestep_s,
            "still_ratio": round(ep.still_ratio, 4),
            "still_longest_s": round(ep.still_longest_s, 3),
            "still_ctrl_delta_rad": config.STILL_CTRL_DELTA_RAD,
            "block_start_m": list(ep.block_start_m),
            "target_m": list(ep.target_m),
            "created_at": ep.created_at,
            "is_intervention_any": bool(ep.is_intervention.any()),
            "qpos_layout": QPOS_LAYOUT,
            "joint_names": JOINT_NAMES,
            "record_camera": {
                "azimuth_deg": config.RECORD_CAMERA_AZIMUTH_DEG,
                "elevation_deg": config.RECORD_CAMERA_ELEVATION_DEG,
                "distance_m": config.RECORD_CAMERA_DISTANCE_M,
                "lookat_m": list(config.RECORD_CAMERA_LOOKAT_M),
                "height_px": config.RECORD_IMAGE_HEIGHT_PX,
                "width_px": config.RECORD_IMAGE_WIDTH_PX,
            },
            **ctx,
        }

    def save(self, ep: FinishedEpisode, timestep_s: float) -> dict[str, Any]:
        """npz → json 순서로 쓴다. 실패하면 임시 파일을 지우고 예외(OSError 등)를 그대로 던진다."""
        self.root.mkdir(parents=True, exist_ok=True)
        stem = stem_of(ep.episode_index)
        meta = self.build_meta(ep, timestep_s)
        tmp_npz = self.root / f".{stem}.tmp.npz"
        tmp_json = self.root / f".{stem}.tmp.json"
        try:
            np.savez_compressed(
                tmp_npz,
                sim_time_s=ep.sim_time_s,
                qpos=ep.qpos,
                ctrl=ep.ctrl,
                is_intervention=ep.is_intervention,
            )
            os.replace(tmp_npz, self.root / f"{stem}.npz")
            tmp_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp_json, self.root / f"{stem}.json")
        except BaseException:
            for tmp in (tmp_npz, tmp_json):
                tmp.unlink(missing_ok=True)
            # json이 없는 npz는 저장된 것이 아니므로 남기지 않는다.
            if not (self.root / f"{stem}.json").exists():
                (self.root / f"{stem}.npz").unlink(missing_ok=True)
            raise
        return meta

    def delete(self, index: int) -> None:
        """json을 먼저 지운다(목록에서 사라지는 것이 삭제의 기준). 없는 파일은 무시."""
        stem = stem_of(index)
        (self.root / f"{stem}.json").unlink(missing_ok=True)
        (self.root / f"{stem}.npz").unlink(missing_ok=True)

    def load_arrays(self, index: int) -> dict[str, np.ndarray]:
        with np.load(self.root / f"{stem_of(index)}.npz", allow_pickle=False) as data:
            return {key: data[key] for key in data.files}


class EpisodeWriter:
    """디스크 쓰기·삭제를 순서대로 하는 스레드. submit_*는 기다리지 않고 바로 돌아온다."""

    def __init__(
        self,
        store: EpisodeStore,
        timestep_s: float,
        *,
        on_saved: Callable[[dict[str, Any]], None],
        on_failed: Callable[[int, str], None],
        on_deleted: Callable[[int, str | None], None],
    ) -> None:
        self._store = store
        self._timestep_s = timestep_s
        self._on_saved = on_saved
        self._on_failed = on_failed
        self._on_deleted = on_deleted
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="episode-writer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit_save(self, ep: FinishedEpisode) -> None:
        self._queue.put(("save", ep))

    def submit_delete(self, index: int) -> None:
        self._queue.put(("delete", index))

    def close(self, timeout_s: float = 10.0) -> None:
        """대기 중인 쓰기를 끝내고 멈춘다."""
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
                    self._handle_delete(payload)
            except Exception:  # noqa: BLE001 - 쓰기 스레드는 죽지 않는다
                log.exception("episode writer task failed")

    def _handle_save(self, ep: FinishedEpisode) -> None:
        try:
            meta = self._store.save(ep, self._timestep_s)
        except Exception as err:  # noqa: BLE001 - 디스크 가득, 권한 등 무엇이든 화면에 알린다
            log.error("saving episode %d failed: %r", ep.episode_index, err)
            self._on_failed(ep.episode_index, f"{type(err).__name__}: {err}")
            return
        self._on_saved(meta)

    def _handle_delete(self, index: int) -> None:
        try:
            self._store.delete(index)
        except Exception as err:  # noqa: BLE001
            self._on_deleted(index, f"{type(err).__name__}: {err}")
            return
        self._on_deleted(index, None)
