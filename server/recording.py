"""녹화 서비스: 브라우저의 record/reset 명령을 규칙대로 받아들이거나 거절하고, 저장 결과를 세고, 화면용 상태를 만든다.

규칙
- 블록 리셋(reset): 녹화 중이거나 이미 리셋 중이면 거절. 다음 시연 시드(0~99)의 위치로 블록을 놓고, 성공하면 그 시드가 "현재 시드"가 된다.
- 녹화 시작(start): 리셋 직후에만 허용한다(블록이 시드 위치에 있어야 seed 기록이 사실이다). 시작 때 블록이 시드 위치에서
  RECORD_START_BLOCK_TOLERANCE_M보다 벗어나 있으면 거절한다. 시작하면 "리셋 직후" 표시가 꺼진다(다음 녹화는 다시 리셋해야 한다).
- 끝(stop_success / stop_fail): 내가 정한 결과로 저장한다. 최대 길이(RECORD_MAX_S)를 넘으면 실패로 자동 종료하고 저장한다.
- 취소(discard): 저장하지 않고 버린다. 시드는 되돌리지 않는다(건너뛴 번호가 생길 수 있다).
- 마지막 에피소드 삭제(delete_last): 화면이 알고 있는 번호가 지금 가장 최근 번호일 때만 지운다(오래된 화면으로 엉뚱한 것을 지우지 않게).
- 저장은 쓰기 스레드가 하고, 성공/실패 수는 파일이 실제로 써진 뒤에만 센다. 쓰기가 실패하면(디스크 가득 등) 센 수에 넣지 않고
  화면에 오류를 띄운다(그 에피소드의 데이터는 메모리에서 사라진다. 다시 시도하는 기능은 없다).
- 다음 시연 시드: 서버를 켤 때 저장된 시연 에피소드의 가장 큰 시드 + 1(없으면 0). 시연 시드를 다 쓰면 리셋을 거절한다.

락은 짧은 상태 변경에만 잡고, 락 안에서 디스크 쓰기나 기다림을 하지 않는다.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

from server import config
from server.control import ControlWorker
from server.episode_store import EpisodeStore, EpisodeWriter
from server.recorder import EpisodeRecorder, FinishedEpisode
from server.owner import ControlOwner
from server.seeds import DEMO, block_xy_for_seed, seed_kind
from server.sim_worker import SimWorker

log = logging.getLogger(__name__)

INFO, WARN, ERROR = "info", "warn", "error"


class RecordingService:
    def __init__(
        self,
        worker: SimWorker,
        control: ControlWorker,
        *,
        store: EpisodeStore | None = None,
        max_s: float = config.RECORD_MAX_S,
        timestep_s: float | None = None,
        owner: ControlOwner | None = None,
        extra_seeds: Callable[[], list[int]] | None = None,
    ) -> None:
        """extra_seeds: data/raw 밖에서 이미 쓴 시드를 알려 주는 함수(교정 폴더와 auto_runs.csv). 다음 시드가 서버를 다시 켠 뒤에도 겹치지 않게 한다."""
        self._worker = worker
        self._control = control
        self._extra_seeds = extra_seeds
        # 끝난 교정을 저장할지 버릴지 고르는 중이면 True를 돌려주는 함수(AutoRunner가 연결한다). 고르기 전에는 블록 리셋도 받지 않는다.
        self.decision_pending: Callable[[], bool] = lambda: False
        # 제어 주체(server/owner.py). AUTO가 로봇을 쓰는 동안은 녹화와 블록 리셋을 받지 않는다(기본은 시뮬레이션이 가진 것).
        self._owner = owner or getattr(worker, "owner", None) or ControlOwner()
        self._store = store or EpisodeStore()
        timestep = timestep_s if timestep_s is not None else worker.timestep_s
        if timestep is None:
            raise ValueError("timestep is unknown: start the sim worker first")
        self._timestep_s = timestep
        self._max_s = max_s

        self._lock = threading.RLock()
        self._episodes: list[dict[str, Any]] = self._store.scan()  # 저장된 에피소드 메타(번호 순서)
        self._pending = 0  # 쓰기 스레드에 맡겼지만 아직 안 끝난 일
        self._save_error_count = 0
        self._event_id = 0
        self._last_event: dict[str, Any] | None = None
        self._resetting = False
        self._block_ready = False
        self._current_seed: int | None = None
        self._next_seed = self._compute_next_seed()
        self._next_index = self._compute_next_index()

        self.recorder = EpisodeRecorder(sink=self._on_finished, timestep_s=timestep, max_s=max_s)
        self._writer = EpisodeWriter(
            self._store,
            timestep,
            on_saved=self._on_saved,
            on_failed=self._on_save_failed,
            on_deleted=self._on_deleted,
        )

    def start(self) -> None:
        self._writer.start()
        self._worker.attach_recorder(self.recorder)

    def close(self) -> None:
        """서버 종료: 녹화 중이던 것은 버리고(저장하지 않는다), 대기 중인 쓰기를 끝낸다."""
        self._worker.attach_recorder(None)
        self.recorder.discard()
        self._writer.close()

    # --- 브라우저 명령 (asyncio 스레드). 돌려주는 값: None이면 받아들임, 문자열이면 거절 이유 ---

    AUTO_BUSY = "AUTO 실행 중에는 사용할 수 없습니다. 먼저 AUTO를 중지하세요"
    DECISION_PENDING = "지난 교정을 저장할지 버릴지 먼저 고르세요(AUTO 상자)"

    def handle_record(self, msg: dict[str, Any]) -> str | None:
        action = msg["action"]
        if self._owner.is_busy:  # AUTO, HUMAN_WAIT, HUMAN 모두
            return self._refuse_or(self.AUTO_BUSY)
        if action == "start":
            return self._refuse_or(self._start())
        if action in ("stop_success", "stop_fail"):
            return self._refuse_or(self._stop(action == "stop_success"))
        if action == "discard":
            return self._refuse_or(self._discard())
        if action == "delete_last":
            return self._refuse_or(self._delete_last(msg["episode"]))
        return self._refuse_or(f"unknown action: {action}")

    def handle_reset(self) -> str | None:
        if self._owner.is_busy:
            return self._refuse_or(self.AUTO_BUSY)
        if self.decision_pending():
            return self._refuse_or(self.DECISION_PENDING)
        return self._refuse_or(self._reset())

    def seed_info(self) -> dict[str, int | None]:
        """화면 안내용: 지금 놓여 있는 블록의 시드(리셋 직후일 때만)와 다음 리셋이 낼 시드(시연 시드를 다 썼으면 None)."""
        with self._lock:
            demo_hi = config.DEMO_SEED_RANGE[1]
            return {
                "ready_seed": self._current_seed if self._block_ready else None,
                "next_seed": self._next_seed if self._next_seed < demo_hi else None,
            }

    # --- AUTO(4-1): 블록 리셋 직후에만 시작하고, 시작하면 "리셋 직후" 표시를 소비한다(녹화 시작과 같은 규칙) ---

    def auto_precheck(self) -> str | None:
        """AUTO를 시작할 수 있는가(상태를 바꾸지 않는다). 못 하면 이유."""
        with self._lock:
            if self.recorder.active:
                return "녹화 중에는 AUTO를 시작할 수 없습니다"
            if self._resetting or self._control.resetting:
                return "블록 리셋이 진행 중입니다"
            if not self._block_ready or self._current_seed is None:
                return "블록 리셋을 먼저 하세요 (AUTO는 리셋 직후에만 시작할 수 있습니다)"
            if seed_kind(self._current_seed) != DEMO:
                return "AUTO는 시연 시드(0~99)의 블록에서만 시작할 수 있습니다(평가 시드는 막혀 있습니다)"
        return None

    def begin_auto(self) -> tuple[int | None, str | None]:
        """AUTO 에피소드를 시작한다: (시드, None) 또는 (None, 거절 이유). 블록이 시드 위치에 그대로 있는지 다시 확인하고 "리셋 직후"를 소비한다."""
        with self._lock:
            reason = self.auto_precheck()
            if reason is not None:
                return None, reason
            snapshot = self._worker.snapshot
            if snapshot is None:
                return None, "시뮬레이션이 아직 준비되지 않았습니다"
            block_m = snapshot["block_m"]
            seed_x, seed_y = block_xy_for_seed(self._current_seed)
            off_m = float(((block_m[0] - seed_x) ** 2 + (block_m[1] - seed_y) ** 2) ** 0.5)
            if off_m > config.RECORD_START_BLOCK_TOLERANCE_M:
                self._block_ready = False
                return None, f"블록이 시드 {self._current_seed} 위치에서 {off_m * 100:.1f} cm 벗어났습니다. 블록 리셋을 다시 하세요"
            self._block_ready = False
            return self._current_seed, None

    def _refuse_or(self, reason: str | None) -> str | None:
        if reason is not None:
            self._event(WARN, f"거절됨: {reason}")
        return reason

    def _start(self) -> str | None:
        with self._lock:
            if self.recorder.active:
                return "이미 녹화 중입니다"
            if self._resetting:
                return "블록 리셋이 진행 중입니다"
            if not self._block_ready or self._current_seed is None:
                return "블록 리셋을 먼저 하세요 (녹화는 리셋 직후에만 시작할 수 있습니다)"
            snapshot = self._worker.snapshot
            if snapshot is None:
                return "시뮬레이션이 아직 준비되지 않았습니다"
            block_m = snapshot["block_m"]
            seed_x, seed_y = block_xy_for_seed(self._current_seed)
            off_m = float(((block_m[0] - seed_x) ** 2 + (block_m[1] - seed_y) ** 2) ** 0.5)
            if off_m > config.RECORD_START_BLOCK_TOLERANCE_M:
                self._block_ready = False
                return f"블록이 시드 {self._current_seed} 위치에서 {off_m * 100:.1f} cm 벗어났습니다. 블록 리셋을 다시 하세요"
            index = self._next_index
            self._next_index += 1
            self._block_ready = False
            self.recorder.begin(
                index,
                self._current_seed,
                (float(block_m[0]), float(block_m[1]), float(block_m[2])),
                config.TARGET_CENTER_M,
            )
        self._event(INFO, f"녹화 시작: 에피소드 {index}, 시드 {self._current_seed}")
        return None

    def _stop(self, success: bool) -> str | None:
        if not self.recorder.active:
            return "녹화 중이 아닙니다"
        finished = self.recorder.stop(success)
        if finished is None:
            self._event(WARN, "샘플이 너무 적어(2개 미만) 저장하지 않고 버렸습니다")
        return None

    def _discard(self) -> str | None:
        if not self.recorder.discard():
            return "녹화 중이 아닙니다"
        self._event(INFO, "녹화를 취소하고 버렸습니다 (저장 안 됨)")
        return None

    def _delete_last(self, episode: int) -> str | None:
        with self._lock:
            if self.recorder.active:
                return "녹화 중에는 삭제할 수 없습니다"
            if self._pending > 0:
                return "저장 중인 에피소드가 있습니다. 잠시 뒤에 다시 하세요"
            if not self._episodes:
                return "삭제할 에피소드가 없습니다"
            latest = self._episodes[-1]["episode_index"]
            if episode != latest:
                return f"가장 최근 에피소드는 {latest}번입니다 (요청한 번호 {episode}). 화면이 오래됐을 수 있습니다"
            self._pending += 1
        self._writer.submit_delete(episode)
        return None

    def _reset(self) -> str | None:
        with self._lock:
            if self.recorder.active:
                return "녹화 중에는 블록 리셋을 할 수 없습니다"
            if self._resetting or self._control.resetting:
                return "블록 리셋이 이미 진행 중입니다"
            seed = self._next_seed
            if seed_kind(seed) != DEMO:
                return "시연 시드(0~99)를 모두 사용했습니다"
            xy_m = block_xy_for_seed(seed)
            self._resetting = True
            self._block_ready = False  # 블록이 곧 옮겨지므로 "리셋 직후" 표시는 새 리셋이 끝날 때까지 꺼 둔다
        reason = self._control.request_reset(seed, xy_m, self._on_reset_done)
        if reason is not None:
            with self._lock:
                self._resetting = False
            return f"리셋을 시작하지 못했습니다: {reason}"
        self._event(INFO, f"블록 리셋 시작: 시드 {seed}")
        return None

    # --- 다른 스레드에서 불리는 콜백 ---

    def _on_reset_done(self, seed: int, ok: bool, reason: str) -> None:
        with self._lock:
            self._resetting = False
            if ok:
                self._current_seed = seed
                self._next_seed = seed + 1
                self._block_ready = True
            else:
                self._block_ready = False
        if ok:
            self._event(INFO, f"블록 리셋 완료: 시드 {seed}. 녹화를 시작할 수 있습니다")
        else:
            self._event(ERROR, f"블록 리셋 실패: {reason}")

    def _on_finished(self, ep: FinishedEpisode) -> None:
        """녹화기가 에피소드를 끝냈다(시뮬레이션 스레드일 수도 있다). 기다리지 않고 쓰기 스레드에 넘기기만 한다."""
        with self._lock:
            self._pending += 1
        if ep.end_reason == "max_length":
            self._event(WARN, f"최대 길이({self._max_s:.0f}초)를 넘어 실패로 자동 종료했습니다 (에피소드 {ep.episode_index}). 저장 중…")
        self._writer.submit_save(ep)

    def _on_saved(self, meta: dict[str, Any]) -> None:
        with self._lock:
            self._pending -= 1
            self._episodes.append(meta)
            self._episodes.sort(key=lambda m: m["episode_index"])
        result = "성공" if meta["success"] else "실패"
        self._event(
            INFO,
            f"에피소드 {meta['episode_index']} 저장됨 ({result}, {meta['n_ticks']}틱, 정지 구간 {meta['still_ratio'] * 100:.0f}%)",
        )

    def _on_save_failed(self, index: int, error: str) -> None:
        with self._lock:
            self._pending -= 1
            self._save_error_count += 1
        self._event(ERROR, f"에피소드 {index} 저장 실패: {error}. 성공/실패 수에 넣지 않았습니다. 디스크 공간과 폴더 권한을 확인하세요")

    def _on_deleted(self, index: int, error: str | None) -> None:
        with self._lock:
            self._pending -= 1
            if error is None:
                self._episodes = [m for m in self._episodes if m["episode_index"] != index]
                self._next_seed = self._compute_next_seed()
                self._next_index = self._compute_next_index()
        if error is None:
            self._event(INFO, f"에피소드 {index} 삭제됨")
        else:
            self._event(ERROR, f"에피소드 {index} 삭제 실패: {error}")

    # --- 내부 ---

    def _event(self, level: str, text: str) -> None:
        with self._lock:
            self._event_id += 1
            self._last_event = {"id": self._event_id, "level": level, "text": text}
        log.info("record: %s", text)

    def _compute_next_seed(self) -> int:
        seeds = [m["seed"] for m in self._episodes if seed_kind(m.get("seed")) == DEMO]
        if self._extra_seeds is not None:
            try:
                seeds += [s for s in self._extra_seeds() if seed_kind(s) == DEMO]
            except Exception:  # noqa: BLE001 - 교정 폴더를 못 읽어도 녹화는 계속된다(대신 시드가 겹칠 수 있다)
                log.exception("could not read the seeds used by corrections")
        return max(seeds) + 1 if seeds else config.DEMO_SEED_RANGE[0]

    def _compute_next_index(self) -> int:
        return max((m["episode_index"] for m in self._episodes), default=-1) + 1

    def state(self) -> dict[str, Any]:
        """상태 메시지의 record 필드(JSON으로 보낼 수 있는 값만)."""
        rec = self.recorder.state()
        phase = self._control.reset_phase
        with self._lock:
            ok_count = sum(1 for m in self._episodes if m.get("success"))
            last = self._episodes[-1] if self._episodes else None
            demo_hi = config.DEMO_SEED_RANGE[1]
            rec.update(
                {
                    "episode_index": self._next_index,
                    "current_seed": self._current_seed,
                    "next_seed": self._next_seed if self._next_seed < demo_hi else None,
                    "seeds_left": max(0, demo_hi - self._next_seed),
                    "block_ready": self._block_ready,
                    "resetting": self._resetting,
                    "reset_phase": phase,
                    "saved_success": ok_count,
                    "saved_fail": len(self._episodes) - ok_count,
                    "saved_total": len(self._episodes),
                    "pending_writes": self._pending,
                    "save_error_count": self._save_error_count,
                    "last_episode": None
                    if last is None
                    else {
                        "index": last["episode_index"],
                        "seed": last["seed"],
                        "success": last["success"],
                        "ticks": last["n_ticks"],
                        "still_ratio": last.get("still_ratio"),
                        "still_longest_s": last.get("still_longest_s"),
                    },
                    "last_event": self._last_event,
                    "tick_ms": self._worker.tick_stats.summary() if hasattr(self._worker, "tick_stats") else None,
                }
            )
        return rec
