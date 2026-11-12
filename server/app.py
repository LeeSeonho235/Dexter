"""브라우저 ↔ 서버 WebSocket + 손 → 로봇 제어(2-2).

실행 (계속 켜져 있는 명령이므로 직접 띄운다):
    cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.app

- 127.0.0.1:PORT(server/config.py)에서만 연다. 브라우저는 Vite 프록시 '/ws'로 붙는다.
- 브라우저 → 서버: hand(숫자만), ping, view(보기 방향), height(키보드 높이), record(녹화 명령), reset(블록 리셋).
  바이너리(영상·이미지)는 받지 않는다.
- 서버 → 브라우저: JPEG 프레임(바이너리), state(녹화 상태는 state.record), pong, error(JSON 텍스트).
- 연결마다 받기 작업과 보내기 작업을 따로 돌려서, 프레임 전송이 느려도 메시지 받기가 막히지 않는다.
  WebSocket에 send하는 곳은 보내기 작업 하나뿐이다.
- 받은 hand는 제어 스레드(server/control.py)에 "최신 한 개"로 넘기기만 한다. 매핑과 역기구학 계산은 거기서 하므로
  받기, 보내기, 렌더링이 막히지 않는다.
"""

import os

# mujoco를 import하기 전에 정해야 한다. 이 서버는 GPU가 없고 Mesa EGL(CPU)로 렌더링된다.
os.environ.setdefault("MUJOCO_GL", "egl")

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager, suppress  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

from fastapi import FastAPI, WebSocket  # noqa: E402

import numpy as np  # noqa: E402

from server import auto_config, config  # noqa: E402
from server.auto import AutoRunner  # noqa: E402
from pathlib import Path  # noqa: E402

from server.control import ControlWorker  # noqa: E402
from server.correction_store import CorrectionStore  # noqa: E402
from server.episode_store import EpisodeStore  # noqa: E402
from server.owner import ControlOwner  # noqa: E402
from server.policy_ipc import PolicyClient  # noqa: E402
from server.protocol import ProtocolError, hand_summary, parse_client_message  # noqa: E402
from server.recording import RecordingService  # noqa: E402
from server.sim_worker import SimWorker  # noqa: E402
from server.view import CameraView  # noqa: E402

log = logging.getLogger(__name__)


@dataclass(eq=False)
class Connection:
    """브라우저 연결 하나의 상태. asyncio 루프 안에서만 바꾼다."""

    wake: asyncio.Event = field(default_factory=asyncio.Event)
    hand_count: int = 0
    invalid_count: int = 0
    last_hand: dict[str, Any] | None = None
    pending_pongs: list[int] = field(default_factory=list)
    pending_error: str | None = None
    frame: tuple[int, bytes] | None = None  # 아직 안 보낸 최신 프레임 (밀리면 이전 것은 버림)


class Hub:
    """연결 목록과 시뮬레이션 스레드를 잇는다."""

    def __init__(
        self,
        worker: SimWorker,
        loop: asyncio.AbstractEventLoop,
        control: ControlWorker,
        recording: RecordingService | None = None,
        auto: AutoRunner | None = None,
    ) -> None:
        self.worker = worker
        self.loop = loop
        self.control = control
        self.recording = recording
        self.auto = auto
        self.connections: set[Connection] = set()

    def add(self, conn: Connection) -> None:
        self.connections.add(conn)
        self.worker.set_viewer_count(len(self.connections))
        log.info("browser connected (%d open)", len(self.connections))

    def remove(self, conn: Connection) -> None:
        self.connections.discard(conn)
        self.worker.set_viewer_count(len(self.connections))
        log.info("browser disconnected (%d open)", len(self.connections))

    def on_frame_from_thread(self, seq: int, jpeg: bytes) -> None:
        """시뮬레이션 스레드에서 불린다. asyncio 루프로 넘기기만 한다."""
        with suppress(RuntimeError):  # 종료 중 루프가 닫힌 경우
            self.loop.call_soon_threadsafe(self._publish_frame, seq, jpeg)

    def _publish_frame(self, seq: int, jpeg: bytes) -> None:
        for conn in self.connections:
            conn.frame = (seq, jpeg)
            conn.wake.set()


def create_app(
    worker_factory=None, *, record_dir=None, record_max_s=None, record_store=None, policy_client=None, auto_max_s=None,
    correction_dir=None, correction_store=None,
) -> FastAPI:
    """worker_factory(on_frame) -> SimWorker. 테스트에서 바꿔 끼울 수 있게 둔다.
    record_dir: 에피소드 저장 폴더(기본 config.RECORD_DIR). record_max_s: 최대 길이(기본 config.RECORD_MAX_S).
    record_store: 저장소 자체를 바꿔 끼울 때(테스트에서 쓰기 실패를 흉내 낼 때).
    policy_client: AUTO가 쓰는 정책 클라이언트(기본: train/.venv의 정책 프로세스를 AUTO 첫 시작 때 띄우는 PolicyClient). 테스트에서 가짜로 바꾼다.
    auto_max_s: AUTO 최대 시간(기본 server/auto_config.py AUTO_MAX_S).
    correction_dir: 교정 에피소드와 auto_runs.csv를 저장할 폴더(기본 server/correction_config.py CORRECTION_DIR = data/corrections).
        record_dir만 주면(테스트) 그 옆의 corrections 폴더를 쓴다. 그래서 테스트가 실제 data/corrections를 건드리지 않는다.
    correction_store: 저장소 자체를 바꿔 끼울 때(테스트에서 쓰기 실패를 흉내 낼 때)."""

    def default_factory(on_frame):
        return SimWorker(
            config.SCENE_PATH,
            keyframe=config.START_KEYFRAME,
            camera=config.CAMERA,
            width_px=config.RENDER_WIDTH_PX,
            height_px=config.RENDER_HEIGHT_PX,
            frame_fps=config.FRAME_FPS,
            jpeg_quality=config.JPEG_QUALITY,
            max_catch_up_s=config.MAX_CATCH_UP_S,
            shadow=config.RENDER_SHADOW,
            reflection=config.RENDER_REFLECTION,
            on_frame=on_frame,
            tick_hz=config.SIM_TICK_HZ,
        )

    factory = worker_factory or default_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        hub_ref: list[Hub] = []
        worker = factory(lambda seq, jpeg: hub_ref[0].on_frame_from_thread(seq, jpeg))
        control = ControlWorker(worker)
        hub = Hub(worker, loop, control)
        hub_ref.append(hub)
        # Renderer 생성(수백 ms)과 IK 모델 읽기가 루프를 막지 않게 스레드에서 기다린다.
        await asyncio.to_thread(worker.start)
        await asyncio.to_thread(control.start)
        # 녹화 서비스: 저장된 에피소드 목록을 읽는다(폴더가 없으면 빈 목록. 폴더는 첫 저장 때 만든다).
        store = record_store or EpisodeStore(record_dir)
        if correction_store is not None:
            corrections = correction_store
        elif correction_dir is not None:
            corrections = CorrectionStore(correction_dir)
        elif record_dir is not None or record_store is not None:
            corrections = CorrectionStore(Path(store.root).parent / "corrections")
        else:
            corrections = CorrectionStore()
        kwargs = {} if record_max_s is None else {"max_s": record_max_s}
        recording = await asyncio.to_thread(
            lambda: RecordingService(worker, control, store=store, extra_seeds=corrections.used_seeds, **kwargs)
        )
        recording.start()
        hub.recording = recording

        # AUTO(4-1): 정책 프로세스는 여기서 띄우지 않는다(AUTO 첫 시작 때 띄운다).
        owner = getattr(worker, "owner", None) or ControlOwner()

        def arm_check() -> str | None:
            """AUTO는 리셋이 끝난 home 자세에서 시작한다(평가와 같은 시작 자세). 리셋 뒤 손으로 움직였으면 거절."""
            snapshot = worker.snapshot
            if snapshot is None:
                return "시뮬레이션이 아직 준비되지 않았습니다"
            deviation_rad = float(np.max(np.abs(np.asarray(snapshot["joints_rad"]) - control.park_ctrl())))
            if deviation_rad > auto_config.AUTO_START_ARM_TOLERANCE_RAD:
                return f"팔이 home 자세에서 {deviation_rad:.2f} rad 벗어나 있습니다. 블록 리셋을 다시 하세요"
            return None

        auto = AutoRunner(
            worker, owner, policy_client or PolicyClient(), recording, control=control, arm_check=arm_check,
            max_s=auto_config.AUTO_MAX_S if auto_max_s is None else auto_max_s, store=corrections,
        )
        recording.decision_pending = auto.has_pending_decision  # 저장/버리기를 고르기 전에는 블록 리셋도 받지 않는다
        hub.auto = auto
        app.state.hub = hub
        log.info(
            "sim worker started (camera=%s, %d fps, tick %d Hz), control %d Hz, recording dir %s",
            config.CAMERA, config.FRAME_FPS, config.SIM_TICK_HZ, config.CONTROL_HZ, store.root,
        )
        try:
            yield
        finally:
            await asyncio.to_thread(auto.close)  # 실행 중이면 멈추고 정책 프로세스를 끈다
            await asyncio.to_thread(recording.close)
            await asyncio.to_thread(control.stop)
            await asyncio.to_thread(worker.stop)

    app = FastAPI(lifespan=lifespan)

    @app.websocket(config.WS_PATH)
    async def ws_endpoint(websocket: WebSocket) -> None:
        hub: Hub = websocket.app.state.hub
        await websocket.accept()
        conn = Connection()
        hub.add(conn)
        receiver = asyncio.create_task(_receive_loop(websocket, conn, hub.control, hub.worker, hub.recording, hub.auto))
        sender = asyncio.create_task(_send_loop(websocket, conn, hub.worker, hub.control, hub.recording, hub.auto))
        try:
            done, _ = await asyncio.wait({receiver, sender}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    log.info("connection ended: %r", task.exception())
        finally:
            for task in (receiver, sender):
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            hub.remove(conn)
            # 연결이 끊기면 손이 없는 것으로 본다(마지막 목표 유지, 다시 연결되면 시작 조건부터 다시 센다).
            if not hub.connections:
                hub.control.submit_hand(None)

    return app


async def _receive_loop(
    websocket: WebSocket,
    conn: Connection,
    control: ControlWorker,
    worker: SimWorker,
    recording: RecordingService | None = None,
    auto: AutoRunner | None = None,
) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        raw = message.get("text")
        if raw is None:
            raw = message.get("bytes") or b""
        try:
            msg = parse_client_message(raw, config.MAX_MESSAGE_BYTES)
        except ProtocolError as err:
            conn.invalid_count += 1
            conn.pending_error = str(err)
            conn.wake.set()
            continue
        if msg["type"] == "hand":
            conn.hand_count += 1
            conn.last_hand = msg
            control.submit_hand(msg, time.monotonic())
        elif msg["type"] == "ping":
            conn.pending_pongs.append(msg["id"])
            conn.wake.set()
        elif msg["type"] == "height":
            # 키보드 높이: 받은 시각과 함께 제어 스레드에 넘긴다(여러 연결이면 마지막 값이 이긴다). 계산은 제어 스레드가 한다.
            control.submit_height(msg["dir"], msg["slow"], time.monotonic())
        elif msg["type"] == "view":
            # 시뮬레이션이 하나라 보기 방향도 서버에 하나뿐이다. 마지막에 받은 값이 모든 연결에 적용된다.
            worker.set_view(CameraView(msg["azimuth"], msg["elevation"], msg["distance"]))
        elif msg["type"] in ("record", "reset") and recording is not None:
            # 형식은 맞지만 지금은 할 수 없는 요청(녹화 중 리셋 등)은 이유를 error로 돌려보낸다. 잘못된 메시지(invalid_count)는 아니다.
            refusal = recording.handle_reset() if msg["type"] == "reset" else recording.handle_record(msg)
            if refusal is not None:
                conn.pending_error = refusal
                conn.wake.set()
        elif msg["type"] == "auto" and auto is not None:
            # AUTO 명령(4-1). 형식은 맞지만 지금은 할 수 없는 요청은 이유를 error로 돌려보낸다(invalid_count는 올리지 않는다).
            refusal = auto.handle(msg["action"])
            if refusal is not None:
                conn.pending_error = refusal
                conn.wake.set()


async def _send_loop(
    websocket: WebSocket,
    conn: Connection,
    worker: SimWorker,
    control: ControlWorker,
    recording: RecordingService | None = None,
    auto: AutoRunner | None = None,
) -> None:
    next_state_s = time.monotonic()
    while True:
        timeout_s = max(0.0, next_state_s - time.monotonic())
        with suppress(TimeoutError):
            await asyncio.wait_for(conn.wake.wait(), timeout_s)
        conn.wake.clear()

        # pong을 먼저 보내야 RTT에 프레임 전송 시간이 덜 섞인다.
        while conn.pending_pongs:
            await websocket.send_text(json.dumps({"type": "pong", "id": conn.pending_pongs.pop(0)}))
        if conn.pending_error is not None:
            error, conn.pending_error = conn.pending_error, None
            await websocket.send_text(json.dumps({"type": "error", "reason": error}))
        if conn.frame is not None:
            _, jpeg = conn.frame
            conn.frame = None
            await websocket.send_bytes(jpeg)
        if time.monotonic() >= next_state_s:
            await websocket.send_text(json.dumps(_state_message(conn, worker, control, recording, auto)))
            next_state_s = time.monotonic() + config.STATE_SEND_INTERVAL_S


def _state_message(
    conn: Connection,
    worker: SimWorker,
    control: ControlWorker,
    recording: RecordingService | None = None,
    auto: AutoRunner | None = None,
) -> dict[str, Any]:
    message = {
        "type": "state",
        "hand_count": conn.hand_count,
        "invalid_count": conn.invalid_count,
        "last_hand": hand_summary(conn.last_hand) if conn.last_hand else None,
        "sim_time_s": round(worker.sim_time_s, 3),
        "frames_rendered": worker.frames_rendered,
        "control": control.state(),
        # 지금 적용 중인 보기 방향(서버에 하나뿐, 마지막에 보낸 값)
        "view": worker.view.as_dict(),
    }
    if recording is not None:
        message["record"] = recording.state()  # 녹화 상태(3-2): docs/08-recording.md
    if auto is not None:
        message["auto"] = auto.state()  # AUTO 상태(4-1): docs/12-auto.md
    return message


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=config.HOST, port=config.PORT, ws_max_size=config.WS_MAX_SIZE_BYTES)
