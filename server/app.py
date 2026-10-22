"""브라우저 ↔ 서버 WebSocket (2-1: 연결만, 손 → 로봇 매핑은 2-2).

실행 (계속 켜져 있는 명령이므로 직접 띄운다):
    cd ~/handover && MUJOCO_GL=egl server/.venv/bin/python -m server.app

- 127.0.0.1:PORT(server/config.py)에서만 연다. 브라우저는 Vite 프록시 '/ws'로 붙는다.
- 브라우저 → 서버: hand(숫자만), ping. 바이너리(영상·이미지)는 받지 않는다.
- 서버 → 브라우저: JPEG 프레임(바이너리), state, pong, error(JSON 텍스트).
- 연결마다 받기 작업과 보내기 작업을 따로 돌려서, 프레임 전송이 느려도 메시지 받기가 막히지 않는다.
  WebSocket에 send하는 곳은 보내기 작업 하나뿐이다.
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

from server import config  # noqa: E402
from server.protocol import ProtocolError, hand_summary, parse_client_message  # noqa: E402
from server.sim_worker import SimWorker  # noqa: E402

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

    def __init__(self, worker: SimWorker, loop: asyncio.AbstractEventLoop) -> None:
        self.worker = worker
        self.loop = loop
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


def create_app(worker_factory=None) -> FastAPI:
    """worker_factory(on_frame) -> SimWorker. 테스트에서 바꿔 끼울 수 있게 둔다."""

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
        )

    factory = worker_factory or default_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        hub_ref: list[Hub] = []
        worker = factory(lambda seq, jpeg: hub_ref[0].on_frame_from_thread(seq, jpeg))
        hub = Hub(worker, loop)
        hub_ref.append(hub)
        # Renderer 생성(수백 ms)이 루프를 막지 않게 스레드에서 기다린다.
        await asyncio.to_thread(worker.start)
        app.state.hub = hub
        log.info("sim worker started (camera=%s, %d fps)", config.CAMERA, config.FRAME_FPS)
        try:
            yield
        finally:
            await asyncio.to_thread(worker.stop)

    app = FastAPI(lifespan=lifespan)

    @app.websocket(config.WS_PATH)
    async def ws_endpoint(websocket: WebSocket) -> None:
        hub: Hub = websocket.app.state.hub
        await websocket.accept()
        conn = Connection()
        hub.add(conn)
        receiver = asyncio.create_task(_receive_loop(websocket, conn))
        sender = asyncio.create_task(_send_loop(websocket, conn, hub.worker))
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

    return app


async def _receive_loop(websocket: WebSocket, conn: Connection) -> None:
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
        elif msg["type"] == "ping":
            conn.pending_pongs.append(msg["id"])
            conn.wake.set()


async def _send_loop(websocket: WebSocket, conn: Connection, worker: SimWorker) -> None:
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
            await websocket.send_text(json.dumps(_state_message(conn, worker)))
            next_state_s = time.monotonic() + config.STATE_SEND_INTERVAL_S


def _state_message(conn: Connection, worker: SimWorker) -> dict[str, Any]:
    return {
        "type": "state",
        "hand_count": conn.hand_count,
        "invalid_count": conn.invalid_count,
        "last_hand": hand_summary(conn.last_hand) if conn.last_hand else None,
        "sim_time_s": round(worker.sim_time_s, 3),
        "frames_rendered": worker.frames_rendered,
    }


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=config.HOST, port=config.PORT, ws_max_size=config.WS_MAX_SIZE_BYTES)
