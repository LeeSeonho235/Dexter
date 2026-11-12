"""4-1: 학습된 ACT 정책을 따로 띄워 서버(server/.venv)가 소켓으로 부르게 한다. train/.venv에서 실행한다.

서버가 AUTO를 처음 시작할 때 띄우고(server/policy_ipc.py default_launcher), 유휴 시간이 지나거나 서버 연결이 끊기면 스스로 끝난다.
직접 띄울 일은 거의 없지만 확인용 명령은 이렇다(소켓 폴더는 0700으로 먼저 만든다):
  mkdir -m 700 /tmp/dexter-policy-try && train/.venv/bin/python train/policy_server.py --socket /tmp/dexter-policy-try/policy.sock \\
      --checkpoint checkpoints/dexter_act_v1/pretrained_model --cleanup-dir /tmp/dexter-policy-try

행동은 train/evaluate.py의 ActPolicy와 같아야 한다: 같은 로딩(device_overrides 포함), 같은 전처리·후처리, 같은 lerobot 함수.
  chunk-mode "predict"(기본): 전처리 → predict_action_chunk()[:, :n_action_steps] → 후처리를 한 번에 해서 청크를 돌려준다(서버가 큐에서 하나씩 쓴다).
  chunk-mode "loop": 같은 관측으로 select_action을 n_action_steps번 불러 모은다(ActPolicy.act와 똑같은 코드 경로. predict가 select_action과
    비트까지 같지 않을 때를 위한 대안, 추론 시간이 늘어난다).
어느 쪽이든 select_action을 하나씩 부른 결과와 같은지 train/tests/test_policy_server.py가 확인한다.

끝나는 조건(모두 소켓과 임시 폴더를 지우고 끝난다): SHUTDOWN 요청 / 요청이 idle-timeout-s 동안 없음 / 서버가 소켓을 닫음(서버가 kill -9로 죽어도 커널이 닫는다) /
부모 프로세스(--parent-pid)가 바뀜(모델을 읽는 동안에도 확인).
"""

import os
import sys

# torch를 import하기 전에 정한다(evaluate.py도 setdefault로 같은 값을 둔다).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from copy import copy  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # server 패키지
sys.path.insert(0, str(Path(__file__).resolve().parent))  # dataset_config, evaluate

from server import policy_ipc as ipc  # noqa: E402


def load_act_policy_class():
    """evaluate.ActPolicy를 상속한 ChunkPolicy. (torch, lerobot을 읽으므로 함수 안에서 만든다)"""
    import dataset_config as dcfg
    import evaluate as ev
    from lerobot.policies import prepare_observation_for_inference

    class ChunkPolicy(ev.ActPolicy):
        """ActPolicy와 같은 로딩. 관측 하나로 행동 청크(n_action_steps개)를 낸다."""

        name = "act-chunk"

        def __init__(self, checkpoint, device="cpu", chunk_mode="predict"):
            super().__init__(checkpoint, device, skip_render=True)
            if self.temporal_ensemble:
                raise SystemExit("시간 앙상블 체크포인트는 지원하지 않습니다(매 샘플 관측이 필요해 AUTO의 청크 방식과 맞지 않음)")
            if chunk_mode not in ("predict", "loop"):
                raise SystemExit(f"unknown chunk mode {chunk_mode!r}")
            self.chunk_mode = chunk_mode

        def predict_chunk(self, state: np.ndarray, image: np.ndarray, reset: bool = False) -> np.ndarray:
            """float32 (n_action_steps, 6): 실제 관절 단위(후처리 끝)."""
            torch = self._torch
            if reset:
                self.policy.reset()
            observation = {"observation.state": state, dcfg.IMAGE_KEY: image}
            if self.chunk_mode == "loop":
                # evaluate.ActPolicy.act와 같은 호출을 n번: 첫 호출이 청크를 예측하고 나머지는 큐에서 꺼내기만 한다.
                from lerobot.common.control_utils import predict_action

                self.policy.reset()
                rows = []
                for _ in range(self.n_action_steps):
                    action = predict_action(observation, self.policy, self.device, self.pre, self.post, use_amp=False, task=dcfg.TASK)
                    rows.append(action.detach().cpu().numpy().astype(np.float32).reshape(-1))
                return np.stack(rows)
            # predict: control_utils.predict_action(use_amp=False)의 순서를 그대로 따르되 select_action 대신 청크를 직접 부른다.
            with torch.inference_mode():
                batch = prepare_observation_for_inference(copy(observation), self.device, dcfg.TASK, None)
                batch = self.pre(batch)
                actions = self.policy.predict_action_chunk(batch)[:, : self.n_action_steps]
                actions = self.post(actions)
            return actions.detach().cpu().numpy().astype(np.float32).reshape(-1, actions.shape[-1])

    return ChunkPolicy


def serve(conn: socket.socket, predict, idle_timeout_s: float, ready_info: dict, clock=time.monotonic) -> str:
    """연결 하나를 처리한다. 끝난 이유(문자열)를 돌려준다: shutdown / idle / client_closed / protocol_error."""
    ipc.send_message(conn, ipc.T_READY, json.dumps(ready_info).encode("utf-8"))
    last_s = clock()
    while True:
        remaining = idle_timeout_s - (clock() - last_s)
        if remaining <= 0:
            return "idle"
        try:
            got = ipc.recv_message(conn, min(1.0, remaining), io_timeout_s=10.0)
        except ipc.ConnectionClosed:
            return "client_closed"
        except (ipc.ProtocolViolation, TimeoutError):
            return "protocol_error"
        if got is None:
            continue
        last_s = clock()
        msg_type, payload = got
        if msg_type == ipc.T_PING:
            ipc.send_message(conn, ipc.T_PONG)
        elif msg_type == ipc.T_SHUTDOWN:
            ipc.send_message(conn, ipc.T_BYE)
            return "shutdown"
        elif msg_type == ipc.T_REQUEST:
            try:
                seq, reset, state, image = ipc.decode_request(payload)
            except ipc.ProtocolViolation as err:
                ipc.send_message(conn, ipc.T_ERROR, str(err).encode("utf-8"))
                continue
            if image is None:
                ipc.send_message(conn, ipc.T_ERROR, b"image is required")
                continue
            t0 = time.perf_counter()
            try:
                actions = predict(state, image, reset)
            except Exception as err:  # noqa: BLE001 - 오류를 서버에 알리고 계속 기다린다
                traceback.print_exc()
                ipc.send_message(conn, ipc.T_ERROR, f"{type(err).__name__}: {err}".encode("utf-8"))
                continue
            infer_ms = (time.perf_counter() - t0) * 1000.0
            ipc.send_message(conn, ipc.T_CHUNK, ipc.encode_chunk(seq, actions, infer_ms))
        else:
            ipc.send_message(conn, ipc.T_ERROR, b"unknown message type")


def _watch_parent(parent_pid: int, cleanup_dir: str | None) -> None:
    """부모(서버)가 죽어 부모 번호가 바뀌면 바로 끝난다. 소켓이 아직 안 열린 모델 읽기 중에도 동작한다.
    os._exit는 finally를 건너뛰므로 소켓 폴더는 여기서 지운다."""
    while True:
        time.sleep(0.5)
        if os.getppid() != parent_pid:
            print("parent process is gone; exiting", flush=True)
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            os._exit(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--idle-timeout-s", type=float, default=600.0)
    parser.add_argument("--nice", type=int, default=19)
    parser.add_argument("--parent-pid", type=int, default=0, help="0이면 부모를 확인하지 않는다")
    parser.add_argument("--cleanup-dir", help="끝날 때 지울 폴더(소켓이 들어 있는 0700 폴더)")
    parser.add_argument("--chunk-mode", default="predict", choices=["predict", "loop"])
    parser.add_argument("--accept-timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)

    try:
        os.nice(args.nice)
    except OSError as err:
        print(f"nice 설정 실패(계속): {err}", file=sys.stderr)
    if args.parent_pid:
        threading.Thread(target=_watch_parent, args=(args.parent_pid, args.cleanup_dir), daemon=True).start()

    reason = "error"
    server = None
    try:
        t0 = time.perf_counter()
        policy_cls = load_act_policy_class()
        policy = policy_cls(args.checkpoint, args.device, args.chunk_mode)
        import lerobot

        ready_info = {
            "n_action_steps": policy.n_action_steps,
            "chunk_size": int(policy.policy.config.chunk_size),
            "device": args.device,
            "chunk_mode": args.chunk_mode,
            "load_s": round(time.perf_counter() - t0, 2),
            "lerobot": getattr(lerobot, "__version__", "?"),
            "pid": os.getpid(),
        }
        # 소켓은 모델을 다 읽은 뒤에 연다: 서버는 소켓 파일이 생기면 곧 READY를 받는다.
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)  # 소켓 파일을 0600으로 만든다
        try:
            server.bind(args.socket)
        finally:
            os.umask(old_umask)
        os.chmod(args.socket, 0o600)
        server.listen(1)
        server.settimeout(args.accept_timeout_s)
        try:
            conn, _ = server.accept()
        except TimeoutError:
            print("no client connected in time; exiting", flush=True)
            reason = "no_client"
            return 1
        with conn:
            reason = serve(conn, policy.predict_chunk, args.idle_timeout_s, ready_info)
        print(f"exit: {reason}", flush=True)
        return 0
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    finally:
        if server is not None:
            server.close()
        try:
            os.unlink(args.socket)
        except OSError:
            pass
        # 시작 실패(error, no_client)면 서버가 로그(policy.log)를 읽고 폴더를 지운다. 서버가 이미 없으면 우리가 지운다.
        parent_alive = not args.parent_pid or os.getppid() == args.parent_pid
        keep_for_parent = reason in ("error", "no_client") and parent_alive
        if args.cleanup_dir and not keep_for_parent:
            shutil.rmtree(args.cleanup_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
