"""AUTO(학습된 정책이 혼자 움직이기, 4-1) 설정값 한 곳. server/config.py(사용자가 따로 수정 중인 파일)는 건드리지 않는다.

단위: 시간 s, 각도 rad, 길이 m. 설명은 docs/12-auto.md.
"""

from server import config

# --- 정책 프로세스(train/.venv의 lerobot). 서버(server/.venv)와 환경을 합치지 않고 따로 띄운다 ---
POLICY_CHECKPOINT = config.REPO_ROOT / "checkpoints" / "dexter_act_v1" / "pretrained_model"
POLICY_PYTHON = config.REPO_ROOT / "train" / ".venv" / "bin" / "python"
POLICY_SCRIPT = config.REPO_ROOT / "train" / "policy_server.py"
POLICY_DEVICE = "cpu"

# 공용 서버(46명 로그인)라서 낮은 우선순위(nice)와 스레드 2개로 돌린다. train/evaluate.py와 같은 값.
POLICY_NICE = 19
POLICY_THREADS = 2

# ★ 마지막 요청(AUTO가 끝나며 보내는 PING 포함) 뒤 이 시간(s) 동안 쓰지 않으면 정책 프로세스가 스스로 끝난다.
POLICY_IDLE_TIMEOUT_S = 600.0
# 프로세스 시작부터 READY까지 기다리는 최대 시간(s). 넘으면 프로세스를 죽이고 오류.
POLICY_START_TIMEOUT_S = 120.0
# 요청 하나(렌더 제외 전송 + 추론 + 응답)의 최대 시간(s). 정상은 약 0.1초. 넘으면 AUTO를 오류로 멈춘다.
POLICY_REQUEST_TIMEOUT_S = 3.0
# 화면에 "정책 불러오는 중, 약 N초"로 보이는 값(s). 이 서버에서 잰 프로세스 시작~READY 시간(docs/12-auto.md). 실행 중에는 직전에 잰 실제 값으로 바뀐다.
POLICY_LOAD_ESTIMATE_S = 3  # 실측 2.35~2.86초(3번, nice 19, 스레드 2개, 서버 한가한 때)

# --- 통신 ---
SOCKET_DIR_PARENT = "/tmp"  # 여기에 0700 폴더를 만들고 그 안에 소켓(0600)을 둔다
SOCKET_DIR_PREFIX = "dexter-policy-"
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024  # 메시지 하나의 최대 크기(이미지 240x320x3 = 230 KB)

# --- AUTO 루프 ---
# 에피소드 최대 시간(s). 평가(server/eval_config.py EVAL_MAX_S)와 같은 값을 기본으로 쓴다.
AUTO_MAX_S = 45.0
# AUTO 중 한 틱에 따라잡을 수 있는 최대 시뮬레이션 시간(s). 화면 렌더링 틱(약 55ms) 정도의 지연만 따라잡는다.
# 정책을 기다린 시간(약 140ms)은 따라잡지 않고 시뮬레이션 시간을 멈춘 채로 둔다(server/sim_worker.py).
AUTO_CATCH_UP_S = 0.07
# AUTO 시작 때 팔이 home(리셋이 끝난 자세)에서 이만큼(rad) 넘게 벗어나 있으면 거절한다(리셋 뒤 손으로 움직인 경우).
AUTO_START_ARM_TOLERANCE_RAD = 0.1
