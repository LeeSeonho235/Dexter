"""서버 설정값을 한 곳에 모은다. (브라우저 쪽 설정은 web/src/hand/config.js, web/src/net/config.js)

단위: 시간은 초(s), 크기는 픽셀(px). 손 → 로봇 매핑 값(2-2)도 여기에 추가한다.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- 네트워크 ---
# 127.0.0.1에서만 연다. 브라우저는 Vite 프록시(web/vite.config.js의 '/ws')를 거쳐 붙으므로
# 이 포트는 VS Code Ports로 전달하지 않는다. 값을 바꾸면 web/vite.config.js의 SIM_SERVER_PORT도 같이 바꾼다.
HOST = "127.0.0.1"
PORT = 18007
WS_PATH = "/ws"
# 브라우저 → 서버 메시지 한 개의 최대 크기(byte). hand 메시지는 약 300 byte.
MAX_MESSAGE_BYTES = 4096
# uvicorn이 받는 WebSocket 프레임 최대 크기(byte). 이보다 크면 uvicorn이 연결을 끊는다.
WS_MAX_SIZE_BYTES = 64 * 1024

# --- 시뮬레이션 ---
SCENE_PATH = REPO_ROOT / "sim" / "scene.xml"
START_KEYFRAME = "home"
# 한 번 늦어졌을 때 따라잡기 위해 한 틱에 진행할 수 있는 최대 시뮬레이션 시간(s). 넘으면 늦은 만큼 버린다.
MAX_CATCH_UP_S = 0.25

# --- 시뮬레이션 화면(서버 → 브라우저 JPEG) ---
CAMERA = "top"
RENDER_WIDTH_PX = 640
RENDER_HEIGHT_PX = 480
# 시뮬레이션 스레드의 틱 주기이자 화면 전송 fps.
FRAME_FPS = 15
JPEG_QUALITY = 80
# 렌더링 때만 쓰는 그리기 옵션(scene.xml은 그대로). CPU EGL(llvmpipe) 640x480에서 잰 한 장 시간 / CPU 시간:
#   그림자+반사 켬 151ms / 425ms(약 6fps 상한)   그림자만 끔 93ms / 194ms
#   반사만 끔     120ms / 355ms                   둘 다 끔     54ms / 107ms (15fps 가능)
# 그림자를 끄면 바닥 색이 달라 보이고 그리퍼 높이를 알려주는 그림자가 사라진다. 3단계 학습 이미지와 같은 설정을 써야 한다.
RENDER_SHADOW = False
RENDER_REFLECTION = False

# --- 서버 → 브라우저 상태 메시지 ---
STATE_SEND_INTERVAL_S = 0.5
