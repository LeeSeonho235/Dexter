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
# "free"면 마우스로 돌릴 수 있는 자유 카메라(MjvCamera, VIEW_* 설정)로 그린다.
# scene.xml에 있는 카메라 이름(예: "top")을 쓰면 그 고정 카메라로 그린다(3단계 학습 이미지에 쓸 수 있다).
CAMERA = "free"
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

# --- 자유 카메라 보기 방향 (2-3) ---
# MuJoCo 자유 카메라 규칙: 방위각 azimuth(도, 위에서 봤을 때 카메라가 도는 각도), 고도각 elevation(도, 음수 = 위에서 내려다봄),
# 거리 distance(m). 방위각 90°에서는 로봇 앞쪽(+x)이 화면 오른쪽, 방위각 0°에서는 +x가 화면 안쪽(위)이다.
# 바라보는 점은 작업 상자 중심 근처로 고정한다(마우스로 옮기지 않는다). 값은 PNG를 직접 열어 보고 정했다:
# home 자세와 집게를 블록 위로 내린 자세에서 모두 빨간 블록, 초록 목표, 집게가 보이고 팔 끝이 화면 밖으로 안 나가는 값.
VIEW_LOOKAT_M = (0.23, 0.0, 0.10)
VIEW_DEFAULT = {"azimuth_deg": 110.0, "elevation_deg": -35.0, "distance_m": 0.8}
VIEW_PRESETS = {
    "default": VIEW_DEFAULT,
    # 고정 top 카메라와 같은 방향(화면 위 = 로봇 앞 +x, 화면 오른쪽 = -y)
    "top": {"azimuth_deg": 0.0, "elevation_deg": -89.0, "distance_m": 0.85},
    # 옆에서: 로봇 앞쪽(+x)이 화면 오른쪽. 집게 높이가 보인다. 정확히 90°면 초록 목표가 블록 바로 뒤라 가려져서 105°로 정했다.
    "side": {"azimuth_deg": 105.0, "elevation_deg": -10.0, "distance_m": 0.8},
}
# 브라우저가 보낸 값을 이 범위로 자른다(방위각은 -180~180으로 돌려서 쓴다). 고도각 -3° 이상은 바닥 아래/수평선이라 막는다.
VIEW_ELEVATION_RANGE_DEG = (-89.0, -3.0)
VIEW_DISTANCE_RANGE_M = (0.3, 2.0)

# --- 서버 → 브라우저 상태 메시지 ---
# 집게 높이 막대와 숫자를 부드럽게 보이려고 10Hz로 보낸다(2-1에서는 0.5초).
STATE_SEND_INTERVAL_S = 0.1

# --- 손 → 로봇 (2-2) ---
# 물리 진행과 ctrl 반영 주기(Hz). 화면 렌더링은 FRAME_FPS 그대로(틱 N번에 한 번). 연결이 없으면 렌더링 안 함.
SIM_TICK_HZ = 30
# 제어 스레드(매핑 → 역기구학 → 변화량 제한 → ctrl) 주기(Hz)
CONTROL_HZ = 30

# 조종 시작 조건: 조종 손이 이 프레임 수만큼 연속으로 보여야 목표를 따라가기 시작한다(잘못 인식된 손 무시).
# 손이 사라지면 다시 센다. 0 또는 1이면 바로 시작.
START_CONSECUTIVE_FRAMES = 3

# 손목 위치(MediaPipe 정규화 좌표, 좌우 뒤집지 않은 원본) → 로봇 작업 공간. 구간은 (입력 a, 입력 b) → (출력 a, 출력 b)로 직선 대응.
# 영상이 원본이라 손을 오른쪽으로 옮기면 이미지 x가 줄어든다. top 카메라 화면 오른쪽 = 로봇 -y 이므로
# 이미지 x가 줄면 y도 줄어야 한다. 손을 위로 올리면 이미지 y가 줄고, 화면 위쪽 = 로봇 앞(+x)이므로 x는 늘어야 한다.
# 방향을 바꾸려면 출력 쪽 두 값의 순서를 바꾼다.
HAND_IMG_X_RANGE = (0.2, 0.8)  # 이미지 x(0~1) 중 쓰는 구간. 가운데만 써서 화면 끝까지 손을 뻗지 않아도 되게 한다.
HAND_IMG_Y_RANGE = (0.2, 0.8)  # 이미지 y(0~1)
ROBOT_Y_FOR_IMG_X_M = (-0.18, 0.18)  # 이미지 x 구간 (a, b) → 로봇 y(m)
ROBOT_X_FOR_IMG_Y_M = (0.28, 0.18)  # 이미지 y 구간 (a, b) → 로봇 x(m)
# 높이: hand_scale(손목~중지 뿌리 거리, 화면 높이 = 1) 구간 (가까이, 멀리) → 집게 높이(m). 손을 가까이 하면 집게가 내려간다.
HAND_SCALE_RANGE = (0.07, 0.19)  # (멀리, 가까이) 실측: 멀리 0.07, 가까이 0.19
ROBOT_Z_FOR_SCALE_M = (0.12, 0.015)  # (멀리일 때 높이, 가까이일 때 높이)

# 로봇이 닿는 작업 상자(m). 목표 위치는 이 상자 안으로 잘린다(clamp).
# IK(server/ik.py)로 격자 점마다 풀어 재서 정했다(집게가 수직 아래에서 30° 안으로 기운 자세까지 허용):
#   x < 0.18에서 높이 0.10 이상이면 베이스에 너무 가까워 안 닿고, x > 0.28에서는 |y| > 0.18 모서리가 안 닿는다.
#   z 0.12 아래, x 0.18~0.28, |y| <= 0.18은 모두 닿았다. 블록(0.20, 0)과 목표(0.20, 0.12)가 안에 있다.
WORKSPACE_X_M = (0.18, 0.28)
WORKSPACE_Y_M = (-0.18, 0.18)
WORKSPACE_Z_M = (0.015, 0.12)

# --- 손 → 로봇 매핑 방식 (2-4) ---
# "relative": 상대 이동(클러치). 조종 시작 시점(손이 START_CONSECUTIVE_FRAMES 연속으로 보인 때)을 기준점으로 잡고,
#             이후에는 손이 움직인 양 × 이득만큼 목표를 옮긴다. 손이 화면 밖으로 나가면 로봇은 멈추고, 다시 들어와
#             연속으로 보이면 그 순간이 새 기준점이 된다(마우스를 들어 옮기는 것과 같다). 방향은 현재 보기 방위각 기준이다.
# "absolute": 손 위치 = 로봇 위치(위의 HAND_IMG_*, ROBOT_* 대응). 보기 방위각을 쓰지 않는 월드 고정 방향.
MAPPING_MODE = "relative"

# 웹캠 영상 가로 / 세로. 손 y 이동을 가로 길이 단위로 바꿀 때 쓴다(hand 메시지에는 영상 크기가 없다).
# web/src/hand/config.js의 VIDEO_WIDTH_PX / VIDEO_HEIGHT_PX와 같아야 한다(server/tests/test_mapping_relative.py가 확인).
IMAGE_ASPECT = 640 / 480

# ★ 상대 모드 이득: 손이 움직인 양 → 로봇 목표가 움직이는 양. 느낌을 바꾸고 싶으면 아래 두 값을 바꾼다.
#   수평(앞뒤·좌우): 손이 영상 "너비" 1만큼 움직일 때 로봇이 가는 거리(m). 영상 가로 세로 방향 모두 같은 거리 단위(너비 기준).
#     기본: 영상 너비의 RELATIVE_FULL_TRAVEL_FRACTION(35%)만 움직이면 작업 상자의 가장 긴 변을 지나간다.
#     (0.36 m ÷ 0.35 ≈ 1.03 m/너비: 640 px 영상에서 손이 64 px 움직이면 로봇 약 10.3 cm)
#     올리면 더 민감하다(손을 조금만 움직여도 많이 감). 숫자를 직접 적어도 된다. 예: RELATIVE_GAIN_XY_M_PER_WIDTH = 1.5
#   높이: (Z_FROM_HAND_SCALE = True일 때만 쓰인다. 2-5부터 기본은 키보드 높이라 False)
#     hand_scale(손 크기) 1만큼 변할 때 로봇 높이가 변하는 양(m). 음수 = 손을 가까이(커지게) 하면 내려간다.
#     기본: 절대 모드와 같은 감도(0.105 m ÷ 0.12 = -0.875 m/scale: hand_scale이 0.05 변하면 약 4.4 cm).
#     부호를 바꾸면 방향이 뒤집힌다. 숫자를 직접 적어도 된다. 예: RELATIVE_GAIN_Z_M_PER_SCALE = -1.2
RELATIVE_FULL_TRAVEL_FRACTION = 0.35
_WORKSPACE_LONGEST_SIDE_M = max(
    WORKSPACE_X_M[1] - WORKSPACE_X_M[0],
    WORKSPACE_Y_M[1] - WORKSPACE_Y_M[0],
    WORKSPACE_Z_M[1] - WORKSPACE_Z_M[0],
)
RELATIVE_GAIN_XY_M_PER_WIDTH = _WORKSPACE_LONGEST_SIDE_M / RELATIVE_FULL_TRAVEL_FRACTION
RELATIVE_GAIN_Z_M_PER_SCALE = (ROBOT_Z_FOR_SCALE_M[1] - ROBOT_Z_FOR_SCALE_M[0]) / (
    HAND_SCALE_RANGE[1] - HAND_SCALE_RANGE[0]
)
# 처음 조종을 시작할 때(아직 목표가 없을 때) 로봇 쪽 기준: 작업 상자 중심. home 자세 끝(x 0.03, z 0.35)은 상자 밖이라 쓰지 않는다.
RELATIVE_START_POSITION_M = (
    sum(WORKSPACE_X_M) / 2,
    sum(WORKSPACE_Y_M) / 2,
    sum(WORKSPACE_Z_M) / 2,
)

# --- 높이는 키보드로 (2-5) ---
# 손은 좌우·앞뒤와 집게 열고 닫기만 맡고, 집게 높이는 브라우저 키보드(↓/S 내리기, ↑/W 올리기, Shift = 천천히)로 바꾼다.
# 브라우저가 {"type":"height","dir":-1|0|1,"slow":bool}을 키를 누르고 있는 동안 20Hz로 보내고, 서버는
# "목표 z += dir × 속도 × 흐른 시간"으로 목표 높이를 바꾼다(작업 상자 WORKSPACE_Z_M로 자름). 변화량 제한과 IK를 그대로 거친다.

# 손 크기(hand_scale)로도 높이를 정할지. False(기본)면 손 크기는 높이에 영향이 없다.
# True면 2-4 동작(손 크기로 정한 높이 — 상대 모드는 RELATIVE_GAIN_Z_M_PER_SCALE, 절대 모드는 ROBOT_Z_FOR_SCALE_M)에
# 키보드 높이가 더해진다.
Z_FROM_HAND_SCALE = False

# ★ 키보드 높이 속도(m/s): 높이를 바꾸고 싶으면 아래 두 값을 바꾼다(서버를 다시 띄워야 반영된다).
#   HEIGHT_SPEED_M_S      키를 누르는 동안 올라가고 내려가는 속도. 상자 맨 위(0.12 m)에서 맨 아래(0.015 m)까지
#                         걸리는 시간 = 0.105 m ÷ 속도. 0.08 → 1.31초, 0.05 → 2.1초, 0.04 → 2.6초, 0.03 → 3.5초.
#   HEIGHT_SLOW_SPEED_M_S Shift를 같이 누를 때(미세 조정). 0.02 → 5.25초.
# 블록을 집으려고 내릴 때 너무 빠르면 HEIGHT_SPEED_M_S를 줄인다(예: 0.04).
HEIGHT_SPEED_M_S = 0.08
HEIGHT_SLOW_SPEED_M_S = 0.02
# 마지막 height 메시지를 받은 지 이 시간(s)이 지나면 키가 떼어진 것으로 본다(브라우저가 죽거나 메시지가 끊겨도 로봇이
# 계속 내려가지 않게). 이 사이에 움직이는 거리는 최대 HEIGHT_SPEED_M_S × 이 값(0.08 m/s에서 2.4 cm)이다.
HEIGHT_COMMAND_TIMEOUT_S = 0.3
# 제어 스레드가 한 번 멈칫해도(CPU가 바쁠 때) 높이가 한꺼번에 뛰지 않게, 한 번에 계산하는 경과 시간(s)의 상한.
# 멈칫한 동안의 이동은 줄어들 뿐 튀지 않는다.
HEIGHT_MAX_DT_S = 0.1

# One Euro Filter(떨림 제거). 목표 위치 x, y, z(m)에 각각 적용한다.
#   min_cutoff_hz: 멈춰 있을 때 거르는 세기(낮을수록 많이 거름)
#   beta: 빠르게 움직일 때 덜 거르는 정도(클수록 빨리 따라옴, 단위 1/(m/s))
#   d_cutoff_hz: 속도 추정에 쓰는 거르기
ONE_EURO_MIN_CUTOFF_HZ = 1.5
ONE_EURO_BETA = 20.0
ONE_EURO_D_CUTOFF_HZ = 1.0

# 집게: pinch_ratio가 CLOSE 아래면 닫고, OPEN 위면 열고, 사이는 직전 상태 유지(깜빡임 방지).
PINCH_CLOSE_BELOW = 0.3
PINCH_OPEN_ABOVE = 0.5
# 집게 모터 목표 각도(rad). 값은 server/grasp_check.py와 실측으로 정했다.
GRIPPER_OPEN_RAD = 0.9
GRIPPER_CLOSED_RAD = -0.1

# 역기구학(mink). 위치를 우선하고, 집게가 아래를 향하게 하는 것은 약하게.
IK_SITE = "gripperframe"
IK_POSITION_COST = 1.0
# 집게 방향을 아래로 맞추는 비용. 위치 오차는 m, 방향 오차는 단위 없는 값이라 0.1이면 위치와 비슷하게 경쟁해서
# 풀이가 막혔다(홈 자세에서 목표 오차 73mm에 멈춤). 0.01로 낮추면 같은 점에서 0.1mm까지 수렴했다.
IK_DOWN_COST = 0.01
IK_MAX_ITERATIONS = 30  # 한 번 풀 때 최대 반복
IK_TOLERANCE_M = 0.001  # 목표와 이 거리 안이면 멈춘다(m)
IK_FAIL_ERROR_M = 0.02  # 반복 후에도 이보다 멀면 실패로 보고 ctrl을 바꾸지 않는다(m)
# home 자세는 집게가 위를 향해 있어 "아래 보기"와 정반대라 풀이 나쁜 해에 빠진다(격자 175점 중 40점 실패).
# IK 내부 시작 자세(rad: pan, lift, elbow, wrist_flex, wrist_roll). 집게가 수직 아래를 향하면서 (0.23, 0, 0.08) m에 닿는
# 자세를 최소제곱으로 구한 값이다. 막히면 이 자세에서 한 번 다시 푼다.
# 실제 로봇(ctrl)은 home에서 출발해 변화량 제한만큼씩 움직이므로 이 자세와 상관없이 튀지 않는다.
IK_SEED_ARM_RAD = (-0.005, -0.123, 0.106, 1.588, 0.0)
IK_SOLVER = "daqp"
IK_DT_S = 0.1  # mink 적분 시간. 한 반복에서 오차를 얼마나 줄이는지에만 영향
# 한 틱(1/CONTROL_HZ)에 ctrl이 바뀔 수 있는 최대량. 관절 각도(rad) / 집게(rad)
MAX_JOINT_STEP_RAD = 0.08
MAX_GRIPPER_STEP_RAD = 0.25

# 통계: 최근 몇 개의 값으로 평균, 최대를 낼지 / 목표 흔들림(표준편차)을 볼 구간(s)
STATS_WINDOW_SAMPLES = 300
JITTER_WINDOW_S = 1.0
JITTER_MIN_SAMPLES = 5
