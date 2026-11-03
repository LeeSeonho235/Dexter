"""데이터셋 변환(3-3) 설정값 한 곳. 저장소 루트 기준 경로, 데이터셋 이름, 품질 점검 기준.

단위: 시간 s / ms, 각도 rad. 값을 바꾸면 docs/09-dataset.md에도 반영한다.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = REPO_ROOT / "data" / "raw"
FRAMES_DIR = REPO_ROOT / "data" / "frames"
DATASET_NAME = "dexter_pick_place_v1"
DATASET_ROOT = REPO_ROOT / "data" / "lerobot" / DATASET_NAME
REPO_ID = f"local/{DATASET_NAME}"

FPS = 30
ROBOT_TYPE = "so101_sim"
TASK = "pick the red block and place it on the green target"
IMAGE_KEY = "observation.images.main"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
N_JOINTS = 6  # qpos[:6] = 팔 관절 6개(state), ctrl 6개(action). qpos 나머지 7개는 블록(이미지 렌더링에만 쓴다)
QPOS_DIM = 13
CTRL_DIM = 6

# AV1 인코더. lerobot 기본값은 libsvtav1 + yuv420p + crf 30이다. 블록이 13x15px로 작아서 바꿨다(실측은 docs/09-dataset.md):
#   yuv420p는 색을 2x2 픽셀로 줄여서 블록 가장자리 색이 번진다(코덱 없이 색 축소만 해도 블록 픽셀 오차 평균 7.5/255).
#   libsvtav1은 yuv444p를 열지 못해서(avcodec_open2 실패) libaom-av1 + yuv444p를 쓴다. 에피소드 21 실측: 용량 1.74 MB(svtav1 420은 2.45 MB),
#   블록 오차 2.1(420은 9.1). crf는 낮을수록 화질이 좋고 용량이 크다.
VIDEO_CODEC = "libaom-av1"
VIDEO_PIX_FMT = "yuv444p"
VIDEO_CRF = 18
VIDEO_GOP = 2
# 공용 서버라 인코더 스레드를 2개로 제한한다. cpu-used는 클수록 빠르고 화질이 조금 떨어진다(0~8).
VIDEO_EXTRA_OPTIONS = {"cpu-used": "8", "threads": "2"}

# --- 품질 점검 기준(보고용. 제외 여부는 사람이 정한다) ---
# 샘플 간격(시뮬레이션 시간)은 격자 때문에 30ms/35ms가 번갈아 나온다(server/grid.py). 이 밖이면 "틱 간격 이상".
TICK_MS_EXPECTED = (30.0, 35.0)
TICK_MS_TOLERANCE = 1.0
# 길이가 이보다 짧거나 길면 보고 목록에 올린다(ACT chunk_size 100 = 3.3초, 녹화 최대 60초)
LENGTH_S_WARN = (5.0, 55.0)
# 03-2 임시 경고 기준. 실제 분포를 보고 사람이 정한다.
STILL_RATIO_WARN = 0.5
STILL_LONGEST_S_WARN = 3.0

# 프레임 대응 검증: 영상으로 디코딩한 프레임을 PNG와 비교할 때 에피소드당 뽑는 개수, 어긋남을 찾는 이웃 간격(프레임)
VERIFY_FRAMES_PER_EPISODE = 8
VERIFY_NEIGHBOR_OFFSETS = (-3, 3)
# 이웃(k±3) PNG와 "바뀐 픽셀"(세 채널 중 최대 차이가 DELTA 초과)이 MIN_PX개 이상일 때만 그 이웃으로 판정한다. 판정은 바뀐 픽셀 영역에서만
# "디코딩 이미지가 PNG(k)에 더 가까운가"를 본다. 로봇이 화면에서 차지하는 면적이 작아서 전체 평균으로 재면 움직임이 묻힌다
# (전체 평균 2.0 기준을 썼더니 160프레임 중 157개가 빠졌다). 바뀐 픽셀이 거의 없는 정지 장면은 압축 잡음에 묻혀 가릴 수 없어 뺀다.
VERIFY_CHANGED_PIXEL_DELTA = 24
VERIFY_MIN_CHANGED_PX = 20
