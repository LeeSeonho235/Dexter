"""보조 카메라(2-6, 보기 전용 작은 화면 2개) 설정값 한 곳. server/config.py(사용자가 따로 수정 중인 파일)는 건드리지 않는다.

이 파일의 값은 보조 화면에만 쓴다. 학습·녹화·AUTO·평가는 이 파일을 읽지 않는다(격리: server/tests/test_aux_isolation.py).
server/config.py에 이미 있는 값(장면 경로, 보기 방향 프리셋, 그림자·반사)은 여기서 다시 정하지 않고 import해서 쓴다.

단위: 시간 s·ms, 각도 deg, 길이 m, 화면 px. 설명은 docs/14-aux-view.md.

★ AUX_FPS, AUX_MAX_RENDER_MS, 크기, JPEG 품질은 아직 잠정값이다. 렌더 비용은 프로브(server/aux_view_probe.py)로 쟀지만
  (두 시점 합쳐 렌더+JPEG p50 약 13 ms, p95 약 15 ms), 시뮬레이션 틱에 주는 영향을 bench(server/aux_view_bench.py)로 잰 뒤에 확정한다.
"""

import os

from server import config

# 1단: 꺼져 있으면 스레드도 렌더러(EGL)도 만들지 않는다. 기본 꺼짐. 켜려면 AUX_ENABLED=1 환경변수 또는 이 값을 바꾼다.
AUX_ENABLED = os.environ.get("AUX_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")

# 보조 화면 갱신 주기(Hz). 메인 화면은 config.FRAME_FPS(15)다. 한 번에 두 시점을 그린다. (잠정: 10 Hz면 렌더 13 ms × 10 ≈ 13% 점유)
AUX_FPS = 10.0
# 렌더가 AUX_MAX_RENDER_MS(두 시점 렌더 + JPEG 인코딩, ms)를 넘으면 fps를 반으로 줄인다. 아래 AUX_MIN_FPS 밑으로는 내리지 않는다. (잠정: 측정 p95 15 ms의 약 2.5배)
AUX_MAX_RENDER_MS = 40.0
AUX_MIN_FPS = 1.0

# 시점 하나의 크기 두 단계(작게/보통) (너비, 높이) px. 이미지 한 장은 시점 수만큼 가로로 이어 붙인다(보통: 400x150, 모델 오프스크린 버퍼 640x480 안).
# 보통 크기는 웹의 칸 크기(web/src/aux/config.js AUX_TILE_SIZE_PX)와 같아야 한다(web/src/aux/view.test.js가 비교한다).
AUX_SIZE_LEVELS = {"small": (160, 120), "normal": (200, 150)}
AUX_SIZE_DEFAULT = "normal"
AUX_JPEG_QUALITY = 70

# 로봇의 보이는 메시 삼각형을 이 비율로 줄인 모델로 그린다(시작할 때 한 번 만든다, server/aux_mesh.py). 10%에서 렌더 약 50 ms → 약 10 ms,
# 가장자리 모양 변화 최대 약 2 mm(프로브 측정). 1.0이면 줄이지 않는다(느림).
AUX_MESH_FRACTION = 0.10

# 카메라 두 개(마우스 보기와 연결하지 않는 고정 카메라, 망원). 왼쪽에서 오른쪽 순서가 이미지의 가로 순서다.
# 시야각을 20°로 좁히고(aux 전용 모델에만 적용, 메인과 학습용 StateRenderer는 config의 45° 그대로) 작업 영역(블록과 집게)으로 다가갔다.
#  - 검은 띠: 시야 반각(10°)보다 고도가 더 가파르면 하늘이 화면 밖으로 나간다(옆에서 고도 -12°, 비스듬히 -25°). 이전(시야 45°, 고도 -10°)은 위 27%가 검었다.
#  - 블록 크기: 시연 프레임 기준 블록(빨간 픽셀 상자) 높이가 칸 높이의 중앙값 옆에서 15%, 비스듬히 12.7%(사용자 메인 화면 확대 기준 12.5%)
#    (docs/14-aux-view.md). 비스듬히는 거리 0.8 m·방위각 70°가 더 낫다(보이는 프레임 100%, 중앙값 15%): 바꾸려면 아래 값만 고친다.
# config.py와 겹치는 값은 import해서 쓴다: 옆에서 방위각은 config.VIEW_PRESETS["side"], 바라보는 점의 x, y는 config.VIEW_LOOKAT_M. (test_aux_isolation.py가 확인한다)
AUX_FOVY_DEG = 20.0  # 세로 시야각(deg). 0 < 값 < 45
_SIDE_AZIMUTH_DEG = config.VIEW_PRESETS["side"]["azimuth_deg"]
_LOOKAT_X_M, _LOOKAT_Y_M = config.VIEW_LOOKAT_M[0], config.VIEW_LOOKAT_M[1]
AUX_CAMERAS = (
    # 옆에서: 집게 높이가 보인다
    {"name": "side", "azimuth_deg": _SIDE_AZIMUTH_DEG, "elevation_deg": -12.0, "distance_m": 0.68, "lookat_m": (_LOOKAT_X_M, _LOOKAT_Y_M, 0.05)},
    # 비스듬히
    {"name": "oblique", "azimuth_deg": 45.0, "elevation_deg": -25.0, "distance_m": 1.0, "lookat_m": (_LOOKAT_X_M, _LOOKAT_Y_M, 0.03)},
)

# AUTO(정책 실행) 중에 보조 화면을 멈출지. 기본 False: 교정은 AUTO 중에 봐야 의미가 있다.
# True로 하면 AUTO의 시간 흐름에 영향이 없게 멈춘다. AUTO 중 틱 영향은 bench에서 한 조건으로 잰다.
AUX_PAUSE_DURING_AUTO = False

# 그림자·반사는 메인 화면과 같은 config 값을 쓴다(config.RENDER_SHADOW, config.RENDER_REFLECTION).
