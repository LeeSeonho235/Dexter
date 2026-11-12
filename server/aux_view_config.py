"""보조 카메라(2-6, 보기 전용 작은 화면) 설정값 한 곳. server/config.py(사용자가 따로 수정 중인 파일)는 건드리지 않는다.

이 파일의 값은 보조 화면에만 쓴다. 학습·녹화·AUTO·평가는 이 파일을 읽지 않는다(격리: server/tests/test_aux_isolation.py).
server/config.py에 이미 있는 값(장면 경로, 보기 방향 프리셋, 그림자·반사)은 여기서 다시 정하지 않고 import해서 쓴다.

단위: 시간 s·ms, 각도 deg, 길이 m, 화면 px. 설명은 docs/14-aux-view.md.

★ 아래 fps, 크기, JPEG 품질, AUX_MAX_RENDER_MS는 측정 전의 임시값(추측)이다. 렌더 비용 측정(server/aux_view_bench.py) 뒤에 정한다.
"""

import os

from server import config

# 1단: 꺼져 있으면 스레드도 렌더러(EGL)도 만들지 않는다. 기본 꺼짐. 켜려면 AUX_ENABLED=1 환경변수 또는 이 값을 바꾼다.
AUX_ENABLED = os.environ.get("AUX_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")

# 보조 화면 갱신 주기(Hz). 메인 화면은 config.FRAME_FPS(15)다.
AUX_FPS = 5.0
# 렌더가 AUX_MAX_RENDER_MS(렌더 + JPEG 인코딩, ms)를 넘으면 fps를 반으로 줄인다. 아래 AUX_MIN_FPS 밑으로는 내리지 않는다.
AUX_MAX_RENDER_MS = 40.0
AUX_MIN_FPS = 1.0

# 화면 크기 두 단계(작게/보통) (너비, 높이) px. 브라우저가 고른다.
AUX_SIZE_LEVELS = {"small": (128, 96), "normal": (160, 120)}
AUX_SIZE_DEFAULT = "normal"
AUX_JPEG_QUALITY = 70

# 카메라: 옆에서 본 모습(방위각 105°, 고도 -10°, 거리 0.8 m). 값은 config.VIEW_PRESETS["side"]를 그대로 쓴다. 마우스 보기와 연결하지 않는 고정 카메라다.
AUX_CAMERA_VIEW = config.VIEW_PRESETS["side"]
AUX_CAMERA_LOOKAT_M = config.VIEW_LOOKAT_M

# AUTO(정책 실행) 중에는 보조 화면을 멈춘다: AUTO의 시간 흐름에 영향이 없게.
AUX_PAUSE_DURING_AUTO = True

# 그림자·반사는 메인 화면과 같은 config 값을 쓴다(config.RENDER_SHADOW, config.RENDER_REFLECTION).
