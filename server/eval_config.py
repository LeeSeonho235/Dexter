"""평가(3-4) 설정값 한 곳. server/config.py(사용자가 따로 수정 중인 파일)는 건드리지 않고 평가 값만 여기에 둔다.

단위: 길이 m, 각도 rad, 시간 s. 평가 시드(1000~1049), 목표 위치, 성공 조건은 server/config.py의 값을 그대로 쓴다.
"""

from server import config

# 평가 샘플 주파수는 녹화와 같다(30). 샘플 k는 server/grid.py의 시뮬레이션 시간 격자에서 찍는다.
EVAL_FPS = config.RECORD_FPS

# ★ 에피소드 최대 시간(s). 이 시간 안에 성공(0.5초 유지 포함)하면 성공, 아니면 실패.
# 주의: 이 값은 **시연 20개를 찍은 뒤에** 정했다(06-experiment.md는 "기준은 녹화 전에 고정"이라고 했다).
# 근거(성공 시연 20개, data/raw): 길이 중앙값 19.7초, 최대 36.7초 / 첫 자동 성공까지 중앙값 17.8초, 최대 34.0초.
# 45초 = 가장 느린 성공(34.0초)의 1.3배, 중앙값(19.7초)의 2.3배. "중앙값의 3배"는 59.0초로 녹화 최대(60초)와 거의 같아서 쓰지 않았다.
EVAL_MAX_S = 45.0

# 리셋 뒤 팔과 블록이 가라앉기를 기다리는 시간(s). 시연 16개의 첫 샘플(qpos[0])과 같은 자세가 되는지는 테스트가 확인한다.
EVAL_SETTLE_S = 1.0

# 평가 시작 자세: home(리셋이 끝나는 자세, 집게 열림). 시연 20개 중 4개(raw 4, 5, 7, 15번)는 다른 자세에서 시작했다(한계).
EVAL_START_POSE = "home"

# 평가에서 변화량 제한(server/ik.py의 limit_joint_step)을 쓸지. 기본 끔: 시연의 ctrl은 샘플 사이 변화가 0.08 rad를 자주 넘었다
# (성공 시연 20개에서 샘플 사이 변화 최대: 팔 0.248 rad, 집게 0.715 rad, 에피소드별 최대 8.2%의 샘플이 0.08 초과).
EVAL_LIMIT_STEP_DEFAULT = False
# 켰을 때 샘플당 최대 변화: 시연에서 관찰한 최대값을 올림한 느슨한 값(안전 장치)
LIMIT_STEP_ARM_RAD = 0.25
LIMIT_STEP_GRIPPER_RAD = 0.72

# --- 실패 유형 판정 (docs/07-lerobot-notes.md "평가"). 임계값은 임시이고 실제 실패 사례를 보며 맞춘다 ---
LIFT_Z_MIN_M = 0.03  # 블록 중심 높이가 한 번이라도 이 이상이면 "들렸다"(바닥 위 중심 0.015)
PUSHED_XY_M = 0.02  # 안 들렸는데 블록이 이만큼 이상 밀렸으면 "밀침"으로 표시
NEAR_TARGET_M = 0.06  # 바닥에 놓인 블록이 목표 중심에서 이 안쪽이면 "목표 빗나감", 바깥이면 "옮기다 떨어뜨림"

# 실패 에피소드의 마지막 프레임을 PNG로 저장할 때 기본 장수(옵션 --save-fail-frames)
FAIL_FRAMES_DEFAULT = 4

# 공용 서버: 평가 프로세스는 nice와 스레드 제한을 건다.
NICE_LEVEL = 19
THREADS = 2

FAILURE_KEYS = ("grasp_fail", "timeout", "target_miss", "dropped", "other")
FAILURE_LABELS_KO = {
    "success": "성공",
    "grasp_fail": "집기 실패(안 들림)",
    "timeout": "시간 초과",
    "target_miss": "목표 빗나감",
    "dropped": "옮기다 떨어뜨림",
    "other": "기타",
}
