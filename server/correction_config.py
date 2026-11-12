"""교정(HG-DAgger, 4-2) 설정값 한 곳. server/config.py(사용자가 따로 수정 중인 파일)와 server/auto_config.py는 건드리지 않는다.

단위: 시간 s, 각도 rad, 길이 m. 설명은 docs/13-correction.md.
"""

from server import config

# --- 저장 위치: data/raw(시연)와 섞이지 않게 따로 둔다. build_dataset은 data/raw만 읽는다 ---
CORRECTION_DIR = config.REPO_ROOT / "data" / "corrections"
AUTO_RUNS_CSV_NAME = "auto_runs.csv"
CORRECTION_FORMAT_VERSION = 1
CORRECTION_SOURCE = "correction"
CORRECTION_FPS = config.RECORD_FPS  # 3-2 녹화와 같은 30Hz 시간 격자(server/grid.py)

# --- 시간 제한 ---
# AUTO 시간(HUMAN, HUMAN_WAIT 시간은 빼고)의 상한은 server/auto_config.py AUTO_MAX_S(45초)를 그대로 쓴다.
# 한 에피소드의 전체 상한(AUTO + 손을 기다린 시간 + 사람 시간): 넘기면 자동으로 끝내고 저장 전에 사용자에게 묻는다.
TOTAL_MAX_S = 120.0

# --- owner 번호(npz의 owner 배열) ---
OWNER_AUTO = 0
OWNER_HUMAN = 1
OWNER_HUMAN_WAIT = 2

# --- 집게 넘겨받기 ---
# 로봇 집게의 닫힘/열림은 "AUTO가 마지막으로 낸 집게 ctrl"로 판단한다(블록을 쥔 qpos는 약 0.17이라 qpos로는 애매하다).
# 닫힘 목표와 열림 목표의 가운데보다 작으면 닫힘이다.
GRIPPER_SYNC_MID_RAD = (config.GRIPPER_OPEN_RAD + config.GRIPPER_CLOSED_RAD) / 2.0
# True면 사용자의 pinch 판정이 로봇 집게 상태와 처음 같아질 때까지 집게 ctrl을 그대로 둔다. False면 바로 손을 따라간다(비교 측정용).
GRIPPER_SYNC_REQUIRE_MATCH = True

# --- IK 시작 자세 ---
# True: 넘겨받을 때 IK를 "지금 관절 각도"에서 시작한다(ik.set_arm). False: 기본 시작 자세로 되돌린다(ik.reset, 이전 방식. 비교 측정용).
HUMAN_IK_WARM_START = True

# --- 돌려주기 ---
# 정책 첫 행동과 직전 사람 ctrl의 최대 관절 차이(rad)가 이 값을 넘으면 보고에 경고로 적는다(보간은 아직 구현하지 않았다).
HANDBACK_JUMP_WARN_RAD = 0.15

# --- 시드 배분(안내일 뿐 서버가 강제하지 않는다) ---
# 06 문서의 "교정 조건"과 "추가 시연 조건"이 서로 겹치지 않게, 학습에 쓴 시연 시드(0~27)를 빼고 12개씩 번갈아 쓴다.
# 번갈아 쓰는 이유: 연습 효과(뒤로 갈수록 잘하는 것)가 한 조건에만 몰리지 않게. 리셋 버튼은 시드를 순서대로 하나씩만 내므로 덩어리로 나눈다.
# (시작, 끝(미포함), 용도). 평가 시드(1000~1049)는 여기에 없다.
PURPOSE_CORRECTION = "correction"
PURPOSE_EXTRA_DEMO = "extra_demo"
SEED_BLOCKS = (
    (28, 40, PURPOSE_CORRECTION),
    (40, 52, PURPOSE_EXTRA_DEMO),
    (52, 64, PURPOSE_CORRECTION),
    (64, 76, PURPOSE_EXTRA_DEMO),
    (76, 88, PURPOSE_CORRECTION),
    (88, 100, PURPOSE_EXTRA_DEMO),
)
PURPOSE_LABELS = {PURPOSE_CORRECTION: "교정용", PURPOSE_EXTRA_DEMO: "추가 시연용"}


def purpose_for_seed(seed: int | None) -> str | None:
    """시드의 권장 용도("correction" / "extra_demo"). 배분표에 없는 시드(0~27, 평가 시드 등)는 None."""
    if seed is None or isinstance(seed, bool):
        return None
    for start, end, purpose in SEED_BLOCKS:
        if start <= seed < end:
            return purpose
    return None
