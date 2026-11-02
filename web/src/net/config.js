// 서버 연결 설정값. (서버 포트는 server/config.py, 프록시는 vite.config.js)

// 브라우저는 같은 주소(location.host)의 이 경로로 붙고, Vite 프록시가 파이썬 서버로 넘긴다.
export const WS_PATH = '/ws'

// hand 메시지 최대 전송 빈도 (Hz). 손 추적은 약 60FPS로 돌지만 서버에는 30Hz까지만 보낸다.
export const HAND_SEND_MAX_HZ = 30

// 보기 방향(view) 메시지 최대 전송 빈도 (Hz). 마우스를 계속 끌어도 이보다 자주 보내지 않고, 마지막 값은 반드시 보낸다.
export const VIEW_SEND_MAX_HZ = 15

// 키보드 높이(height) 메시지 반복 전송 빈도 (Hz). 키를 누르고 있는 동안 같은 메시지를 이 빈도로 되풀이해 보낸다
// (서버는 마지막 메시지를 받은 지 server/config.py HEIGHT_COMMAND_TIMEOUT_S(0.3초)가 지나면 키가 떼어진 것으로 본다).
export const HEIGHT_REPEAT_HZ = 20

// 왕복 지연(RTT) 측정용 ping 간격 (ms)
export const PING_INTERVAL_MS = 1000

// 연결이 끊긴 뒤 다시 연결을 시도하기까지 기다리는 시간 (ms)
export const RECONNECT_DELAY_MS = 2000

// "초당 개수"를 셀 때 보는 구간 (ms)
export const RATE_WINDOW_MS = 1000

// 화면 숫자를 새로 그리는 간격 (ms)
export const STATS_REFRESH_MS = 500
