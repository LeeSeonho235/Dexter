// 서버 연결 설정값. (서버 포트는 server/config.py, 프록시는 vite.config.js)

// 브라우저는 같은 주소(location.host)의 이 경로로 붙고, Vite 프록시가 파이썬 서버로 넘긴다.
export const WS_PATH = '/ws'

// hand 메시지 최대 전송 빈도 (Hz). 손 추적은 약 60FPS로 돌지만 서버에는 30Hz까지만 보낸다.
export const HAND_SEND_MAX_HZ = 30

// 왕복 지연(RTT) 측정용 ping 간격 (ms)
export const PING_INTERVAL_MS = 1000

// 연결이 끊긴 뒤 다시 연결을 시도하기까지 기다리는 시간 (ms)
export const RECONNECT_DELAY_MS = 2000

// "초당 개수"를 셀 때 보는 구간 (ms)
export const RATE_WINDOW_MS = 1000

// 화면 숫자를 새로 그리는 간격 (ms)
export const STATS_REFRESH_MS = 500
