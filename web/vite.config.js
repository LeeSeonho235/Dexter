import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 파이썬 서버 포트. 원본은 server/config.py의 PORT (src/net/serverPort.test.js가 같은지 확인한다).
// 이 포트는 127.0.0.1에서만 열리고 VS Code로 전달하지 않는다. 브라우저는 아래 '/ws' 프록시로만 붙는다.
export const SIM_SERVER_PORT = 18007

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // dev 서버: 내 계정에 배정된 포트. 사용 중이면 다른 포트로 넘어가지 않고 멈춘다.
  server: {
    host: '127.0.0.1',
    port: 3007,
    strictPort: true,
    proxy: {
      // 브라우저 ws://localhost:3007/ws → 파이썬 서버. 같은 주소라 index.html의 CSP(connect-src 'self')를 바꾸지 않는다.
      '/ws': {
        target: `ws://127.0.0.1:${SIM_SERVER_PORT}`,
        ws: true,
      },
    },
  },
})
