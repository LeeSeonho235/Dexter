import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // dev 서버: 내 계정에 배정된 포트. 사용 중이면 다른 포트로 넘어가지 않고 멈춘다.
  server: {
    host: '127.0.0.1',
    port: 3007,
    strictPort: true,
  },
})
