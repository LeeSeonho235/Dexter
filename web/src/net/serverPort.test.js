// 서버 포트는 server/config.py가 원본이다. vite.config.js의 프록시 포트가 같은지 확인한다.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { SIM_SERVER_PORT } from '../../vite.config.js'

describe('서버 포트', () => {
  it('vite.config.js의 SIM_SERVER_PORT == server/config.py의 PORT', () => {
    const configPy = readFileSync(new URL('../../../server/config.py', import.meta.url), 'utf8')
    const match = configPy.match(/^PORT = (\d+)$/m)
    expect(match).not.toBeNull()
    expect(SIM_SERVER_PORT).toBe(Number(match[1]))
  })
})
