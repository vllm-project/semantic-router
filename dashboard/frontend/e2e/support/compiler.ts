import { test as base, type Page } from '@playwright/test'
import { execFile, spawn } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createInterface } from 'node:readline'
import { promisify } from 'node:util'

import { mockAuthenticatedAppShell as mockAppShell } from './auth'

let compilerURL: string | undefined

// One production Go compiler server per worker; configuration and authentication
// remain explicit browser fixtures. No AST, diagnostics, or output is mocked.
export const test = base.extend<object, { compilerServer: string }>({
  compilerServer: [
    async ({ browserName }, use) => {
      const directory = await mkdtemp(path.join(os.tmpdir(), `dashboard-dsl-${browserName}-`))
      const executable = process.env.DASHBOARD_TEST_DSL_SERVER ?? path.join(directory, 'dslserver')
      try {
        if (!process.env.DASHBOARD_TEST_DSL_SERVER) {
          await promisify(execFile)('go', ['build', '-o', executable, './testsupport/dslserver'], {
            cwd: path.resolve(import.meta.dirname, '../../../backend'),
            timeout: 120_000,
          })
        }
        const server = spawn(executable, [], { stdio: ['ignore', 'pipe', 'pipe'] })
        let diagnostics = ''
        server.stderr.on('data', (chunk) => {
          diagnostics = (diagnostics + chunk.toString()).slice(-16_384)
        })
        const lines = createInterface({ input: server.stdout })
        try {
          compilerURL = await new Promise<string>((resolve, reject) => {
            const timeout = setTimeout(() => {
              reject(new Error(`Compiler fixture did not start: ${diagnostics}`))
            }, 10_000)
            const fail = (error: Error) => {
              clearTimeout(timeout)
              reject(error)
            }
            server.once('error', fail)
            server.once('exit', (code) =>
              fail(new Error(`Compiler fixture exited (${code}): ${diagnostics}`)),
            )
            lines.once('line', (line) => {
              clearTimeout(timeout)
              if (!/^http:\/\/127\.0\.0\.1:\d+$/.test(line)) {
                reject(new Error(`Unexpected compiler fixture address: ${line}`))
              } else resolve(line)
            })
          })
          await use(compilerURL)
        } finally {
          compilerURL = undefined
          lines.close()
          if (server.exitCode === null && server.signalCode === null) {
            const closed = new Promise<void>((resolve) => server.once('exit', () => resolve()))
            server.kill('SIGTERM')
            const timer = setTimeout(() => server.kill('SIGKILL'), 5_000)
            await closed
            clearTimeout(timer)
          }
        }
      } finally {
        await rm(directory, { recursive: true, force: true })
      }
    },
    { scope: 'worker', auto: true, timeout: 150_000 },
  ],
})

export async function mockAuthenticatedAppShell(page: Page): Promise<void> {
  await mockAppShell(page)
  if (!compilerURL) throw new Error('Use the compiler fixture test export')
  const origin = compilerURL
  // Register after the shell's deny-by-default API route. Forward just the real
  // compiler transport, retaining the browser's request body and method.
  await page.route(/\/api\/(?:dsl\/[^/]+|decision-model\/tasks)$/, async (route) => {
    const response = await route.fetch({ url: origin + new URL(route.request().url()).pathname })
    await route.fulfill({ response })
  })
}
