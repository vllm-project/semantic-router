import { execFileSync } from 'node:child_process'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import { expect, it } from 'vitest'

const frontend = fileURLToPath(new URL('../', import.meta.url))

function inventoryFiles(config) {
  const output = execFileSync(
    process.execPath,
    [
      path.join(frontend, 'node_modules/@playwright/test/cli.js'),
      'test',
      'e2e/evaluation',
      '--config',
      config,
      '--list',
      '--reporter=json',
    ],
    {
      cwd: frontend,
      encoding: 'utf8',
      timeout: 15000,
      env: {
        ...process.env,
        SR_BENCH_LIVE_PLAN: '',
        SR_BENCH_LIFECYCLE_PLAN: '',
      },
    },
  )
  const report = JSON.parse(output)
  const files = new Set()
  function visit(suite) {
    if (suite.file) files.add(path.basename(suite.file))
    suite.suites?.forEach(visit)
  }
  report.suites.forEach(visit)
  return files
}

it('keeps real deployment acceptance out of fixture CI and in dedicated inventories', () => {
  const fixture = inventoryFiles('playwright.config.ts')
  expect(fixture.has('sr-bench.spec.ts')).toBe(true)
  expect(fixture.has('sr-bench-live.spec.ts')).toBe(false)
  expect(fixture.has('sr-bench-lifecycle.spec.ts')).toBe(false)
  expect(inventoryFiles('playwright.sr-bench-live.config.ts')).toEqual(
    new Set(['sr-bench-live.spec.ts']),
  )
  expect(inventoryFiles('playwright.sr-bench-lifecycle.config.ts')).toEqual(
    new Set(['sr-bench-lifecycle.spec.ts']),
  )
}, 45000)
