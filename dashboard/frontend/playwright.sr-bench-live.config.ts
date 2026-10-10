import { defineConfig, devices } from '@playwright/test'

// Opt-in integration checks against an already-running, isolated Dashboard.
// No development server, retries, traces, saved cookies or model jobs are created.
export default defineConfig({
  testDir: './e2e',
  testMatch: 'evaluation/sr-bench-live.spec.ts',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 180000,
  expect: { timeout: 30000 },
  outputDir: process.env.SR_BENCH_LIVE_OUTPUT ?? 'test-results/sr-bench-live',
  reporter: [['line']],
  use: {
    ...devices['Desktop Chrome'],
    trace: 'off',
    screenshot: 'off',
    video: 'off',
  },
  projects: [{ name: 'chromium' }],
})
