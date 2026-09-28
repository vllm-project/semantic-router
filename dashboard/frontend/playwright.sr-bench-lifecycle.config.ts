import { defineConfig, devices } from '@playwright/test'

// Explicitly authorized synthetic lifecycle acceptance, separate from read-only QA.
// Each run creates a disposable browser context. No retries or service bootstrap.
export default defineConfig({
  testDir: './e2e',
  testMatch: 'evaluation/sr-bench-lifecycle.spec.ts',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 240000,
  expect: { timeout: 30000 },
  outputDir: process.env.SR_BENCH_LIFECYCLE_OUTPUT ?? 'test-results/sr-bench-lifecycle',
  reporter: [['line']],
  use: {
    ...devices['Desktop Chrome'],
    trace: 'off',
    screenshot: 'off',
    video: 'off',
  },
  projects: [{ name: 'chromium' }],
})
