import { defineConfig, devices } from '@playwright/test';

// For tests against already-running stack (no webServer).
export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: { baseURL: 'http://localhost:3001' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
