import { expect, test } from '@playwright/test'
import { dashboardSettingsResponse, mockAuthenticatedAppShell } from './support/auth'

test('sign-in remains usable while its decorative animation module is stalled', async ({
  page,
}) => {
  await page.route('**/api/auth/me', (route) =>
    route.fulfill({ status: 401, body: 'Unauthorized' }),
  )
  await page.route('**/api/setup/state', (route) =>
    route.fulfill({
      json: { setupMode: false, hasModels: true, hasDecisions: true, canActivate: true },
    }),
  )
  await page.route('**/api/auth/bootstrap/can-register', (route) =>
    route.fulfill({ json: { canRegister: false } }),
  )
  await page.route('**/components/ColorBends.tsx', () => new Promise(() => {}))
  await page.goto('/login')
  await expect(page.getByLabel('Email')).toBeEditable()
  await expect(page.getByLabel('Password', { exact: true })).toBeEditable()
  await expect(page.getByRole('button', { name: 'Continue', exact: true })).toBeVisible()
})

test('stalled settings and ML fallback release the access retry gate and a retry recovers', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page)
  await page.clock.install()
  let requests = 0
  let recover = false
  await page.route('**/api/settings', async (route) => {
    requests += 1
    if (!recover) return new Promise(() => {})
    await route.fulfill({
      json: dashboardSettingsResponse({
        srBenchAvailable: false,
        srBenchUnavailableReason: 'Evaluation is intentionally offline for this test.',
      }),
    })
  })
  await page.route('**/api/ml-pipeline/availability', () => new Promise(() => {}))
  await page.goto('/evaluation')
  await expect.poll(() => requests).toBeGreaterThan(0)
  await page.clock.fastForward(15_001)
  const retry = page.getByRole('button', { name: 'Refresh access', exact: true })
  await expect(retry).toBeVisible()
  recover = true
  await retry.click()
  await expect(page.getByText('Evaluation is intentionally offline for this test.')).toBeVisible()
  await expect(retry).toHaveCount(0)
})
