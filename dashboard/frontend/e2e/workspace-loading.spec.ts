import { expect, test } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

test('a stalled setup check releases the workspace retry action and can recover', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page)
  await page.clock.install()
  let checks = 0
  let recovered = false
  await page.route('**/api/setup/state', async (route) => {
    checks += 1
    if (!recovered) return // Keep initial checks open until their deadlines.
    await route.fulfill({ json: { setupMode: false, canActivate: false } })
  })
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: {} }))
  await page.route('**/api/status', (route) =>
    route.fulfill({
      json: {
        overall: 'healthy',
        serving_mode: 'router',
        services: [],
      },
    }),
  )
  await page.goto('/dashboard')
  await expect.poll(() => checks).toBeGreaterThan(0)
  await page.clock.runFor(15_001)
  await expect(page.getByRole('heading', { name: 'Unable to load setup state' })).toBeVisible()
  const initialChecks = checks
  recovered = true
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await expect(page.getByTestId('dashboard-serving-mode')).toContainText('Router mode')
  expect(checks).toBe(initialChecks + 1)
})

test('a stalled session check stays on the requested route and exposes a working retry', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page)
  await page.clock.install()
  let checks = 0
  let recovered = false
  await page.route('**/api/auth/me', async (route) => {
    checks += 1
    if (!recovered) return
    await route.fulfill({
      json: {
        user: {
          id: 'user-admin',
          role: 'admin',
          name: 'Admin',
          email: 'admin@example.test',
        },
      },
    })
  })
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: {} }))
  await page.route('**/api/status', (route) =>
    route.fulfill({
      json: {
        overall: 'healthy',
        serving_mode: 'router',
        services: [],
      },
    }),
  )
  await page.goto('/dashboard')
  await expect.poll(() => checks).toBeGreaterThan(0)
  await page.clock.runFor(15_001)
  await expect(page.getByRole('heading', { name: 'Unable to verify your session' })).toBeVisible()
  await expect(page).toHaveURL(/\/dashboard$/)
  recovered = true
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await expect(page.getByTestId('dashboard-serving-mode')).toContainText('Router mode')
})
