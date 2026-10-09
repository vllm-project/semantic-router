import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const config = {
  version: 'v0.3',
  providers: {
    defaults: { default_model: 'model-a' },
    models: [{ name: 'model-a' }],
  },
  routing: {
    modelCards: [{ name: 'model-a' }],
    decisions: [],
  },
}

test('dashboard overview reports a failed config request and recovers on retry', async ({
  page,
}) => {
  let configStatus = 500
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', async (route) => {
    if (configStatus !== 200) {
      await route.fulfill({
        status: configStatus,
        contentType: 'text/plain',
        body: 'Failed to read config: permission denied',
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(config),
    })
  })

  await page.goto('/dashboard')

  await expect(
    page.getByText('Failed to load data: Router config request failed (HTTP 500)'),
  ).toBeVisible()
  await expect(page.getByText(/^Updated /)).toHaveCount(0)

  configStatus = 200
  await page.getByRole('button', { name: 'Retry' }).click()

  await expect(page.getByText(/^Failed to load data:/)).toHaveCount(0)
  await expect(page.getByText(/^Updated /)).toBeVisible()
  await expect(page.getByRole('button').filter({ hasText: 'Models' }).first()).toContainText('1')
})

test('dashboard overview keeps the last loaded config when a refresh fails and recovers automatically', async ({
  page,
}) => {
  let configStatus = 200
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', async (route) => {
    if (configStatus !== 200) {
      await route.fulfill({ status: configStatus, contentType: 'text/plain', body: 'Bad Gateway' })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(config),
    })
  })

  await page.clock.install()
  await page.goto('/dashboard')
  const updated = page.getByText(/^Updated /)
  await expect(updated).toBeVisible()
  const updatedBefore = await updated.textContent()

  configStatus = 502
  await page.clock.fastForward(5_000)
  await page.getByRole('button', { name: 'Refresh' }).click()

  await expect(
    page.getByText('Failed to load data: Router config request failed (HTTP 502)'),
  ).toBeVisible()
  await expect(updated).toHaveText(updatedBefore ?? '')
  await expect(page.getByRole('button').filter({ hasText: 'Models' }).first()).toContainText('1')

  configStatus = 200
  await page.clock.fastForward(30_000)

  await expect(page.getByText(/^Failed to load data:/)).toHaveCount(0)
  await expect(updated).not.toHaveText(updatedBefore ?? '')
})

test('dashboard overview keeps the previous update time when only the status request fails', async ({
  page,
}) => {
  let statusCode = 200
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(config),
    })
  })
  await page.route('**/api/status', async (route) => {
    if (statusCode === 200) {
      await route.fallback()
      return
    }
    await route.fulfill({
      status: statusCode,
      contentType: 'text/plain',
      body: 'Internal Server Error',
    })
  })

  await page.clock.install()
  await page.goto('/dashboard')
  const updated = page.getByText(/^Updated /)
  await expect(updated).toBeVisible()
  const updatedBefore = await updated.textContent()

  statusCode = 500
  await page.clock.fastForward(5_000)
  await page.getByRole('button', { name: 'Refresh' }).click()

  await expect(
    page.getByText('Failed to load data: System status request failed (HTTP 500)'),
  ).toBeVisible()
  await expect(updated).toHaveText(updatedBefore ?? '')
})
