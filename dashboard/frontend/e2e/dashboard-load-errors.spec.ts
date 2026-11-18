import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const config = {
  version: 'v0.5',
  providers: {
    defaults: { model: 'model-a' },
    models: [{ name: 'model-a' }],
  },
  recipes: [{ name: 'default', decisions: [] }],
}

// The page refreshes itself; the range control's tooltip says when it last did.
async function refreshNote(page: Page) {
  const tooltip = page.getByRole('tooltip')
  await expect(async () => {
    await page.getByRole('button', { name: 'Open playground' }).focus()
    await page.getByRole('radio', { name: '24h' }).focus()
    await expect(tooltip).toBeAttached({ timeout: 1_000 })
  }).toPass()
  return tooltip
}

// Under 20 seconds ago, and at least the 30 seconds a failed refresh leaves it.
const RECENT = /Updated (just now|1?\d s ago)/
const STALE = /Updated ([3-5]\d s|\d+ min) ago/

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
  await expect(page.getByRole('button', { name: 'Refresh' })).toHaveCount(0)
  const before = await refreshNote(page)
  await expect(before).toContainText('Refreshes every 30 seconds')
  await expect(before).not.toContainText('Updated')

  configStatus = 200
  await page.getByRole('button', { name: 'Retry' }).click()

  await expect(page.getByText(/^Failed to load data:/)).toHaveCount(0)
  await expect(page.getByTestId('home-count-models')).toContainText('1')
  await expect(await refreshNote(page)).toContainText(RECENT)
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
  await expect(page.getByTestId('home-count-models')).toContainText('1')
  await expect(await refreshNote(page)).toContainText(RECENT)

  configStatus = 502
  await page.clock.fastForward(30_000)

  await expect(
    page.getByText('Failed to load data: Router config request failed (HTTP 502)'),
  ).toBeVisible()
  await expect(page.getByTestId('home-count-models')).toContainText('1')
  await expect(await refreshNote(page)).toContainText(STALE)

  configStatus = 200
  await page.clock.fastForward(30_000)

  await expect(page.getByText(/^Failed to load data:/)).toHaveCount(0)
  await expect(await refreshNote(page)).toContainText(RECENT)
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
  await expect(await refreshNote(page)).toContainText(RECENT)

  statusCode = 500
  await page.clock.fastForward(30_000)

  await expect(
    page.getByText('Failed to load data: System status request failed (HTTP 500)'),
  ).toBeVisible()
  await expect(await refreshNote(page)).toContainText(STALE)
})
