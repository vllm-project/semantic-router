import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

test('renders centrally redacted bounded-spool logs', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  let failRefresh = false
  await page.route('**/api/logs**', async (route) => {
    if (failRefresh) {
      await route.fulfill({ status: 503, body: 'temporarily unavailable' })
      return
    }
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
      body: JSON.stringify({
        deployment_type: 'local-spool',
        service: 'all',
        supported: true,
        logs: [
          { service: 'router', line: '[router] [INFO] request completed' },
          { service: 'dashboard', line: 'Authorization: [REDACTED]' },
        ],
        count: 2,
      }),
    })
  })

  await page.goto('/logs')

  await expect(page.getByRole('heading', { name: 'System Logs' })).toBeVisible()
  await expect(page.getByText('Available', { exact: true })).toBeVisible()
  await expect(page.getByText('[router] [INFO] request completed')).toBeVisible()
  await expect(page.getByText('Authorization: [REDACTED]')).toBeVisible()

  failRefresh = true
  await page.getByRole('button', { name: 'Refresh' }).click()
  await expect(page.getByText(/Failed to fetch logs/)).toBeVisible()
  await expect(page.getByText('[router] [INFO] request completed')).toBeVisible()
})

test('explains when the deployment has no bounded log spool', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/logs**', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
      body: JSON.stringify({
        deployment_type: 'unsupported',
        service: 'all',
        supported: false,
        logs: [],
        count: 0,
        message:
          'Runtime logs are unsupported for this deployment because the bounded shared log spool is not mounted.',
      }),
    })
  })

  await page.goto('/logs')

  await expect(page.getByText('Unavailable', { exact: true })).toBeVisible()
  await expect(page.getByText('Runtime logs unavailable')).toBeVisible()
  await expect(page.getByText(/bounded shared log spool is not mounted/)).toBeVisible()
})
