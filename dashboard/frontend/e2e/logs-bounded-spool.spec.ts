import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

test('renders centrally redacted bounded-spool logs and keeps them when a follow poll fails', async ({
  page,
}) => {
  await page.clock.install()
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
        components: ['router', 'envoy', 'dashboard'],
        logs: [
          { service: 'router', line: '[INFO] request completed' },
          { service: 'dashboard', line: 'Authorization: [REDACTED]' },
        ],
        count: 2,
      }),
    })
  })

  await page.goto('/observe/logs')

  await expect(page.getByRole('heading', { name: 'Logs', exact: true, level: 1 })).toBeVisible()
  await expect(page.getByText('Log spool', { exact: true })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Envoy', exact: true })).toBeVisible()
  await expect(page.getByText('[INFO] request completed')).toBeVisible()
  await expect(page.getByText('Authorization: [REDACTED]')).toBeVisible()
  // Follow is the only way to refresh; there is no separate refresh control.
  await expect(page.getByRole('button', { name: 'Refresh' })).toHaveCount(0)
  await expect(page.getByRole('switch', { name: 'Follow' })).toBeChecked()

  failRefresh = true
  await page.clock.fastForward(5_100)
  await expect(page.getByText(/Showing the last lines received/)).toBeVisible()
  await expect(page.getByText(/HTTP 503/)).toBeVisible()
  await expect(page.getByText('[INFO] request completed')).toBeVisible()
})

test('explains when the deployment has no log source', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/logs**', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
      body: JSON.stringify({
        deployment_type: 'unsupported',
        service: 'all',
        supported: false,
        components: [],
        logs: [],
        count: 0,
        message:
          'Runtime logs are unavailable: no bounded log spool is mounted, and the Dashboard is not running in Kubernetes.',
      }),
    })
  })

  await page.goto('/observe/logs')

  await expect(page.getByText('No log source')).toBeVisible()
  await expect(page.getByText(/no bounded log spool is mounted/)).toBeVisible()
  await expect(page.getByRole('switch', { name: 'Follow' })).toBeDisabled()
})
