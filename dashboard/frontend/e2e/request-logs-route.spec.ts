import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const managementMediaType = 'application/vnd.vllm-semantic-router.management.v1+json'

test('opens the shared Request Logs workspace instead of a host log spool', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  let legacySpoolRequested = false
  await page.route('**/api/logs**', async (route) => {
    legacySpoolRequested = true
    await route.abort()
  })
  await page.route('**/api/router/management/v1/request-logs**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: managementMediaType,
      body: JSON.stringify({ data: [], page: { hasMore: false, pageSize: 20 } }),
    })
  })

  await page.goto('/logs')

  await expect(page.getByRole('heading', { name: 'Request Logs' })).toBeVisible()
  await expect(page.getByText('No requests in this scope')).toBeVisible()
  await expect(page.getByRole('navigation', { name: 'Access control' })).toBeVisible()
  expect(legacySpoolRequested).toBe(false)
})
