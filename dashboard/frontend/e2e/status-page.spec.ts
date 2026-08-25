import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

async function mockStatus(page: Page, status: unknown) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(status),
    })
  })
}

test.describe('System status', () => {
  test('presents the public availability contract without internal model metadata', async ({
    page,
  }) => {
    await mockStatus(page, {
      overall: 'healthy',
      services: [
        { name: 'Router', status: 'operational', healthy: true },
        { name: 'Dashboard', status: 'operational', healthy: true },
      ],
      models: [{ model_path: 'must-not-render' }],
      endpoint: 'https://must-not-render.invalid',
    })

    await page.goto('/status')

    await expect(page.getByRole('heading', { name: 'System status' })).toBeVisible()
    await expect(page.getByTestId('status-overview')).toContainText('All systems operational')
    await expect(page.getByTestId('status-services-section')).toContainText('Router')
    await expect(page.getByTestId('status-services-section')).toContainText('Dashboard')
    await expect(page.getByText('must-not-render', { exact: false })).toHaveCount(0)
  })

  test('keeps degraded service state visible and refreshable', async ({ page }) => {
    await mockStatus(page, {
      overall: 'degraded',
      services: [
        { name: 'Router', status: 'operational', healthy: true },
        { name: 'Dashboard', status: 'unavailable', healthy: false },
      ],
    })

    await page.goto('/status')

    await expect(page.getByTestId('status-overview')).toContainText('Degraded')
    await expect(page.getByTestId('status-overview')).toContainText('1/2')
    await expect(page.getByTestId('status-services-section')).toContainText('Unavailable')
    await expect(page.getByLabel('Refresh system status')).toBeVisible()
  })

  test('stays readable without services at compact-phone, phone, and tablet widths', async ({
    page,
  }) => {
    await mockStatus(page, { overall: 'not_running', services: [] })

    for (const viewport of [
      { width: 320, height: 568 },
      { width: 390, height: 844 },
      { width: 768, height: 1024 },
    ]) {
      await page.setViewportSize(viewport)
      await page.goto('/status')

      await expect(page.getByTestId('status-overview')).toContainText(
        'No running services detected',
      )
      await expect(page.getByTestId('status-services-section')).toContainText(
        'No services reported',
      )
      const width = await page.evaluate(() => ({
        viewport: window.innerWidth,
        body: document.body.scrollWidth,
        document: document.documentElement.scrollWidth,
      }))
      expect(width.body).toBeLessThanOrEqual(width.viewport)
      expect(width.document).toBeLessThanOrEqual(width.viewport)
    }
  })
})
