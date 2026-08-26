import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'
import { withStatusHistory } from './support/status'

interface StatusFixture {
  overall: string
  services: Array<{
    name: string
    status: 'operational' | 'starting' | 'unavailable'
    healthy: boolean
  }>
  [key: string]: unknown
}

async function mockStatus(page: Page, status: StatusFixture) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(withStatusHistory(status)),
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
    await expect(page.getByTestId('status-services-section')).toContainText(
      '90-hour observed history',
    )
    await expect(page.getByTestId('status-services-section')).toContainText('Normal')
    await expect(page.getByTestId('status-services-section')).toContainText('100% uptime')
    const desktopGridColumns = await page
      .getByTestId('status-service-grid')
      .evaluate(
        (element) =>
          getComputedStyle(element).gridTemplateColumns.trim().split(/\s+/).filter(Boolean).length,
      )
    expect(desktopGridColumns).toBe(1)
    const routerHistory = page.getByRole('group', {
      name: /Router, 90-hour observed history/,
    })
    await expect(routerHistory).toBeVisible()
    await expect(routerHistory.getByRole('button')).toHaveCount(90)
    await expect(page.getByText('89 hours ago', { exact: true }).first()).toBeVisible()

    const currentHour = routerHistory.getByRole('button').last()
    await expect(currentHour).toHaveAccessibleName(
      'History hour 90 of 90, 2026-08-26 14:00 UTC: Operational',
    )
    await currentHour.focus()
    await expect(page.getByRole('tooltip')).toContainText('2026-08-26 14:00 UTC: Operational')
    await page.keyboard.press('ArrowLeft')
    await expect(routerHistory.getByRole('button').nth(88)).toBeFocused()
    await expect(page.getByRole('tooltip')).toContainText('2026-08-26 13:00 UTC: Unknown')

    await routerHistory.getByRole('button').first().click()
    await expect(page.getByRole('tooltip')).toContainText('2026-08-22 21:00 UTC: Unknown')
    await expect(page.getByText('must-not-render', { exact: false })).toHaveCount(0)
  })

  test('shows only service availability when Router Management is unavailable', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await mockAuthenticatedAppShell(page)
    await page.route('**/api/router/management/v1/me', async (route) => {
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Request failed (HTTP 503).' }),
      })
    })
    await page.route('**/api/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(
          withStatusHistory({
            overall: 'healthy',
            services: [{ name: 'Dashboard', status: 'operational', healthy: true }],
          }),
        ),
      })
    })

    await page.goto('/dashboard')

    await expect(page.getByTestId('routing-access-status-only')).toBeVisible()
    await expect(page.getByTestId('status-services-section')).toContainText('Routing access')
    await expect(page.getByTestId('status-services-section')).toContainText('Unavailable')
    await expect(
      page.getByRole('group', {
        name: /Routing access, 90-hour observed history: 0 operational, 0 starting, 1 unavailable, 89 unknown/,
      }),
    ).toBeVisible()
    await expect(
      page.getByRole('heading', { name: 'Your model system, at a glance.' }),
    ).toHaveCount(0)
    await expect(page.getByRole('heading', { name: 'Models' })).toHaveCount(0)
    await expect(page.getByText('Routing access unavailable.', { exact: true })).toHaveCount(0)
  })

  test('recovers automatically after a temporary Router Management 503', async ({ page }) => {
    await mockAuthenticatedAppShell(page)
    let identityAttempts = 0
    await page.route('**/api/router/management/v1/me', async (route) => {
      identityAttempts += 1
      if (identityAttempts === 1) {
        await route.fulfill({
          status: 503,
          contentType: 'application/json',
          body: JSON.stringify({ message: 'Request failed (HTTP 503).' }),
        })
        return
      }
      await route.fallback()
    })
    await page.route('**/api/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(
          withStatusHistory({
            overall: 'healthy',
            services: [{ name: 'Dashboard', status: 'operational', healthy: true }],
          }),
        ),
      })
    })

    await page.goto('/dashboard')
    await expect.poll(() => identityAttempts, { timeout: 6_000 }).toBeGreaterThanOrEqual(2)
    await expect(page.getByTestId('routing-access-status-only')).toHaveCount(0)
    await expect(
      page.getByRole('heading', { name: 'Your model system, at a glance.' }),
    ).toBeVisible()
  })

  test('does not present schema or non-503 identity failures as a service outage', async ({
    page,
  }) => {
    await mockAuthenticatedAppShell(page)
    await page.route('**/api/router/management/v1/me', async (route) => {
      await route.fulfill({
        status: 502,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Invalid identity response.' }),
      })
    })
    await page.route('**/api/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(
          withStatusHistory({
            overall: 'healthy',
            services: [{ name: 'Dashboard', status: 'operational', healthy: true }],
          }),
        ),
      })
    })

    await page.goto('/dashboard')
    await expect(page.getByTestId('routing-access-status-only')).toHaveCount(0)
    await expect(
      page.getByRole('heading', { name: 'Your model system, at a glance.' }),
    ).toBeVisible()
    await expect(page.getByRole('alert')).toContainText('HTTP 502')
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

  test('withdraws stale green health when a refresh fails', async ({ page }) => {
    await mockAuthenticatedAppShell(page)
    let statusAttempts = 0
    await page.route('**/api/status', async (route) => {
      statusAttempts += 1
      if (statusAttempts === 1) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(
            withStatusHistory({
              overall: 'healthy',
              services: [{ name: 'Router', status: 'operational', healthy: true }],
            }),
          ),
        })
        return
      }
      await route.fulfill({ status: 503, body: 'unavailable' })
    })

    await page.goto('/status')
    await expect(page.getByTestId('status-overview')).toContainText('All systems operational')
    await page.getByLabel('Refresh system status').click()
    await expect(page.getByTestId('status-overview')).toContainText('Status unavailable')
    await expect(page.getByTestId('status-overview')).not.toContainText('All systems operational')
    await expect(page.getByLabel('Refresh system status')).toContainText('Unavailable')
    await expect(page.getByRole('alert')).toContainText('HTTP 503')
  })

  test('keeps the 90-hour service history inside compact viewports', async ({ page }) => {
    await mockStatus(page, {
      overall: 'healthy',
      services: [{ name: 'Router', status: 'operational', healthy: true }],
    })

    for (const viewport of [
      { width: 320, height: 568 },
      { width: 390, height: 844 },
      { width: 768, height: 1024 },
    ]) {
      await page.setViewportSize(viewport)
      await page.goto('/status')
      await expect(
        page.getByRole('group', { name: /Router, 90-hour observed history/ }),
      ).toBeVisible()
      const mobileGridColumns = await page
        .getByTestId('status-service-grid')
        .evaluate(
          (element) =>
            getComputedStyle(element).gridTemplateColumns.trim().split(/\s+/).filter(Boolean)
              .length,
        )
      expect(mobileGridColumns).toBe(1)
      const width = await page.evaluate(() => ({
        viewport: window.innerWidth,
        body: document.body.scrollWidth,
        document: document.documentElement.scrollWidth,
      }))
      expect(width.body).toBeLessThanOrEqual(width.viewport)
      expect(width.document).toBeLessThanOrEqual(width.viewport)
    }
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
