import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

test.describe('Public and transition surfaces on short screens', () => {
  test('keeps every landing action reachable on a short mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 568 })
    await page.route('**/api/setup/state', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ setupMode: false }),
      })
    })
    await page.route('**/api/auth/me', async (route) => {
      await route.fulfill({ status: 401, body: 'Unauthorized' })
    })
    await page.goto('/')

    const learnMore = page.getByRole('button', { name: 'Learn More' })
    await learnMore.scrollIntoViewIfNeeded()
    await expect(learnMore).toBeVisible()

    const viewportHeight = await page.evaluate(() => window.innerHeight)
    const buttonBox = await learnMore.boundingBox()

    expect(buttonBox).not.toBeNull()
    expect((buttonBox?.y ?? 0) + (buttonBox?.height ?? 0)).toBeLessThanOrEqual(viewportHeight + 1)
  })

  test('uses the compact transition layout without clipping progress', async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 568 })
    await page.clock.install({ time: new Date('2026-07-11T00:00:00Z') })
    await mockAuthenticatedAppShell(page)
    await page.goto('/auth/transition?to=/dashboard', { waitUntil: 'domcontentloaded' })

    await expect(page.getByRole('heading', { name: 'Preparing workspace' })).toBeVisible()
    await expect(page.getByText('Loading sequence')).toBeHidden()

    const progress = page.getByRole('progressbar', { name: 'Opening dashboard' })
    await expect(progress).toBeVisible()
    const progressBox = await progress.boundingBox()

    expect(progressBox).not.toBeNull()
    expect((progressBox?.y ?? 0) + (progressBox?.height ?? 0)).toBeLessThanOrEqual(568)
  })
})
