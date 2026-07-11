import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

async function mockPublicVisitor(page: Page) {
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
  await page.route('**/api/auth/bootstrap/can-register', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ canRegister: false }),
    })
  })
  await page.route('**/api/settings', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ readonlyMode: false, platform: '' }),
    })
  })
}

test.describe('Public and transition surfaces on short screens', () => {
  test('renders the public project shell and three-stage routing story', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 })
    await mockPublicVisitor(page)
    await page.goto('/')

    const header = page.getByTestId('public-header')
    await expect(header).toBeVisible()
    await expect(header.getByRole('link', { name: 'vLLM Semantic Router home' })).toBeVisible()
    await expect(header.getByRole('link', { name: 'Docs', exact: true })).toHaveAttribute(
      'href',
      'https://vllm-semantic-router.com/docs/intro/',
    )
    await expect(header.getByRole('link', { name: 'GitHub', exact: true })).toHaveAttribute(
      'href',
      'https://github.com/vllm-project/semantic-router',
    )
    await expect(header.getByRole('link', { name: 'Enter Dashboard' })).toHaveAttribute(
      'href',
      '/login',
    )

    await expect(
      page.getByRole('heading', { name: 'Intelligence, composed for you.' }),
    ).toBeVisible()
    await expect(page.getByRole('heading', { name: 'From request to model path.' })).toBeVisible()

    const routingHeadings = [
      page.getByRole('heading', { name: 'Extract signals', exact: true }),
      page.getByRole('heading', { name: 'Compose decisions', exact: true }),
      page.getByRole('heading', { name: 'Route one—or coordinate many', exact: true }),
    ]
    const routingBoxes = await Promise.all(
      routingHeadings.map(async (heading) => heading.boundingBox()),
    )

    routingBoxes.forEach((box) => expect(box).not.toBeNull())
    expect(routingBoxes[0]?.x ?? 0).toBeLessThan(routingBoxes[1]?.x ?? 0)
    expect(routingBoxes[1]?.x ?? 0).toBeLessThan(routingBoxes[2]?.x ?? 0)

    const footer = page.getByTestId('public-footer')
    await footer.scrollIntoViewIfNeeded()
    await expect(footer).toBeVisible()
    await expect(footer.locator('[data-footer-group]')).toHaveCount(3)
    await expect(footer.getByRole('link', { name: 'Hugging Face' })).toHaveAttribute(
      'href',
      'https://huggingface.co/LLM-Semantic-Router',
    )
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
  })

  test('keeps expressive landing and login surfaces reachable on a short mobile viewport', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 320, height: 568 })
    await mockPublicVisitor(page)
    await page.goto('/')

    const landingMotion = page.getByTestId('landing-motion-background')
    await expect(landingMotion).toBeVisible()
    await expect(landingMotion.locator('canvas')).toBeVisible()

    await expect(
      page.getByRole('heading', { name: 'Intelligence, composed for you.' }),
    ).toBeVisible()
    await expect(
      page.getByText('One model or many, across compute and locations—shaped by your priorities.'),
    ).toBeVisible()
    const exploreDocs = page.getByRole('button', { name: 'Explore the Docs' })
    await exploreDocs.scrollIntoViewIfNeeded()
    await expect(exploreDocs).toBeVisible()

    const viewportHeight = await page.evaluate(() => window.innerHeight)
    const buttonBox = await exploreDocs.boundingBox()

    expect(buttonBox).not.toBeNull()
    expect((buttonBox?.y ?? 0) + (buttonBox?.height ?? 0)).toBeLessThanOrEqual(viewportHeight + 1)

    const routeStep = page.getByRole('heading', {
      name: 'Route one—or coordinate many',
      exact: true,
    })
    await routeStep.scrollIntoViewIfNeeded()
    await expect(routeStep).toBeVisible()

    const publicFooter = page.getByTestId('public-footer')
    await publicFooter.scrollIntoViewIfNeeded()
    const footerGroups = publicFooter.locator('[data-footer-group]')
    await expect(footerGroups).toHaveCount(3)
    const footerGroupBoxes = await Promise.all(
      [0, 1, 2].map(async (index) => footerGroups.nth(index).boundingBox()),
    )
    footerGroupBoxes.forEach((box) => expect(box).not.toBeNull())
    expect(footerGroupBoxes[0]?.y ?? 0).toBeLessThan(footerGroupBoxes[1]?.y ?? 0)
    expect(footerGroupBoxes[1]?.y ?? 0).toBeLessThan(footerGroupBoxes[2]?.y ?? 0)

    await page.getByRole('button', { name: 'Enter Dashboard' }).click()
    await expect(page).toHaveURL(/\/login$/)
    await expect(page.getByRole('heading', { name: 'Sign in', exact: true })).toBeVisible()

    const loginMotion = page.getByTestId('login-motion-background')
    await expect(loginMotion).toBeVisible()
    await expect(loginMotion.locator('canvas')).toBeVisible()
    await page.getByPlaceholder('you@example.com').fill('admin@example.com')

    const layoutWidth = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
    }))
    expect(layoutWidth.scrollWidth).toBeLessThanOrEqual(layoutWidth.innerWidth)
  })

  test('uses the compact transition layout without clipping progress', async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 568 })
    const transitionTime = new Date('2026-07-11T00:00:00Z')
    await page.clock.install({ time: transitionTime })
    await page.clock.pauseAt(transitionTime)
    await mockAuthenticatedAppShell(page)
    await page.goto('/auth/transition?to=/dashboard', { waitUntil: 'domcontentloaded' })

    await expect(page.getByRole('heading', { name: 'Entering control plane' })).toBeVisible()
    await expect(page.getByTestId('auth-transition-scene')).toBeVisible()

    const progress = page.getByRole('progressbar', { name: 'Opening workspace' })
    await expect(progress).toBeVisible()
    const progressBox = await progress.boundingBox()

    expect(progressBox).not.toBeNull()
    expect((progressBox?.y ?? 0) + (progressBox?.height ?? 0)).toBeLessThanOrEqual(568)
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
  })

  test('uses a static decision plane and completes immediately with reduced motion', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 320, height: 568 })
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await mockAuthenticatedAppShell(page)
    await page.route('**/api/auth/me', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 700))
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          user: {
            id: 'user-admin-1',
            email: 'admin@example.com',
            name: 'Admin User',
            role: 'admin',
          },
        }),
      })
    })

    await page.goto('/auth/transition?to=/dashboard', { waitUntil: 'domcontentloaded' })

    const scene = page.getByTestId('auth-transition-scene')
    await expect(scene).toBeVisible()
    await expect(scene).toHaveAttribute('data-motion', 'static')
    await expect(page.getByRole('progressbar', { name: 'Opening workspace' })).toHaveAttribute(
      'aria-valuenow',
      '100',
    )
    await expect(page).toHaveURL(/\/dashboard$/, { timeout: 5000 })
  })
})
