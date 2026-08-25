import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

async function mockPublicVisitor(page: Page) {
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
      body: JSON.stringify({ readonlyMode: false, serverReadonly: false, platform: '' }),
    })
  })
}

test.describe('Public and transition surfaces on short screens', () => {
  test('keeps landing, sign-in, and invitation surfaces fluid at every product breakpoint', async ({
    page,
  }) => {
    const viewports = [
      { width: 320, height: 568 },
      { width: 390, height: 844 },
      { width: 768, height: 1024 },
      { width: 1440, height: 900 },
    ]
    await mockPublicVisitor(page)
    await page.route('**/api/auth/invitations/info?*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          email: 'ada@example.com',
          name: 'Ada Lovelace',
          expiresAt: Math.floor(Date.now() / 1000) + 7 * 24 * 60 * 60,
        }),
      })
    })

    for (const viewport of viewports) {
      await page.setViewportSize(viewport)
      for (const route of ['/', '/login', '/login?invite=1&token=responsive-invite']) {
        await page.goto(route, { waitUntil: 'domcontentloaded' })
        await expect(page.locator('#root')).toBeVisible()
        expect(
          await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
          `${route} overflowed at ${viewport.width}px`,
        ).toBe(true)
      }
      await expect(page.getByRole('heading', { name: 'Choose your password' })).toBeVisible()
      await expect(page.locator('form').getByText('Ada Lovelace', { exact: true })).toBeVisible()
    }
  })

  test('renders the public project shell and three-stage routing story', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 })
    await mockPublicVisitor(page)
    await page.goto('/')

    const header = page.getByTestId('public-header')
    await expect(header).toBeVisible()
    await expect(header.getByRole('link', { name: 'vLLM Semantic Router home' })).toBeVisible()
    await expect(header.getByRole('link', { name: 'Docs', exact: true })).toHaveAttribute(
      'href',
      'https://vllm-sr.ai/docs/intro/',
    )
    await expect(header.getByRole('link', { name: 'GitHub', exact: true })).toHaveAttribute(
      'href',
      'https://github.com/vllm-project/semantic-router',
    )
    await expect(header.getByRole('link', { name: 'Enter Dashboard' })).toHaveAttribute(
      'href',
      '/login',
    )

    await expect(page.getByRole('heading', { name: 'Build your Mixture-of-Models.' })).toBeVisible()
    await expect(
      page.getByRole('heading', { name: 'Match workload with right model on right hardware' }),
    ).toBeVisible()

    const routingHeadings = [
      page.getByRole('heading', { name: 'Understand every request', exact: true }),
      page.getByRole('heading', { name: 'Make preference executable', exact: true }),
      page.getByRole('heading', { name: 'Compose the model path', exact: true }),
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

  test('aligns the public and authenticated header shells', async ({ page, context }) => {
    const viewports = [
      { width: 2048, height: 1152, brandVisible: true },
      { width: 1024, height: 800, brandVisible: false },
      { width: 961, height: 720, brandVisible: false },
      { width: 390, height: 844, brandVisible: false },
    ]

    await page.setViewportSize(viewports[0])
    await mockPublicVisitor(page)
    await page.goto('/')

    const publicHeader = page.getByTestId('public-header')
    const publicBrand = publicHeader.getByRole('link', { name: 'vLLM Semantic Router home' })

    const authenticatedPage = await context.newPage()
    await authenticatedPage.setViewportSize(viewports[0])
    await mockAuthenticatedAppShell(authenticatedPage)
    await authenticatedPage.goto('/dashboard')

    const authenticatedHeader = authenticatedPage.getByTestId('layout-header-content')
    const authenticatedBrand = authenticatedHeader.getByRole('link').first()
    await expect(authenticatedHeader).toBeVisible()

    for (const viewport of viewports) {
      await page.setViewportSize(viewport)
      await authenticatedPage.setViewportSize(viewport)

      const publicHeaderBox = await page.getByTestId('public-header-content').boundingBox()
      const publicLogoBox = await publicBrand.locator('img').boundingBox()
      const publicDashboardBox = await publicHeader
        .getByRole('link', { name: 'Enter Dashboard' })
        .boundingBox()
      const authenticatedHeaderBox = await authenticatedHeader.boundingBox()
      const authenticatedLogoBox = await authenticatedBrand.locator('img').boundingBox()

      expect(publicHeaderBox).not.toBeNull()
      expect(publicLogoBox).not.toBeNull()
      expect(publicDashboardBox).not.toBeNull()
      expect(authenticatedHeaderBox).not.toBeNull()
      expect(authenticatedLogoBox).not.toBeNull()

      expect(publicHeaderBox?.height).toBeCloseTo(authenticatedHeaderBox?.height ?? 0, 0)
      expect(publicHeaderBox?.width).toBeCloseTo(authenticatedHeaderBox?.width ?? 0, 0)
      expect(publicLogoBox?.x).toBeCloseTo(authenticatedLogoBox?.x ?? 0, 0)
      expect(publicLogoBox?.y).toBeCloseTo(authenticatedLogoBox?.y ?? 0, 0)
      expect(publicLogoBox?.width).toBeCloseTo(authenticatedLogoBox?.width ?? 0, 0)
      expect(publicLogoBox?.height).toBeCloseTo(authenticatedLogoBox?.height ?? 0, 0)
      expect((publicDashboardBox?.x ?? 0) + (publicDashboardBox?.width ?? 0)).toBeCloseTo(
        (authenticatedHeaderBox?.x ?? 0) +
          (authenticatedHeaderBox?.width ?? 0) -
          ((authenticatedLogoBox?.x ?? 0) - (authenticatedHeaderBox?.x ?? 0)),
        0,
      )

      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
        viewport.width,
      )
    }

    await authenticatedPage.close()
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

    await expect(page.getByRole('heading', { name: 'Build your Mixture-of-Models.' })).toBeVisible()
    await expect(
      page.getByText('Compose heterogeneous LLMs into personalized model paths.'),
    ).toBeVisible()
    const exploreDocs = page.getByRole('button', { name: 'Explore the Docs' })
    await exploreDocs.scrollIntoViewIfNeeded()
    await expect(exploreDocs).toBeVisible()

    const viewportHeight = await page.evaluate(() => window.innerHeight)
    const buttonBox = await exploreDocs.boundingBox()

    expect(buttonBox).not.toBeNull()
    expect((buttonBox?.y ?? 0) + (buttonBox?.height ?? 0)).toBeLessThanOrEqual(viewportHeight + 1)

    const routeStep = page.getByRole('heading', {
      name: 'Compose the model path',
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
    await mockAuthenticatedAppShell(page)
    let releaseAuthentication: () => void = () => undefined
    const authenticationGate = new Promise<void>((resolve) => {
      releaseAuthentication = resolve
    })
    await page.route('**/api/auth/me', async (route) => {
      await authenticationGate
      await route.fallback()
    })
    await page.goto('/auth/transition?to=/dashboard', { waitUntil: 'domcontentloaded' })

    try {
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
    } finally {
      releaseAuthentication()
    }
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
