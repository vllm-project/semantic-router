import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const productViewports = [
  { name: 'compact phone', width: 320, height: 568 },
  { name: 'phone', width: 390, height: 844 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'desktop', width: 1440, height: 900 },
] as const

async function expectNoPublicOverflow(page: Page, surface: string) {
  const dimensions = await page.evaluate(() => ({
    body: document.body.scrollWidth,
    document: document.documentElement.scrollWidth,
    viewport: window.innerWidth,
  }))
  expect(dimensions.body, `${surface} body overflow`).toBeLessThanOrEqual(dimensions.viewport)
  expect(dimensions.document, `${surface} document overflow`).toBeLessThanOrEqual(
    dimensions.viewport,
  )
}

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

    for (const viewport of productViewports) {
      await page.setViewportSize(viewport)
      for (const route of ['/', '/login', '/login?invite=1&token=responsive-invite']) {
        await page.goto(route, { waitUntil: 'domcontentloaded' })
        await expect(page.locator('#root')).toBeVisible()
        if (route === '/') {
          await expect(
            page.getByRole('heading', { name: 'Build your Mixture-of-Models.' }),
          ).toBeVisible()
        } else if (route.includes('invite=')) {
          await expect(page.getByRole('heading', { name: 'Choose your password' })).toBeVisible()
        } else {
          await expect(page.getByRole('heading', { name: 'Sign in', exact: true })).toBeVisible()
        }
        await expectNoPublicOverflow(page, `${route} at ${viewport.name}`)
      }
      await expect(page.getByRole('heading', { name: 'Choose your password' })).toBeVisible()
      await expect(page.locator('form').getByText('Ada Lovelace', { exact: true })).toBeVisible()
    }
  })

  test('keeps the authenticated handoff composed at every product breakpoint', async ({ page }) => {
    await mockAuthenticatedAppShell(page)
    let releaseAuthentication: () => void = () => undefined
    const authenticationGate = new Promise<void>((resolve) => {
      releaseAuthentication = resolve
    })
    await page.route('**/api/auth/me', async (route) => {
      await authenticationGate
      await route.fallback()
    })

    try {
      for (const viewport of productViewports) {
        await page.setViewportSize(viewport)
        await page.goto('/auth/transition?to=/dashboard', { waitUntil: 'domcontentloaded' })
        const loading = page.getByRole('status', { name: 'Opening dashboard' })
        await expect(loading).toBeVisible()
        await expect(loading.locator('img')).toHaveAttribute(
          'src',
          '/vllm-sr-logo.white.png',
        )
        await expectNoPublicOverflow(page, `authenticated handoff at ${viewport.name}`)
      }
    } finally {
      releaseAuthentication()
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

  test('keeps the unified loading state inside a compact viewport', async ({ page }) => {
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
      const loading = page.getByRole('status', { name: 'Opening dashboard' })
      await expect(loading).toBeVisible()
      const loadingBox = await loading.boundingBox()

      expect(loadingBox).not.toBeNull()
      expect((loadingBox?.y ?? 0) + (loadingBox?.height ?? 0)).toBeLessThanOrEqual(568)
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
      ).toBe(true)
    } finally {
      releaseAuthentication()
    }
  })

  test('opens the dashboard when authentication becomes available', async ({ page }) => {
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

    await expect(page.getByRole('status', { name: 'Opening dashboard' })).toBeVisible()
    await expect(page).toHaveURL(/\/dashboard$/, { timeout: 5000 })
  })
})
