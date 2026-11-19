import { expect, test, type Page } from '@playwright/test'

import { dashboardSettingsResponse } from './support/auth'

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
      body: JSON.stringify(dashboardSettingsResponse()),
    })
  })
}

async function expectNoHorizontalOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
}

const personalInvitation = {
  email: 'ada@example.com',
  name: 'Ada Router',
  role: 'write',
  kind: 'personal',
  maxUses: 1,
  remainingUses: 1,
  expiresAt: Math.floor(Date.now() / 1000) + 3 * 24 * 3600,
}

test.describe('Sign-in and invitation surfaces on short screens', () => {
  test('keeps the sign-in card reachable on a short mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 568 })
    await mockPublicVisitor(page)
    await page.goto('/login')

    await expect(
      page.getByRole('heading', { name: 'Sign in to Semantic Router', exact: true }),
    ).toBeVisible()
    const motion = page.getByTestId('login-motion-background')
    await expect(motion).toBeAttached()
    await expect(motion.locator('canvas')).toBeAttached()

    await page.getByPlaceholder('you@example.com').fill('admin@example.com')
    const passwordInput = page.getByLabel('Password', { exact: true })
    await passwordInput.fill('secret-password')
    await page.getByRole('button', { name: 'Show password' }).click()
    await expect(passwordInput).toHaveAttribute('type', 'text')
    await page.getByRole('button', { name: 'Hide password' }).click()
    await expect(passwordInput).toHaveAttribute('type', 'password')

    const submit = page.getByRole('button', { name: 'Continue', exact: true })
    await submit.scrollIntoViewIfNeeded()
    await expect(submit).toBeVisible()
    await expectNoHorizontalOverflow(page)
  })

  test('offers docs and theme controls without leaving the sign-in page', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 })
    await mockPublicVisitor(page)
    await page.goto('/login')

    await expect(page.getByRole('link', { name: 'Docs', exact: true })).toHaveAttribute(
      'href',
      'https://vllm-sr.ai/docs/intro/',
    )
    const html = page.locator('html')
    const wasDark = await html.evaluate((element) => element.classList.contains('dark'))
    await page.getByRole('button', { name: 'Toggle theme' }).click()
    await expect
      .poll(() => html.evaluate((element) => element.classList.contains('dark')))
      .toBe(!wasDark)
    await expect(page).toHaveURL(/\/login$/)
  })

  test('fits a personal invitation on a short mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 568 })
    await mockPublicVisitor(page)
    await page.route('**/api/auth/invitations/invite-token', (route) =>
      route.fulfill({ json: personalInvitation }),
    )
    await page.goto('/invite/invite-token')

    await expect(page.getByRole('heading', { name: 'You’re in, Ada.' })).toBeVisible()
    await expect(page.getByText('Builder', { exact: true })).toBeVisible()
    await expect(page.getByText('ada@example.com', { exact: true })).toBeVisible()
    const join = page.getByRole('button', { name: 'Join the workspace' })
    await join.scrollIntoViewIfNeeded()
    await expect(join).toBeVisible()
    await expectNoHorizontalOverflow(page)
  })

  test('explains an unavailable invitation and returns to sign in', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await mockPublicVisitor(page)
    await page.route('**/api/auth/invitations/used-token', (route) =>
      route.fulfill({ status: 410, body: 'This invitation has already been used.' }),
    )
    await page.goto('/invite/used-token')

    await expect(page.getByRole('heading', { name: 'This invitation can’t be used' })).toBeVisible()
    await expect(page.getByText('This invitation has already been used.')).toBeVisible()
    await page.getByRole('link', { name: 'Back to sign in' }).click()
    await expect(page).toHaveURL(/\/login$/)
  })
})
