import { expect, test } from '@playwright/test'
import { mockAuthenticatedAppShell, mockLoggedOutAuthShell } from './support/auth'

test.describe('Dashboard auth flow', () => {
  test('keeps the mobile sign-in form reachable below the story panel', async ({ page }) => {
    let loginRequestCount = 0
    await page.setViewportSize({ width: 390, height: 844 })
    await mockLoggedOutAuthShell(page)
    await page.route('**/api/auth/login', async (route) => {
      loginRequestCount += 1
      await route.fulfill({ status: 401, body: 'Unauthorized' })
    })

    await page.goto('/login')
    const continueButton = page.getByRole('button', { name: 'Continue' })
    const emailInput = page.getByLabel('Email')
    const passwordInput = page.getByLabel('Password', { exact: true })
    await continueButton.scrollIntoViewIfNeeded()
    await expect(emailInput).toBeVisible()
    await expect(emailInput).toHaveAttribute('name', 'email')
    await expect(emailInput).toHaveAttribute('autocomplete', 'username')
    await expect(passwordInput).toBeVisible()
    await expect(passwordInput).toHaveAttribute('id', 'current-password')
    await expect(passwordInput).toHaveAttribute('name', 'password')
    await expect(passwordInput).toHaveAttribute('autocomplete', 'current-password')
    const showPassword = page.getByRole('button', {
      name: 'Show sign-in password',
    })
    await expect(showPassword).toHaveAttribute('type', 'button')
    await expect(showPassword).toHaveAttribute('aria-controls', 'current-password')
    await expect(showPassword).toHaveAttribute('aria-pressed', 'false')

    await emailInput.fill('admin@example.com')
    await passwordInput.fill('unique-sign-in-password')
    await showPassword.click()
    await expect(passwordInput).toHaveAttribute('type', 'text')
    await expect(passwordInput).toHaveValue('unique-sign-in-password')

    const hidePassword = page.getByRole('button', {
      name: 'Hide sign-in password',
    })
    await expect(hidePassword).toHaveAttribute('aria-pressed', 'true')
    await hidePassword.click()
    await expect(passwordInput).toHaveAttribute('type', 'password')
    await expect(passwordInput).toHaveValue('unique-sign-in-password')
    await expect.poll(() => loginRequestCount).toBe(0)
    await expect(continueButton).toBeVisible()

    await page.getByRole('button', { name: 'Show sign-in password' }).click()
    await continueButton.click()
    await expect.poll(() => loginRequestCount).toBe(1)
    await expect(passwordInput).toHaveAttribute('type', 'password')
    await expect(passwordInput).toHaveValue('unique-sign-in-password')
  })

  test('does not carry bootstrap password state into sign-in after a registration conflict', async ({
    page,
  }) => {
    await mockLoggedOutAuthShell(page, {
      canRegister: true,
      setupState: { setupMode: true },
    })
    await page.route('**/api/auth/bootstrap/register', async (route) => {
      await route.fulfill({
        status: 409,
        headers: { 'Content-Type': 'text/plain' },
        body: 'already registered',
      })
    })
    await page.goto('/login')
    await page.getByLabel('What should we call you?').fill('Ada Router')
    await page.getByRole('button', { name: 'Next' }).click()
    await page.getByLabel('Admin email').fill('ada@example.com')
    await page.getByRole('button', { name: 'Next' }).click()

    const bootstrapPassword = page.locator('#new-password')
    await bootstrapPassword.fill('bootstrap-conflict-password')
    await page.getByRole('button', { name: 'Show first administrator password' }).click()
    await expect(bootstrapPassword).toHaveAttribute('type', 'text')
    await page.getByRole('button', { name: 'Create admin and continue' }).click()

    await expect(page.getByRole('heading', { name: 'Sign in', exact: true })).toBeVisible()
    await expect(page.locator('#new-password')).toHaveCount(0)
    const loginPassword = page.locator('#current-password')
    await expect(loginPassword).toHaveAttribute('type', 'password')
    await expect(loginPassword).toHaveValue('')
    await expect(page.getByRole('button', { name: 'Show sign-in password' })).toHaveAttribute(
      'aria-pressed',
      'false',
    )
    await expect(page.getByLabel('Email')).toHaveValue('ada@example.com')
  })

  test('changes the signed-in password through password-manager-compatible fields', async ({
    page,
  }) => {
    const { user } = await mockAuthenticatedAppShell(page, {
      token: 'password-change-old-token',
    })
    const passwordPayloads: Array<Record<string, unknown>> = []

    await page.route('**/api/status', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          overall: 'healthy',
          deployment_type: 'local',
          services: [],
        }),
      })
    })
    await page.route('**/api/router/config/all', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          signals: {},
          decisions: [],
          providers: { models: [] },
          plugins: {},
        }),
      })
    })
    await page.route('**/api/auth/password', async (route) => {
      passwordPayloads.push(route.request().postDataJSON() as Record<string, unknown>)
      if (passwordPayloads.length === 1) {
        await route.fulfill({
          status: 403,
          headers: { 'Content-Type': 'text/plain' },
          body: 'current password is invalid',
        })
        return
      }
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: 'password-change-rotated-token',
          user,
        }),
      })
    })

    await page.goto('/dashboard', { waitUntil: 'domcontentloaded' })
    await page.getByRole('button', { name: 'Open account menu for Admin User' }).click()
    await page.getByRole('link', { name: 'Password & security' }).click()
    await expect(page).toHaveURL(/\/account\/security$/)

    const username = page.getByRole('textbox', {
      name: 'Account',
      exact: true,
    })
    const currentPassword = page.getByRole('textbox', {
      name: 'Current password',
      exact: true,
    })
    const newPassword = page.getByRole('textbox', {
      name: 'New password',
      exact: true,
    })
    await expect(username).toHaveValue('admin@example.com')
    await expect(username).toHaveAttribute('autocomplete', 'username')
    await expect(currentPassword).toHaveAttribute('autocomplete', 'current-password')
    await expect(newPassword).toHaveAttribute('autocomplete', 'new-password')

    await currentPassword.fill('wrong-current-password')
    await newPassword.fill('new-unique-password-value')
    await page.getByRole('button', { name: 'Change password' }).click()

    await expect(page.getByText('current password is invalid')).toBeVisible()
    await expect(page).toHaveURL(/\/account\/security$/)
    await expect(currentPassword).toHaveValue('')
    await expect(newPassword).toHaveValue('new-unique-password-value')
    await expect(page.getByTestId('account-security-form')).toBeVisible()

    await currentPassword.fill('correct-current-password')
    await page.getByRole('button', { name: 'Change password' }).click()

    await expect(page.getByRole('heading', { name: 'Password changed' })).toBeVisible()
    await expect(page.getByTestId('account-security-form')).toHaveCount(0)
    await expect
      .poll(() => passwordPayloads)
      .toEqual([
        {
          currentPassword: 'wrong-current-password',
          newPassword: 'new-unique-password-value',
        },
        {
          currentPassword: 'correct-current-password',
          newPassword: 'new-unique-password-value',
        },
      ])
    await expect
      .poll(() => page.evaluate(() => window.localStorage.getItem('vsr_auth_token')))
      .toBeNull()
  })
})
