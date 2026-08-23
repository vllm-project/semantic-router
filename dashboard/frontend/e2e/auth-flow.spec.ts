import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const managementMediaType = 'application/vnd.vllm-semantic-router.management.v1+json'
const transitionCopyPattern = /Entering control plane/i

test.describe('Dashboard auth flow', () => {
  test('keeps the mobile sign-in form reachable', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.route('**/api/auth/bootstrap/can-register', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '{"canRegister":false}',
      })
    })
    await page.route('**/api/auth/me', async (route) => {
      await route.fulfill({ status: 401, body: 'Unauthorized' })
    })

    await page.goto('/login')
    const continueButton = page.getByRole('button', { name: 'Continue' })
    await continueButton.scrollIntoViewIfNeeded()
    await expect(page.getByLabel('Email')).toHaveAttribute('autocomplete', 'username')
    await expect(page.getByLabel('Password')).toHaveAttribute('autocomplete', 'current-password')
    await expect(continueButton).toBeVisible()
  })

  test('redirects an unauthenticated protected route to sign in', async ({ page }) => {
    await page.route('**/api/auth/bootstrap/can-register', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '{"canRegister":false}',
      })
    })
    await page.route('**/api/auth/me', async (route) => {
      await route.fulfill({ status: 401, body: 'Unauthorized' })
    })

    await page.goto('/playground', { waitUntil: 'domcontentloaded' })
    await expect(page).toHaveURL(/\/login$/)
    await expect(page.getByRole('heading', { name: 'Sign in', exact: true })).toBeVisible()
  })

  test('uses the server session and returns to the requested page', async ({ page }) => {
    test.slow()
    const sessionCookie = 'dashboard-session'
    await page.route('**/api/auth/bootstrap/can-register', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '{"canRegister":false}',
      })
    })
    await page.route('**/api/auth/login', async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Set-Cookie': `vsr_session=${sessionCookie}; Path=/; HttpOnly; SameSite=Lax`,
        },
        body: JSON.stringify({
          user: {
            id: 'user-admin-1',
            email: 'admin@example.com',
            name: 'Admin',
            role: 'admin',
            permissions: ['status.read'],
          },
        }),
      })
    })
    await page.route('**/api/auth/me', async (route) => {
      if (!route.request().headers().cookie?.includes(`vsr_session=${sessionCookie}`)) {
        await route.fulfill({ status: 401, body: 'Unauthorized' })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          user: {
            id: 'user-admin-1',
            email: 'admin@example.com',
            name: 'Admin',
            role: 'admin',
            permissions: ['status.read'],
          },
        }),
      })
    })
    await page.route('**/api/router/management/v1/me', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': managementMediaType },
        body: JSON.stringify({
          principal: {
            principalId: 'principal-1',
            displayName: 'Admin',
            kind: 'human',
            status: 'active',
          },
          session: {
            sessionId: 'session-1',
            authenticatedAt: '2026-08-23T00:00:00Z',
            expiresAt: '2099-08-23T00:00:00Z',
            evidenceKind: 'human',
          },
          clusterPermissions: [],
          namespaces: [
            {
              namespace: {
                namespaceId: 'namespace-1',
                name: 'Default',
                status: 'active',
                desiredRevision: 1,
                appliedRevision: 1,
              },
              permissions: ['routing.read'],
              roleBindings: [],
              user: {
                userId: 'router-user-1',
                email: 'admin@example.com',
                displayName: 'Admin',
                status: 'active',
              },
              teams: [],
            },
          ],
        }),
      })
    })
    await page.route('**/api/settings', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          readonlyMode: false,
          serverReadonly: false,
          platform: '',
          envoyUrl: '',
          routerPublicUrl: '',
        }),
      })
    })
    await page.route('**/api/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ overall: 'healthy', deployment_type: 'local', services: [] }),
      })
    })

    await page.goto('/status', { waitUntil: 'domcontentloaded' })
    await expect(page).toHaveURL(/\/login$/)
    await page.getByPlaceholder('you@example.com').fill('admin@example.com')
    await page.getByPlaceholder('••••••••').fill('secret-password')
    await Promise.all([
      page.waitForURL(/\/auth\/transition\?to=%2Fstatus$/),
      page.getByText(transitionCopyPattern).waitFor({ state: 'visible' }),
      page.getByRole('button', { name: 'Continue' }).click(),
    ])
    await expect(page).toHaveURL(/\/status$/, { timeout: 12000 })
  })

  test('first administrator bootstrap enters the Dashboard without setup authority', async ({
    page,
  }) => {
    await page.route('**/api/auth/bootstrap/can-register', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '{"canRegister":true}',
      })
    })
    await page.route('**/api/auth/me', async (route) => {
      await route.fulfill({ status: 401, body: 'Unauthorized' })
    })
    await page.route('**/api/auth/bootstrap/register', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          user: {
            id: 'user-admin-1',
            email: 'ada@example.com',
            name: 'Ada Router',
            role: 'admin',
          },
        }),
      })
    })

    await page.goto('/login')
    await page.getByLabel('What should we call you?').fill('Ada Router')
    await page.getByRole('button', { name: 'Next' }).click()
    await page.getByLabel('Admin email').fill('ada@example.com')
    await page.getByRole('button', { name: 'Next' }).click()
    await page.getByLabel('Password').fill('future-password')
    await page.getByRole('button', { name: 'Create admin and continue' }).click()

    await expect(page).toHaveURL(/\/auth\/transition\?to=%2Fdashboard$/)
  })

  test('routing.read without routing.manage exposes read-only topology', async ({ page }) => {
    await mockAuthenticatedAppShell(page, {
      user: {
        id: 'user-read-1',
        email: 'reader@example.com',
        name: 'Reader',
        role: 'custom-role',
        permissions: [],
      },
      managementPermissions: ['routing.read'],
    })
    await page.route('**/api/router/management/v1/routing/**', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': managementMediaType },
        body: JSON.stringify({ data: [], page: { hasMore: false, pageSize: 100 } }),
      })
    })

    await page.goto('/config/entrypoints-recipes')
    await expect(page).toHaveURL(/\/config\/entrypoints-recipes$/)
    await expect(page.getByRole('button', { name: /new mixture/i })).toHaveCount(0)
    await page.goto('/topology')
    await expect(page).toHaveURL(/\/topology$/)
  })
})
