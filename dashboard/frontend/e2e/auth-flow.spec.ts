import { expect, test, type Locator } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const managementMediaType = 'application/vnd.vllm-semantic-router.management.v1+json'
const transitionCopyPattern = /Entering control plane/i

async function expectDialogInsideViewport(
  dialog: Locator,
  viewport: { width: number; height: number },
) {
  const bounds = await dialog.boundingBox()
  expect(bounds).not.toBeNull()
  expect(bounds?.x ?? -1).toBeGreaterThanOrEqual(0)
  expect(bounds?.y ?? -1).toBeGreaterThanOrEqual(0)
  expect(bounds?.width ?? Infinity).toBeLessThanOrEqual(viewport.width)
  expect(bounds?.height ?? Infinity).toBeLessThanOrEqual(viewport.height)
}

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
    await page.getByLabel('Password', { exact: true }).fill('future-password')
    await page.getByRole('button', { name: 'Create admin and continue' }).click()

    await expect(page).toHaveURL(/\/auth\/transition\?to=%2Fdashboard$/)
  })

  test('delivers an invited user from welcome to the one-time API key reveal', async ({
    page,
    context,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await context.grantPermissions(['clipboard-read', 'clipboard-write'])
    const dashboardUser = {
      id: 'dashboard-invite-1',
      email: 'ada@example.com',
      name: 'Ada Lovelace',
      role: 'read',
      permissions: [],
    }
    const routerUserId = '10000000-0000-4000-8000-000000000011'
    const keyId = '10000000-0000-4000-8000-000000000003'
    const secret = 'vsr_invited_user_secret'
    let accepted = false

    await page.route('**/api/auth/me', async (route) => {
      await route.fulfill(
        accepted
          ? {
              status: 200,
              contentType: 'application/json',
              body: JSON.stringify({ user: dashboardUser }),
            }
          : { status: 401, body: 'Unauthorized' },
      )
    })
    await page.route('**/api/auth/bootstrap/can-register', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' })
    })
    await page.route('**/api/auth/invitations/info?*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          email: dashboardUser.email,
          name: dashboardUser.name,
          expiresAt: Math.floor(Date.now() / 1000) + 86_400,
        }),
      })
    })
    await page.route('**/api/auth/invitations/accept', async (route) => {
      accepted = true
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          user: dashboardUser,
          onboarding: {
            userId: routerUserId,
            teamId: '10000000-0000-4000-8000-000000000020',
            apiKeyId: keyId,
            apiKey: secret,
            deliveryExpiresAt: '2099-08-23T00:00:00Z',
          },
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
          routerPublicUrl: '',
        }),
      })
    })
    await page.route('**/api/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ overall: 'healthy', services: [] }),
      })
    })
    await page.route('**/api/router/management/v1/**', async (route) => {
      const path = new URL(route.request().url()).pathname
      let body: unknown = { data: [], page: { hasMore: false, pageSize: 100 } }
      if (path.endsWith('/me')) {
        body = {
          principal: {
            principalId: '10000000-0000-4000-8000-000000000010',
            displayName: dashboardUser.name,
            kind: 'human',
            status: 'active',
          },
          session: {
            sessionId: '10000000-0000-4000-8000-000000000002',
            authenticatedAt: '2026-08-23T00:00:00Z',
            expiresAt: '2099-08-23T00:00:00Z',
            evidenceKind: 'human',
          },
          clusterPermissions: [],
          namespaces: [
            {
              namespace: {
                namespaceId: '10000000-0000-4000-8000-000000000001',
                name: 'Test',
                status: 'active',
                desiredRevision: 1,
                appliedRevision: 1,
              },
              permissions: [
                'access_policy.read',
                'agent.read',
                'agent.use',
                'delegation.use',
                'key.read',
                'log.read',
                'routing_context.read',
                'usage.read',
              ],
              roleBindings: [],
              user: {
                userId: routerUserId,
                email: dashboardUser.email,
                displayName: dashboardUser.name,
                status: 'active',
              },
              teams: [
                {
                  teamId: '10000000-0000-4000-8000-000000000020',
                  name: 'Research',
                  role: 'member',
                  status: 'active',
                },
              ],
              selfServicePolicy: {
                maxKeysPerUser: 1,
                maxDelegatedSessions: 3,
                delegatedSessionTtlSeconds: 900,
                allowTeamKeyDelegation: false,
                automaticFirstKey: true,
                revision: 1,
              },
            },
          ],
        }
      } else if (path.endsWith('/self/inference-keys')) {
        body = {
          data: [
            {
              keyId,
              name: "Ada Lovelace's API key",
              owner: { type: 'user', id: routerUserId },
              contextTeamId: '10000000-0000-4000-8000-000000000020',
              expiresAt: '2099-08-23T00:00:00Z',
            },
          ],
          page: { hasMore: false, pageSize: 100 },
        }
      } else if (path.endsWith('/routing-catalog')) {
        body = {
          keyId,
          policyRevision: 1,
          policyDigest: 'a'.repeat(64),
          routingRevision: 1,
          routingDigest: 'b'.repeat(64),
          models: [],
          recipes: [],
          entrypoints: [],
        }
      } else if (path.endsWith('/statistics')) {
        body = {
          users: '1',
          teams: '1',
          activeApiKeys: '1',
          expiringApiKeys: '0',
          accessPolicies: '1',
          activeRatePolicies: '1',
        }
      } else if (path.endsWith('/usage/series')) {
        body = { grain: 'hour', points: [] }
      } else if (path.endsWith('/usage/breakdowns')) {
        body = { dimension: 'logical_model', rows: [] }
      } else if (path.endsWith('/usage')) {
        body = {
          grain: 'hour',
          totals: {
            requests: '0',
            successfulRequests: '0',
            inputTokens: '0',
            outputTokens: '0',
            totalTokens: '0',
            costs: [],
          },
        }
      }
      await route.fulfill({
        status: 200,
        contentType: managementMediaType,
        body: JSON.stringify(body),
      })
    })

    await page.goto('/login?invite=1&token=welcome-token')
    await page.getByLabel('Password', { exact: true }).fill('future-password')
    await page.getByRole('button', { name: 'Join the workspace' }).click()

    await expect(page).toHaveURL(/\/dashboard$/)
    const welcome = page.getByRole('dialog', { name: 'You’re in, Ada.' })
    await expect(welcome).toBeVisible()
    await expect(welcome.locator('img[src="/vllm.png"]')).toBeVisible()
    await expectDialogInsideViewport(welcome, { width: 390, height: 844 })
    await welcome.getByRole('button', { name: 'Reveal my API key' }).click()

    await expect(page).toHaveURL(/\/access\/api-keys$/)
    const keyDialog = page.getByRole('dialog', { name: 'Your key is ready.' })
    await expect(keyDialog).toContainText(secret)
    await expect(keyDialog.locator('img[src="/vllm.png"]')).toBeVisible()
    await keyDialog.getByRole('button', { name: 'Copy' }).click()
    await expect(keyDialog.getByRole('button', { name: 'Copied' })).toBeVisible()
    await expect(keyDialog.getByRole('button', { name: 'View details' })).toBeVisible()
    await expectDialogInsideViewport(keyDialog, { width: 390, height: 844 })
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
