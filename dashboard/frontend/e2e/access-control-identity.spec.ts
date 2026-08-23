import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const user = {
  userId: 'access-user-1',
  email: 'ada@example.com',
  displayName: 'Ada Lovelace',
  status: 'active',
  revision: 1,
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
}

const team = {
  teamId: 'team-1',
  name: 'Platform',
  description: 'Production inference',
  status: 'active',
  revision: 1,
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
}

const group = {
  policyId: 'group-1',
  name: 'Production models',
  description: 'Approved production routes',
  status: 'active',
  revision: 1,
  grants: [
    {
      resourceType: 'model',
      resourceId: 'vllm-sr/mom-*',
      permission: 'invoke',
      effect: 'allow',
    },
  ],
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
}

const budget = {
  policyId: 'budget-1',
  name: 'Production quota',
  description: 'Default Team capacity',
  status: 'active',
  revision: 1,
  rules: [],
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
}

const membership = {
  teamId: team.teamId,
  userId: user.userId,
  role: 'admin',
  status: 'active',
  revision: 1,
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
}

const managementMediaType = 'application/vnd.vllm-semantic-router.management.v1+json'
const namespaceId = '10000000-0000-4000-8000-000000000001'
const requestLog = {
  admissionId: 'admission-1',
  eventId: 'request-1',
  occurredAt: '2026-08-23T00:00:00Z',
  completedAt: '2026-08-23T00:00:01Z',
  protocol: 'openai',
  path: '/v1/chat/completions',
  statusCode: 200,
  usageState: 'recorded',
  inputTokens: '12',
  outputTokens: '8',
  latencyMilliseconds: 420,
  ttftMilliseconds: 90,
  stream: true,
  toolCall: false,
  apiKeyId: 'key-1',
  userId: user.userId,
  teamId: team.teamId,
  entrypointId: 'vllm-sr/balanced',
  recipeId: 'balanced',
  metadata: { externalRequestId: 'chatcmpl-1', request: '{"prompt":"hello"}' },
  costs: [],
}
const emptyTotals = {
  requests: '0',
  successfulRequests: '0',
  inputTokens: '0',
  outputTokens: '0',
  totalTokens: '0',
  cost: '0',
}

const managementPage = (data: unknown[]) => ({
  data,
  page: { hasMore: false, pageSize: 20 },
})

async function mockAccessControl(page: Page) {
  await mockAuthenticatedAppShell(page, {
    user: {
      id: 'dashboard-admin-1',
      email: 'admin@example.com',
      name: 'Admin User',
      role: 'admin',
      permissions: ['users.manage', 'users.view'],
    },
  })

  await page.route('**/api/admin/users**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        users: [
          {
            id: 'dashboard-user-1',
            email: user.email,
            name: user.displayName,
            role: 'admin',
            status: 'active',
          },
        ],
        total: 1,
      }),
    })
  })
  await page.route('**/api/admin/invitations**', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{"items":[]}' })
  })
  await page.route('**/api/router/management/v1/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (
      path.endsWith('/me') ||
      path.includes('/self/inference-keys') ||
      path.includes('/self/inference-sessions')
    ) {
      await route.fallback()
      return
    }
    let body: unknown = managementPage([])
    if (path.endsWith('/users/access-user-1')) body = { data: user }
    else if (path.endsWith('/teams/team-1')) body = { data: team }
    else if (path.endsWith(`/namespaces/${namespaceId}/request-logs/admission-1`)) {
      body = { data: { request: requestLog, routing: {}, quotaReceipts: [], dispatches: [] } }
    } else if (path.endsWith('/teams/team-1/members')) body = managementPage([membership])
    else if (path.endsWith('/users')) body = managementPage([user])
    else if (path.endsWith('/teams')) body = managementPage([team])
    else if (path.endsWith('/access-policies')) body = managementPage([group])
    else if (path.endsWith('/rate-limit-policies')) body = managementPage([budget])
    else if (path.endsWith('/request-logs')) body = managementPage([requestLog])
    else if (path.endsWith('/usage/series')) body = { grain: 'hour', points: [] }
    else if (path.endsWith('/usage/breakdowns')) body = { dimension: 'logical_model', rows: [] }
    else if (path.endsWith('/usage')) body = { grain: 'hour', totals: emptyTotals }
    await route.fulfill({
      status: 200,
      contentType: managementMediaType,
      body: JSON.stringify(body),
    })
  })
}

test.describe('Access control identity', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockAccessControl(page)
  })

  test('makes API key ownership explicit and keeps overrides advanced', async ({ page }) => {
    await page.goto('/access/api-keys')
    await page.getByRole('button', { name: 'Create key' }).click()

    const dialog = page.getByRole('dialog', { name: 'Create API key' })
    await expect(dialog.getByText('Owned by')).toBeVisible()
    await expect(dialog.getByRole('radio', { name: /Personal/ })).toBeVisible()
    await expect(dialog.getByRole('radio', { name: /Team/ })).toBeVisible()
    await expect(dialog.getByRole('searchbox', { name: 'Search users' })).toBeVisible()
    const advanced = dialog.locator('details').filter({ hasText: 'Advanced settings' })
    await expect(advanced).not.toHaveAttribute('open', '')

    await advanced.locator('summary').click()
    await expect(advanced).toHaveAttribute('open', '')
    await expect(advanced.getByText('Key override · optional').first()).toBeVisible()
  })

  test('keeps Dashboard invitations independent from Router team assignment', async ({ page }) => {
    await page.goto('/access/users')
    await page.getByRole('button', { name: 'Invite user' }).click()

    const dialog = page.getByRole('dialog', { name: 'Invite a user' })
    await expect(dialog.getByLabel('Dashboard role')).toBeVisible()
    await expect(dialog.getByText('Team (optional)')).toHaveCount(0)
    await expect(dialog.getByRole('radiogroup', { name: 'Team role' })).toHaveCount(0)
  })

  test('opens Team membership in a centered detail dialog', async ({ page }) => {
    await page.goto('/access/teams')
    await page.getByRole('link', { name: /Platform/ }).click()

    const dialog = page.getByRole('dialog', { name: 'Platform' })
    await expect(dialog).toBeVisible()
    await expect(dialog.getByText('Ada Lovelace')).toBeVisible()
    await expect(dialog.getByText('Team admin', { exact: true })).toBeVisible()
  })

  test('opens a request log in the shared detail experience', async ({ page }) => {
    await page.goto('/logs')
    await page.getByText('vllm-sr/balanced', { exact: true }).click()

    const dialog = page.getByRole('dialog', { name: 'vllm-sr/balanced' })
    await expect(dialog).toBeVisible()
    await expect(dialog.getByRole('button', { name: 'Copy request ID' })).toBeVisible()
    await expect(dialog.getByRole('link', { name: 'Open in Insights' })).toBeVisible()
    await expect(dialog.getByRole('button', { name: 'Done' })).toBeVisible()
  })

  test('searches routing grants in bounded pages instead of loading a full catalog', async ({
    page,
  }) => {
    const routingReads: URL[] = []
    await page.route('**/api/router/management/v1/routing/entrypoints?*', async (route) => {
      routingReads.push(new URL(route.request().url()))
      await route.fulfill({
        status: 200,
        contentType: managementMediaType,
        body: JSON.stringify({
          data: [
            {
              id: 'entrypoint-1',
              name: 'Balanced',
              status: 'active',
              revision: 1,
              entrypointRevision: 1,
              aliases: ['vllm-sr/balanced'],
              ruleCount: 1,
              assignedModelCount: 2,
              createdAt: '2026-08-23T00:00:00Z',
              updatedAt: '2026-08-23T00:00:00Z',
            },
          ],
          page: { hasMore: false, pageSize: 20 },
        }),
      })
    })
    await page.route('**/api/router/management/v1/routing/models?*', async (route) => {
      routingReads.push(new URL(route.request().url()))
      await route.fulfill({
        status: 200,
        contentType: managementMediaType,
        body: JSON.stringify({ data: [], page: { hasMore: false, pageSize: 20 } }),
      })
    })

    await page.goto('/access/access-groups')
    await page.getByRole('button', { name: 'New group' }).click()
    const dialog = page.getByRole('dialog', { name: 'Create access group' })
    const search = dialog.getByRole('searchbox', { name: 'Search Mixture-of-Models' })
    await expect(search).toBeVisible()
    await search.fill('balanced')

    await expect
      .poll(() => routingReads.some((url) => url.searchParams.get('search') === 'balanced'))
      .toBe(true)
    const matchingRequest = routingReads.find(
      (url) => url.searchParams.get('search') === 'balanced',
    )!
    expect(matchingRequest.searchParams.get('pageSize')).toBe('20')
    expect(matchingRequest.searchParams.get('status')).toBe('active')
    const entrypointReads = routingReads.filter((url) =>
      url.pathname.endsWith('/routing/entrypoints'),
    )
    expect(entrypointReads.every((url) => url.searchParams.get('pageSize') === '20')).toBe(true)
    expect(entrypointReads.every((url) => !url.searchParams.has('cursor'))).toBe(true)
    await expect(dialog.getByRole('option', { name: /Balanced/ })).toBeVisible()
  })
})
