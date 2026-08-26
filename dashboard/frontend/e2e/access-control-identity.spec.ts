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
  displayName: user.displayName,
  email: user.email,
  userStatus: user.status,
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
  models: [{ id: 'model-balanced', name: 'vllm-sr/balanced', revision: 1 }],
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
  latency: { averageMilliseconds: 0, p95Milliseconds: 0 },
  ttft: { averageMilliseconds: 0, p95Milliseconds: 0 },
  costs: [],
}

const managementPage = (data: unknown[]) => ({
  data,
  page: { hasMore: false, pageSize: 20 },
})

async function expectCenteredProductDialog(page: Page, dialogName: string) {
  const dialog = page.getByRole('dialog', { name: dialogName })
  await expect(dialog).toBeVisible()
  await dialog.evaluate(async (element) => {
    await Promise.all(element.getAnimations().map((animation) => animation.finished))
  })
  const metrics = await dialog.evaluate((element) => {
    const bounds = element.getBoundingClientRect()
    const style = window.getComputedStyle(element)
    const backdropStyle = element.parentElement
      ? window.getComputedStyle(element.parentElement)
      : null
    return {
      centerX: bounds.left + bounds.width / 2,
      centerY: bounds.top + bounds.height / 2,
      borderWidth: Number.parseFloat(style.borderTopWidth),
      left: bounds.left,
      top: bounds.top,
      right: bounds.right,
      bottom: bounds.bottom,
      dialogBackdropFilter: style.backdropFilter || style.webkitBackdropFilter || '',
      backdropFilter: backdropStyle?.backdropFilter || backdropStyle?.webkitBackdropFilter || '',
    }
  })
  const viewport = page.viewportSize()!

  expect(Math.abs(metrics.centerX - viewport.width / 2)).toBeLessThanOrEqual(2)
  expect(Math.abs(metrics.centerY - viewport.height / 2)).toBeLessThanOrEqual(4)
  expect(metrics.borderWidth).toBeGreaterThanOrEqual(1.5)
  expect(metrics.left).toBeGreaterThanOrEqual(0)
  expect(metrics.top).toBeGreaterThanOrEqual(0)
  expect(metrics.right).toBeLessThanOrEqual(viewport.width)
  expect(metrics.bottom).toBeLessThanOrEqual(viewport.height)
  expect(metrics.dialogBackdropFilter).toContain('blur')
  expect(metrics.backdropFilter).toContain('blur')
  return dialog
}

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

  test('makes API key ownership and policy visible while keeping lifecycle settings advanced', async ({
    page,
  }) => {
    await page.goto('/access/api-keys')
    await page.getByRole('button', { name: 'Create key' }).click()

    const dialog = page.getByRole('dialog', { name: 'Create API key' })
    await expect(dialog.getByText('Owned by')).toBeVisible()
    await expect(dialog.getByRole('radio', { name: /Personal/ })).toBeVisible()
    await expect(dialog.getByRole('radio', { name: /Team/ })).toBeVisible()
    await expect(dialog.getByRole('combobox', { name: 'Search users' })).toBeVisible()
    await expect(dialog.getByRole('group', { name: 'Model access' })).toBeVisible()
    const advanced = dialog.locator('details').filter({ hasText: 'Advanced settings' })
    await expect(advanced).not.toHaveAttribute('open', '')

    await advanced.locator('summary').click()
    await expect(advanced).toHaveAttribute('open', '')
    await expect(advanced.getByText('Expiration')).toBeVisible()
    await expect(advanced.getByText('Status')).toBeVisible()
  })

  test('keeps Team assignment optional while making Dashboard access explicit', async ({
    page,
  }) => {
    await page.goto('/access/users')
    await page.getByRole('button', { name: 'Invite user' }).click()

    const dialog = page.getByRole('dialog', { name: 'Invite a user' })
    await expect(dialog.getByLabel('Dashboard role')).toBeVisible()
    await expect(dialog.getByRole('button', { name: 'Choose a Team' })).toBeVisible()
    await expect(dialog.getByRole('radiogroup', { name: 'Team role' })).toHaveCount(0)
  })

  test('keeps every access creation dialog centered and usable at each breakpoint', async ({
    page,
  }) => {
    const dialogs = [
      { path: '/access/api-keys', action: 'Create key', title: 'Create API key' },
      { path: '/access/users', action: 'Invite user', title: 'Invite a user' },
      { path: '/access/teams', action: 'New team', title: 'Create team' },
      { path: '/access/access-groups', action: 'New group', title: 'Create access group' },
      { path: '/access/budgets', action: 'New budget', title: 'Create budget' },
    ] as const

    for (const viewport of [
      { width: 320, height: 568 },
      { width: 390, height: 844 },
      { width: 768, height: 1024 },
      { width: 1440, height: 900 },
    ]) {
      await page.setViewportSize(viewport)
      for (const target of dialogs) {
        await page.goto(target.path)
        await page.getByRole('button', { name: target.action }).click()
        const dialog = await expectCenteredProductDialog(page, target.title)
        await dialog.getByRole('button', { name: 'Close' }).click()
      }
    }
  })

  test('keeps access picker headings and Budget actions visually restrained', async ({ page }) => {
    for (const viewport of [
      { width: 390, height: 844 },
      { width: 1440, height: 900 },
    ]) {
      await page.setViewportSize(viewport)

      await page.goto('/access/teams')
      await page.getByRole('button', { name: 'New team' }).click()
      const teamDialog = page.getByRole('dialog', { name: 'Create team' })
      await expect(teamDialog.getByRole('group', { name: 'Members' })).toBeVisible()
      const modelAccess = teamDialog.getByRole('group', { name: 'Model access' })
      await expect(modelAccess).toBeVisible()
      await expect(teamDialog.getByRole('group', { name: 'Quota' })).toBeVisible()
      const pickerGeometry = await modelAccess.evaluate((element) => {
        const heading = element.querySelector('header')?.getBoundingClientRect()
        const headingCopy = Array.from(element.querySelector('header')?.children ?? []).map(
          (child) => child.getBoundingClientRect(),
        )
        const search = element.querySelector('input[role="combobox"]')?.getBoundingClientRect()
        const copyOverlaps =
          headingCopy.length === 2 &&
          headingCopy[0].left < headingCopy[1].right &&
          headingCopy[0].right > headingCopy[1].left &&
          headingCopy[0].top < headingCopy[1].bottom &&
          headingCopy[0].bottom > headingCopy[1].top
        return {
          spacing: heading && search ? search.top - heading.bottom : -1,
          copyOverlaps,
        }
      })
      expect(pickerGeometry.spacing).toBeGreaterThanOrEqual(8)
      expect(pickerGeometry.copyOverlaps).toBe(false)

      await page.goto('/access/budgets')
      await page.getByRole('button', { name: 'New budget' }).click()
      const budgetDialog = page.getByRole('dialog', { name: 'Create budget' })
      const actionSizes = await Promise.all(
        [
          budgetDialog.getByRole('button', { name: 'Add limit' }),
          budgetDialog.getByRole('button', { name: 'Remove limit 1' }),
        ].map((button) =>
          button.evaluate((element) => {
            const bounds = element.getBoundingClientRect()
            return { width: bounds.width, height: bounds.height }
          }),
        ),
      )
      expect(actionSizes[0].height).toBeLessThanOrEqual(32)
      expect(actionSizes[1].height).toBeLessThanOrEqual(32)
      expect(actionSizes[1].width).toBeLessThanOrEqual(32)
    }
  })

  test('presents member and administrator roles as a compact choice, not a raw select', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto('/access/users')
    await page.getByRole('button', { name: 'Invite user' }).click()
    const inviteDialog = page.getByRole('dialog', { name: 'Invite a user' })
    await inviteDialog.getByRole('button', { name: 'Choose a Team' }).click()
    await inviteDialog.getByRole('option', { name: /Platform/ }).click()

    const roles = inviteDialog.getByRole('radiogroup', { name: 'Team role' })
    await expect(roles.getByRole('radio', { name: 'Member' })).toBeVisible()
    await expect(roles.getByRole('radio', { name: 'Admin' })).toBeVisible()
    await expect(roles.getByRole('radio', { name: 'Member' })).toHaveAttribute(
      'aria-checked',
      'true',
    )
  })

  test('keeps Team membership in a centered detail dialog at each breakpoint', async ({ page }) => {
    for (const viewport of [
      { width: 320, height: 568 },
      { width: 390, height: 844 },
      { width: 768, height: 1024 },
      { width: 1440, height: 900 },
    ]) {
      await page.setViewportSize(viewport)
      await page.goto('/access/teams')
      await page.getByRole('link', { name: /Platform/ }).click()

      const dialog = await expectCenteredProductDialog(page, 'Platform')
      await expect(dialog.getByText('Ada Lovelace')).toBeVisible()
      await expect(dialog.getByText('Team admin', { exact: true })).toBeVisible()
      await dialog.getByRole('button', { name: 'Close' }).click()
    }
  })

  test('keeps request details in the shared dialog at each breakpoint', async ({ page }) => {
    for (const viewport of [
      { width: 320, height: 568 },
      { width: 390, height: 844 },
      { width: 768, height: 1024 },
      { width: 1440, height: 900 },
    ]) {
      await page.setViewportSize(viewport)
      await page.goto('/logs')
      await page.getByText('vllm-sr/balanced', { exact: true }).click()

      const dialog = await expectCenteredProductDialog(page, 'vllm-sr/balanced')
      await expect(dialog.getByRole('button', { name: 'Copy request ID' })).toBeVisible()
      await expect(dialog.getByRole('link', { name: 'Open in Insights' })).toBeVisible()
      await expect(dialog.getByRole('button', { name: 'Done' })).toBeVisible()
      await dialog.getByRole('button', { name: 'Close' }).click()
    }
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
              recipeIds: ['recipe-balanced'],
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
    const search = dialog.getByRole('combobox', { name: 'Search Mixture-of-Models' })
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
