import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const user = {
  id: 'access-user-1',
  email: 'ada@example.com',
  name: 'Ada Lovelace',
  status: 'active',
  accessGroupIds: [],
  memberships: [{ teamId: 'team-1', userId: 'access-user-1', role: 'admin' }],
}

const team = {
  id: 'team-1',
  name: 'Platform',
  description: 'Production inference',
  status: 'active',
  members: [{ teamId: 'team-1', userId: 'access-user-1', role: 'admin' }],
  accessGroupIds: ['group-1'],
  budgetId: 'budget-1',
}

const group = {
  id: 'group-1',
  name: 'Production models',
  description: 'Approved production routes',
  modelPatterns: ['vllm-sr/mom-*'],
  assignmentCount: 1,
}

const budget = {
  id: 'budget-1',
  name: 'Production quota',
  description: 'Default Team capacity',
  rpm: 120,
  tpm: 250000,
  dailyTokens: 5000000,
  enabled: true,
  assignmentCount: 1,
}

const overview = {
  users: 1,
  teams: 1,
  activeKeys: 0,
  expiringKeys: 0,
  accessGroups: 1,
  enabledBudgets: 1,
  requestsToday: 0,
  successfulToday: 0,
  tokensToday: 0,
  p95LatencyMs: 0,
}

const emptyUsage = {
  granularity: 'hour',
  requests: 0,
  successful: 0,
  failed: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  activeKeys: 0,
  averageLatencyMs: 0,
  p95LatencyMs: 0,
  averageTtftMs: 0,
  p95TtftMs: 0,
  series: [],
  byModel: [],
  byUser: [],
  byTeam: [],
  byKey: [],
}

async function mockAccessControl(page: Page) {
  await mockAuthenticatedAppShell(page, {
    user: {
      id: 'dashboard-admin-1',
      email: 'admin@example.com',
      name: 'Admin User',
      role: 'admin',
      permissions: ['access.manage', 'access.read', 'users.manage', 'users.view'],
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
            name: user.name,
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
  await page.route('**/api/v1/access-control/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    let body: unknown
    if (path.endsWith('/overview')) body = overview
    else if (path.endsWith('/usage')) body = emptyUsage
    else if (path.endsWith('/users/access-user-1')) body = user
    else if (path.endsWith('/teams/team-1')) body = team
    else if (path.endsWith('/users')) body = { items: [user], total: 1, limit: 100, offset: 0 }
    else if (path.endsWith('/teams')) body = { items: [team], total: 1, limit: 100, offset: 0 }
    else if (path.endsWith('/access-groups'))
      body = { items: [group], total: 1, limit: 100, offset: 0 }
    else if (path.endsWith('/budgets')) body = { items: [budget], total: 1, limit: 100, offset: 0 }
    else if (path.endsWith('/api-keys')) body = { items: [], total: 0, limit: 100, offset: 0 }
    else body = { items: [], total: 0, limit: 10, offset: 0, hasMore: false }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
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
    await expect(dialog.getByRole('combobox').first()).toBeVisible()
    const advanced = dialog.locator('details').filter({ hasText: 'Advanced settings' })
    await expect(advanced).not.toHaveAttribute('open', '')

    await advanced.locator('summary').click()
    await expect(advanced).toHaveAttribute('open', '')
    await expect(advanced.getByText('Key override · optional').first()).toBeVisible()
  })

  test('invites users with independent Dashboard and Team roles', async ({ page }) => {
    await page.goto('/access/users')
    await page.getByRole('button', { name: 'Invite user' }).click()

    const dialog = page.getByRole('dialog', { name: 'Invite a user' })
    await expect(dialog.getByLabel('Dashboard role')).toBeVisible()
    await dialog
      .getByText('Team (optional)')
      .locator('..')
      .getByRole('combobox')
      .selectOption(team.id)
    await expect(dialog.getByRole('radiogroup', { name: 'Team role' })).toBeVisible()
    await expect(dialog.getByRole('radio', { name: /Member/ })).toHaveAttribute(
      'aria-checked',
      'true',
    )
    await expect(dialog.getByRole('radio', { name: /Team admin/ })).toBeVisible()
  })

  test('opens Team membership in a centered detail dialog', async ({ page }) => {
    await page.goto('/access/teams')
    await page.getByRole('link', { name: /Platform/ }).click()

    const dialog = page.getByRole('dialog', { name: 'Platform' })
    await expect(dialog).toBeVisible()
    await expect(dialog.getByText('Ada Lovelace')).toBeVisible()
    await expect(dialog.getByText('Team admin', { exact: true })).toBeVisible()
  })
})
