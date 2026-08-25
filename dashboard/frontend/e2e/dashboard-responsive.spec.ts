import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const managementMediaType = 'application/vnd.vllm-semantic-router.management.v1+json'

const viewports = [
  { name: 'compact phone', width: 320, height: 568 },
  { name: 'phone', width: 390, height: 844 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'desktop', width: 1440, height: 900 },
] as const

const routes = [
  '/dashboard',
  '/access/usage',
  '/access/api-keys',
  '/access/users',
  '/access/teams',
  '/access/access-groups',
  '/access/budgets',
  '/logs',
  '/config/models',
  '/config/entrypoints-recipes',
  '/config/signals',
  '/config/projections',
  '/config/decisions',
  '/config/agent',
  '/topology',
  '/playground',
  '/insights',
  '/status',
  '/evaluation',
  '/builder',
] as const

const emptyTotals = {
  requests: '0',
  successfulRequests: '0',
  inputTokens: '0',
  outputTokens: '0',
  totalTokens: '0',
  latency: { averageMilliseconds: 0, p95Milliseconds: 0 },
  ttft: { averageMilliseconds: 0, p95Milliseconds: 0 },
  costs: [],
}

const emptyManagementPage = {
  data: [],
  page: { hasMore: false, pageSize: 100 },
}

const keyScopedCatalog = {
  keyId: '10000000-0000-4000-8000-000000000003',
  policyRevision: 1,
  policyDigest: 'a'.repeat(64),
  routingRevision: 1,
  routingDigest: 'b'.repeat(64),
  models: [
    {
      id: 'model_consumer',
      revision: 1,
      name: 'local/consumer',
      aliases: [],
      capabilities: ['text'],
      loras: [],
      tags: [],
      pricing: {
        inputCostPerMillionTokens: null,
        outputCostPerMillionTokens: null,
        cacheReadCostPerMillionTokens: null,
        cacheWriteCostPerMillionTokens: null,
      },
    },
  ],
  recipes: [
    {
      id: 'recipe_consumer',
      revision: 1,
      name: 'Consumer',
      decisions: [{ id: 'decision_default', name: 'Default', dispatchCardinality: 'single' }],
    },
  ],
  entrypoints: [
    {
      id: 'entrypoint_consumer',
      revision: 1,
      name: 'consumer',
      aliases: ['vllm-sr/consumer'],
      rules: [
        {
          id: 'rule_default',
          name: 'Default',
          recipeId: 'recipe_consumer',
          recipeRevision: 1,
          assignments: {
            decision_default: {
              models: [
                {
                  modelId: 'model_consumer',
                  modelRevision: 1,
                  priority: 0,
                  weight: '1',
                },
              ],
            },
          },
        },
      ],
    },
  ],
}

async function mockResponsiveDashboardData(page: Page) {
  await page.route('**/v1/models*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ object: 'list', data: [] }),
    })
  })
  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ overall: 'healthy', deployment_type: 'local', services: [] }),
    })
  })
  await page.route('**/api/admin/users**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ users: [], total: 0 }),
    })
  })
  await page.route('**/api/admin/invitations**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ items: [] }),
    })
  })
  await page.route('**/api/evaluation/tasks**', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  })
  await page.route('**/api/evaluation/datasets**', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' })
  })
  await page.route('**/api/logs**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ logs: [], total: 0 }),
    })
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

    let body: unknown = emptyManagementPage
    if (path.endsWith('/routing-catalog')) {
      body = keyScopedCatalog
    } else if (path.endsWith('/statistics')) {
      body = {
        users: '0',
        teams: '0',
        activeApiKeys: '0',
        expiringApiKeys: '0',
        accessPolicies: '0',
        activeRatePolicies: '0',
      }
    } else if (path.endsWith('/usage/series')) {
      body = { grain: 'hour', points: [] }
    } else if (path.endsWith('/usage/breakdowns')) {
      body = { dimension: 'logical_model', rows: [] }
    } else if (path.endsWith('/usage')) {
      body = { grain: 'hour', totals: emptyTotals }
    }

    await route.fulfill({
      status: 200,
      contentType: managementMediaType,
      body: JSON.stringify(body),
    })
  })
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (
      path === '/api/status' ||
      path.startsWith('/api/admin/') ||
      path.startsWith('/api/auth/') ||
      path.startsWith('/api/evaluation/') ||
      path.startsWith('/api/logs') ||
      path.startsWith('/api/router/management/') ||
      path === '/api/settings'
    ) {
      await route.fallback()
      return
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' })
  })

  await mockAuthenticatedAppShell(page)
}

async function expectNoViewportOverflow(page: Page, route: string) {
  await expect(page.locator('#root')).toBeVisible()
  await page.evaluate(() => document.fonts.ready)
  const dimensions = await page.evaluate(() => ({
    body: document.body.scrollWidth,
    document: document.documentElement.scrollWidth,
    viewport: window.innerWidth,
  }))
  expect(dimensions.body, `${route} body overflow`).toBeLessThanOrEqual(dimensions.viewport)
  expect(dimensions.document, `${route} document overflow`).toBeLessThanOrEqual(dimensions.viewport)
}

async function expectCenteredDialog(page: Page) {
  const dialog = page.getByRole('dialog')
  await expect(dialog).toBeVisible()
  const [dialogBox, viewport, styles] = await Promise.all([
    dialog.boundingBox(),
    page.evaluate(() => ({ width: window.innerWidth, height: window.innerHeight })),
    dialog.evaluate((element) => {
      const computed = window.getComputedStyle(element)
      return {
        borderWidth: Number.parseFloat(computed.borderTopWidth),
        backdropFilter: computed.backdropFilter || computed.webkitBackdropFilter,
      }
    }),
  ])

  expect(dialogBox).not.toBeNull()
  expect(
    Math.abs((dialogBox?.x ?? 0) + (dialogBox?.width ?? 0) / 2 - viewport.width / 2),
  ).toBeLessThan(2)
  expect(
    Math.abs((dialogBox?.y ?? 0) + (dialogBox?.height ?? 0) / 2 - viewport.height / 2),
  ).toBeLessThan(2)
  expect(styles.borderWidth).toBeGreaterThanOrEqual(1.5)
  expect(styles.backdropFilter).toContain('blur')
}

test.describe('Dashboard responsive route matrix', () => {
  for (const viewport of viewports) {
    test(`${viewport.name} keeps every product surface inside the viewport`, async ({ page }) => {
      test.slow()
      await page.setViewportSize(viewport)
      await mockResponsiveDashboardData(page)

      for (const route of routes) {
        await page.goto(route, { waitUntil: 'domcontentloaded' })
        await expect(page).toHaveURL(new RegExp(`${route.replaceAll('/', '\\/')}(?:[?#]|$)`))
        await expectNoViewportOverflow(page, route)
      }
    })
  }

  test('lowest consumer sees Playground and only the Routing build group', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockResponsiveDashboardData(page)
    await mockAuthenticatedAppShell(page, {
      user: {
        id: 'consumer-1',
        email: 'consumer@example.com',
        name: 'Consumer',
        role: 'read',
        permissions: ['config.read', 'topology.read'],
      },
      managementPermissions: [
        'agent.read',
        'agent.use',
        'access_policy.read',
        'delegation.use',
        'key.read',
        'routing_context.read',
        'usage.read',
      ],
    })
    await page.goto('/dashboard')

    const overview = page.getByRole('region', { name: 'Your model system, at a glance.' })
    await expect(overview.getByRole('button', { name: /Models/ })).toHaveCount(0)
    await expect(overview.getByRole('button', { name: /Signals/ })).toHaveCount(0)
    await expect(overview.getByRole('button', { name: /Decisions/ })).toHaveCount(0)
    await expect(overview.getByRole('button', { name: /API Keys/ })).toBeVisible()
    await expect(overview.getByRole('button', { name: /Try a request/ })).toBeVisible()
    await expect(page.getByRole('link', { name: 'Playground', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: /Operate/ })).toHaveCount(0)
    await page.getByRole('button', { name: /Build/ }).click()
    const buildMenu = page.getByRole('navigation', { name: 'Build' })
    await expect(buildMenu.getByRole('tab', { name: /Routing/ })).toBeVisible()
    await expect(buildMenu.getByRole('tab', { name: /Integrations/ })).toHaveCount(0)
    await expect(buildMenu.getByText('vLLM-SR Agent', { exact: true })).toHaveCount(0)
    await expect(buildMenu.getByText('Mixture-of-Models', { exact: true })).toBeVisible()
    await expect(buildMenu.getByText('Brain Topology', { exact: true })).toBeVisible()
    await expect(buildMenu.getByRole('button', { name: 'Models', exact: true })).toHaveCount(0)
    await expect(buildMenu.getByText('Signals', { exact: true })).toHaveCount(0)
  })

  test('mobile consumer menu preserves the same visibility contract', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await mockResponsiveDashboardData(page)
    await mockAuthenticatedAppShell(page, {
      user: {
        id: 'consumer-1',
        email: 'consumer@example.com',
        name: 'Consumer',
        role: 'read',
        permissions: ['config.read', 'topology.read'],
      },
      managementPermissions: [
        'agent.read',
        'agent.use',
        'access_policy.read',
        'delegation.use',
        'key.read',
        'routing_context.read',
        'usage.read',
      ],
    })
    await page.goto('/dashboard')
    await page.getByRole('button', { name: 'Toggle menu' }).click()
    const navigation = page.getByRole('navigation', { name: 'Mobile navigation' })

    await expect(navigation.getByRole('link', { name: 'Playground', exact: true })).toBeVisible()
    await expect(navigation.getByRole('button', { name: /Operate/ })).toHaveCount(0)
    await navigation.getByRole('button', { name: /Build/ }).click()
    await expect(navigation.getByText('Routing', { exact: true })).toBeVisible()
    await expect(navigation.getByText('Integrations', { exact: true })).toHaveCount(0)
    await expect(navigation.getByText('vLLM-SR Agent', { exact: true })).toHaveCount(0)
    await expect(navigation.getByText('Mixture-of-Models', { exact: true })).toBeVisible()
    await expect(navigation.getByText('Brain Topology', { exact: true })).toBeVisible()
    await expect(navigation.getByRole('button', { name: 'Models', exact: true })).toHaveCount(0)
    await expectNoViewportOverflow(page, 'consumer mobile navigation')
  })

  test('product guide stays centered and usable across compact viewports', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await mockResponsiveDashboardData(page)

    for (const viewport of viewports.slice(0, 3)) {
      await page.setViewportSize(viewport)
      await page.goto('/dashboard')
      await page.getByRole('button', { name: 'Open product guide' }).click()
      const dialog = page.getByRole('dialog')
      await expect(dialog.getByRole('heading', { name: 'Connect your models' })).toBeVisible()
      await expect(dialog.locator('img[src="/vllm.png"]')).toBeVisible()
      await expectCenteredDialog(page)
      await expectNoViewportOverflow(page, '/dashboard product guide')
      await page.getByRole('button', { name: 'Pause product guide' }).click()
      await page.evaluate(() => window.localStorage.clear())
    }
  })

  test('mobile account details use the centered dialog contract', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await mockResponsiveDashboardData(page)

    for (const viewport of viewports.slice(0, 3)) {
      await page.setViewportSize(viewport)
      await page.goto('/dashboard')
      await page.getByRole('button', { name: 'Open account menu for Admin User' }).click()
      await expect(page.getByRole('heading', { name: 'Admin User' })).toBeVisible()
      await expectCenteredDialog(page)
      await expectNoViewportOverflow(page, '/dashboard account dialog')
      await page.getByRole('button', { name: 'Close account dialog' }).click()
    }
  })
})
