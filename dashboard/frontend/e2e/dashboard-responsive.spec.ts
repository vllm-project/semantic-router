import { expect, test, type Locator, type Page, type Route } from '@playwright/test'

import { shellRouteDefinitions } from '../src/app/routeManifest'
import { ACCESS_NAV_ITEMS } from '../src/pages/AccessControlPageSupport'
import { mockAuthenticatedAppShell } from './support/auth'

const managementMediaType = 'application/vnd.vllm-semantic-router.management.v1+json'
const namespaceId = '10000000-0000-4000-8000-000000000001'
const insightAdmissionId = '10000000-0000-4000-8000-000000000005'
const insightRecordRoute = `/insights/${namespaceId}:${insightAdmissionId}`

const viewports = [
  { name: 'compact phone', width: 320, height: 568 },
  { name: 'phone', width: 390, height: 844 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'desktop', width: 1440, height: 900 },
] as const

const shellRouteSample = (path: string) => {
  if (path === '/plugins') return '/plugins/context-compression'
  return path
    .replace(':view', 'usage')
    .replace(':recordId', `${namespaceId}:${insightAdmissionId}`)
    .replace(':plugin', 'context-compression')
}

const routes = Array.from(
  new Set([
    ...shellRouteDefinitions.map(({ path }) => shellRouteSample(path)),
    ...ACCESS_NAV_ITEMS.map(({ id }) => (id === 'request-logs' ? '/logs' : `/access/${id}`)),
    '/config',
    '/config/models',
    '/config/entrypoints-recipes',
    '/config/signals',
    '/config/projections',
    '/config/decisions',
    '/config/agent',
    '/playground/fullscreen',
    '/ml-setup',
  ]),
)

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
    if (path.endsWith(`/namespaces/${namespaceId}/request-logs/${insightAdmissionId}`)) {
      body = {
        data: {
          request: {
            admissionId: insightAdmissionId,
            eventId: 'chatcmpl-responsive',
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
            apiKeyId: '10000000-0000-4000-8000-000000000003',
            entrypointId: 'vllm-sr/consumer',
            recipeId: 'consumer',
            metadata: { externalRequestId: 'chatcmpl-responsive' },
            costs: [],
          },
          routing: {},
          quotaReceipts: [],
          dispatches: [],
        },
      }
    } else if (path.endsWith('/routing-catalog')) {
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
      path === '/api/fleet-sim/api/workloads' ||
      path === '/api/fleet-sim/api/traces' ||
      path === '/api/fleet-sim/api/gpu-profiles' ||
      path === '/api/fleet-sim/api/fleets' ||
      path === '/api/fleet-sim/api/jobs'
    ) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
      return
    }
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

async function mockTopologyPreviewData(page: Page) {
  const now = '2026-08-23T00:00:00Z'
  const pageEnvelope = (data: unknown[]) => ({
    data,
    page: { hasMore: false, pageSize: 100 },
  })
  const model = {
    id: 'model_fast',
    name: 'local/fast',
    card: { aliases: [], capabilities: ['text'], loras: [], tags: [] },
  }
  const recipe = {
    id: 'recipe_responsive',
    name: 'responsive',
    description: 'Responsive topology fixture',
    status: 'active',
    revision: 1,
    recipeRevision: 1,
    origin: 'custom',
    immutable: false,
    decisions: [{ id: 'decision_simple', name: 'Simple', dispatchCardinality: 'single' }],
    document: {
      signals: {},
      projections: {},
      decisions: [
        {
          id: 'decision_simple',
          name: 'Simple',
          priority: 1,
          rules: { operator: 'AND', conditions: [] },
        },
      ],
    },
    createdAt: now,
    updatedAt: now,
  }
  const entrypoint = {
    id: 'entrypoint_responsive',
    name: 'responsive',
    status: 'active',
    revision: 1,
    entrypointRevision: 1,
    aliases: ['vllm-sr/responsive'],
    ruleCount: 1,
    assignedModelCount: 1,
    createdAt: now,
    updatedAt: now,
  }
  const topology = {
    ...entrypoint,
    rules: [
      {
        id: 'rule_default',
        name: 'Default',
        recipeId: recipe.id,
        recipeRevision: 1,
        assignments: {
          decision_simple: {
            models: [
              { modelId: model.id, modelRevision: 1, priority: 0, weight: '1' },
            ],
          },
        },
      },
    ],
  }
  const fulfill = (route: Route, body: unknown) =>
    route.fulfill({
      status: 200,
      contentType: managementMediaType,
      body: JSON.stringify(body),
    })

  await page.route('**/api/router/management/v1/routing/model-cards?*', (route) =>
    fulfill(route, pageEnvelope([model])),
  )
  await page.route('**/api/router/management/v1/routing/recipes?*', (route) =>
    fulfill(route, pageEnvelope([recipe])),
  )
  await page.route('**/api/router/management/v1/routing/entrypoints?*', (route) =>
    fulfill(route, pageEnvelope([entrypoint])),
  )
  await page.route(
    '**/api/router/management/v1/routing/entrypoints/entrypoint_responsive?includeTopology=true',
    (route) => fulfill(route, { data: topology }),
  )
}

async function expectNoViewportOverflow(page: Page, route: string) {
  await expect(page.locator('#root')).toBeVisible()
  await expect(page.getByText('Loading control plane', { exact: true })).toHaveCount(0)
  await expect(page.getByTestId('route-load-error')).toHaveCount(0)
  await page.evaluate(() => document.fonts.ready)
  const dimensions = await page.evaluate(() => ({
    body: document.body.scrollWidth,
    document: document.documentElement.scrollWidth,
    viewport: window.innerWidth,
  }))
  expect(dimensions.body, `${route} body overflow`).toBeLessThanOrEqual(dimensions.viewport)
  expect(dimensions.document, `${route} document overflow`).toBeLessThanOrEqual(dimensions.viewport)
}

async function expectUnobstructedControl(page: Page, control: Locator, label: string) {
  await expect(control, `${label} is visible`).toBeVisible()
  const box = await control.boundingBox()
  expect(box, `${label} has a layout box`).not.toBeNull()
  if (!box) return

  const viewport = page.viewportSize()
  expect(viewport, `${label} has a viewport`).not.toBeNull()
  if (!viewport) return
  expect(box.x, `${label} starts inside the viewport`).toBeGreaterThanOrEqual(0)
  expect(box.y, `${label} starts inside the viewport`).toBeGreaterThanOrEqual(0)
  expect(box.x + box.width, `${label} ends inside the viewport`).toBeLessThanOrEqual(
    viewport.width,
  )
  expect(box.y + box.height, `${label} ends inside the viewport`).toBeLessThanOrEqual(
    viewport.height,
  )
  expect(
    Math.min(box.width, box.height),
    `${label} keeps a usable hit target`,
  ).toBeGreaterThanOrEqual(28)
  const ownsCenterPoint = await control.evaluate((element) => {
    const bounds = element.getBoundingClientRect()
    const hit = document.elementFromPoint(
      bounds.left + bounds.width / 2,
      bounds.top + bounds.height / 2,
    )
    return Boolean(hit && (hit === element || element.contains(hit)))
  })
  expect(ownsCenterPoint, `${label} is not covered by another surface`).toBe(true)
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
  expect(dialogBox?.x ?? -1).toBeGreaterThanOrEqual(0)
  expect(dialogBox?.y ?? -1).toBeGreaterThanOrEqual(0)
  expect((dialogBox?.x ?? 0) + (dialogBox?.width ?? 0)).toBeLessThanOrEqual(viewport.width)
  expect((dialogBox?.y ?? 0) + (dialogBox?.height ?? 0)).toBeLessThanOrEqual(viewport.height)
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
        if (route === insightRecordRoute) {
          await expect(page.getByRole('button', { name: 'Copy link' })).toBeVisible()
        }
        await expectNoViewportOverflow(page, route)
      }
    })
  }

  test('keeps core Build dialogs centered at each product breakpoint', async ({ page }) => {
    test.slow()
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await mockResponsiveDashboardData(page)

    for (const viewport of viewports) {
      await page.setViewportSize(viewport)

      await page.goto('/config/models')
      await page.getByRole('button', { name: 'Add model' }).click()
      await expect(page.getByRole('heading', { name: 'Choose a provider' })).toBeVisible()
      await expectCenteredDialog(page)
      await expectNoViewportOverflow(page, `/config/models dialog at ${viewport.name}`)
      await page.getByRole('button', { name: 'Close' }).click()

      await page.goto('/config/entrypoints-recipes')
      await page.getByRole('button', { name: 'Create recipe' }).click()
      await expect(page.getByRole('heading', { name: 'Create a recipe' })).toBeVisible()
      await expectCenteredDialog(page)
      await expectNoViewportOverflow(page, `/config/entrypoints-recipes recipe at ${viewport.name}`)
      await page.getByRole('button', { name: 'Close' }).click()

      await page.getByRole('tab', { name: 'Models' }).click()
      await page.getByRole('button', { name: 'Create model' }).click()
      await expect(page.getByRole('heading', { name: 'Create a mixture' })).toBeVisible()
      await expectCenteredDialog(page)
      await expectNoViewportOverflow(page, `/config/entrypoints-recipes model at ${viewport.name}`)
      await page.getByRole('button', { name: 'Close' }).click()
    }
  })

  test('keeps Playground controls unobstructed and the mobile conversation drawer accessible', async ({
    page,
  }) => {
    test.slow()
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await mockResponsiveDashboardData(page)

    for (const viewport of viewports) {
      await page.setViewportSize(viewport)
      await page.goto('/playground/fullscreen')
      await expect(page.getByTestId('agent-playground')).toBeVisible()

      await expectUnobstructedControl(
        page,
        page.getByTestId('playground-composer-add'),
        `${viewport.name} add control`,
      )
      await expectUnobstructedControl(
        page,
        page.getByTestId('playground-composer-model-select'),
        `${viewport.name} model control`,
      )
      await expectUnobstructedControl(
        page,
        page.getByRole('textbox', { name: 'Message' }),
        `${viewport.name} message input`,
      )
      await expectUnobstructedControl(
        page,
        page.getByRole('button', { name: 'Send message' }),
        `${viewport.name} send control`,
      )

      const mobileTrigger = page.getByTestId('agent-mobile-conversation-trigger')
      if (viewport.width < 960) {
        await expectUnobstructedControl(
          page,
          mobileTrigger,
          `${viewport.name} conversation trigger`,
        )
        await expect(page.getByTestId('agent-conversation-sidebar')).toHaveCount(0)
        await mobileTrigger.click()

        const drawer = page.getByRole('dialog', { name: 'Conversations' })
        await expect(drawer).toBeVisible()
        await expect(drawer.getByRole('button', { name: 'Close conversations' })).toBeFocused()
        await page.keyboard.press('Shift+Tab')
        expect(
          await drawer.evaluate((element) => element.contains(document.activeElement)),
          `${viewport.name} drawer traps keyboard focus`,
        ).toBe(true)
        await page.keyboard.press('Escape')
        await expect(drawer).toHaveCount(0)
        await expect(mobileTrigger).toBeFocused()
      } else {
        await expect(mobileTrigger).toBeHidden()
        await expect(page.getByTestId('agent-conversation-sidebar')).toBeVisible()
        await expect(page.getByRole('dialog', { name: 'Conversations' })).toHaveCount(0)
      }
    }
  })

  test('keeps the Topology result dialog centered, bounded, and keyboard accessible', async ({
    page,
  }) => {
    test.slow()
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await mockResponsiveDashboardData(page)
    await mockTopologyPreviewData(page)

    for (const viewport of viewports) {
      await page.setViewportSize(viewport)
      await page.goto('/topology')
      const query = page.getByPlaceholder('Message...')
      await expect(query).toBeVisible()
      await query.fill('Show me the fastest path')
      const preview = page.getByRole('button', { name: 'Preview', exact: true })
      await preview.click()

      const dialog = page.getByRole('dialog', { name: 'Routing result' })
      await expect(dialog).toBeVisible()
      await expect(dialog.getByRole('button', { name: 'Close routing result' })).toBeFocused()
      await expect(dialog).toHaveCSS('border-top-width', '2px')
      await expect(page.getByTestId('topology-result-header')).toHaveCSS('position', 'sticky')
      await expect(page.getByTestId('topology-result-scroll')).toHaveCSS('overflow-y', 'auto')
      await expectCenteredDialog(page)
      await expectNoViewportOverflow(page, `/topology result at ${viewport.name}`)

      await page.keyboard.press('Shift+Tab')
      expect(
        await dialog.evaluate((element) => element.contains(document.activeElement)),
        `${viewport.name} Topology dialog traps keyboard focus`,
      ).toBe(true)
      await page.keyboard.press('Escape')
      await expect(dialog).toHaveCount(0)
      await expect(preview).toBeFocused()
    }
  })

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

  test('consumer routes and actions stay within the key-scoped product boundary', async ({
    page,
  }) => {
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

    for (const target of [
      { path: '/access/usage', heading: 'Usage' },
      { path: '/access/api-keys', heading: 'API Keys' },
      { path: '/config/entrypoints-recipes', heading: 'Mixture-of-Models' },
    ]) {
      await page.goto(target.path)
      await expect(page).toHaveURL(new RegExp(`${target.path.replaceAll('/', '\\/')}$`))
      await expect(
        page.getByRole('heading', { name: target.heading, exact: true }).first(),
      ).toBeVisible()
      await expectNoViewportOverflow(page, target.path)
    }
    await expect(page.getByRole('heading', { name: 'Recipes', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Create recipe' })).toHaveCount(0)
    await page.getByRole('tab', { name: 'Models' }).click()
    await expect(page.getByRole('heading', { name: 'Models', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Create model' })).toHaveCount(0)
    await page.goto('/access/api-keys')
    await expect(page.getByRole('heading', { name: 'API Keys', exact: true }).first()).toBeVisible()
    await expect(page.getByRole('button', { name: 'Create key' })).toHaveCount(0)

    for (const route of [
      '/access/users',
      '/access/teams',
      '/access/access-groups',
      '/access/budgets',
      '/access/audit-logs',
      '/logs',
      '/config/models',
      '/config/signals',
      '/config/projections',
      '/config/decisions',
      '/config/agent',
      '/builder',
      '/evaluation',
      '/fleet-sim',
      '/insights',
      '/ml-setup',
      '/monitoring',
      '/openclaw',
      '/plugins',
      '/status',
      '/tracing',
    ]) {
      await page.goto(route)
      await expect(page).toHaveURL(/\/dashboard$/)
    }
  })

  test('read-only operators can inspect resources without seeing mutation controls', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockResponsiveDashboardData(page)
    await mockAuthenticatedAppShell(page, {
      user: {
        id: 'viewer-1',
        email: 'viewer@example.com',
        name: 'Viewer',
        role: 'read',
        permissions: ['config.read', 'topology.read', 'users.view'],
      },
      managementPermissions: [
        'access_policy.read',
        'audit.read',
        'key.read',
        'log.read',
        'rate_policy.read',
        'routing.read',
        'team.read',
        'usage.read',
        'user.read',
      ],
    })

    for (const target of [
      { path: '/access/api-keys', heading: 'API Keys', action: 'Create key' },
      { path: '/access/users', heading: 'Users', action: 'Invite user' },
      { path: '/access/teams', heading: 'Teams', action: 'New team' },
      { path: '/access/access-groups', heading: 'Access Groups', action: 'New group' },
      { path: '/access/budgets', heading: 'Budgets', action: 'New budget' },
    ]) {
      await page.goto(target.path)
      await expect(page).toHaveURL(new RegExp(`${target.path.replaceAll('/', '\\/')}$`))
      await expect(
        page.getByRole('heading', { name: target.heading, exact: true }).first(),
      ).toBeVisible()
      await expect(page.getByRole('button', { name: target.action })).toHaveCount(0)
    }

    await page.goto('/config/models')
    await expect(page.getByRole('heading', { name: 'Models', exact: true }).first()).toBeVisible()
    await expect(page.getByRole('button', { name: 'Add model' })).toHaveCount(0)
    await expect(page.getByRole('columnheader', { name: 'Live' })).toHaveCount(0)

    await page.goto('/config/entrypoints-recipes')
    await expect(page.getByRole('heading', { name: 'Recipes', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Create recipe' })).toHaveCount(0)
    await page.getByRole('tab', { name: 'Models' }).click()
    await expect(page.getByRole('heading', { name: 'Models', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Create model' })).toHaveCount(0)
  })

  test('product guide stays centered and usable at each breakpoint', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await mockResponsiveDashboardData(page)

    for (const viewport of viewports) {
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

  test('account details stay composed at each breakpoint', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await mockResponsiveDashboardData(page)

    for (const viewport of viewports) {
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
