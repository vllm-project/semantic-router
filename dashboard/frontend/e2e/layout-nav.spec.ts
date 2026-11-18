import { expect, test, type Locator, type Page } from '@playwright/test'
import {
  dashboardSettingsResponse,
  mockAuthenticatedSession,
  unavailableCapability,
} from './support/auth'

const setupState = {
  setupMode: false,
  listenerPort: 8000,
  models: 1,
  decisions: 1,
  hasModels: true,
  hasDecisions: true,
  canActivate: true,
}

const settingsResponse = dashboardSettingsResponse()

const readUser = {
  id: 'user-read-1',
  email: 'viewer@example.com',
  name: 'Viewer User',
  role: 'read',
  permissions: [
    'config.read',
    'evaluation.read',
    'logs.read',
    'mcp.read',
    'tools.use',
    'topology.read',
  ],
}

const minimalPermissionUser = {
  id: 'user-minimal-1',
  email: 'minimal@example.com',
  name: 'Minimal User',
  role: 'read',
  permissions: ['tools.use'],
}

const configResponse = {
  version: 'v0.5',
  listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
  providers: {
    defaults: { model: 'test-model' },
    models: [
      {
        name: 'test-model',
        provider_model_id: 'test-model',
        backend_refs: [
          {
            name: 'endpoint1',
            endpoint: '127.0.0.1:8000',
            protocol: 'http',
          },
        ],
      },
    ],
  },
  recipes: [
    {
      name: 'default',
      signals: {
        domains: [
          { name: 'business', description: 'Business' },
          { name: 'economics', description: 'Economics' },
        ],
        keywords: [
          {
            name: 'business_analysis',
            operator: 'OR',
            keywords: ['revenue', 'margin'],
            case_sensitive: false,
          },
        ],
        context: [{ name: 'long_context', min_tokens: '4K' }],
        complexity: [{ name: 'general_reasoning' }],
      },
      decisions: [
        {
          name: 'medium_business',
          priority: 215,
          rules: {
            operator: 'AND',
            conditions: [
              {
                operator: 'OR',
                conditions: [
                  { type: 'domain', name: 'business' },
                  { type: 'domain', name: 'economics' },
                ],
              },
              {
                operator: 'OR',
                conditions: [
                  { type: 'keyword', name: 'business_analysis' },
                  { type: 'complexity', name: 'general_reasoning:easy' },
                  { type: 'complexity', name: 'general_reasoning:medium' },
                  { type: 'context', name: 'long_context' },
                ],
              },
            ],
          },
          modelRefs: [{ model: 'test-model', use_reasoning: true, reasoning_effort: 'medium' }],
        },
        {
          name: 'business-route',
          priority: 1,
          rules: {
            operator: 'AND',
            conditions: [{ type: 'domain', name: 'business' }],
          },
          modelRefs: [{ model: 'test-model' }],
        },
      ],
    },
  ],
  global: {
    router: { strategy: 'priority' },
    services: {
      response_api: { enabled: true },
    },
  },
}

const rawGlobalYaml = `router:
  strategy: priority
services:
  response_api:
    enabled: true
`

const statusResponse = {
  overall: 'healthy',
  deployment_type: 'local',
  services: [
    {
      name: 'router',
      status: 'healthy',
      healthy: true,
    },
  ],
  models: {
    models: [],
    summary: {
      loaded_models: 0,
      total_models: 0,
    },
  },
}

const replayRecord = {
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 1,
  recipe: 'default',
  lifecycle_state: 'completed',
  response_status: 200,
  duration_ms: 820,
}

const replayRecordsResponse = {
  object: 'router_replay.list',
  count: 2,
  total: 2,
  limit: 100,
  offset: 0,
  has_more: false,
  data: [
    {
      ...replayRecord,
      id: 'replay-1',
      timestamp: '2026-03-17T10:00:00Z',
      request_id: 'req-1',
      decision: 'business-route',
      original_model: 'test-model',
      selected_model: 'test-model',
      reasoning_mode: 'off',
      selection_method: 'static',
      signals: {
        domain: ['business'],
        keyword: ['finance'],
      },
      from_cache: false,
      streaming: false,
      prompt_tokens: 1000,
      completion_tokens: 500,
      total_tokens: 1500,
      actual_cost: 0.002,
      baseline_cost: 0.008,
      cost_savings: 0.006,
      currency: 'USD',
      baseline_model: 'premium-model',
    },
    {
      ...replayRecord,
      id: 'replay-2',
      timestamp: '2026-03-17T10:05:00Z',
      request_id: 'req-2',
      decision: 'business-route',
      original_model: 'test-model',
      selected_model: 'test-model',
      reasoning_mode: 'on',
      selection_method: 'static',
      signals: {
        domain: ['business'],
        preference: ['low-latency'],
      },
      from_cache: true,
      streaming: true,
    },
  ],
}

const replayAggregateResponse = {
  object: 'router_replay.aggregate',
  record_count: 2,
  lifecycle: { completed: 2, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
  summary: {
    total_saved: 0.006,
    baseline_spend: 0.008,
    actual_spend: 0.002,
    currency: 'USD',
    cost_record_count: 1,
    excluded_record_count: 1,
  },
  model_selection: [{ name: 'test-model', value: 2 }],
  decision_distribution: [{ name: 'business-route', value: 2 }],
  signal_distribution: [
    { name: 'domain', value: 2 },
    { name: 'keyword', value: 1 },
    { name: 'preference', value: 1 },
  ],
  token_volume: {
    input_tokens: 1000,
    output_tokens: 500,
    total_tokens: 1500,
    excluded_record_count: 1,
  },
  token_breakdown: {
    by_decision: [
      {
        name: 'business-route',
        input_tokens: 1000,
        output_tokens: 500,
        total_tokens: 1500,
      },
    ],
    by_selected_model: [
      {
        name: 'test-model',
        input_tokens: 1000,
        output_tokens: 500,
        total_tokens: 1500,
      },
    ],
  },
  available_recipes: ['default'],
  available_decisions: ['business-route'],
  available_models: ['test-model'],
}

async function mockCommon(
  page: Page,
  options: {
    user?: {
      id: string
      email: string
      name: string
      role?: string
      permissions?: string[]
    }
    settings?: Record<string, unknown>
    capabilities?: Record<string, Record<string, unknown>>
  } = {},
) {
  await mockAuthenticatedSession(page, options)

  await page.route('**/api/setup/state', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(setupState),
    })
  })

  await page.route('**/api/settings', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ...settingsResponse, ...(options.settings ?? {}) }),
    })
  })

  await page.route('**/api/mcp/tools', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ tools: [] }),
    })
  })

  await page.route('**/api/mcp/servers', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) })
  })

  await page.route('**/api/router/config/all', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(configResponse),
    })
  })

  await page.route('**/api/router/config/global', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(configResponse.global),
    })
  })

  await page.route('**/api/router/config/global/raw', async (route) => {
    await route.fulfill({ status: 200, contentType: 'text/yaml', body: rawGlobalYaml })
  })

  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(statusResponse),
    })
  })

  await page.route('**/api/auth/bootstrap/can-register', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ canRegister: false }),
    })
  })

  await page.route(/\/api\/router\/v1\/router_replay(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(replayRecordsResponse),
    })
  })

  await page.route(/\/api\/router\/v1\/router_replay\/aggregate(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(replayAggregateResponse),
    })
  })

  await page.route(/\/api\/router\/api\/v1\/observability\/replays(?:\?.*)?$/, async (route) => {
    await route.fulfill({ json: replayRecordsResponse })
  })

  await page.route('**/api/router/api/v1/observability/replays/*', async (route) => {
    const requestURL = new URL(route.request().url())
    const replayID = requestURL.pathname.split('/').pop()
    if (replayID === 'aggregate') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(replayAggregateResponse),
      })
      return
    }
    const record = replayRecordsResponse.data.find((item) => item.id === replayID)

    await route.fulfill({
      status: record ? 200 : 404,
      contentType: 'application/json',
      body: JSON.stringify(record ?? { error: { message: 'not found' } }),
    })
  })
}

async function expectInsideViewport(page: Page, locator: Locator) {
  await expect(locator).toBeVisible()
  const bounds = await locator.boundingBox()
  const viewport = page.viewportSize()
  expect(bounds).not.toBeNull()
  expect(viewport).not.toBeNull()
  expect(bounds!.x).toBeGreaterThanOrEqual(0)
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport!.width)
}

async function expectNoHorizontalScroll(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
}

const sidebar = (page: Page) => page.getByRole('navigation', { name: 'Primary' })
const settingsNav = (page: Page) => page.getByRole('navigation', { name: 'Settings', exact: true })
// The sidebar marks the item that owns the current page with an accent bar.
const activeBar = (link: Locator) => link.locator('span.bg-primary')

test.describe('Shell navigation', () => {
  test('lists the three decision model views as tabs on one desktop row', async ({ page }) => {
    await mockCommon(page)
    for (const width of [1024, 1440]) {
      await page.setViewportSize({ width, height: 900 })
      await page.goto('/models/decision')
      const tabs = page.getByRole('tablist').first().getByRole('tab')
      await expect(tabs).toHaveText([/Deploy/, /Playground/, /Monitoring/])
      const boxes = await tabs.evaluateAll((elements) =>
        elements.map((element) => {
          const { x, y, right } = element.getBoundingClientRect()
          return { x, y, right }
        }),
      )
      expect(
        Math.max(...boxes.map((box) => box.y)) - Math.min(...boxes.map((box) => box.y)),
      ).toBeLessThan(2)
      expect(boxes[0].right).toBeLessThanOrEqual(boxes[1].x)
      expect(boxes[1].right).toBeLessThanOrEqual(boxes[2].x)
      expect(boxes[2].right).toBeLessThan(width)
    }
  })

  test('hovering navigation does not fetch an unvisited Model Hub route', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)
    const hubRequests: string[] = []
    page.on('request', (request) => {
      if (/ModelHubPage|\/api\/models\/catalog|model_catalog\.json/.test(request.url()))
        hubRequests.push(request.url())
    })
    await page.goto('/routing/signals')
    const modelHub = sidebar(page).getByRole('link', { name: 'Model Hub', exact: true })
    await modelHub.hover()
    // A hover used to trigger an import while a user was only scanning menus.
    await page.waitForTimeout(350)
    expect(hubRequests).toEqual([])
    await expect(page).toHaveURL(/\/routing\/signals$/)
  })

  test('hides Evaluations and explains direct access when the service is unavailable', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page, {
      capabilities: {
        evaluation: unavailableCapability('Evaluation is disabled for this deployment.'),
      },
    })
    await page.goto('/routing/signals')

    await expect(sidebar(page).getByRole('link', { name: 'Status' })).toBeVisible()
    await expect(sidebar(page).getByRole('link', { name: 'Evaluations' })).toHaveCount(0)

    await page.goto('/evaluation')
    await expect(page).toHaveURL(/\/observe\/evaluations$/)
    await expect(page.getByRole('heading', { name: 'Evaluation is not available' })).toBeVisible()
    await expect(page.getByText('Evaluation is disabled for this deployment.')).toBeVisible()
  })

  test('keeps every mobile header control inside 320px and 375px viewports', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 700 })
    await mockCommon(page)

    await page.goto('/routing/signals')

    for (const width of [320, 375]) {
      await page.setViewportSize({ width, height: 700 })
      for (const name of ['Open navigation', 'Search', 'Notifications', 'Toggle theme', 'Help']) {
        await expectInsideViewport(page, page.getByRole('button', { name, exact: true }))
      }
      await expectNoHorizontalScroll(page)
    }
  })

  test('switches to the complete navigation drawer before tablet controls clip', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 900, height: 700 })
    await mockCommon(page)

    await page.goto('/routing/signals')

    await expect(sidebar(page)).toBeHidden()
    const menuButton = page.getByRole('button', { name: 'Open navigation' })
    await expect(menuButton).toBeVisible()
    await expectNoHorizontalScroll(page)

    await menuButton.click()
    const drawer = page.getByRole('dialog', { name: 'Navigation' })
    await expect(drawer).toBeVisible()
    const navigation = drawer.getByRole('navigation', { name: 'Primary' })
    for (const group of ['Routing', 'Models', 'Observe']) {
      await expect(navigation.getByText(group, { exact: true })).toBeVisible()
    }
    for (const label of [
      'Home',
      'Playground',
      'Recipes',
      'Config editor',
      'Backend models',
      'Decision models',
      'Requests',
      'Traces',
    ]) {
      await expect(navigation.getByRole('link', { name: label, exact: true })).toBeVisible()
    }
    await expect(drawer.getByRole('link', { name: 'Settings' })).toBeVisible()
    const activeLink = navigation.getByRole('link', { name: 'Signals' })
    await expect(activeLink).toHaveAttribute('aria-current', 'page')
    expect(
      await activeLink.evaluate(() => document.activeElement?.closest('[role="dialog"]') !== null),
    ).toBe(true)

    await page.keyboard.press('Escape')
    await expect(drawer).toBeHidden()
    await expect(menuButton).toBeFocused()

    await menuButton.click()
    await expect(drawer).toBeVisible()
    await navigation.getByRole('link', { name: 'Decisions' }).click()
    await expect(page).toHaveURL(/\/routing\/decisions$/)
    await expect(drawer).toBeHidden()
  })

  test('omits empty navigation groups after permission filtering', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page, { user: minimalPermissionUser })

    await page.goto('/playground')

    const primary = sidebar(page)
    await expect(primary.getByRole('link')).toHaveText(['Home', 'Get started', 'Playground'])
    for (const group of ['Routing', 'Models', 'Observe']) {
      await expect(primary.getByText(group, { exact: true })).toHaveCount(0)
    }
    await expect(page.getByRole('link', { name: 'Settings', exact: true })).toHaveCount(0)

    await page.setViewportSize({ width: 900, height: 700 })
    await page.getByRole('button', { name: 'Open navigation' }).click()
    const drawer = page.getByRole('dialog', { name: 'Navigation' })
    await expect(drawer.getByRole('navigation', { name: 'Primary' }).getByRole('link')).toHaveCount(
      3,
    )
    await expect(drawer.getByRole('link', { name: 'Settings', exact: true })).toHaveCount(0)
  })

  test('organizes the control plane into routing, models, observe and settings', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/playground')

    const primary = sidebar(page)
    await expect(primary.getByRole('link')).toHaveText([
      'Home',
      'Get started',
      'Playground',
      'Recipes',
      'Decisions',
      'Signals',
      'Projections',
      'Topology',
      'Config editor',
      'Backend models',
      'Decision models',
      'Model Hub',
      'Requests',
      'Evaluations',
      'Status',
      'Logs',
      'Metrics',
      'Traces',
    ])
    for (const group of ['Routing', 'Models', 'Observe']) {
      await expect(primary.getByText(group, { exact: true })).toBeVisible()
    }
    const playground = primary.getByRole('link', { name: 'Playground' })
    await expect(playground).toHaveAttribute('aria-current', 'page')
    await expect(activeBar(playground)).toHaveCount(1)
    await expect(activeBar(primary.getByRole('link', { name: 'Home' }))).toHaveCount(0)
    await expect(page.getByRole('navigation', { name: 'Breadcrumb' })).toContainText('Playground')

    await playground.focus()
    await page.keyboard.press('Tab')
    await expect(primary.getByRole('link', { name: 'Recipes' })).toBeFocused()

    const sidebarBounds = await primary.boundingBox()
    expect(sidebarBounds).not.toBeNull()
    expect(sidebarBounds!.x).toBeGreaterThanOrEqual(0)
    expect(sidebarBounds!.x + sidebarBounds!.width).toBeLessThanOrEqual(1440)

    await primary.getByRole('link', { name: 'Decision models' }).click()
    await expect(page).toHaveURL(/\/models\/decision$/)
    await expect(page.getByRole('heading', { name: 'Decision models', exact: true })).toBeVisible()
    for (const [tab, query] of [
      ['Playground', 'playground'],
      ['Monitoring', 'monitoring'],
    ] as const) {
      await page.getByRole('tab', { name: tab }).click()
      await expect(page).toHaveURL(new RegExp(`/models/decision\\?tab=${query}$`))
      await expect(page.getByRole('tab', { name: tab })).toHaveAttribute('aria-selected', 'true')
    }
    for (const [legacy, query] of [
      ['/decision-model/playground', 'playground'],
      ['/decision-model/monitoring', 'monitoring'],
    ] as const) {
      await page.goto(legacy)
      await expect(page).toHaveURL(new RegExp(`/models/decision\\?tab=${query}$`))
    }

    await page.getByRole('link', { name: 'Settings', exact: true }).click()
    await expect(page).toHaveURL(/\/settings\/router$/)
    await expect(settingsNav(page).getByRole('link')).toHaveText([
      'Router settings',
      'Integrations',
      'Users & access',
      'Schema reference',
    ])
  })

  test('navigates from the command menu and closes it', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/routing/signals')

    await page.getByRole('button', { name: /Search or jump to/ }).click()
    const commandMenu = page.getByRole('dialog')
    await expect(commandMenu).toBeVisible()
    await page.getByPlaceholder('Search pages, recipes, decisions, models…').fill('Playground')
    await commandMenu.getByRole('option', { name: /^Playground/ }).click()
    await expect(page).toHaveURL(/\/playground$/)
    await expect(commandMenu).toBeHidden()

    await page.keyboard.press('ControlOrMeta+k')
    await expect(commandMenu).toBeVisible()
    await page.getByPlaceholder('Search pages, recipes, decisions, models…').fill('Router settings')
    await commandMenu.getByRole('option', { name: /^Router settings/ }).click()
    await expect(page).toHaveURL(/\/settings\/router$/)
    await expect(commandMenu).toBeHidden()

    await page.keyboard.press('ControlOrMeta+k')
    await expect(commandMenu).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(commandMenu).toBeHidden()
    await expect(page).toHaveURL(/\/settings\/router$/)
  })

  test('navigates every configuration page through its canonical URL', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/playground')

    await sidebar(page).getByRole('link', { name: 'Signals' }).click()
    await expect(page).toHaveURL(/\/routing\/signals$/)
    await sidebar(page).getByRole('link', { name: 'Decisions' }).click()
    await expect(page).toHaveURL(/\/routing\/decisions$/)

    await page.getByRole('link', { name: 'Settings', exact: true }).click()
    await expect(page).toHaveURL(/\/settings\/router$/)
    for (const [label, path] of [
      ['Integrations', '/settings/integrations'],
      ['Users & access', '/settings/access'],
      ['Schema reference', '/settings/schema'],
      ['Router settings', '/settings/router'],
    ] as const) {
      await settingsNav(page).getByRole('link', { name: label }).click()
      await expect(page).toHaveURL(new RegExp(`${path}$`))
    }

    for (const [legacy, path] of [
      ['/config/signals', '/routing/signals'],
      ['/config/decisions', '/routing/decisions'],
      ['/config/global-config', '/settings/router'],
      ['/config/mcp', '/settings/integrations'],
    ] as const) {
      await page.goto(legacy)
      await expect(page).toHaveURL(new RegExp(`${path}$`))
    }
  })

  test('keeps the desktop sidebar inside the viewport at 1024px', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 })
    await mockCommon(page)

    await page.goto('/routing/signals')

    await expectInsideViewport(page, sidebar(page))
    await expect(page.getByRole('button', { name: 'Open navigation' })).toBeHidden()
    await expectNoHorizontalScroll(page)
  })

  test('frames the page in one balanced shell that the sidebar can collapse', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await page.setViewportSize({ width: 2048, height: 1152 })
    await mockCommon(page)

    await page.goto('/routing/signals')

    const panel = page.locator('main#main').locator('..')
    const header = page.locator('header').first()
    const frame = async () => {
      const box = await panel.boundingBox()
      return box
        ? {
            left: Math.round(box.x),
            top: Math.round(box.y),
            right: Math.round(2048 - box.x - box.width),
            bottom: Math.round(1152 - box.y - box.height),
          }
        : null
    }
    await expect.poll(frame).toEqual({ left: 248, top: 8, right: 8, bottom: 8 })
    expect((await header.boundingBox())?.height).toBe(52)
    await expectNoHorizontalScroll(page)

    await page.getByRole('button', { name: 'Collapse sidebar' }).click()
    await expect.poll(async () => (await frame())?.left).toBe(60)
    await page.getByRole('button', { name: 'Expand sidebar' }).click()
    await expect.poll(async () => (await frame())?.left).toBe(248)
  })

  test('switches to compact navigation at 961px with full access and a long account name', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 961, height: 768 })
    await mockCommon(page, {
      user: {
        id: 'user-long-name',
        email: 'alexandria.montgomery@example.com',
        name: 'Alexandria Catherine Montgomery',
        role: 'admin',
        permissions: [
          'config.deploy',
          'config.read',
          'config.write',
          'evaluation.read',
          'evaluation.run',
          'evaluation.write',
          'feedback.submit',
          'logs.read',
          'mcp.manage',
          'mcp.read',
          'mlpipeline.manage',
          'replay.read',
          'tools.use',
          'topology.read',
          'users.manage',
          'users.view',
        ],
      },
    })

    await page.goto('/routing/signals')

    await expect(sidebar(page)).toBeHidden()
    for (const name of ['Open navigation', 'Notifications', 'Toggle theme', 'Help']) {
      await expectInsideViewport(page, page.getByRole('button', { name, exact: true }))
    }
    await expectNoHorizontalScroll(page)

    await page.getByRole('button', { name: 'Open navigation' }).click()
    const drawer = page.getByRole('dialog', { name: 'Navigation' })
    const account = drawer.getByRole('button', { name: /Alexandria Catherine Montgomery/ })
    await expect(account).toBeVisible()
    const [drawerBox, accountBox] = await Promise.all([drawer.boundingBox(), account.boundingBox()])
    expect(drawerBox).not.toBeNull()
    expect(accountBox).not.toBeNull()
    expect(accountBox!.x).toBeGreaterThanOrEqual(drawerBox!.x)
    expect(accountBox!.x + accountBox!.width).toBeLessThanOrEqual(drawerBox!.x + drawerBox!.width)
  })

  test('keeps Settings selected while moving between settings pages', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/builder')
    await expect(page).toHaveURL(/\/routing\/editor$/)
    await expect(sidebar(page).getByRole('link', { name: 'Config editor' })).toHaveAttribute(
      'aria-current',
      'page',
    )

    const settingsLink = page.getByRole('link', { name: 'Settings', exact: true })
    await settingsLink.click()
    await expect(settingsNav(page).getByRole('link', { name: 'Router settings' })).toHaveAttribute(
      'aria-current',
      'page',
    )

    await settingsNav(page).getByRole('link', { name: 'Integrations' }).click()
    await expect(page).toHaveURL(/\/settings\/integrations$/)
    await expect(settingsNav(page).getByRole('link', { name: 'Integrations' })).toHaveAttribute(
      'aria-current',
      'page',
    )
    await expect(
      settingsNav(page).getByRole('link', { name: 'Router settings' }),
    ).not.toHaveAttribute('aria-current', 'page')
    await expect(activeBar(settingsLink)).toHaveCount(1)
    await expect(activeBar(sidebar(page).getByRole('link', { name: 'Config editor' }))).toHaveCount(
      0,
    )
  })

  test('keeps drawer links clickable when Safari reports a null blur target', async ({ page }) => {
    await page.setViewportSize({ width: 900, height: 700 })
    await mockCommon(page)

    await page.goto('/routing/signals')
    await page.getByRole('button', { name: 'Open navigation' }).click()

    const drawer = page.getByRole('dialog', { name: 'Navigation' })
    await expect(drawer).toBeVisible()
    await drawer.evaluate((element) => {
      element.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: null }))
    })

    await expect(drawer).toBeVisible()
    await drawer.getByRole('link', { name: 'Config editor' }).click()
    await expect(page).toHaveURL(/\/routing\/editor$/)
  })

  test('keeps every navigation group reachable in a short desktop window', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 400 })
    await mockCommon(page)

    await page.goto('/routing/signals')

    const traces = sidebar(page).getByRole('link', { name: 'Traces' })
    await traces.scrollIntoViewIfNeeded()
    await expect(traces).toBeInViewport()
    await expect(page.getByRole('link', { name: 'Settings', exact: true })).toBeInViewport()
    await expect(page.getByRole('button', { name: /Admin User/ })).toBeInViewport()
    await traces.click()
    await expect(page).toHaveURL(/\/observe\/traces$/)
  })

  test('opens Requests from the sidebar with its summary and keeps runs under it', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/routing/signals')
    await sidebar(page).getByRole('link', { name: 'Requests' }).click()

    await expect(page).toHaveURL(/\/observe\/requests$/)
    await expect(page.getByRole('heading', { name: 'Requests', exact: true })).toBeVisible()
    const summary = page.getByRole('region', { name: 'Requests summary' })
    for (const label of ['Requests', 'Estimated spend', 'Saved by routing', 'Tokens', 'Failed']) {
      await expect(summary.getByText(label, { exact: true })).toBeVisible()
    }
    await expect(page.getByRole('columnheader', { name: 'Decision' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: 'Cost' })).toBeVisible()
    await expect(page.getByRole('row', { name: /req-1/ })).toBeVisible()
    await expect(page.getByTitle('business-route')).toHaveCount(2)

    await page.goto('/insights')
    await expect(page).toHaveURL(/\/observe\/requests$/)
    await page.goto('/insights/run?session=session-1')
    await expect(page).toHaveURL(/\/observe\/runs\?session=session-1$/)
    await expect(activeBar(sidebar(page).getByRole('link', { name: 'Requests' }))).toHaveCount(1)

    await page.goto('/replay')
    await expect(page.getByRole('heading', { name: "This page doesn't exist" })).toBeVisible()
  })

  test('shows nested canonical decision groups in the topology inspector', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/topology')
    await expect(page).toHaveURL(/\/routing\/topology$/)

    await page.getByTestId('rf__node-decision-medium_business').click()
    const inspector = page
      .getByRole('button', { name: 'Close details' })
      .locator('xpath=ancestor::aside[1]')
    await expect(inspector).toContainText('medium_business')
    await expect(inspector).toContainText('WHEN')
    await expect(inspector).toContainText('(domain("business") OR domain("economics")) AND (')
    await expect(inspector).toContainText('keyword("business_analysis")')
    await expect(inspector).toContainText('complexity("general_reasoning:medium")')
    await expect(inspector).toContainText('context("long_context")')
    await expect(inspector).toContainText('P215')
    await expect(inspector.getByText('Referenced signals not configured')).toHaveCount(0)
  })

  test('retired knowledge routes show the not-found page without fetching KB or embedded-map resources', async ({
    page,
  }) => {
    await mockCommon(page)
    const featureRequests: string[] = []
    page.on('request', (request) => {
      if (
        /\/api\/.*knowledge-bases|\/embedded\/wizmap|KnowledgeMapPage|TaxonomyPage/.test(
          request.url(),
        )
      ) {
        featureRequests.push(request.url())
      }
    })
    for (const path of [
      '/knowledge-bases/bases',
      '/knowledge-bases/privacy_kb/map',
      '/taxonomy/labels',
    ]) {
      await page.goto(path)
      await expect(page.getByRole('heading', { name: "This page doesn't exist" })).toBeVisible()
      await expect(page.getByRole('button', { name: 'Go to Home' })).toBeVisible()
      await expect(page.locator('iframe')).toHaveCount(0)
    }
    expect(featureRequests).toEqual([])
  })

  test('opens Integrations from Settings with its existing permission and URL', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page, { user: readUser })
    await page.goto('/routing/signals')
    await page.getByRole('link', { name: 'Settings', exact: true }).click()
    await settingsNav(page).getByRole('link', { name: 'Integrations' }).click()
    await expect(page).toHaveURL(/\/settings\/integrations$/)
    await expect(
      page.getByRole('heading', { name: 'Integrations', exact: true }).first(),
    ).toBeVisible()
  })

  test('hides admin pages from read users', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page, { user: readUser })

    await page.goto('/settings/router')

    await expect(settingsNav(page).getByRole('link', { name: 'Users & access' })).toHaveCount(0)
    await expect(settingsNav(page).getByRole('link', { name: 'Router settings' })).toBeVisible()
    await expect(settingsNav(page).getByRole('link', { name: 'Integrations' })).toBeVisible()

    await page.keyboard.press('ControlOrMeta+k')
    const commandMenu = page.getByRole('dialog')
    await expect(commandMenu.getByRole('option', { name: /^Integrations/ })).toBeVisible()
    await expect(commandMenu.getByRole('option', { name: /^Users & access/ })).toHaveCount(0)
  })

  test('opens the anchored sidebar account menu and signs out from it', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page, {
      user: {
        id: 'user-admin-2',
        email: 'ada@example.com',
        name: 'Ada Lovelace',
        role: 'admin',
        permissions: ['config.read', 'config.write', 'users.manage'],
      },
    })

    await page.goto('/routing/signals')

    const accountTrigger = page.getByRole('button', { name: /Ada Lovelace/ })
    await expect(accountTrigger).toHaveAttribute('aria-haspopup', 'menu')
    const triggerBox = await accountTrigger.boundingBox()
    await accountTrigger.click()

    const accountMenu = page.getByRole('menu')
    await expect(accountMenu).toBeVisible()
    await expect(accountMenu.getByText('Ada Lovelace')).toBeVisible()
    await expect(accountMenu.getByText('ada@example.com')).toBeVisible()
    await expect(accountMenu.getByText('admin', { exact: true })).toBeVisible()
    await expect(accountMenu.getByRole('menuitem', { name: 'Sign out' })).toBeVisible()

    const menuBox = await accountMenu.boundingBox()
    const viewport = page.viewportSize()
    expect(menuBox).not.toBeNull()
    expect(triggerBox).not.toBeNull()
    expect(menuBox!.x).toBeLessThan(viewport!.width / 2)
    expect(menuBox!.y + menuBox!.height).toBeLessThanOrEqual(triggerBox!.y + 1)
    expect(menuBox!.width).toBeLessThanOrEqual(400)

    await page.keyboard.press('ArrowDown')
    await expect(accountMenu.getByRole('menuitem').first()).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(accountMenu).toBeHidden()
    await expect(accountTrigger).toBeFocused()

    await accountTrigger.click()
    await accountMenu.getByRole('menuitem', { name: 'Sign out' }).click()

    await expect(page).toHaveURL(/\/login$/)
    await expect(page.getByRole('heading', { name: 'Sign in to Semantic Router' })).toBeVisible()
  })

  test('sends /config to router settings and loads the effective global YAML', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/config')

    await expect(page).toHaveURL(/\/settings\/router$/)
    await expect(
      page.getByRole('heading', { name: 'Router settings', exact: true }).first(),
    ).toBeVisible()

    await page.getByRole('radio', { name: 'YAML' }).click()
    await page.getByRole('button', { name: 'Edit effective YAML' }).click()
    const dialog = page.getByRole('dialog', { name: 'Effective global YAML' })
    await expect(dialog).toBeVisible()
    await expect(dialog.getByRole('textbox', { name: 'Effective global YAML' })).toHaveValue(
      rawGlobalYaml,
    )
  })
})
