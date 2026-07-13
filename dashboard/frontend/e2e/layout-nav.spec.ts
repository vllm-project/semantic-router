import { expect, test, type Locator, type Page } from '@playwright/test'
import { mockAuthenticatedSession } from './support/auth'

const setupState = {
  setupMode: false,
  listenerPort: 8000,
  models: 1,
  decisions: 1,
  hasModels: true,
  hasDecisions: true,
  canActivate: true,
}

const settingsResponse = {
  readonlyMode: false,
  setupMode: false,
  platform: '',
  envoyUrl: '',
  fleetSimEnabled: true,
}

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
    'openclaw.read',
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
  version: 'v0.3',
  listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
  providers: {
    defaults: {
      default_model: 'test-model',
      reasoning_families: {
        qwen3: {
          type: 'reasoning_effort',
          parameter: 'reasoning_effort',
        },
      },
    },
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
  routing: {
    modelCards: [{ name: 'test-model' }],
    signals: {
      domains: [
        { name: 'business', description: 'Business' },
        { name: 'economics', description: 'Economics' },
      ],
      embeddings: [
        {
          name: 'business_analysis',
          threshold: 0.72,
          candidates: ['business'],
          aggregation_method: 'max',
        },
      ],
      preferences: [
        {
          name: 'structured_delivery',
          threshold: 0.55,
          examples: ['Use a structured answer'],
        },
      ],
      complexity: [
        { name: 'general_reasoning:easy', threshold: 0.25 },
        { name: 'general_reasoning:medium', threshold: 0.5 },
      ],
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
                { type: 'embedding', name: 'business_analysis' },
                { type: 'complexity', name: 'general_reasoning:easy' },
                { type: 'complexity', name: 'general_reasoning:medium' },
                { type: 'preference', name: 'structured_delivery' },
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

const replayRecordsResponse = {
  object: 'router_replay.list',
  count: 2,
  total: 2,
  limit: 100,
  offset: 0,
  has_more: false,
  data: [
    {
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
      response_status: 200,
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
      response_status: 200,
      from_cache: true,
      streaming: true,
    },
  ],
}

const replayAggregateResponse = {
  object: 'router_replay.aggregate',
  record_count: 2,
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
      body: JSON.stringify(settingsResponse),
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

  await page.route('**/api/router/v1/router_replay/*', async (route) => {
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

async function expectBalancedDesktopFrame(page: Page, locator: Locator) {
  await expect(locator).toBeVisible()
  const bounds = await locator.boundingBox()
  const viewport = page.viewportSize()
  const documentFrame = await page.evaluate(() => {
    const rect = document.documentElement.getBoundingClientRect()
    return { left: rect.left, right: rect.right, width: rect.width }
  })
  expect(bounds).not.toBeNull()
  expect(viewport).not.toBeNull()

  const leftGutter = bounds!.x - documentFrame.left
  const rightGutter = documentFrame.right - (bounds!.x + bounds!.width)
  expect(Math.abs(leftGutter - rightGutter)).toBeLessThanOrEqual(2)
  expect(leftGutter).toBeGreaterThanOrEqual(24)
  expect(leftGutter).toBeLessThanOrEqual(72)
  expect(bounds!.width / documentFrame.width).toBeGreaterThanOrEqual(0.9)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )

  return bounds!
}

test.describe('Layout top navigation', () => {
  test('keeps every mobile header control inside 320px and 375px viewports', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 700 })
    await mockCommon(page)

    await page.goto('/dashboard')

    for (const width of [320, 375]) {
      await page.setViewportSize({ width, height: 700 })
      await expectInsideViewport(page, page.getByRole('link', { name: 'Documentation' }))
      await expectInsideViewport(page, page.getByRole('link', { name: 'GitHub' }))
      await expectInsideViewport(page, page.getByRole('button', { name: /Open account menu/i }))
      await expectInsideViewport(page, page.getByRole('button', { name: 'Toggle menu' }))
    }
  })

  test('switches to the complete mobile navigation before tablet controls clip', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 900, height: 700 })
    await mockCommon(page)

    await page.goto('/dashboard')

    await expect(page.getByRole('navigation', { name: 'Global navigation' })).toBeHidden()
    const menuButton = page.getByRole('button', { name: 'Toggle menu' })
    await expect(menuButton).toBeVisible()
    await expect(menuButton).toHaveAttribute('aria-controls', 'mobile-navigation')
    await expect(menuButton).toHaveAttribute('aria-expanded', 'false')
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)

    await menuButton.click()
    await expect(menuButton).toHaveAttribute('aria-expanded', 'true')
    const mobileNavigation = page.getByRole('navigation', { name: 'Mobile navigation' })
    await expect(mobileNavigation).toHaveAttribute('id', 'mobile-navigation')
    const mobileDashboardLink = mobileNavigation.getByRole('link', { name: 'Dashboard' })
    await expect(mobileDashboardLink).toHaveAttribute('aria-current', 'page')
    await expect(mobileDashboardLink).toBeFocused()
    await page.keyboard.press('ArrowDown')
    await expect(mobileNavigation.getByRole('link', { name: 'Playground' })).toBeFocused()
    await page.keyboard.press('Home')
    await expect(mobileDashboardLink).toBeFocused()
    await page.keyboard.press('End')
    await expect(mobileNavigation.getByRole('button', { name: 'Operate' })).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(mobileNavigation).toBeHidden()
    await expect(menuButton).toBeFocused()

    await menuButton.click()
    await expect(mobileNavigation).toBeVisible()
    expect(
      await mobileDashboardLink.evaluate(
        (element) => window.getComputedStyle(element).backgroundColor,
      ),
    ).not.toBe('rgba(0, 0, 0, 0)')
    const buildToggle = mobileNavigation.getByRole('button', { name: 'Build' })
    const analyzeToggle = mobileNavigation.getByRole('button', { name: 'Analyze' })
    const operateToggle = mobileNavigation.getByRole('button', { name: 'Operate' })
    await expect(buildToggle).toBeVisible()
    await expect(analyzeToggle).toBeVisible()
    await expect(operateToggle).toBeVisible()
    await expect(buildToggle).toHaveAttribute('aria-expanded', 'false')

    await buildToggle.click()
    await expect(buildToggle).toHaveAttribute('aria-expanded', 'true')
    await expect(mobileNavigation.getByText('Routing', { exact: true })).toBeVisible()
    await expect(mobileNavigation.getByText('Knowledge', { exact: true })).toBeVisible()
    await expect(mobileNavigation.getByText('Integrations & Policy', { exact: true })).toBeVisible()
    await expect(mobileNavigation.getByRole('link', { name: 'Config Builder' })).toBeVisible()
  })

  test('omits empty workflow navigation after permission filtering', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page, { user: minimalPermissionUser })

    await page.goto('/dashboard')

    await expect(page.getByRole('group', { name: 'Workflow navigation' })).toHaveCount(0)
    await expect(page.getByRole('button', { name: /^(Build|Analyze|Operate)$/ })).toHaveCount(0)

    await page.setViewportSize({ width: 900, height: 700 })
    const menuButton = page.getByRole('button', { name: 'Toggle menu' })
    await menuButton.click()

    const mobileNavigation = page.getByRole('navigation', { name: 'Mobile navigation' })
    await expect(
      mobileNavigation.getByRole('button', { name: /^(Build|Analyze|Operate)$/ }),
    ).toHaveCount(0)
    await expect(mobileNavigation.getByRole('link')).toHaveCount(2)
    await page.keyboard.press('End')
    await expect(mobileNavigation.getByRole('link', { name: 'Playground' })).toBeFocused()
  })

  test('organizes the control plane around Build, Analyze, and Operate mega navigation', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/playground')

    const primaryGroup = page.getByRole('group', { name: 'Primary navigation' })
    const workflowGroup = page.getByRole('group', { name: 'Workflow navigation' })

    await expect(primaryGroup.getByRole('link', { name: 'Dashboard' })).toBeVisible()
    await expect(primaryGroup.getByRole('link', { name: 'Playground' })).toBeVisible()
    await expect(primaryGroup.getByRole('link')).toHaveCount(2)

    const buildTrigger = workflowGroup.getByRole('button', { name: 'Build' })
    const analyzeTrigger = workflowGroup.getByRole('button', { name: 'Analyze' })
    const operateTrigger = workflowGroup.getByRole('button', { name: 'Operate' })
    await expect(buildTrigger).toHaveAttribute('aria-controls', 'layout-mega-menu-build')
    await expect(analyzeTrigger).toHaveAttribute('aria-controls', 'layout-mega-menu-analyze')
    await expect(operateTrigger).toHaveAttribute('aria-controls', 'layout-mega-menu-operate')
    await expect(buildTrigger).not.toHaveAttribute('aria-haspopup')
    await expect(workflowGroup.getByRole('button')).toHaveCount(3)

    const header = page.locator('header').first()
    await expect(header).toHaveCSS('background-color', 'rgb(0, 0, 0)')

    const playgroundLink = primaryGroup.getByRole('link', { name: 'Playground' })
    await expect(playgroundLink).toHaveCSS('background-color', 'rgba(0, 0, 0, 0)')
    const activeUnderline = await playgroundLink.evaluate((element) => {
      const underline = window.getComputedStyle(element, '::after')
      return { backgroundColor: underline.backgroundColor, height: underline.height }
    })
    expect(activeUnderline).toEqual({ backgroundColor: 'rgb(227, 27, 35)', height: '2px' })

    await buildTrigger.hover()
    await expect(buildTrigger).toHaveCSS('background-color', 'rgba(0, 0, 0, 0)')
    await buildTrigger.click()

    const buildMenu = page.getByRole('navigation', { name: 'Build' })
    await expect(buildMenu).toHaveAttribute('id', 'layout-mega-menu-build')
    await expect(buildTrigger).toHaveAttribute('aria-expanded', 'true')
    await expect(buildMenu.getByTestId('layout-mega-menu-rail')).toHaveCSS(
      'background-color',
      'rgb(8, 8, 8)',
    )
    await expect(buildMenu.getByTestId('layout-mega-menu-content')).toHaveCSS(
      'background-color',
      'rgb(244, 244, 241)',
    )
    const routingTab = buildMenu.getByRole('tab', { name: /Routing/ })
    const knowledgeTab = buildMenu.getByRole('tab', { name: /Knowledge/ })
    const integrationsTab = buildMenu.getByRole('tab', { name: /Integrations & Policy/ })
    await expect(routingTab).toBeFocused()
    await expect(routingTab).toHaveAttribute('aria-selected', 'true')
    await expect(routingTab).toHaveAttribute(
      'aria-controls',
      'layout-mega-menu-build-routing-panel',
    )
    await expect(knowledgeTab).not.toHaveAttribute('aria-controls', /.+/)
    await expect(buildMenu.getByRole('link', { name: 'Config Builder' })).toBeVisible()
    await expect(buildMenu.getByRole('link', { name: 'Brain Topology' })).toBeVisible()
    await expect(buildMenu.getByRole('button', { name: 'Signals' })).toBeVisible()
    await expect(buildMenu.getByRole('button', { name: 'Decisions' })).toBeVisible()

    await routingTab.focus()
    await page.keyboard.press('ArrowDown')
    await expect(knowledgeTab).toBeFocused()
    await expect(knowledgeTab).toHaveAttribute('aria-selected', 'true')
    await expect(buildMenu.getByRole('link', { name: 'Bases' })).toBeVisible()
    await page.keyboard.press('End')
    await expect(integrationsTab).toBeFocused()
    await expect(buildMenu.getByRole('button', { name: 'MCP Servers' })).toBeVisible()
    await expect(buildMenu.getByRole('link', { name: 'Security Policy' })).toBeVisible()
    await page.keyboard.press('Home')
    await expect(routingTab).toBeFocused()
    await page.keyboard.press('ArrowRight')
    await expect(buildMenu.getByRole('link', { name: 'Config Builder' })).toBeFocused()

    const buildBounds = await buildMenu.boundingBox()
    expect(buildBounds).not.toBeNull()
    expect(buildBounds!.x).toBeGreaterThanOrEqual(0)
    expect(buildBounds!.x + buildBounds!.width).toBeLessThanOrEqual(1440)

    await page.keyboard.press('Escape')
    await expect(buildMenu).toBeHidden()
    await expect(buildTrigger).toHaveAttribute('aria-expanded', 'false')
    await expect(buildTrigger).toBeFocused()

    await analyzeTrigger.click()
    const analyzeMenu = page.getByRole('navigation', { name: 'Analyze' })
    await expect(analyzeMenu.getByRole('tab', { name: /Outcomes/ })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    await expect(analyzeMenu.getByRole('link', { name: 'Insights' })).toBeVisible()
    await expect(analyzeMenu.getByRole('link', { name: 'Evaluation' })).toBeVisible()
    await expect(analyzeMenu.getByRole('link', { name: 'ML Setup' })).toBeVisible()
    await analyzeMenu.getByRole('tab', { name: /Fleet Simulation/ }).click()
    await expect(analyzeMenu.getByRole('link', { name: 'Overview' })).toBeVisible()
    await expect(analyzeMenu.getByRole('link', { name: 'Runs' })).toBeVisible()

    await operateTrigger.click()
    const operateMenu = page.getByRole('navigation', { name: 'Operate' })
    await expect(operateMenu.getByRole('link', { name: 'Status' })).toBeVisible()
    await expect(operateMenu.getByRole('link', { name: 'Logs' })).toBeVisible()
    await operateMenu.getByRole('tab', { name: /Observability/ }).click()
    await expect(operateMenu.getByRole('link', { name: 'Grafana' })).toBeVisible()
    await expect(operateMenu.getByRole('link', { name: 'Tracing' })).toBeVisible()
    await operateMenu.getByRole('tab', { name: /Platform & Access/ }).click()
    await expect(operateMenu.getByRole('button', { name: 'Global Config' })).toBeVisible()
    await expect(operateMenu.getByRole('link', { name: 'Users' })).toBeVisible()

    await page.keyboard.press('Escape')
    await expect(operateTrigger).toBeFocused()

    await page.keyboard.press('ArrowUp')
    await expect(operateMenu).toBeVisible()
    await expect(operateMenu.getByRole('link', { name: 'Logs' })).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(operateTrigger).toBeFocused()

    await page.keyboard.press('ArrowDown')
    await expect(operateMenu.getByRole('tab', { name: /Runtime/ })).toBeFocused()
    await page.keyboard.press('Shift+Tab')
    await expect(operateTrigger).toBeFocused()
    await expect(operateTrigger).toHaveAttribute('aria-expanded', 'true')
    await page.keyboard.press('Shift+Tab')
    await expect(operateMenu).toBeHidden()
    await expect(operateTrigger).toHaveAttribute('aria-expanded', 'false')
  })

  test('keeps the desktop mega menu inside the viewport at 1024px', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 })
    await mockCommon(page)

    await page.goto('/dashboard')

    const globalNav = page.getByRole('navigation', { name: 'Global navigation' })
    await expect(globalNav).toBeVisible()
    await expect(page.getByRole('button', { name: 'Toggle menu' })).toBeHidden()

    const buildTrigger = globalNav.getByRole('button', { name: 'Build' })
    await buildTrigger.click()
    const buildMenu = page.getByRole('navigation', { name: 'Build' })
    await expect(buildMenu).toBeVisible()

    const bounds = await buildMenu.boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.x).toBeGreaterThanOrEqual(0)
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(1024)
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
  })

  test('aligns every desktop mega menu to one balanced AMD-style shell', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await page.setViewportSize({ width: 2048, height: 1152 })
    await mockCommon(page)

    await page.goto('/dashboard')

    const headerFrame = page.getByTestId('layout-header-content')
    const headerBounds = await headerFrame.boundingBox()
    const headerPadding = await headerFrame.evaluate((element) => {
      const style = window.getComputedStyle(element)
      return {
        left: Number.parseFloat(style.paddingLeft),
        right: Number.parseFloat(style.paddingRight),
      }
    })
    expect(headerBounds).not.toBeNull()
    expect(headerBounds!.height).toBe(80)
    expect(Math.abs(headerPadding.left - headerPadding.right)).toBeLessThanOrEqual(1)

    const workflowGroup = page.getByRole('group', { name: 'Workflow navigation' })
    const menuWidths: number[] = []

    for (const label of ['Build', 'Analyze', 'Operate'] as const) {
      const trigger = workflowGroup.getByRole('button', { name: label })
      await trigger.click()
      const menu = page.getByRole('navigation', { name: label })
      const menuBounds = await expectBalancedDesktopFrame(page, menu)
      menuWidths.push(menuBounds.width)
      expect(Math.abs(menuBounds.x - headerBounds!.x - headerPadding.left)).toBeLessThanOrEqual(1)
      expect(Math.abs(menuBounds.y - headerBounds!.height)).toBeLessThanOrEqual(1)
    }

    expect(Math.max(...menuWidths) - Math.min(...menuWidths)).toBeLessThanOrEqual(2)

    await workflowGroup.getByRole('button', { name: 'Build' }).click()
    const buildMenu = page.getByRole('navigation', { name: 'Build' })
    const initialBounds = await expectBalancedDesktopFrame(page, buildMenu)
    const railBounds = await buildMenu.getByTestId('layout-mega-menu-rail').boundingBox()
    const contentBounds = await buildMenu.getByTestId('layout-mega-menu-content').boundingBox()
    expect(railBounds).not.toBeNull()
    expect(contentBounds).not.toBeNull()
    expect(railBounds!.width / initialBounds.width).toBeGreaterThanOrEqual(0.15)
    expect(railBounds!.width / initialBounds.width).toBeLessThanOrEqual(0.2)
    expect(contentBounds!.width / initialBounds.width).toBeGreaterThanOrEqual(0.78)

    const firstTabBounds = await buildMenu.getByRole('tab').first().boundingBox()
    expect(firstTabBounds).not.toBeNull()
    expect(Math.abs(firstTabBounds!.y - initialBounds.y)).toBeLessThanOrEqual(2)

    for (const category of [/Knowledge/, /Integrations & Policy/]) {
      await buildMenu.getByRole('tab', { name: category }).click()
      const nextBounds = await buildMenu.boundingBox()
      expect(nextBounds).not.toBeNull()
      expect(Math.abs(nextBounds!.x - initialBounds.x)).toBeLessThanOrEqual(1)
      expect(Math.abs(nextBounds!.width - initialBounds.width)).toBeLessThanOrEqual(1)
    }

    await page.mouse.click(2040, 1144)
    await expect(buildMenu).toBeHidden()
    await expect(workflowGroup.getByRole('button', { name: 'Build' })).toHaveAttribute(
      'aria-expanded',
      'false',
    )
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
          'openclaw.manage',
          'openclaw.read',
          'openclaw.use',
          'replay.read',
          'security.manage',
          'tools.use',
          'topology.read',
          'users.manage',
          'users.view',
        ],
      },
    })

    await page.goto('/dashboard')

    await expect(page.getByRole('navigation', { name: 'Global navigation' })).toBeHidden()
    await expect(page.getByRole('button', { name: 'Toggle menu' })).toBeVisible()
    await expectInsideViewport(page, page.getByRole('link', { name: 'Documentation' }))
    await expectInsideViewport(page, page.getByRole('link', { name: 'GitHub' }))
    await expectInsideViewport(page, page.getByRole('button', { name: /Open account menu/i }))
    await expectInsideViewport(page, page.getByRole('button', { name: 'Toggle menu' }))
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
  })

  test('keeps a manually selected category open from an active route', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/builder')

    const buildTrigger = page
      .getByRole('group', { name: 'Workflow navigation' })
      .getByRole('button', { name: 'Build' })
    await buildTrigger.click()

    const buildMenu = page.getByRole('navigation', { name: 'Build' })
    const routingTab = buildMenu.getByRole('tab', { name: /Routing/ })
    const knowledgeTab = buildMenu.getByRole('tab', { name: /Knowledge/ })
    await expect(routingTab).toHaveAttribute('aria-selected', 'true')

    await knowledgeTab.click()
    await expect(knowledgeTab).toHaveAttribute('aria-selected', 'true')
    await expect(buildMenu.getByRole('link', { name: 'Bases' })).toBeVisible()
    await expect(routingTab).toHaveAttribute('aria-selected', 'false')
  })

  test('keeps every mega-menu category reachable in a short desktop window', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 400 })
    await mockCommon(page)

    await page.goto('/dashboard')
    const buildTrigger = page
      .getByRole('group', { name: 'Workflow navigation' })
      .getByRole('button', { name: 'Build' })
    await buildTrigger.click()

    const buildMenu = page.getByRole('navigation', { name: 'Build' })
    const routingTab = buildMenu.getByRole('tab', { name: /Routing/ })
    const integrationsTab = buildMenu.getByRole('tab', { name: /Integrations & Policy/ })
    await routingTab.focus()
    await page.keyboard.press('End')

    await expect(integrationsTab).toBeFocused()
    await expect(integrationsTab).toBeInViewport()
    expect(
      await buildMenu.evaluate(
        (element) => element.getBoundingClientRect().bottom <= window.innerHeight,
      ),
    ).toBe(true)
  })

  test('loads Insights charts and does not preserve the legacy replay route', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/dashboard')
    await page.getByRole('button', { name: 'Analyze' }).click()
    await page
      .getByRole('navigation', { name: 'Analyze' })
      .getByRole('link', { name: 'Insights' })
      .click()

    await expect(page).toHaveURL(/\/insights$/)
    await expect(page.getByRole('heading', { name: 'Insights', exact: true })).toBeVisible()
    await expect(page.getByText('Total Saved')).toBeVisible()
    await expect(page.getByText('Saved %')).toBeVisible()
    await expect(page.getByText('Baseline Spend')).toBeVisible()
    await expect(page.getByText('Actual Spend')).toBeVisible()
    await expect(
      page.getByText('See what the router picked, what signals fired, and how much it saved.'),
    ).toBeVisible()
    await expect(
      page.getByText('Decisions, model picks, token usage, and savings in one view.'),
    ).toBeVisible()
    await expect(
      page.getByText(
        'Replay-backed routing records with spend, savings, and token details per request.',
      ),
    ).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Model Selection', exact: true })).toBeVisible()
    await expect(
      page.getByRole('heading', { name: 'Decision Distribution', exact: true }),
    ).toBeVisible()
    await expect(
      page.getByRole('heading', { name: 'Signal Distribution', exact: true }),
    ).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Token Volume', exact: true })).toBeVisible()
    await expect(
      page.getByRole('heading', { name: 'Tokens by Decision', exact: true }),
    ).toBeVisible()
    await expect(
      page.getByRole('heading', { name: 'Tokens by Selected Model', exact: true }),
    ).toBeVisible()
    await expect(page.getByText('Input Tokens').first()).toBeVisible()
    await expect(page.getByText('Output Tokens').first()).toBeVisible()
    await expect(page.getByText('Total Tokens').first()).toBeVisible()
    await expect(
      page.getByText(
        '1 filtered record excluded from cost totals because usage or pricing data is incomplete.',
      ),
    ).toBeVisible()
    await expect(
      page.getByText(
        '1 filtered record excluded from token totals because usage data is incomplete.',
      ),
    ).toBeVisible()
    await expect(
      page.getByRole('article').filter({ hasText: 'Total Saved' }).getByRole('strong'),
    ).toHaveText('$0.0060')
    await expect(
      page.getByRole('article').filter({ hasText: 'Saved %' }).getByRole('strong'),
    ).toHaveText('75.0%')
    await expect(page.getByRole('columnheader', { name: 'Actual Cost' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: 'Saved vs Baseline' })).toBeVisible()

    await page.goto('/replay')
    await expect(page).toHaveURL(/\/dashboard$/)
  })

  test('renders nested canonical decision groups in Brain topology', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/topology')

    const decisionNode = page
      .locator('[class*="decisionNode"]')
      .filter({ has: page.getByText('medium_business', { exact: true }) })
      .first()

    await expect(decisionNode).toBeVisible()
    await expect(decisionNode).toContainText('AND')
    await expect(decisionNode).toContainText('OR')
    await expect(decisionNode).toContainText('domain: business')
    await expect(decisionNode).toContainText('domain: economics')
    await expect(decisionNode).toContainText('embedding: business_analysis')
    await expect(decisionNode.getByText('Referenced signals not configured')).toHaveCount(0)
  })

  test('hides ML Setup for read users and redirects direct access back to the dashboard', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page, { user: readUser })

    await page.goto('/dashboard')

    await expect(page.getByRole('button', { name: 'ML Setup', exact: true })).toHaveCount(0)

    const workflowGroup = page.getByRole('group', { name: 'Workflow navigation' })
    await workflowGroup.getByRole('button', { name: 'Operate' }).click()

    const operateMenu = page.getByRole('navigation', { name: 'Operate' })
    await operateMenu.getByRole('tab', { name: /Platform & Access/ }).click()
    await expect(operateMenu.getByRole('link', { name: 'Users' })).toHaveCount(0)
    await expect(operateMenu.getByRole('button', { name: 'Global Config' })).toBeVisible()

    await workflowGroup.getByRole('button', { name: 'Analyze' }).click()

    const analyzeMenu = page.getByRole('navigation', { name: 'Analyze' })
    await expect(analyzeMenu.getByRole('link', { name: 'ML Setup' })).toHaveCount(0)

    await workflowGroup.getByRole('button', { name: 'Build' }).click()
    const buildMenu = page.getByRole('navigation', { name: 'Build' })
    await buildMenu.getByRole('tab', { name: /Integrations & Policy/ }).click()
    await expect(buildMenu.getByRole('button', { name: 'MCP Servers' })).toBeVisible()
    await expect(buildMenu.getByRole('link', { name: 'ClawOS' })).toBeVisible()

    await page.goto('/ml-setup')

    await expect(page).toHaveURL(/\/dashboard$/)
    await expect(page.getByRole('button', { name: 'ML Setup', exact: true })).toHaveCount(0)
  })

  test('opens an anchored account dropdown and allows logging out from the header', async ({
    page,
  }) => {
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

    await page.goto('/dashboard')

    const accountTrigger = page.getByRole('button', { name: /Open account menu/i })
    await accountTrigger.click()

    const accountDialog = page.getByTestId('layout-account-dialog')
    const closeDialogButton = accountDialog.getByRole('button', { name: 'Close account dialog' })
    const logoutButton = accountDialog.getByRole('button', { name: 'Logout' })
    await expect(accountDialog).toBeVisible()
    await expect(accountDialog).toHaveAttribute('aria-modal', 'true')
    await expect(accountDialog.getByText('Ada Lovelace')).toBeVisible()
    await expect(accountDialog.getByText('ada@example.com')).toBeVisible()
    await expect(accountDialog.getByText('Admin')).toBeVisible()
    await expect(accountDialog.getByText('config.read')).toBeVisible()
    await expect(accountDialog.getByText('config.write')).toBeVisible()
    await expect(accountDialog.getByText('users.manage')).toBeVisible()
    await expect(closeDialogButton).toBeFocused()

    const accountBounds = await accountDialog.boundingBox()
    const viewport = page.viewportSize()
    expect(accountBounds).not.toBeNull()
    expect(viewport).not.toBeNull()
    expect(accountBounds!.x).toBeGreaterThan(viewport!.width / 2)
    expect(accountBounds!.y).toBeLessThan(120)
    expect(accountBounds!.width).toBeLessThanOrEqual(400)

    await page.keyboard.press('Shift+Tab')
    await expect(logoutButton).toBeFocused()
    await page.keyboard.press('Tab')
    await expect(closeDialogButton).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(accountDialog).toBeHidden()
    await expect(accountTrigger).toBeFocused()

    await accountTrigger.click()
    await accountDialog.getByRole('button', { name: 'Logout' }).click()

    await expect(page).toHaveURL(/\/login$/)
    await expect(page.getByRole('heading', { name: 'Sign in', exact: true })).toBeVisible()
  })

  test('defaults /config to Global Config and loads raw global YAML', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockCommon(page)

    await page.goto('/config')

    await expect(page.getByRole('heading', { name: 'Global Config', exact: true })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Global Config Overview' })).toBeVisible()

    await page.getByRole('button', { name: 'Raw YAML' }).click()
    await expect(page.getByRole('heading', { name: 'Raw Global YAML' })).toBeVisible()
    await expect(page.locator('textarea')).toHaveValue(rawGlobalYaml)
  })
})
