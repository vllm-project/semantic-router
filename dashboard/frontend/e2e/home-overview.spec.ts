import { expect, test, type Locator, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const config = {
  version: 'v0.5',
  listeners: [{ name: 'http-8899', port: 8899 }],
  providers: {
    defaults: { model: 'qwen3-8b' },
    models: [
      { name: 'qwen3-8b', backend_refs: [{ name: 'local', provider: 'vllm' }] },
      {
        name: 'deepseek/deepseek-v4.1-flash',
        backend_refs: [{ name: 'api', provider: 'deepseek' }],
      },
    ],
  },
  entrypoints: [{ model_names: ['vllm-sr/balanced'], recipe: 'default', default: true }],
  recipes: [
    {
      name: 'default',
      signals: { keywords: [{ name: 'code' }], domains: [{ name: 'law' }] },
      decisions: [
        { name: 'agentic', priority: 300, modelRefs: [{ model: 'deepseek/deepseek-v4.1-flash' }] },
        { name: 'fast', priority: 200, modelRefs: [{ model: 'qwen3-8b' }] },
      ],
    },
  ],
}

const minutesAgo = (minutes: number) => new Date(Date.now() - minutes * 60_000).toISOString()

const base = {
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 300,
  recipe: 'default',
  original_model: 'vllm-sr/balanced',
  lifecycle_state: 'completed',
  response_status: 200,
  currency: 'USD',
  signals: {},
}

const claudeSession = 'key_1:49b948ac-48b9-4fd5-b4e8-ccb8f6fba483'

const records = [
  {
    ...base,
    id: 'claude-subagent',
    timestamp: minutesAgo(2),
    session_id: claudeSession,
    decision: 'agentic',
    selected_model: 'deepseek/deepseek-v4.1-flash',
    duration_ms: 170,
    actual_cost: 0.002,
    agent: { harness: 'claude_code', id: '3fa9c2d81e07' },
  },
  {
    ...base,
    id: 'codex-call',
    timestamp: minutesAgo(5),
    session_id: 'key_1:01a1ea71-6532-7fc1-9137-1fd7cf9e9851',
    decision: 'fast',
    selected_model: 'qwen3-8b',
    duration_ms: 1395,
    actual_cost: 0.001,
    agent: { harness: 'codex' },
  },
  {
    ...base,
    id: 'api-call',
    timestamp: minutesAgo(9),
    decision: 'fast',
    selected_model: 'qwen3-8b',
    duration_ms: 88,
  },
]

const aggregate = {
  object: 'router_replay.aggregate',
  record_count: records.length,
  lifecycle: { completed: 3, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
  summary: {
    total_saved: 0.004,
    baseline_spend: 0.007,
    actual_spend: 0.003,
    total_spend: 0.003,
    currency: 'USD',
    cost_record_count: 2,
    excluded_record_count: 1,
  },
  model_selection: [
    { name: 'qwen3-8b', value: 2 },
    { name: 'deepseek/deepseek-v4.1-flash', value: 1 },
  ],
  decision_distribution: [
    { name: 'fast', value: 2 },
    { name: 'agentic', value: 1 },
  ],
  signal_distribution: [],
  token_volume: { input_tokens: 0, output_tokens: 0, total_tokens: 0, excluded_record_count: 0 },
  token_breakdown: { by_decision: [], by_selected_model: [] },
  available_recipes: ['default'],
  available_decisions: ['agentic', 'fast'],
  available_models: ['qwen3-8b', 'deepseek/deepseek-v4.1-flash'],
}

// The traffic series a router with interval support returns for the asked window.
function trafficSeries(since: string, intervalSeconds: number) {
  const start = Date.parse(since)
  const count = Math.ceil((Date.now() - start) / (intervalSeconds * 1000))
  return {
    since,
    interval_seconds: intervalSeconds,
    latency_p50_ms: 170,
    latency_p95_ms: 1500,
    buckets: Array.from({ length: count }, (_, index) => {
      const last = index === count - 1
      return {
        start: new Date(start + index * intervalSeconds * 1000).toISOString(),
        requests: last ? 3 : 0,
        cached: 0,
        failed: 0,
        ...(last ? { latency_p50_ms: 170, latency_p95_ms: 1500 } : {}),
        cost: last
          ? [
              {
                currency: 'USD',
                total_spend: 0.003,
                actual_spend: 0.003,
                total_saved: 0.004,
                baseline_spend: 0.007,
                cost_record_count: 2,
              },
            ]
          : [],
      }
    }),
  }
}

async function mockHome(
  page: Page,
  { replay = true, series = true }: { replay?: boolean; series?: boolean } = {},
) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: config }))
  await page.route('**/api/status', (route) =>
    route.fulfill({
      json: {
        overall: 'healthy',
        serving_mode: 'router',
        deployment_type: 'docker',
        version: 'v0.5.0',
        services: [
          { name: 'Router', status: 'running', healthy: true, message: 'ExtProc routing engine' },
          { name: 'Dashboard', status: 'running', healthy: true },
        ],
        models: {
          models: [
            {
              name: 'decision_model',
              type: 'decision',
              loaded: true,
              state: 'ready',
              metadata: { provider: 'model_runtime', deployment: 'primary' },
            },
          ],
        },
      },
    }),
  )
  await page.route('**/api/router/api/v1/inventory/model-runtime', (route) =>
    route.fulfill({ json: { deployments: [] } }),
  )
  const aggregateQueries: string[] = []
  const listQueries: URLSearchParams[] = []
  const runQueries: URLSearchParams[] = []
  // A stalled router: replay reads stay open, as a slow aggregate does.
  const replies = { stalled: false }
  await page.route('**/api/router/api/v1/observability/replays**', (route) => {
    const url = new URL(route.request().url())
    if (url.pathname.endsWith('/models'))
      return route.fulfill({ json: { object: 'list', data: [] } })
    if (replies.stalled) return
    if (!replay) return route.fulfill({ status: 404, json: { error: 'not found' } })
    if (url.pathname.endsWith('/aggregate')) {
      aggregateQueries.push(url.search)
      const since = url.searchParams.get('since')
      const interval = url.searchParams.get('interval')
      const seconds = interval
        ? Number.parseInt(interval, 10) * (interval.endsWith('m') ? 60 : 3600)
        : 0
      return route.fulfill({
        json:
          series && since && seconds
            ? { ...aggregate, traffic: trafficSeries(since, seconds) }
            : aggregate,
      })
    }
    if (url.pathname.endsWith('/runs')) {
      runQueries.push(url.searchParams)
      return route.fulfill({
        json: {
          object: 'router_replay.runs',
          data: [{ session_id: claudeSession, session_name: 'Fix the login redirect' }],
        },
      })
    }
    listQueries.push(url.searchParams)
    // Without a series, a range holding more records than the newest page charts only those.
    const total = series ? records.length : 500
    const limit = Number(url.searchParams.get('limit') ?? records.length)
    return route.fulfill({
      json: { object: 'list', data: records.slice(0, limit), count: records.length, total },
    })
  })
  const windowReads = () => listQueries.filter((query) => query.get('limit') === '100')
  return { aggregateQueries, listQueries, runQueries, windowReads, replies }
}

test('summarizes traffic, recent requests and the routing configuration', async ({ page }) => {
  const { aggregateQueries, listQueries, runQueries, windowReads } = await mockHome(page)
  await page.goto('/')

  await expect(page.getByTestId('dashboard-serving-mode')).toContainText('Router mode')
  await expect(page.getByRole('button', { name: 'Refresh' })).toHaveCount(0)
  await expect(page.getByText(/^Updated /)).toHaveCount(0)
  await expect(page.getByRole('link', { name: 'Requests: open Requests' })).toContainText('3')
  await expect(page.getByRole('link', { name: 'p95 latency: open Requests' })).toContainText(
    '1.50s',
  )
  await expect(page.getByTestId('home-saved')).toContainText('$0.00')

  const traffic = page.getByTestId('home-traffic')
  await expect(traffic).toContainText('Last 24 hours across every public model')
  await expect(traffic.locator('.recharts-surface')).toBeVisible()
  await traffic.getByRole('radio', { name: 'Latency' }).click()
  await expect(traffic).toContainText('p95')
  await expect(traffic).toContainText('1.50s')
  await traffic.getByRole('radio', { name: 'Cost' }).click()
  await expect(traffic).toContainText('Without routing')
  expect(aggregateQueries.some((query) => query.includes('interval=1h'))).toBe(true)

  await expect(page.getByTestId('home-count-decisions')).toContainText('2')
  await expect(page.getByTestId('home-count-signals')).toContainText('2')
  await expect(page.getByTestId('home-count-models')).toContainText('2')

  const recent = page.getByTestId('home-recent-requests')
  await expect(recent.getByRole('img', { name: 'Claude Code' })).toBeVisible()
  await expect(recent.getByRole('img', { name: 'Codex' })).toBeVisible()
  await expect(recent.locator('[data-request-class="subagent"]')).toBeVisible()
  await expect(recent.getByText('API client')).toBeVisible()
  // The session title arrives last, from one bounded read of the newest runs.
  await expect(recent.getByRole('link', { name: 'Fix the login redirect' })).toBeVisible()
  expect(runQueries).toHaveLength(1)
  expect(runQueries[0].get('limit')).toBe('20')
  expect(runQueries[0].get('since')).toBeTruthy()
  // A router with traffic intervals charts from the aggregate alone.
  expect(listQueries.every((query) => query.get('limit') === '7' && query.get('since'))).toBe(true)
  expect(windowReads()).toHaveLength(0)

  await expect(page.getByTestId('home-routing-mix')).toContainText('66.7%')
  await expect(page.getByTestId('home-models')).toContainText('qwen3-8b')
  await expect(page.getByTestId('start-checklist')).toContainText('Everything is in place')

  await page.getByRole('radio', { name: '7d' }).click()
  await expect(page).toHaveURL(/range=7d/)
  await expect
    .poll(() => aggregateQueries.some((query) => /since=.*interval=6h/.test(query)))
    .toBe(true)
  await expect(traffic).toContainText('Last 7 days across every public model')

  await recent.getByRole('link', { name: /01a1ea71/ }).click()
  await expect(page).toHaveURL(/\/observe\/requests\/codex-call$/)
})

test('explains missing traffic when Router Replay is off and keeps the configuration', async ({
  page,
}) => {
  await mockHome(page, { replay: false })
  await page.goto('/')

  const traffic = page.getByTestId('home-traffic')
  await expect(traffic.getByText('Traffic metrics need Router Replay')).toBeVisible()
  await expect(traffic.getByRole('button', { name: 'Open recipes' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Requests: open Requests' })).toHaveCount(0)
  await expect(page.getByTestId('home-recent-requests')).toContainText(
    'No recorded requests to show',
  )
  await expect(page.getByTestId('home-count-decisions')).toContainText('2')
  await expect(page.getByTestId('home-routing-mix')).toContainText('agentic')
  await expect(
    page.getByTestId('start-checklist').locator('[data-step="first-request"]'),
  ).toHaveAttribute('data-state', 'todo')
})

test('charts the newest requests when the router returns no traffic series', async ({ page }) => {
  const { windowReads } = await mockHome(page, { series: false })
  await page.goto('/')

  const traffic = page.getByTestId('home-traffic')
  await expect(traffic).toContainText('Last 3 requests')
  await expect(traffic.locator('.recharts-surface')).toBeVisible()
  await expect(traffic).toContainText('Served from cache')
  await expect(page.getByRole('link', { name: 'p95 latency: open Requests' })).toContainText(
    'newest 3',
  )

  // One read of the newest 100 records per range, however often the range comes back.
  await page.getByRole('radio', { name: '7d' }).click()
  await expect.poll(() => windowReads().length).toBe(2)
  await page.getByRole('radio', { name: '24h' }).click()
  await expect(traffic).toContainText('Last 3 requests')
  expect(windowReads().map((query) => query.get('limit'))).toEqual(['100', '100'])
})

test('shows the last traffic at once on a return visit while the router answers slowly', async ({
  page,
}) => {
  const { replies } = await mockHome(page)
  await page.goto('/')
  const requests = page.getByRole('link', { name: 'Requests: open Requests' })
  await expect(requests).toContainText('3')
  await expect(page.getByTestId('home-recent-requests')).toContainText('Fix the login redirect')

  replies.stalled = true
  await page.reload()
  await expect(requests).toContainText('3')
  await expect(page.getByRole('link', { name: 'p95 latency: open Requests' })).toContainText(
    '1.50s',
  )
  await expect(page.getByTestId('home-traffic').locator('.recharts-surface')).toBeVisible()
  await expect(
    page.getByTestId('home-recent-requests').getByRole('link', { name: 'Fix the login redirect' }),
  ).toBeVisible()
})

test('keeps the Home grid rows level and leaves the runtime inventory off', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1400 })
  await mockHome(page)
  await page.goto('/')
  await expect(page.getByRole('link', { name: 'Requests: open Requests' })).toContainText('3')

  const intelligence = page.getByRole('region', { name: 'Router intelligence' })
  await expect(intelligence).toBeVisible()
  await expect(page.getByText('Runtime inventory')).toHaveCount(0)
  await expect(page.getByText('No runtime inventory reported yet.')).toHaveCount(0)

  const box = async (cell: Locator) => {
    const bounds = await cell.boundingBox()
    expect(bounds).not.toBeNull()
    return bounds!
  }
  const level = async (...cells: Locator[]) => {
    const boxes = await Promise.all(cells.map(box))
    for (const bounds of boxes.slice(1)) {
      expect(Math.abs(bounds.y - boxes[0].y)).toBeLessThanOrEqual(1)
      expect(Math.abs(bounds.height - boxes[0].height)).toBeLessThanOrEqual(1)
    }
  }
  await level(page.getByTestId('home-traffic'), page.getByTestId('start-checklist'))
  await level(
    page.getByTestId('home-routing-mix'),
    page.getByTestId('home-models'),
    page.getByTestId('home-public-models'),
  )
  await level(page.getByTestId('home-recent-requests'), page.getByTestId('home-services'))
  await level(intelligence, page.getByRole('navigation', { name: 'Routing configuration' }))
  const [first, last] = await Promise.all([
    box(page.getByTestId('home-count-recipes')),
    box(page.getByTestId('home-count-models')),
  ])
  const panel = await box(intelligence)
  expect(Math.abs(first.y - panel.y)).toBeLessThanOrEqual(1)
  expect(Math.abs(last.y + last.height - (panel.y + panel.height))).toBeLessThanOrEqual(1)
})
