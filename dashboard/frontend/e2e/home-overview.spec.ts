import { expect, test, type Page } from '@playwright/test'
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

const records = [
  {
    ...base,
    id: 'claude-subagent',
    timestamp: minutesAgo(2),
    session_id: 'key_1:49b948ac-48b9-4fd5-b4e8-ccb8f6fba483',
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

async function mockHome(page: Page, { replay = true }: { replay?: boolean } = {}) {
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
  const aggregateQueries: string[] = []
  await page.route('**/api/router/api/v1/observability/replays**', (route) => {
    const url = new URL(route.request().url())
    if (!replay) return route.fulfill({ status: 404, json: { error: 'not found' } })
    if (url.pathname.endsWith('/aggregate')) {
      aggregateQueries.push(url.search)
      return route.fulfill({ json: aggregate })
    }
    if (url.pathname.endsWith('/runs')) return route.fulfill({ json: { object: 'list', data: [] } })
    if (url.pathname.endsWith('/models'))
      return route.fulfill({ json: { object: 'list', data: [] } })
    return route.fulfill({
      json: { object: 'list', data: records, count: records.length, total: records.length },
    })
  })
  return { aggregateQueries }
}

test('summarizes traffic, recent requests and the routing configuration', async ({ page }) => {
  const { aggregateQueries } = await mockHome(page)
  await page.goto('/')

  await expect(page.getByText('Router healthy').first()).toBeVisible()
  await expect(page.getByText('v0.5.0').first()).toBeVisible()
  await expect(page.getByRole('link', { name: 'Requests: open Requests' })).toContainText('3')
  await expect(page.getByTestId('home-saved')).toContainText('$0.00')

  await expect(page.getByTestId('home-count-decisions')).toContainText('2')
  await expect(page.getByTestId('home-count-signals')).toContainText('2')
  await expect(page.getByTestId('home-count-models')).toContainText('2')

  const recent = page.getByTestId('home-recent-requests')
  await expect(recent.getByRole('img', { name: 'Claude Code' })).toBeVisible()
  await expect(recent.getByRole('img', { name: 'Codex' })).toBeVisible()
  await expect(recent.locator('[data-request-class="subagent"]')).toBeVisible()
  await expect(recent.getByText('API client')).toBeVisible()

  await expect(page.getByTestId('home-routing-mix')).toContainText('66.7%')
  await expect(page.getByTestId('home-models')).toContainText('qwen3-8b')
  await expect(page.getByTestId('start-checklist')).toContainText('Everything is in place')

  await page.getByRole('radio', { name: '7d' }).click()
  await expect(page).toHaveURL(/range=7d/)
  await expect.poll(() => aggregateQueries.some((query) => query.includes('since='))).toBe(true)

  await recent.getByRole('link', { name: /01a1ea71/ }).click()
  await expect(page).toHaveURL(/\/observe\/requests\/codex-call$/)
})

test('explains missing traffic when Router Replay is off and keeps the configuration', async ({
  page,
}) => {
  await mockHome(page, { replay: false })
  await page.goto('/')

  await expect(page.getByText('Traffic metrics need Router Replay')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Open recipes' })).toBeVisible()
  await expect(page.getByTestId('home-recent-requests')).toContainText(
    'No recorded requests to show',
  )
  await expect(page.getByTestId('home-count-decisions')).toContainText('2')
  await expect(page.getByTestId('home-routing-mix')).toContainText('agentic')
  await expect(
    page.getByTestId('start-checklist').locator('[data-step="first-request"]'),
  ).toHaveAttribute('data-state', 'todo')
})
