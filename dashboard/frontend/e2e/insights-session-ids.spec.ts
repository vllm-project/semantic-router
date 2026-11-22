import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

// An OpenCode session as the Router files it: the portal scoped the id the
// harness sent, ses_…, to the caller's key. A second key's caller chose the
// same id for another session.
const RAW = 'ses_e0e5a17c2YCOSIf'
const SCOPED = `key_3fa9c2:${RAW}`
const OTHER = `key_77b104:${RAW}`
const FLASH = 'deepseek/deepseek-v4.1-flash'
const QWEN = 'qwen/qwen3.8-27b'
const REPLAYS = '**/api/router/api/v1/observability/replays'

const cost = {
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 23_000,
  completion_tokens: 806,
  by_currency: [{ currency: 'USD', total: 0.0038, router_call_spend: 0 }],
}
const agent = (label: string, calls: number, fields: Record<string, unknown> = {}) => ({
  label,
  calls,
  first_call_at: '2026-11-19T02:36:38Z',
  last_call_at: '2026-11-19T02:36:42Z',
  models: [FLASH],
  cost,
  ...fields,
})
const runOf = (session_id: string, session_name: string) => ({
  session_id,
  session_name,
  harness: 'opencode',
  started_at: session_id === OTHER ? '2026-11-18T09:10:00Z' : '2026-11-19T02:36:38Z',
  ended_at: session_id === OTHER ? '2026-11-18T09:10:05Z' : '2026-11-19T02:36:42Z',
  duration_ms: 4_500,
  calls: 5,
  agents: [agent('main', 3), agent('subagent 1', 2, { id: 'a1b2c3d4e5f6', parent: 'main' })],
  models: [
    { name: FLASH, value: 4 },
    { name: QWEN, value: 1 },
  ],
  outcome: { status: 'answered', tool_results: 2, tool_failures: 0 },
  cost,
  events: { subagent_start: 1, tool_loop: 2 },
})
const RUN = runOf(SCOPED, 'Find file mentioning release codename')
const OTHER_RUN = runOf(OTHER, 'Rename the billing queue')

const call = (index: number, fields: Record<string, unknown> = {}) => ({
  record_id: `call-${index}`,
  agent: 'main',
  timestamp: `2026-11-19T02:36:${String(38 + index).padStart(2, '0')}Z`,
  duration_ms: 1_150,
  lifecycle_state: 'completed',
  response_status: 200,
  decision: 'agentic',
  model: FLASH,
  prompt_tokens: 7_400,
  completion_tokens: 192,
  cost: 0.0008,
  currency: 'USD',
  signals: ['keyword:code_request', 'agent:opencode'],
  selection_method: 'session_aware',
  session_action: 'stay',
  reason: 'stay_has_best_adjusted_score',
  router_calls: [
    { kind: 'decision_model', model: 'primary', calls: 1, cost: 0.0001, currency: 'USD' },
  ],
  ...fields,
})
const calls = [
  call(0, { request_class: 'housekeeping', decision: 'housekeeping' }),
  call(1),
  call(2, { agent: 'subagent 1', request_class: 'subagent', markers: ['subagent_start'] }),
  call(3, {
    model: QWEN,
    previous_model: FLASH,
    session_action: 'switch',
    reason: 'switch_allowed',
    effort: 'high',
    tool_results: 2,
    tool_failures: 0,
    markers: ['tool_loop'],
  }),
  call(4, { agent: 'subagent 1', markers: ['tool_loop'] }),
]
const records = calls.map((entry) => ({
  id: entry.record_id,
  timestamp: entry.timestamp,
  session_id: SCOPED,
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 300,
  recipe: 'default',
  original_model: 'vllm-sr/auto',
  decision: entry.decision,
  selected_model: entry.model,
  lifecycle_state: 'completed',
  response_status: 200,
  duration_ms: entry.duration_ms,
  agent: { harness: 'opencode' },
}))

const aggregateOf = (count: number) => ({
  object: 'router_replay.aggregate',
  record_count: count,
  lifecycle: { completed: count, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
  summary: {
    total_saved: 0,
    baseline_spend: 0,
    actual_spend: count ? 0.0038 : 0,
    total_spend: count ? 0.0038 : 0,
    currency: 'USD',
    cost_record_count: 0,
    excluded_record_count: count,
  },
  model_selection: count ? [{ name: FLASH, value: count }] : [],
  decision_distribution: count ? [{ name: 'agentic', value: count }] : [],
  signal_distribution: [],
  token_volume: {
    input_tokens: count ? 23_000 : 0,
    output_tokens: count ? 806 : 0,
    total_tokens: count ? 23_806 : 0,
    excluded_record_count: 0,
  },
  token_breakdown: { by_decision: [], by_selected_model: [] },
  available_recipes: ['default'],
  available_decisions: ['agentic'],
  available_models: [FLASH],
})

async function setup(page: Page, { shared = false }: { shared?: boolean } = {}) {
  await mockAuthenticatedAppShell(page)
  const seen = { aggregates: [] as URLSearchParams[], runs: [] as URLSearchParams[] }
  const sessions = shared ? [RUN, OTHER_RUN] : [RUN]
  await page.route(/\/api\/router\/api\/v1\/observability\/replays(\?|$)/, (route) => {
    const query = new URL(route.request().url()).searchParams
    const data = query.get('session_id') === SCOPED ? records : []
    return route.fulfill({
      json: {
        object: 'router_replay.list',
        count: data.length,
        total: data.length,
        has_more: false,
        harness_counts: data.length ? { opencode: data.length } : {},
        data,
      },
    })
  })
  await page.route(`${REPLAYS}/aggregate*`, (route) => {
    const query = new URL(route.request().url()).searchParams
    seen.aggregates.push(query)
    return route.fulfill({ json: aggregateOf(query.get('session_id') === SCOPED ? 5 : 0) })
  })
  await page.route(`${REPLAYS}/models`, (route) =>
    route.fulfill({ json: { object: 'list', data: [] } }),
  )
  await page.route(`${REPLAYS}/runs*`, (route) => {
    const query = new URL(route.request().url()).searchParams
    seen.runs.push(query)
    const session = query.get('session_id')
    const search = query.get('search') ?? ''
    const data = session
      ? sessions.filter((entry) => entry.session_id === session)
      : sessions.filter((entry) => entry.session_id.includes(search))
    return route.fulfill({
      json: {
        object: 'list',
        count: data.length,
        total: data.length,
        limit: 25,
        offset: 0,
        has_more: false,
        data,
      },
    })
  })
  await page.route(`${REPLAYS}/run?*`, (route) => {
    const session = new URL(route.request().url()).searchParams.get('session_id')
    const found = sessions.find((entry) => entry.session_id === session)
    return found
      ? route.fulfill({ json: { object: 'router_replay.run', run: found, calls } })
      : route.fulfill({ status: 404, json: { error: 'run not found' } })
  })
  return seen
}

const fitsWidth = (page: Page) =>
  page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)

test('the summary counts what a session filter by the harness id lists', async ({ page }) => {
  const seen = await setup(page)
  await page.goto(`/observe/requests?view=sessions&session=${RAW}`)

  const summary = page.getByRole('region', { name: 'Requests summary' })
  await expect(summary).toContainText('5 completed')
  await expect(summary).toContainText('23.8K')
  const sessions = page.getByRole('region', { name: 'Sessions' })
  await expect(sessions.locator(`[data-session="${SCOPED}"]`)).toContainText(
    'Find file mentioning release codename',
  )
  // every read matched the session's full id, never the harness id on its own
  expect(seen.aggregates.length).toBeGreaterThan(0)
  expect(seen.aggregates.every((query) => query.get('session_id') === SCOPED)).toBe(true)
  expect(seen.runs.some((query) => query.get('session_id') === SCOPED)).toBe(true)
  await expect(page.locator('body')).not.toContainText('key_')

  await page.getByRole('radio', { name: 'Requests' }).click()
  await expect(page.getByText('1–5 of 5 requests')).toBeVisible()
  await expect(summary).toContainText('5 completed')

  await page.getByRole('button', { name: 'Open run' }).click()
  await expect(page).toHaveURL(new RegExp(`/observe/runs\\?session=${RAW}$`))
  await expect(page.getByRole('heading', { level: 1 })).toContainText(
    'Find file mentioning release codename',
  )
  await expect(page.locator('body')).not.toContainText('key_')
})

test('a run opens by the id its harness shows, and copies that id', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await setup(page)
  await page.goto(`/observe/runs?session=${RAW}`)

  await expect(page.getByRole('heading', { level: 1 })).toContainText(
    'Find file mentioning release codename',
  )
  await expect(page.locator('main')).toContainText(RAW)
  await page.getByRole('button', { name: 'Copy session ID' }).first().click()
  await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(RAW)
  await expect(page.locator('body')).not.toContainText('key_')

  await page.getByRole('button', { name: 'View requests' }).click()
  await expect(page).toHaveURL(new RegExp(`/observe/requests\\?session=${RAW}$`))
  await expect(page.getByRole('region', { name: 'Requests summary' })).toContainText('5 completed')
})

test('a harness id two keys chose opens the newest session and offers the other', async ({
  page,
}) => {
  await setup(page, { shared: true })
  await page.goto(`/observe/runs?session=${RAW}`)

  const title = page.getByRole('heading', { level: 1 })
  await expect(title).toContainText('Find file mentioning release codename')
  await page.getByRole('button', { name: '2 sessions use this id' }).click()
  await page.getByRole('menuitem', { name: /Rename the billing queue/ }).click()
  await expect(title).toContainText('Rename the billing queue')
  await expect(page).toHaveURL(new RegExp(`/observe/runs\\?session=${RAW}$`))
  await expect(page.locator('body')).not.toContainText('key_')
})

test('the call card keeps Routing beside Usage and Outcome', async ({ page }, info) => {
  await page.setViewportSize({ width: 1440, height: 1100 })
  await setup(page)
  await page.goto(`/observe/runs?session=${RAW}`)

  const timeline = page.getByRole('list', { name: 'Run timeline' })
  await timeline
    .getByRole('button', { name: /, 1 call$/ })
    .first()
    .click()
  const details = page.getByRole('complementary', { name: 'Step details' })
  const routing = details.getByRole('region', { name: 'Routing' })
  const usage = details.getByRole('region', { name: 'Usage' })
  const outcome = details.getByRole('region', { name: 'Outcome' })
  await expect(outcome.getByRole('link', { name: 'Open the replay record' })).toBeVisible()
  const [r, u, o, link] = await Promise.all(
    [routing, usage, outcome, outcome.getByRole('link', { name: 'Open the replay record' })].map(
      async (locator) => (await locator.boundingBox())!,
    ),
  )
  expect(r.x).toBeLessThan(u.x)
  expect(Math.abs(u.x - o.x)).toBeLessThanOrEqual(1)
  expect(Math.abs(r.y - u.y)).toBeLessThanOrEqual(1)
  expect(Math.abs(r.y + r.height - (o.y + o.height))).toBeLessThanOrEqual(1)
  // the replay link heads the Outcome section rather than closing an empty column
  expect(link.y - o.y).toBeLessThan(32)
  await details.screenshot({ path: info.outputPath('call-card-1440.png') })

  await page.setViewportSize({ width: 390, height: 844 })
  await expect.poll(() => fitsWidth(page)).toBe(true)
  await details.screenshot({ path: info.outputPath('call-card-390.png') })
})
