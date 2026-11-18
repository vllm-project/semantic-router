import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

// Agent runs as the router files them: a Claude Code session whose main agent
// started an Explore subagent, compacted, read a picture and was refused once
// by the guard; a Cursor conversation the router keyed by how it starts, with a
// request for one of Cursor's own models; and an OpenClaw session.
const CLAUDE = 'key_3fa9c2:49b948ac-48b9-4fd5-b4e8-ccb8f6fba483'
const CURSOR = 'key_3fa9c2:cursor-start-e939a0c78d77f163'
const OPENCLAW = 'key_3fa9c2:openclaw-3e87f001-0e05-4131-be78-0eb121f53ba4'

const cost = {
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 64200,
  completion_tokens: 1800,
}
const priced = { ...cost, by_currency: [{ currency: 'USD', total: 0.0213, router_call_spend: 0 }] }
const agent = (label: string, fields: Record<string, unknown> = {}) => ({
  label,
  calls: 3,
  first_call_at: '2026-11-19T10:00:00Z',
  last_call_at: '2026-11-19T10:02:00Z',
  models: ['deepseek/deepseek-v4.1-flash'],
  cost: priced,
  ...fields,
})
const claudeRun = {
  session_id: CLAUDE,
  session_name: 'Fix the failing mean test',
  harness: 'claude_code',
  started_at: '2026-11-19T10:00:00Z',
  ended_at: '2026-11-19T10:02:00Z',
  duration_ms: 120000,
  calls: 6,
  agents: [
    agent('main'),
    agent('subagent 1', { id: 'ad63effbc9a5', parent: 'main', type: 'Explore' }),
  ],
  models: [{ name: 'deepseek/deepseek-v4.1-flash', value: 6 }],
  outcome: { status: 'answered', tool_results: 4, tool_failures: 1 },
  cost: priced,
  events: { subagent_start: 1, compaction: 1, image: 1, tool_loop: 3, block: 1 },
}
const runs = [
  claudeRun,
  {
    ...claudeRun,
    session_id: CURSOR,
    session_name: undefined,
    prompt: 'Read app/stats.py and tell me what median() returns',
    harness: 'cursor',
    calls: 2,
    agents: [agent('main', { calls: 2 })],
    events: { tool_loop: 1 },
    cost,
  },
  {
    ...claudeRun,
    session_id: OPENCLAW,
    session_name: undefined,
    prompt: 'Run the tests and fix calc.py',
    harness: 'openclaw',
    calls: 5,
    agents: [agent('main', { calls: 5 })],
    events: { compaction: 1, tool_loop: 3 },
  },
]
const call = (fields: Record<string, unknown>) => ({
  lifecycle_state: 'completed',
  response_status: 200,
  duration_ms: 900,
  prompt_tokens: 21000,
  completion_tokens: 120,
  decision: 'agentic',
  model: 'deepseek/deepseek-v4.1-flash',
  ...fields,
})
const claudeCalls = [
  call({
    record_id: 'r1',
    agent: 'main',
    timestamp: '2026-11-19T10:00:00Z',
    request_class: 'main',
  }),
  call({
    record_id: 'r2',
    agent: 'main',
    timestamp: '2026-11-19T10:00:10Z',
    markers: ['tool_loop'],
  }),
  call({
    record_id: 'r3',
    agent: 'subagent 1',
    agent_type: 'Explore',
    timestamp: '2026-11-19T10:00:20Z',
    decision: 'explore',
    request_class: 'subagent',
    markers: ['subagent_start'],
  }),
  call({
    record_id: 'r4',
    agent: 'main',
    timestamp: '2026-11-19T10:01:00Z',
    image_count: 1,
    markers: ['image', 'tool_loop'],
  }),
  call({
    record_id: 'r5',
    agent: 'main',
    timestamp: '2026-11-19T10:01:30Z',
    decision: 'housekeeping',
    request_class: 'compaction',
    markers: ['compaction'],
  }),
  call({
    record_id: 'r6',
    agent: 'main',
    timestamp: '2026-11-19T10:02:00Z',
    decision: 'guard',
    model: undefined,
    duration_ms: 12,
    prompt_tokens: undefined,
    completion_tokens: undefined,
    markers: ['guard', 'block'],
  }),
]

const records = [
  {
    id: 'cursor-own-model',
    timestamp: '2026-11-19T10:05:00Z',
    session_id: CURSOR,
    turn_index: 0,
    decision_tier: 0,
    decision_priority: 700,
    recipe: 'default',
    original_model: 'vllm-sr/auto',
    requested_model: 'gpt-5.6-sol',
    decision: 'fast',
    selected_model: 'qwen/qwen3.8-27b',
    lifecycle_state: 'completed',
    response_status: 200,
    duration_ms: 610,
    agent: { harness: 'cursor' },
  },
  {
    id: 'openclaw-compaction',
    timestamp: '2026-11-19T10:04:00Z',
    session_id: OPENCLAW,
    turn_index: 2,
    decision_tier: 0,
    decision_priority: 860,
    recipe: 'default',
    original_model: 'vllm-sr/auto',
    decision: 'housekeeping',
    selected_model: 'deepseek/deepseek-v4.1-flash',
    lifecycle_state: 'completed',
    response_status: 200,
    duration_ms: 5400,
    agent: { harness: 'openclaw', request_class: 'compaction' },
  },
]

const aggregate = {
  object: 'router_replay.aggregate',
  record_count: 12,
  lifecycle: { completed: 12, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
  summary: {
    total_saved: 0,
    baseline_spend: 0,
    actual_spend: 0,
    cost_record_count: 0,
    excluded_record_count: 12,
    by_currency: [
      {
        currency: 'USD',
        total_saved: 0,
        baseline_spend: 0,
        actual_spend: 0,
        router_call_spend: 0.004,
        cost_record_count: 0,
      },
    ],
  },
  model_selection: [{ name: 'deepseek/deepseek-v4.1-flash', value: 12 }],
  decision_distribution: [{ name: 'agentic', value: 12 }],
  signal_distribution: [],
  token_volume: {
    input_tokens: 130000,
    output_tokens: 3600,
    total_tokens: 133600,
    excluded_record_count: 0,
  },
  token_breakdown: {
    by_decision: [
      { name: 'agentic', input_tokens: 130000, output_tokens: 3600, total_tokens: 133600 },
    ],
    by_selected_model: [
      {
        name: 'deepseek/deepseek-v4.1-flash',
        input_tokens: 130000,
        output_tokens: 3600,
        total_tokens: 133600,
      },
    ],
  },
  available_recipes: ['default'],
  available_decisions: ['agentic', 'fast', 'housekeeping', 'guard'],
  available_models: ['deepseek/deepseek-v4.1-flash', 'qwen/qwen3.8-27b'],
}

async function setup(page: Page) {
  await mockAuthenticatedAppShell(page)
  const replays = '**/api/router/api/v1/observability/replays'
  await page.route(/\/api\/router\/api\/v1\/observability\/replays(\?|$)/, (route) =>
    route.fulfill({
      json: {
        object: 'router_replay.list',
        count: records.length,
        total: records.length,
        has_more: false,
        harness_counts: { cursor: 1, openclaw: 1 },
        data: records,
      },
    }),
  )
  await page.route(`${replays}/aggregate*`, (route) => route.fulfill({ json: aggregate }))
  await page.route(`${replays}/models`, (route) =>
    route.fulfill({ json: { object: 'list', data: [] } }),
  )
  await page.route(`${replays}/runs*`, (route) =>
    route.fulfill({
      json: {
        object: 'list',
        count: runs.length,
        total: runs.length,
        limit: 100,
        offset: 0,
        has_more: false,
        data: runs,
      },
    }),
  )
  await page.route(`${replays}/run?*`, (route) =>
    route.fulfill({ json: { object: 'router_replay.run', run: claudeRun, calls: claudeCalls } }),
  )
}

test('shows sessions as a tree of agents and turns with the shared agent events', async ({
  page,
}, info) => {
  await setup(page)
  await page.goto('/observe/requests?view=sessions')

  const sessions = page.getByRole('region', { name: 'Sessions' })
  const claude = sessions.locator(`[data-session="${CLAUDE}"]`)
  await expect(claude).toContainText('Fix the failing mean test')
  await expect(claude.getByRole('img', { name: 'Claude Code' })).toBeVisible()
  await expect(claude.locator('[data-event="compaction"]').first()).toBeVisible()
  await expect(claude.locator('[data-event="block"]').first()).toBeVisible()

  const cursor = sessions.locator(`[data-session="${CURSOR}"]`)
  await expect(cursor.getByRole('img', { name: 'Cursor' })).toBeVisible()
  await expect(cursor).toContainText('router-keyed')
  await expect(
    sessions.locator(`[data-session="${OPENCLAW}"]`).getByRole('img', { name: 'OpenClaw' }),
  ).toBeVisible()

  await claude.getByRole('button', { expanded: false }).first().click()
  const subagent = claude.locator('[data-agent="subagent 1"]')
  await expect(subagent).toContainText('Explore')
  await expect(subagent).toContainText('started by the main agent')
  await expect(subagent.locator('[data-turn="r3"] [data-event="subagent_start"]')).toBeVisible()
  const main = claude.locator('[data-agent="main"]').first()
  await expect(main.locator('[data-turn="r4"] [data-event="image"]')).toBeVisible()
  await expect(main.locator('[data-turn="r5"]')).toContainText('Compaction')
  await expect(main.locator('[data-turn="r6"] [data-event="block"]')).toBeVisible()
  await expect(main.locator('[data-turn="r6"] [data-event="guard"]')).toHaveCount(0)

  const signals = page.getByRole('table', { name: 'Signals by harness' })
  await expect(
    signals.locator('tr[data-harness="claude_code"] [data-signal="seen"]').first(),
  ).toBeVisible()
  await expect(signals.locator('tr[data-harness="cursor"]')).toContainText('router-keyed')
  await expect(signals.locator('tr[data-harness="cursor"]')).toContainText('not sent')

  await page.screenshot({ path: info.outputPath('sessions-tree.png'), fullPage: true })
})

test('shows a dash with its reason instead of N/A in the totals and the breakdown', async ({
  page,
}, info) => {
  await setup(page)
  await page.goto('/observe/requests')
  await expect(page.getByText('Cost vs reference')).toBeVisible()
  await expect(page.getByText('Saved by routing')).toHaveCount(0)
  await page.getByRole('button', { name: 'Breakdown' }).click()
  const breakdown = page.locator('#requests-breakdown')
  const table = breakdown.getByRole('table', { name: 'USD cost estimate' })
  await expect(table).toContainText('Reference cost')
  await expect(table.locator('[title^="No request in range"]').first()).toBeVisible()
  await expect(breakdown).toContainText('not a quality comparison')
  await expect(page.locator('main')).not.toContainText('N/A')
  await breakdown.screenshot({ path: info.outputPath('breakdown.png') })
})

test('shows the model a client asked for beside the one that answered', async ({ page }) => {
  await setup(page)
  await page.goto('/observe/requests')
  const row = page.getByRole('row').filter({ hasText: 'requested gpt-5.6-sol' })
  await expect(row.getByRole('img', { name: 'Cursor' })).toBeVisible()
  const compaction = page
    .getByRole('row')
    .filter({ has: page.getByRole('img', { name: 'OpenClaw' }) })
  await expect(compaction.locator('[data-event="compaction"]')).toBeVisible()
})
