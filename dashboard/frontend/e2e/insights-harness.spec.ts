import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

// Replay records as the router files agent harness traffic: a Claude Code
// session whose main agent spawned a Task subagent, a Codex thread, an
// OpenCode session with its session-title call, and a DeepSeek Harness session.
const CLAUDE = 'key_3fa9c2:49b948ac-48b9-4fd5-b4e8-ccb8f6fba483'
const CODEX = 'key_3fa9c2:01a1ea71-6532-7fc1-9137-1fd7cf9e9851'
const OPENCODE = 'key_3fa9c2:ses_e158e5346ffel2rgJH7CFjm2zB'
const DSH = 'session-6c1f0d2a'

const base = {
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 700,
  recipe: 'default',
  original_model: 'vllm-sr/norn-1',
  lifecycle_state: 'completed',
  response_status: 200,
  currency: 'USD',
  signals: { conversation: ['has_tools'] },
}

const records = [
  {
    ...base,
    id: 'dsh-call',
    timestamp: '2026-11-17T16:59:02Z',
    session_id: DSH,
    decision: 'fast',
    selected_model: 'qwen/qwen3.8-27b',
    duration_ms: 142,
    agent: { harness: 'deepseek_harness' },
  },
  {
    ...base,
    id: 'opencode-title',
    timestamp: '2026-11-17T16:58:03Z',
    session_id: OPENCODE,
    session_name: 'Subagent reports function name from calc.py',
    request_kind: 'session_title',
    turn_index: 1,
    decision: 'housekeeping',
    selected_model: 'deepseek/deepseek-v4.1-flash',
    duration_ms: 873,
    agent: { harness: 'opencode' },
    signals: { agent: ['housekeeping'] },
  },
  {
    ...base,
    id: 'codex-call',
    timestamp: '2026-11-17T16:57:45Z',
    session_id: CODEX,
    thread_id: '01a1ea71-6532-7fc1-9137-1fd7cf9e9851',
    turn_id: '0199a1b2-0000-4000-8000-00000000000a',
    decision: 'fast',
    selected_model: 'qwen/qwen3.8-27b',
    duration_ms: 1395,
    agent: { harness: 'codex' },
  },
  {
    ...base,
    id: 'claude-subagent',
    timestamp: '2026-11-17T16:57:39Z',
    session_id: CLAUDE,
    decision: 'agentic',
    selected_model: 'deepseek/deepseek-v4.1-flash',
    duration_ms: 170,
    tool_trace: { tool_names: ['Read'] },
    agent: { harness: 'claude_code', id: '3fa9c2d81e07' },
    route_diagnostics: { session_action: 'stay', session_reason: 'session_scope_protect' },
  },
  {
    ...base,
    id: 'claude-main',
    timestamp: '2026-11-17T16:57:32Z',
    session_id: CLAUDE,
    decision: 'agentic',
    selected_model: 'deepseek/deepseek-v4.1-flash',
    duration_ms: 1719,
    tool_trace: { tool_names: ['Agent'] },
    agent: { harness: 'claude_code' },
    route_diagnostics: { session_action: 'establish_baseline', session_reason: 'new_session' },
  },
]

const cost = {
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
}
const agent = (label: string, fields: Record<string, unknown> = {}) => ({
  label,
  calls: 1,
  first_call_at: '2026-11-17T16:57:32Z',
  last_call_at: '2026-11-17T16:57:39Z',
  models: ['deepseek/deepseek-v4.1-flash'],
  cost,
  ...fields,
})
const claudeRun = {
  session_id: CLAUDE,
  session_name: 'Explain the retry loop in replay_store.go',
  harness: 'claude_code',
  started_at: '2026-11-17T16:57:32Z',
  ended_at: '2026-11-17T16:57:40Z',
  duration_ms: 8000,
  calls: 2,
  agents: [agent('main'), agent('subagent 1', { id: '3fa9c2d81e07', parent: 'main' })],
  models: [{ name: 'deepseek/deepseek-v4.1-flash', value: 2 }],
  outcome: { status: 'answered', tool_results: 1, tool_failures: 0 },
  cost,
}
const runs = [
  claudeRun,
  {
    ...claudeRun,
    session_id: CODEX,
    session_name: 'Add a histogram for selection latency',
    harness: 'codex',
    agents: [agent('main')],
  },
  {
    ...claudeRun,
    session_id: OPENCODE,
    session_name: 'Subagent reports function name from calc.py',
    harness: 'opencode',
    agents: [agent('main')],
  },
  {
    ...claudeRun,
    session_id: DSH,
    session_name: undefined,
    prompt: 'Sum the numbers in data.csv',
    harness: 'deepseek_harness',
    agents: [agent('main')],
  },
]

const aggregate = {
  object: 'router_replay.aggregate',
  record_count: records.length,
  lifecycle: { completed: records.length, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
  summary: {
    total_saved: 0,
    baseline_spend: 0,
    actual_spend: 0,
    cost_record_count: 0,
    excluded_record_count: records.length,
  },
  model_selection: [],
  decision_distribution: [],
  signal_distribution: [],
  token_volume: {
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    excluded_record_count: records.length,
  },
  token_breakdown: { by_decision: [], by_selected_model: [] },
  available_recipes: ['default'],
  available_decisions: ['agentic', 'fast', 'housekeeping'],
  available_models: ['deepseek/deepseek-v4.1-flash', 'qwen/qwen3.8-27b'],
}

async function setup(page: Page) {
  await mockAuthenticatedAppShell(page)
  const replays = '**/api/router/api/v1/observability/replays'
  await page.route(/\/api\/router\/api\/v1\/observability\/replays(\?|$)/, (route) => {
    const query = new URL(route.request().url()).searchParams
    const session = query.get('session_id')
    const kind = query.get('request_kind')
    const data = records.filter(
      (record) =>
        (!session || record.session_id === session) && (!kind || record.request_kind === kind),
    )
    return route.fulfill({
      json: {
        object: 'router_replay.list',
        count: data.length,
        total: data.length,
        has_more: false,
        data,
      },
    })
  })
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
    route.fulfill({
      json: {
        object: 'router_replay.run',
        run: claudeRun,
        calls: [
          {
            record_id: 'claude-main',
            agent: 'main',
            timestamp: '2026-11-17T16:57:32Z',
            duration_ms: 1719,
            model: 'deepseek/deepseek-v4.1-flash',
            lifecycle_state: 'completed',
            response_status: 200,
          },
          {
            record_id: 'claude-subagent',
            agent: 'subagent 1',
            timestamp: '2026-11-17T16:57:39Z',
            duration_ms: 170,
            model: 'deepseek/deepseek-v4.1-flash',
            lifecycle_state: 'completed',
            response_status: 200,
          },
        ],
      },
    }),
  )
  await page.route(`${replays}/trajectory*`, (route) =>
    route.fulfill({
      json: {
        object: 'router_replay.trajectory',
        session_id: CLAUDE,
        record_count: 2,
        turn_count: 1,
        messages: [],
        routes: [],
      },
    }),
  )
  await page.route(`${replays}/claude-subagent`, (route) => route.fulfill({ json: records[3] }))
}

test('lists each request with its harness logo and agent, filters by harness and hides housekeeping', async ({
  page,
}, info) => {
  await setup(page)
  await page.goto('/observe/requests')
  const rows = page.getByRole('row')
  const row = (text: string) => rows.filter({ hasText: text })

  await expect(row('DeepSeek Harness').getByRole('img', { name: 'DeepSeek Harness' })).toBeVisible()
  await expect(row('Codex').getByRole('img', { name: 'Codex' })).toBeVisible()
  await expect(row('Codex')).toContainText('main agent')
  await expect(rows.filter({ has: page.locator('[data-request-class="subagent"]') })).toContainText(
    'Claude Code',
  )
  await expect(
    rows
      .filter({ has: page.locator('[data-request-class="subagent"]') })
      .getByRole('img', { name: 'Claude Code' }),
  ).toBeVisible()
  const housekeeping = rows.filter({ has: page.locator('[data-request-class="housekeeping"]') })
  await expect(housekeeping).toContainText('OpenCode')
  await expect(housekeeping).toContainText('housekeeping')
  await expect(housekeeping.getByRole('img', { name: 'OpenCode' })).toBeVisible()
  // the session reads as the harness sent it, its title when the harness named it
  await expect(housekeeping).toContainText('Subagent reports function name from calc.py')
  await expect(page.getByText('key_3fa9c2:', { exact: false })).toHaveCount(0)
  await page.screenshot({ path: info.outputPath('requests-harness-list.png'), fullPage: true })

  await page.getByRole('button', { name: 'Client' }).click()
  const opencode = page.getByRole('menuitemcheckbox', { name: /OpenCode/ })
  await expect(opencode).toContainText('1')
  await expect(opencode.getByRole('img', { name: 'OpenCode' })).toBeVisible()
  await opencode.click()
  await page.keyboard.press('Escape')
  await expect(page).toHaveURL(/harness=opencode/)
  await expect(rows.filter({ hasText: 'Codex' })).toHaveCount(0)
  await expect(housekeeping).toHaveCount(1)

  await page.getByLabel('Hide housekeeping').click()
  await expect(page).toHaveURL(/housekeeping=hide/)
  await expect(housekeeping).toHaveCount(0)
})

test("a session is a pivot: its requests, then its run's agent tree", async ({ page }) => {
  await setup(page)
  await page.goto('/observe/requests')
  await page
    .getByRole('row')
    .filter({ hasText: 'Codex' })
    .getByRole('button', { name: 'Add a histogram for selection latency' })
    .click()
  await expect(page).toHaveURL(/session=key_3fa9c2%3A01a1ea71/)
  const chip = page.getByRole('button', { name: 'Clear session filter' }).locator('..')
  await expect(chip).toContainText('Add a histogram for selection latency')
  await expect(chip.getByRole('button', { name: 'Open run' })).toBeVisible()
  await expect(page.getByRole('row').filter({ hasText: 'OpenCode' })).toHaveCount(0)

  await page.goto('/observe/requests?tab=runs')
  const run = page.getByRole('row').filter({ hasText: 'Explain the retry loop in replay_store.go' })
  await expect(run.getByRole('img', { name: 'Claude Code' })).toBeVisible()
  await expect(
    page
      .getByRole('row')
      .filter({ hasText: 'Sum the numbers in data.csv' })
      .getByRole('img', { name: 'DeepSeek Harness' }),
  ).toBeVisible()
  await run.click()
  await expect(page).toHaveURL(/\/observe\/runs\?session=/)
  const agents = page.getByRole('list', { name: 'Agents' })
  await expect(agents).toContainText('main')
  await expect(agents).toContainText('subagent 1')
  await expect(agents).toContainText('Spawned by main')
})

test('a record names its harness, its agent, the tools it used and how session routing held its model', async ({
  page,
}, info) => {
  await setup(page)
  await page.goto('/observe/requests/claude-subagent')
  const agentCard = page.getByRole('region', { name: 'Agent' })
  await expect(agentCard.getByRole('img', { name: 'Claude Code' })).toBeVisible()
  await expect(agentCard).toContainText('subagent 1')
  await expect(agentCard).toContainText('Spawned by main')
  await expect(agentCard).toContainText('Read')
  await expect(agentCard).toContainText('Kept the session model')
  await expect(agentCard).toContainText('49b948ac-48b9-4fd5-b4e8-ccb8f6fba483')
  await expect(agentCard.getByRole('link', { name: 'Open agent run' })).toHaveAttribute(
    'href',
    /\/observe\/runs\?session=/,
  )
  await page.screenshot({ path: info.outputPath('record-agent.png'), fullPage: true })
})
