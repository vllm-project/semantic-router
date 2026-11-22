import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const TURN = '0199a1b2-0000-4000-8000-00000000000a'
const NEXT_TURN = '0199a1b2-0000-4000-8000-00000000000b'

const base = {
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 0,
  signals: {},
  session_id: 'codex-session',
  recipe: 'coding',
  selected_model: 'gpt-6-luna',
  lifecycle_state: 'completed',
  response_status: 200,
  duration_ms: 900,
}

const call = {
  ...base,
  id: 'codex-call',
  timestamp: '2026-11-15T09:00:00Z',
  thread_id: 'codex-session',
  turn_id: TURN,
}

const review = {
  ...base,
  id: 'codex-review',
  timestamp: '2026-11-15T09:01:30Z',
  selected_model: 'codex-auto-review',
  request_kind: 'codex_approval_review',
  thread_id: 'review-thread',
  turn_id: 'review-turn',
  parent_turn_id: TURN,
  review: {
    schema_version: 1,
    identification_source: 'entrypoint',
    status: 'completed',
    assessment: { outcome: 'allow' },
    action: { tool: 'exec_command', summary: 'make test' },
  },
}

const route = (record_id: string, timestamp: string, fields: Record<string, unknown>) => ({
  record_id,
  timestamp,
  turn_index: 0,
  session_policy_applied: false,
  lifecycle_state: 'completed',
  response_status: 200,
  duration_ms: 900,
  selected_model: 'gpt-6-luna',
  ...fields,
})

const aggregate = {
  object: 'router_replay.aggregate',
  record_count: 2,
  lifecycle: { completed: 2, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
  summary: {
    total_saved: 0,
    baseline_spend: 0,
    actual_spend: 0,
    cost_record_count: 0,
    excluded_record_count: 2,
  },
  model_selection: [],
  decision_distribution: [],
  signal_distribution: [],
  token_volume: { input_tokens: 0, output_tokens: 0, total_tokens: 0, excluded_record_count: 2 },
  token_breakdown: { by_decision: [], by_selected_model: [] },
  available_recipes: ['coding'],
  available_decisions: [],
  available_models: ['gpt-6-luna'],
}

async function setup(page: Page) {
  await mockAuthenticatedAppShell(page)
  const lists: URLSearchParams[] = []
  const aggregates: URLSearchParams[] = []
  await page.route(/\/api\/router\/api\/v1\/observability\/replays\/aggregate/, (request) => {
    aggregates.push(new URL(request.request().url()).searchParams)
    return request.fulfill({ json: aggregate })
  })
  await page.route(/\/api\/router\/api\/v1\/observability\/replays(\?|$)/, (request) => {
    const query = new URL(request.request().url()).searchParams
    if (query.get('request_kind') === 'codex_approval_review') {
      return request.fulfill({ json: { object: 'router_replay.list', count: 1, data: [review] } })
    }
    lists.push(query)
    const data = query.get('turn_id') ? [call, review] : [call]
    return request.fulfill({
      json: { object: 'router_replay.list', count: data.length, total: data.length, data },
    })
  })
  await page.route(/\/api\/router\/api\/v1\/observability\/replays\/codex-call$/, (request) =>
    request.fulfill({ json: call }),
  )
  await page.route(/\/api\/router\/api\/v1\/observability\/replays\/trajectory/, (request) =>
    request.fulfill({
      json: {
        object: 'router_replay.trajectory',
        session_id: 'codex-session',
        recipe: 'coding',
        record_count: 3,
        turn_count: 2,
        messages: [],
        routes: [
          route('codex-call', '2026-11-15T09:00:00Z', { turn_id: TURN }),
          route('codex-subagent', '2026-11-15T09:00:10Z', {
            agent: '9f2c1a',
            turn_id: 'child-turn',
            parent_turn_id: TURN,
          }),
          route('codex-next', '2026-11-15T09:01:00Z', { turn_id: NEXT_TURN, turn_index: 1 }),
        ],
      },
    }),
  )
  return { lists, aggregates }
}

test("a record's Codex turn opens the list filtered to that turn until the filter is cleared", async ({
  page,
}, info) => {
  const { lists, aggregates } = await setup(page)
  await page.goto('/observe/requests/codex-call')

  const history = page.getByRole('region', { name: 'Session routing history' })
  await expect(history.getByRole('link', { name: 'Codex turn 0199a1b2…0000000a' })).toBeVisible()
  await expect(history.getByRole('row', { name: /Codex turn 0199a1b2…0000000a/ })).toContainText(
    '2 calls',
  )
  await expect(history.getByRole('row', { name: /Codex turn 0199a1b2…0000000b/ })).toContainText(
    '1 call',
  )
  await page.screenshot({ path: info.outputPath('insights-turn-record.png'), fullPage: true })

  const identity = page.getByRole('region', { name: 'Session Routing', exact: true })
  await identity.getByRole('button', { name: 'Expand Session Routing' }).click()
  await expect(identity).toContainText('codex-session')
  await identity.getByRole('link', { name: TURN }).click()

  await expect(page).toHaveURL(new RegExp(`/observe/requests\\?turn=${TURN}$`))
  const chip = page.getByRole('button', { name: 'Clear codex turn filter' }).locator('..')
  await expect(chip).toHaveAttribute('title', TURN)
  await expect(chip).toContainText('Codex turn')
  await expect(chip).toContainText('0199a1b2…0000000a')
  await expect.poll(() => lists.some((query) => query.get('turn_id') === TURN)).toBe(true)
  expect(aggregates.at(-1)?.get('turn_id')).toBe(TURN)
  // a turn reads in full, with no time bound
  await expect(page.getByRole('radio', { name: 'All' })).toHaveAttribute('data-state', 'on')
  expect(lists.find((query) => query.get('turn_id') === TURN)?.has('since')).toBe(false)
  await expect(page.getByRole('row').filter({ hasText: 'codex-auto-review' })).toHaveCount(1)
  await page.screenshot({ path: info.outputPath('insights-turn-list.png'), fullPage: true })

  await page.getByRole('button', { name: 'Clear codex turn filter' }).click()
  await expect(page).toHaveURL(/\/observe\/requests$/)
  await expect(chip).toHaveCount(0)
  await expect.poll(() => lists.at(-1)?.has('turn_id')).toBe(false)
  // without the turn the list returns to the last day
  await expect(page.getByRole('radio', { name: '24h' })).toHaveAttribute('data-state', 'on')
  expect(lists.at(-1)?.has('since')).toBe(true)
  await expect(page.getByRole('row').filter({ hasText: 'codex-auto-review' })).toHaveCount(0)
})
