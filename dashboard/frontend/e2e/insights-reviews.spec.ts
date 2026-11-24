import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const base = {
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 0,
  signals: {},
  session_id: 'agent-session',
  selected_model: 'codex-auto-review',
  request_kind: 'codex_approval_review',
  lifecycle_state: 'completed',
  response_status: 200,
}

const denied = {
  ...base,
  id: 'review-deny',
  timestamp: '2026-11-15T09:00:30Z',
  duration_ms: 1200,
  prompt_tokens: 4843,
  completion_tokens: 136,
  actual_cost: 0.004,
  currency: 'USD',
  review: {
    schema_version: 1,
    identification_source: 'entrypoint',
    status: 'completed',
    reviewed_session_id: 'agent-session',
    thread_id: 'review-thread',
    parent_turn_id: 'agent-turn',
    latency_ms: 1200,
    assessment: {
      outcome: 'deny',
      risk_level: 'high',
      user_authorization: 'low',
      rationale: 'Deletes build output the user did not ask to remove.',
    },
    action: {
      tool: 'exec_command',
      summary: 'rm -rf build',
      cwd: '/work/project',
      arguments: '{"command":["/bin/bash","-lc","rm -rf build"],"cwd":"/work/project"}',
    },
  },
}

const allowed = {
  ...base,
  id: 'review-allow',
  timestamp: '2026-11-15T09:01:10Z',
  duration_ms: 400,
  review: {
    schema_version: 1,
    identification_source: 'entrypoint',
    status: 'completed',
    reviewed_session_id: 'agent-session',
    latency_ms: 400,
    assessment: { outcome: 'allow' },
    action: { tool: 'exec_command', summary: "printf 'hi' > hello.txt" },
  },
}

const reviewSummary = {
  reviews: 2,
  allowed: 1,
  denied: 1,
  invalid: 0,
  failed: 0,
  cancelled: 0,
  pending: 0,
  tool_calls: 0,
  approval_rate: 0.5,
  latency_p50_ms: 400,
  latency_p95_ms: 1200,
  cost: [{ currency: 'USD', amount: 0.004, requests: 1 }],
  unpriced_requests: 1,
  bucket: 'minute',
  over_time: [
    { start: '2026-11-15T09:00:00Z', allowed: 0, denied: 1, invalid: 0, failed: 0, cancelled: 0 },
    { start: '2026-11-15T09:01:00Z', allowed: 1, denied: 0, invalid: 0, failed: 0, cancelled: 0 },
  ],
  by_model: [
    {
      model: 'codex-auto-review',
      requests: 2,
      cost: [{ currency: 'USD', amount: 0.004, requests: 1 }],
    },
  ],
}

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
  available_recipes: [],
  available_decisions: [],
  available_models: ['codex-auto-review'],
  available_review_models: ['codex-auto-review'],
  reviews: reviewSummary,
}

async function setup(page: Page) {
  await mockAuthenticatedAppShell(page)
  const queries: URLSearchParams[] = []
  await page.route(/\/api\/router\/api\/v1\/observability\/replays\/aggregate/, (route) =>
    route.fulfill({ json: aggregate }),
  )
  await page.route(/\/api\/router\/api\/v1\/observability\/replays(\?|$)/, (route) => {
    const query = new URL(route.request().url()).searchParams
    queries.push(query)
    const outcome = query.get('review_outcome')
    const data = [allowed, denied].filter(
      (record) => !outcome || record.review.assessment.outcome === outcome,
    )
    return route.fulfill({
      json: { object: 'router_replay.list', count: data.length, total: data.length, data },
    })
  })
  await page.route(/\/api\/router\/api\/v1\/observability\/replays\/review-deny$/, (route) =>
    route.fulfill({ json: denied }),
  )
  await page.route(/\/api\/router\/api\/v1\/observability\/replays\/trajectory/, (route) =>
    route.fulfill({
      json: {
        object: 'router_replay.trajectory',
        session_id: 'agent-session',
        record_count: 0,
        turn_count: 0,
        messages: [],
        routes: [],
      },
    }),
  )
  return queries
}

test('the reviews view summarizes reviews, filters them and opens a review', async ({
  page,
}, info) => {
  const queries = await setup(page)
  await page.goto('/observe/requests')
  await page.getByRole('tab', { name: /Reviews/ }).click()
  await expect(page).toHaveURL(/tab=reviews/)

  const summary = page.getByRole('region', { name: 'Review summary' })
  await expect(summary).toContainText('50.0%')
  // reviews stay live as the request list does
  await expect(page.getByRole('button', { name: 'Live' })).toHaveAttribute('aria-pressed', 'true')
  await expect(summary).toContainText('1 allowed of 2 verdicts')
  await expect(summary).toContainText('1 of 2 requests priced')
  const rows = page.getByRole('row')
  await expect(rows.filter({ hasText: 'rm -rf build' })).toContainText('Denied')
  await expect(rows.filter({ hasText: "printf 'hi' > hello.txt" })).toContainText('Allowed')
  expect(queries.at(-1)?.get('request_kind')).toBe('codex_approval_review')
  await page.screenshot({ path: info.outputPath('insights-reviews.png'), fullPage: true })

  await page.getByRole('button', { name: 'Result' }).click()
  await page.getByRole('menuitemcheckbox', { name: 'Denied' }).click()
  await page.keyboard.press('Escape')
  await expect(rows.filter({ hasText: "printf 'hi' > hello.txt" })).toHaveCount(0)
  expect(queries.at(-1)?.get('review_outcome')).toBe('deny')

  await page.getByPlaceholder('Filter by session ID…').fill('agent-session')
  await expect.poll(() => queries.at(-1)?.get('session_id')).toBe('agent-session')

  await rows.filter({ hasText: 'rm -rf build' }).getByRole('link').first().click()
  await expect(page).toHaveURL(/\/observe\/requests\/review-deny$/)
  const review = page.getByRole('region', { name: 'Approval Review', exact: true })
  await expect(review).toContainText('Deletes build output the user did not ask to remove.')
  await expect(review).toContainText('/work/project')
  await page.screenshot({ path: info.outputPath('insights-review-detail.png'), fullPage: true })
  await review.getByRole('link', { name: 'agent-session' }).click()
  await expect(page).toHaveURL(/tab=reviews&session=agent-session/)
  await expect(page.getByPlaceholder('Filter by session ID…')).toHaveValue('agent-session')
})
