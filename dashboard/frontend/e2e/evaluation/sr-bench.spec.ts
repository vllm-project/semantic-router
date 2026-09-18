import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'

const target = {
  id: 'single',
  kind: 'single',
  model: 'model-a',
  base_url: 'http://localhost:8000/v1',
}
const manifest = {
  version: 'sr-bench-1.0',
  name: 'Baseline test',
  mode: 'live',
  profile: 'quick',
  targets: [target],
  limits: {},
  sampling: {},
  seed: 42,
}
const run = {
  id: 'run-1',
  status: 'completed',
  created_at: '2026-09-18T00:00:00Z',
  updated_at: '2026-09-18T00:00:10Z',
  manifest,
  progress: { total: 2, completed: 2, failed: 0 },
}
const report = {
  version: 'sr-bench-1.0',
  run_id: run.id,
  status: 'completed',
  summary: {
    targets: [
      {
        id: 'single',
        total: 2,
        completed: 2,
        failed: 0,
        correct: 1,
        scored: 2,
        accuracy: 0.5,
        macro_accuracy: 0.5,
        sr_bench_score: null,
        cost_usd: null,
        tokens: 120,
        latency_p50_s: 1,
        latency_p95_s: 2,
        selected_models: { 'model-a': 2 },
        decisions: { reasoning: 2 },
      },
    ],
    wall_time_s: 10,
  },
  benchmarks: [],
  limitations: ['Subset evaluation; not a complete sr-bench score.'],
  provenance: { dataset_sha256: 'a'.repeat(64) },
}

async function mockBench(page: Page, settings: Record<string, unknown> = {}) {
  await mockAuthenticatedAppShell(page, { settings })
  const requests: unknown[] = []
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const path = new URL(route.request().url()).pathname.replace('/api/sr-bench/v1', '')
    let body: unknown
    if (path === '/catalog')
      body = {
        version: 'sr-bench-1.0',
        profiles: ['smoke', 'quick', 'standard'].map((id) => ({ id, purpose: id })),
        benchmarks: [
          'mmlu-pro',
          'gpqa-diamond',
          'hle',
          'livecodebench',
          'scicode',
          'terminal-bench',
          'simpleqa-verified',
          'arc-agi',
          'tau3',
        ].map((id) => ({ id, title: id, kind: 'capability' })),
      }
    else if (path === '/datasets')
      body = {
        datasets: [
          {
            id: 'quick-v1',
            name: 'Frozen quick suite',
            path: '/prepared/quick.jsonl',
            sha256: 'a'.repeat(64),
            case_count: 2,
            profile: 'quick',
            seed: 42,
            split: 'dev',
            benchmarks: ['mmlu-pro'],
          },
        ],
      }
    else if (path === '/datasets/compose')
      body = {
        dataset: {
          id: 'quick-v1',
          path: '/prepared/quick.jsonl',
          sha256: 'a'.repeat(64),
          case_count: 2,
          profile: 'quick',
          benchmarks: ['mmlu-pro'],
        },
      }
    else if (path === '/targets') body = { targets: [target] }
    else if (path === '/plans') {
      requests.push(route.request().postDataJSON())
      body = {
        status: 'validated',
        total: 2,
        plan_sha256: 'b'.repeat(64),
        manifest: route.request().postDataJSON().manifest,
      }
    } else if (path === '/runs' && route.request().method() === 'POST') {
      requests.push(route.request().postDataJSON())
      body = { ...run, manifest: route.request().postDataJSON().manifest }
    } else if (path === '/runs')
      body = {
        runs: [run, { ...run, id: 'run-2', manifest: { ...manifest, name: 'Candidate test' } }],
      }
    else if (path === '/runs/run-1') body = run
    else if (/^\/runs\/[^/]+\/report$/.test(path)) body = report
    else if (path === '/runs/run-1/results')
      body = {
        total: 1,
        limit: 100,
        next_cursor: null,
        results: [
          {
            case_id: 'case-a',
            target_id: 'single',
            benchmark: 'mmlu-pro',
            status: 'completed',
            score: 1,
            answer: 'B',
            cost_usd: null,
            latency_s: 1,
            details: { selected_model: 'model-a', decision: 'reasoning' },
          },
        ],
      }
    else if (path === '/runs/run-1/calls')
      body = { calls: [], total: 0, limit: 100, next_cursor: null }
    else if (path === '/runs/run-1/events') body = { events: [{ seq: 1, type: 'completed' }] }
    else if (path === '/comparisons')
      body = {
        baseline_selection: 'Best observed single model on identical cases.',
        baseline_tied_best_target_ids: ['single', 'another-single'],
        baseline_tie_policy: 'Lowest complete known subject cost, then stable target ID.',
        comparisons: [
          {
            baseline_target_id: 'single',
            candidate_target_id: 'balance',
            paired_cases: 2,
            wins: 1,
            losses: 0,
            ties: 1,
            baseline_cost_usd: 1,
            candidate_cost_usd: 0.8,
            quality_delta: 0.1,
            cost_saving_percent: 20,
            quality_delta_ci95: [-0.1, 0.3],
          },
        ],
      }
    else {
      await route.fulfill({ status: 404, json: { error: 'No fixture for route' } })
      return
    }
    await route.fulfill({ json: body })
  })
  return requests
}

async function chooseDataset(page: Page, id: string) {
  const selector = page.getByRole('combobox', { name: 'Prepared dataset', exact: true })
  if (!(await selector.isVisible()))
    await page.getByText('Prepared source collection', { exact: true }).click()
  await selector.selectOption(id)
}

async function section(page: Page, name: string) {
  await page.getByRole('tab', { name, exact: true }).click()
}

test('plans and launches a reusable frozen dataset from the normal form', async ({ page }) => {
  const requests = await mockBench(page)
  await page.goto('/evaluation?view=new')
  await expect(page.getByRole('heading', { name: 'sr-bench 1.0' })).toBeVisible()
  await chooseDataset(page, 'quick-v1')
  await page.getByLabel('Add configured target').selectOption('single')
  await expect(page.getByRole('button', { name: 'Start evaluation' })).toBeDisabled()
  await page.getByRole('button', { name: 'Review plan' }).click()
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  await page.getByLabel('Budget (USD)').fill('6')
  await expect(page.getByRole('button', { name: 'Start evaluation' })).toBeDisabled()
  await page.getByRole('button', { name: 'Review plan' }).click()
  await page.getByRole('button', { name: 'Start evaluation' }).click()
  await expect(page).toHaveURL(/run=run-1/)
  expect(requests).toHaveLength(3)
  expect(requests[2]).toMatchObject({
    manifest: {
      dataset: { sha256: 'a'.repeat(64) },
      targets: [target],
      limits: { max_cost_usd: 6 },
    },
  })
})

test('preserves registered profiles and blocks output cap conflicts before planning', async ({
  page,
}) => {
  const requests = await mockBench(page)
  const requestParams = { max_tokens: 4096, temperature: 1, top_p: 0.95, seed: 42 }
  await page.route('**/api/sr-bench/v1/targets', (route) =>
    route.fulfill({ json: { targets: [{ ...target, request_params: requestParams }] } }),
  )
  await page.goto('/evaluation?view=new')
  await chooseDataset(page, 'quick-v1')
  await page
    .getByRole('combobox', { name: 'Add configured target', exact: true })
    .selectOption('single')
  const profile = page.getByRole('region', { name: 'single request profile', exact: true })
  await expect(
    profile.getByText('Output tokens: 4,096 (registered override)', { exact: true }),
  ).toBeVisible()
  await profile.getByText('Registered sampling overrides', { exact: true }).click()
  await expect(profile.locator('pre')).toContainText('4096')
  await page.getByText('Advanced manifest', { exact: true }).click()
  await page.getByLabel('Use edited manifest', { exact: true }).check()
  await expect(profile.getByRole('heading', { name: 'Effective request profile' })).toHaveCount(0)
  await expect(profile.getByText('Output tokens:', { exact: false })).toHaveCount(0)
  await expect(
    profile.getByText('Request parameters come from the edited manifest and reviewed plan.'),
  ).toBeVisible()
  await expect(profile.locator('pre')).toContainText('4096')
  await page.getByLabel('Use edited manifest', { exact: true }).uncheck()
  await page.getByLabel('Max output tokens', { exact: true }).fill('512')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText(
    'Target single has a registered output limit of 4096 tokens, above the run cap of 512',
  )
  expect(requests).toHaveLength(0)
  await expect(page.getByLabel('Max output tokens', { exact: true })).toHaveValue('512')
  await page.getByLabel('Max output tokens', { exact: true }).fill('4096')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(
    page.getByRole('heading', { name: 'Plan ready for review', exact: true }),
  ).toBeVisible()
  expect(requests).toHaveLength(1)
  expect(requests[0]).toMatchObject({
    manifest: { targets: [{ request_params: requestParams }], limits: { max_output_tokens: 4096 } },
  })
})

test('shows truthful metrics, routing distribution and case evidence', async ({ page }) => {
  await mockBench(page)
  await page.goto('/evaluation?view=runs&run=run-1')
  await expect(page.getByRole('heading', { name: 'Target comparison' })).toBeVisible()
  await expect(page.getByRole('cell', { name: '50%', exact: true })).toBeVisible()
  await expect(
    page.getByText(
      'Costs apply frozen per-token prices to recorded usage; they are not invoice or hardware-cost measurements.',
      { exact: true },
    ),
  ).toBeVisible()
  await expect(page.getByText('Subset evaluation; not a complete sr-bench score.')).toBeVisible()
  await expect(page.getByRole('cell', { name: '$0.00000' })).toHaveCount(0)
  await section(page, 'Questions')
  await page.getByRole('button', { name: 'case-a' }).click()
  await expect(page.getByRole('heading', { name: 'Final answer' })).toBeVisible()
  await section(page, 'Evidence')
  await expect(page.getByRole('link', { name: 'Open report JSON' })).toHaveAttribute(
    'href',
    '/api/sr-bench/v1/runs/run-1/report',
  )
})

test('reconciles a lost initial submission after reload with the same identity and account', async ({
  page,
}) => {
  await mockBench(page)
  let actorID = 'user-admin-1'
  await page.route('**/api/auth/me', (route) =>
    route.fulfill({
      json: {
        user: {
          id: actorID,
          email: 'test@example.com',
          name: 'Test operator',
          role: 'admin',
          permissions: ['evaluation.read', 'evaluation.write', 'evaluation.run'],
        },
      },
    }),
  )
  const submissions: Record<string, unknown>[] = []
  await page.route('**/api/sr-bench/v1/runs', async (route) => {
    if (route.request().method() !== 'POST') return route.fulfill({ json: { runs: [run] } })
    submissions.push(route.request().postDataJSON())
    if (submissions.length === 1) return route.abort('failed')
    await route.fulfill({ json: run })
  })
  await page.goto('/evaluation?view=new')
  await chooseDataset(page, 'quick-v1')
  await page.getByLabel('Add configured target').selectOption('single')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await page.getByRole('button', { name: 'Start evaluation', exact: true }).click()
  await expect(
    page.getByRole('heading', { name: 'Evaluation submission needs reconciliation' }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Review plan', exact: true })).toHaveCount(0)
  await page.reload()
  await expect(
    page.getByRole('button', { name: 'Check or submit same evaluation', exact: true }),
  ).toBeVisible()
  expect(submissions).toHaveLength(1)
  actorID = 'another-user'
  await page.reload()
  await expect(page.getByRole('button', { name: 'Review plan', exact: true })).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Check or submit same evaluation', exact: true }),
  ).toHaveCount(0)
  actorID = 'user-admin-1'
  await page.reload()
  await page.getByRole('button', { name: 'Check or submit same evaluation', exact: true }).click()
  await expect(page).toHaveURL(/run=run-1/)
  expect(submissions).toHaveLength(2)
  expect(submissions[1]).toEqual(submissions[0])
  expect(
    await page.evaluate(() => sessionStorage.getItem('sr-bench-submission:user-admin-1')),
  ).toBeNull()
})

test('ignores an old response after remount without clearing a newer submission', async ({
  page,
}) => {
  await mockBench(page)
  const submissions: Array<{ idempotency_key: string; manifest: typeof manifest }> = []
  let finishOld!: () => void
  const oldPending = new Promise<void>((resolve) => {
    finishOld = resolve
  })
  let finishNew!: () => void
  const newPending = new Promise<void>((resolve) => {
    finishNew = resolve
  })
  await page.route('**/api/sr-bench/v1/runs', async (route) => {
    if (route.request().method() !== 'POST') return route.fulfill({ json: { runs: [] } })
    submissions.push(route.request().postDataJSON())
    const index = submissions.length
    if (index === 1) await oldPending
    if (index === 3) await newPending
    await route.fulfill({ json: { ...run, id: index === 3 ? 'run-2' : 'run-1' } })
  })
  await page.goto('/evaluation?view=new')
  await chooseDataset(page, 'quick-v1')
  await page.getByLabel('Add configured target').selectOption('single')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await page.getByRole('button', { name: 'Start evaluation', exact: true }).click()
  await expect.poll(() => submissions.length).toBe(1)
  await page.getByRole('button', { name: 'Datasets', exact: true }).click()
  await page.getByRole('button', { name: 'Create evaluation', exact: true }).click()
  await page.getByRole('button', { name: 'Check or submit same evaluation', exact: true }).click()
  await expect(page).toHaveURL(/run=run-1/)
  expect(submissions[1]).toEqual(submissions[0])
  await page.getByRole('button', { name: 'Create evaluation', exact: true }).click()
  await page.getByLabel('Run name', { exact: true }).fill('Newer submission')
  await chooseDataset(page, 'quick-v1')
  await page.getByLabel('Add configured target').selectOption('single')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await page.getByRole('button', { name: 'Start evaluation', exact: true }).click()
  await expect.poll(() => submissions.length).toBe(3)
  const oldResponse = page.waitForResponse(
    (response) =>
      response.request().method() === 'POST' &&
      response.request().postDataJSON().idempotency_key === submissions[0].idempotency_key,
  )
  finishOld()
  await (await oldResponse).finished()
  await page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  )
  await expect(page).toHaveURL(/view=new$/)
  const saved = await page.evaluate(() =>
    JSON.parse(sessionStorage.getItem('sr-bench-submission:user-admin-1')!),
  )
  expect(saved.idempotencyKey).toBe(submissions[2].idempotency_key)
  finishNew()
  await expect(page).toHaveURL(/run=run-2/)
})

test('does not navigate a new account from a prior account delayed submission', async ({
  page,
}) => {
  await mockBench(page)
  let finishOld!: () => void
  const oldPending = new Promise<void>((resolve) => {
    finishOld = resolve
  })
  let submitted = false
  await page.route('**/api/sr-bench/v1/runs', async (route) => {
    if (route.request().method() !== 'POST') return route.fulfill({ json: { runs: [] } })
    submitted = true
    await oldPending
    await route.fulfill({ json: run })
  })
  await page.route('**/api/auth/logout', (route) => route.fulfill({ json: {} }))
  await page.route('**/api/auth/login', (route) =>
    route.fulfill({
      json: {
        token: 'second-test-session',
        user: {
          id: 'second-account',
          name: 'Second account',
          email: 'second@example.com',
          role: 'admin',
        },
      },
    }),
  )
  await page.goto('/evaluation?view=new')
  await chooseDataset(page, 'quick-v1')
  await page.getByLabel('Add configured target').selectOption('single')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await page.getByRole('button', { name: 'Start evaluation', exact: true }).click()
  await expect.poll(() => submitted).toBe(true)
  await page.getByRole('button', { name: 'Open account menu for Admin User', exact: true }).click()
  await page.getByRole('button', { name: 'Logout', exact: true }).click()
  await expect(page).toHaveURL(/\/login/)
  await page.getByLabel('Email', { exact: true }).fill('second@example.com')
  await page.getByLabel('Password', { exact: true }).fill('fixture-password')
  await page.getByRole('button', { name: 'Continue', exact: true }).click()
  await expect(
    page.getByRole('button', { name: 'Open account menu for Second account', exact: true }),
  ).toBeVisible()
  await page.evaluate(() => {
    history.pushState({}, '', '/evaluation?view=new')
    window.dispatchEvent(new PopStateEvent('popstate'))
  })
  await expect(page.getByRole('button', { name: 'Review plan', exact: true })).toBeVisible()
  const oldResponse = page.waitForResponse(
    (response) =>
      response.request().method() === 'POST' && new URL(response.url()).pathname.endsWith('/runs'),
  )
  finishOld()
  await (await oldResponse).finished()
  await page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  )
  await expect(page).toHaveURL(/view=new$/)
  expect(
    await page.evaluate(() => sessionStorage.getItem('sr-bench-submission:user-admin-1')),
  ).not.toBeNull()
  expect(
    await page.evaluate(() => sessionStorage.getItem('sr-bench-submission:second-account')),
  ).toBeNull()
})

test('fails closed when an initial submission cannot be preserved in session storage', async ({
  page,
}) => {
  const requests = await mockBench(page)
  await page.addInitScript(() => {
    const original = Storage.prototype.setItem
    Storage.prototype.setItem = function (key, value) {
      if (key.startsWith('sr-bench-submission:'))
        throw new DOMException('Storage full', 'QuotaExceededError')
      original.call(this, key, value)
    }
  })
  await page.goto('/evaluation?view=new')
  await chooseDataset(page, 'quick-v1')
  await page.getByLabel('Add configured target').selectOption('single')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await page.getByRole('button', { name: 'Start evaluation', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('No request was sent')
  expect(requests).toHaveLength(1)
})

test('clears an initial submission only after the service proves no dispatch occurred', async ({
  page,
}) => {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/runs', async (route) => {
    if (route.request().method() !== 'POST') return route.fulfill({ json: { runs: [] } })
    await route.fulfill({
      status: 400,
      json: { error: 'Frozen plan rejected before dispatch', dispatch_started: false },
    })
  })
  await page.goto('/evaluation?view=new')
  await chooseDataset(page, 'quick-v1')
  await page.getByLabel('Add configured target').selectOption('single')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await page.getByRole('button', { name: 'Start evaluation', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Frozen plan rejected before dispatch')
  await expect(page.getByRole('button', { name: 'Review plan', exact: true })).toBeEnabled()
  await expect(page.getByRole('button', { name: 'Start evaluation', exact: true })).toBeDisabled()
  expect(
    await page.evaluate(() => sessionStorage.getItem('sr-bench-submission:user-admin-1')),
  ).toBeNull()
})

test('shows learning preview selection and snapshot evidence without a capability score', async ({
  page,
}) => {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/runs/run-1', (route) =>
    route.fulfill({ json: { ...run, manifest: { ...manifest, mode: 'preview' } } }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/report', (route) =>
    route.fulfill({
      json: {
        ...report,
        summary: {
          targets: [
            {
              id: 'balance',
              total: 2,
              completed: 2,
              failed: 0,
              selected_models: { 'model-a': 1 },
              decisions: { learned: 2 },
              selection_statuses: { selected: 1, execution_required: 1 },
              selection_reasons: {
                'Selected from captured state': 1,
                'Execution is still required': 1,
              },
            },
          ],
        },
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/results?*', (route) =>
    route.fulfill({
      json: {
        total: 2,
        limit: 100,
        next_cursor: null,
        results: [
          {
            case_id: 'learned-case',
            target_id: 'balance',
            benchmark: 'mmlu-pro',
            status: 'completed',
            details: {
              routing: {
                selected_model: 'model-a',
                selection_status: 'selected',
                selection_method: 'learning',
                selection_reason: 'Selected from captured state',
                decision_result: { decision_name: 'learned' },
                selection_provenance: {
                  mode: 'read_only_snapshot',
                  config_hash: 'c'.repeat(64),
                  state_hash: 'd'.repeat(64),
                  state_dependent: true,
                  captured_at: '2026-09-18T00:00:00Z',
                  sampled: true,
                  sampling_seed: 42,
                  caveat: 'This sampled choice is not a guarantee of a later live selection.',
                },
              },
            },
          },
          {
            case_id: 'unresolved-case',
            target_id: 'balance',
            benchmark: 'mmlu-pro',
            status: 'completed',
            details: {
              routing: {
                selection_status: 'execution_required',
                selection_reason: 'Execution is still required',
                decision_result: { decision_name: 'learned' },
              },
            },
          },
        ],
      },
    }),
  )
  await page.goto('/evaluation?view=runs&run=run-1')
  await expect(page.getByRole('heading', { name: 'Selection status', exact: true })).toBeVisible()
  await expect(
    page.getByText('Some model selections require live execution.', { exact: false }),
  ).toBeVisible()
  await expect(page.getByRole('cell', { name: '50%', exact: true })).toHaveCount(0)
  await section(page, 'Questions')
  await page.getByRole('button', { name: 'learned-case', exact: true }).click()
  const evidence = page.getByRole('region', { name: 'Routing preview evidence' })
  await expect(evidence.getByText('model-a', { exact: true })).toBeVisible()
  await expect(evidence.getByText('c'.repeat(64), { exact: true })).toBeVisible()
  await expect(evidence.getByText('d'.repeat(64), { exact: true })).toBeVisible()
  await expect(evidence.getByText('Yes · not eligible for replay', { exact: true })).toBeVisible()
  await expect(evidence.getByText('42', { exact: true })).toBeVisible()
  await expect(
    evidence.getByText('This sampled choice is not a guarantee of a later live selection.'),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Back to questions', exact: true }).click()
  await page.getByRole('button', { name: 'unresolved-case', exact: true }).click()
  await expect(evidence.getByText('Execution is still required', { exact: true })).toBeVisible()
  await expect(evidence.getByText('model-a', { exact: true })).toHaveCount(0)
})

test('distinguishes pending, unavailable and genuinely empty run evidence', async ({ page }) => {
  await mockBench(page)
  let mode: 'pending' | 'failed' | 'empty' = 'pending'
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  for (const resource of ['report', 'results', 'calls']) {
    await page.route(`**/api/sr-bench/v1/runs/run-1/${resource}*`, async (route) => {
      if (mode === 'pending') await gate
      if (mode === 'failed') {
        await route.fulfill({ status: 503, json: { error: 'Saved evidence read failed' } })
        return
      }
      await route.fulfill({
        json:
          resource === 'report'
            ? { ...report, summary: { ...report.summary, targets: [] } }
            : { [resource]: [], total: 0, limit: 100, next_cursor: null },
      })
    })
  }
  await page.goto('/evaluation?view=runs&run=run-1')
  await expect(page.getByText('Loading report metrics…', { exact: true })).toBeVisible()
  await expect(
    page.getByText('Complete quality and cost evidence is needed for this chart.', { exact: true }),
  ).toHaveCount(0)
  await section(page, 'Questions')
  await expect(page.getByText('Loading persisted case results…', { exact: true })).toBeVisible()
  await section(page, 'Calls')
  await expect(page.getByText('Loading persisted call records…', { exact: true })).toBeVisible()
  await expect(page.getByText('No matching persisted results.', { exact: true })).toHaveCount(0)
  await expect(
    page.getByText('No routing trace recorded for these results.', { exact: true }),
  ).toHaveCount(0)
  await expect(page.getByText('No persisted call records yet.', { exact: true })).toHaveCount(0)
  mode = 'failed'
  release()
  await section(page, 'Results')
  await expect(
    page.getByText('Report metrics are unavailable. Refresh evidence to retry.', { exact: true }),
  ).toBeVisible()
  await section(page, 'Questions')
  await expect(
    page.getByText('Case results are unavailable: Saved evidence read failed', { exact: true }),
  ).toBeVisible()
  await section(page, 'Calls')
  await expect(
    page.getByText('Call records are unavailable: Saved evidence read failed', { exact: true }),
  ).toBeVisible()
  await expect(page.getByText('No persisted case results yet.', { exact: true })).toHaveCount(0)
  await expect(page.getByText('No persisted call records yet.', { exact: true })).toHaveCount(0)
  mode = 'empty'
  await page.getByRole('button', { name: 'Refresh evidence', exact: true }).click()
  await section(page, 'Questions')
  await expect(page.getByText('No persisted case results yet.', { exact: true })).toBeVisible()
  await section(page, 'Calls')
  await expect(page.getByText('No persisted call records yet.', { exact: true })).toBeVisible()
  await section(page, 'Results')
  await expect(
    page.getByText('Summary metrics will appear as results are persisted.', { exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('recovers every event page after a temporary read failure', async ({ page }) => {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/runs/run-1', (route) =>
    route.fulfill({ json: { ...run, status: 'running' } }),
  )
  const cursors: number[] = []
  let failed = false
  await page.route('**/api/sr-bench/v1/runs/run-1/events?*', async (route) => {
    const cursor = Number(new URL(route.request().url()).searchParams.get('after'))
    cursors.push(cursor)
    if (cursor === 1000 && !failed) {
      failed = true
      await route.fulfill({ status: 503, json: { error: 'Event page unavailable' } })
      return
    }
    await route.fulfill({
      json: {
        events:
          cursor === 0
            ? Array.from({ length: 1000 }, (_, index) => ({ seq: index + 1, type: 'progress' }))
            : [{ seq: 1001, type: 'recovered' }],
      },
    })
  })
  await page.goto('/evaluation?view=runs&run=run-1')
  await expect(page.getByRole('alert')).toContainText('Event page unavailable')
  await section(page, 'Evidence')
  await expect(page.getByText('Run events (1001)', { exact: true })).toBeVisible({ timeout: 10000 })
  expect(cursors.slice(0, 4)).toEqual([0, 1000, 0, 1000])
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('loads bounded evidence pages on demand and keeps full report aggregates', async ({
  page,
}) => {
  await mockBench(page)
  const resultReads: number[] = []
  const callReads: number[] = []
  const detailReads: string[] = []
  let failLastPage = true
  await page.route('**/api/sr-bench/v1/runs/run-1/report', (route) =>
    route.fulfill({
      json: {
        ...report,
        summary: {
          ...report.summary,
          targets: [
            {
              ...report.summary.targets[0],
              total: 250,
              completed: 250,
              correct: 125,
              scored: 250,
              selected_models: { 'full-report-model': 250 },
              decisions: { 'full-report-decision': 250 },
            },
          ],
        },
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/results?*', async (route) => {
    const query = new URL(route.request().url()).searchParams
    const after = Number(query.get('after'))
    expect(query.get('limit')).toBe('100')
    resultReads.push(after)
    if (after === 200 && failLastPage) {
      failLastPage = false
      await route.fulfill({ status: 503, json: { error: 'Evidence read temporarily unavailable' } })
      return
    }
    await route.fulfill({
      json: {
        results: Array.from({ length: Math.min(100, 250 - after) }, (_, i) => ({
          case_id: `case-${after + i}`,
          target_id: 'single',
          benchmark: 'mmlu-pro',
          status: 'completed',
          score: 1,
          answer: 'B',
        })),
        total: 250,
        limit: 100,
        next_cursor: after < 200 ? after + 100 : null,
      },
    })
  })
  await page.route('**/api/sr-bench/v1/runs/run-1/calls?*', async (route) => {
    const query = new URL(route.request().url()).searchParams
    const after = Number(query.get('after'))
    expect(query.get('limit')).toBe('100')
    callReads.push(after)
    await route.fulfill({
      json: {
        calls: Array.from({ length: Math.min(100, 201 - after) }, (_, i) => ({
          id: `call-${after + i}`,
          case_id: `case-${after + i}`,
          target_id: 'single',
          role: 'subject',
          status: 'completed',
          model: 'page-only-model',
        })),
        total: 201,
        limit: 100,
        next_cursor: after < 200 ? after + 100 : null,
      },
    })
  })
  await page.route('**/api/sr-bench/v1/runs/run-1/calls/call-*', async (route) => {
    const id = new URL(route.request().url()).pathname.split('/').at(-1)!
    detailReads.push(id)
    await route.fulfill({
      json: {
        id,
        request: { messages: [{ content: 'Saved call prompt' }] },
        final: 'Saved final answer',
      },
    })
  })
  await page.goto('/evaluation?view=runs&run=run-1')
  await section(page, 'Questions')
  await expect(
    page.getByText('Loaded 100 of 250 persisted results.', { exact: false }),
  ).toBeVisible()
  await section(page, 'Calls')
  await expect(
    page.getByText('Showing 100 of 201 persisted call summaries.', { exact: false }),
  ).toBeVisible()
  const callTable = page.getByRole('region', { name: 'Persisted call records', exact: true })
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 844 })
    await callTable.evaluate((element) => {
      element.scrollTop = element.scrollHeight
    })
    expect(
      await callTable.evaluate((element) => {
        const viewport = element.getBoundingClientRect()
        const heading = element.querySelector('thead th')!.getBoundingClientRect()
        return (
          element.clientHeight <= window.innerHeight * 0.6 + 1 &&
          element.scrollHeight > element.clientHeight &&
          heading.top >= viewport.top - 1 &&
          heading.top <= viewport.top + 2
        )
      }),
    ).toBe(true)
  }
  await page.setViewportSize({ width: 1280, height: 844 })
  await section(page, 'Results')
  await expect(page.getByText('single: full-report-model', { exact: true })).toBeVisible()
  await expect(page.getByText('single: full-report-decision', { exact: true })).toBeVisible()
  await expect(page.getByRole('cell', { name: '125 / 250', exact: true })).toBeVisible()
  expect(resultReads).toEqual([0])
  expect(callReads).toEqual([0])
  expect(detailReads).toEqual([])
  await section(page, 'Questions')
  await page.getByRole('button', { name: 'Load more results', exact: true }).click()
  await expect(
    page.getByText('Loaded 200 of 250 persisted results.', { exact: false }),
  ).toBeVisible()
  await page.getByLabel('Filter loaded results').fill('case-150')
  await page.getByRole('button', { name: 'case-150', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'case-150 · single' })).toBeVisible()
  await page.getByRole('button', { name: 'Back to questions', exact: true }).click()
  await page.getByRole('button', { name: 'Load more results', exact: true }).click()
  await expect(page.getByText('Evidence read temporarily unavailable')).toBeVisible()
  expect(resultReads).toEqual([0, 100, 200])
  await expect(
    page.getByText('Loaded 200 of 250 persisted results.', { exact: false }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Load more results', exact: true }).click()
  await expect(
    page.getByText('Loaded 250 of 250 persisted results.', { exact: false }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Load more results', exact: true })).toHaveCount(0)
  await section(page, 'Calls')
  await page.getByRole('button', { name: 'Load more call records', exact: true }).click()
  await expect(
    page.getByText('Showing 200 of 201 persisted call summaries.', { exact: false }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'call-150', exact: true }).click()
  await page.getByText('Original call receipt', { exact: true }).click()
  await expect(page.getByText('Saved call prompt', { exact: false })).toBeVisible()
  expect(detailReads).toEqual(['call-150'])
  await section(page, 'Results')
  await expect(page.getByRole('cell', { name: '125 / 250', exact: true })).toBeVisible()
})

test('compares complete runs using paired results', async ({ page }) => {
  await mockBench(page)
  await page.goto('/evaluation?view=compare')
  await expect(
    page.getByText(
      'Costs apply frozen per-token prices to recorded usage; they are not invoice or hardware-cost measurements.',
      { exact: true },
    ),
  ).toBeVisible()
  await page.getByLabel('Baseline run').selectOption('run-1')
  await page.getByRole('checkbox', { name: /Candidate test/ }).check()
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await expect(page.getByRole('heading', { name: 'Comparison evidence' })).toBeVisible()
  await page.getByText('Technical comparison evidence', { exact: true }).click()
  await page.getByText('Baseline selection policy', { exact: true }).click()
  await expect(
    page.getByText('Tied best single models: single, another-single.', { exact: false }),
  ).toBeVisible()
  await page.getByText('Detailed iteration metrics and uncertainty', { exact: true }).click()
  await expect(page.getByRole('cell', { name: '-10 pp to 30 pp', exact: false })).toBeVisible()
})

test('keeps actions disabled in a server readonly session', async ({ page }) => {
  await mockBench(page, { serverReadonly: true })
  await page.goto('/evaluation?view=new')
  await expect(page.getByRole('button', { name: 'Review plan' })).toBeDisabled()
  await expect(
    page.getByText('Run controls require evaluation permissions and a writable Dashboard session.'),
  ).toBeVisible()
})

test('explains an unavailable worker without exposing a broken workspace', async ({ page }) => {
  await mockBench(page, {
    srBenchAvailable: false,
    srBenchUnavailableReason: 'Start the sr-bench service.',
  })
  await page.goto('/evaluation')
  await expect(page.getByText('Start the sr-bench service.')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Start evaluation' })).toHaveCount(0)
})

test('explains a failed run from saved case evidence when its terminal error is absent', async ({
  page,
}) => {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/runs/run-1', (route) =>
    route.fulfill({ json: { ...run, status: 'failed', error: null } }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/report', (route) =>
    route.fulfill({
      json: {
        ...report,
        status: 'failed',
        failure: {
          case_id: 'case-a',
          target_id: 'balance',
          reason: 'Runtime identity acknowledgement mismatched',
          inferred_from_saved_results: true,
        },
      },
    }),
  )
  await page.goto('/evaluation?view=runs&run=run-1')
  await expect(page.getByRole('alert')).toContainText('Runtime identity acknowledgement mismatched')
  await expect(page.getByRole('alert')).toContainText(
    'Target balance · case case-a · from saved case evidence',
  )
  await expect(
    page.getByText('This run is failed. Partial results are not a completed evaluation.'),
  ).toBeVisible()
})

test('catalog renders every registered adapter and fits mobile width', async ({ page }) => {
  await mockBench(page)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/evaluation?view=catalog')
  await expect(page.getByRole('heading', { name: 'tau3', exact: true })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'hle', exact: true })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'scicode', exact: true })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
})

test('replays saved answers and labels estimated metrics separately', async ({ page }) => {
  await mockBench(page)
  const replayRun = {
    ...run,
    id: 'replay-1',
    manifest: { ...manifest, name: 'Saved replay', mode: 'replay' },
  }
  const submitted: unknown[] = []
  await page.route('**/api/sr-bench/v1/runs', (route) =>
    route.fulfill({
      json: {
        runs: [
          run,
          {
            ...run,
            id: 'preview-1',
            manifest: { ...manifest, name: 'Route preview', mode: 'preview' },
          },
        ],
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/replays', (route) => {
    submitted.push(route.request().postDataJSON())
    return route.fulfill({ json: replayRun })
  })
  await page.route('**/api/sr-bench/v1/runs/replay-1**', (route) => {
    const path = new URL(route.request().url()).pathname
    const body = path.endsWith('/report')
      ? {
          ...report,
          summary: {
            targets: [
              {
                id: 'balance',
                accuracy: null,
                macro_accuracy: null,
                sr_bench_score: null,
                cost_usd: null,
                estimated_macro_accuracy: 0.75,
                estimated_cost_usd: 0.2,
                estimated_latency_p50_s: 1,
              },
            ],
            wall_time_s: 0,
          },
          limitations: ['Replay diagnostics only.'],
        }
      : path.endsWith('/calls')
        ? { calls: [], total: 0, limit: 100, next_cursor: null }
        : path.endsWith('/results')
          ? { results: [], total: 0, limit: 100, next_cursor: null }
          : path.endsWith('/events')
            ? { events: [] }
            : replayRun
    return route.fulfill({ json: body })
  })
  await page.goto('/evaluation?view=compare')
  await page.getByText('Reuse saved answers for diagnostic replay', { exact: true }).click()
  await page.getByLabel('Saved single-model baseline').selectOption('run-1')
  await page.getByLabel('Routing preview').selectOption('preview-1')
  await page.getByRole('button', { name: 'Create diagnostic replay' }).click()
  await expect(page.getByRole('heading', { name: 'Diagnostic replay estimates' })).toBeVisible()
  await expect(page.getByRole('cell', { name: '75%', exact: true })).toBeVisible()
  await expect(
    page.getByText('These estimates reuse saved answers.', { exact: false }),
  ).toBeVisible()
  expect(submitted).toEqual([{ baseline_run_id: 'run-1', preview_run_id: 'preview-1' }])
})

test('regrades saved answers and exports development evidence with holdout errors visible', async ({
  page,
}) => {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/runs/run-1/regrade', (route) =>
    route.fulfill({
      json: { kind: 'offline-regrade', changed_count: 0, model_requests: 0, results: [] },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/export', (route) =>
    route.fulfill({ status: 400, json: { error: 'Holdout rows cannot be exported for training' } }),
  )
  await page.goto('/evaluation?view=runs&run=run-1')
  await section(page, 'Evidence')
  await page.getByRole('button', { name: 'Regrade saved answers' }).click()
  await expect(page.getByText('0 changed grades · 0 model requests')).toBeVisible()
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download artifact JSON' }).click()
  expect((await download).suggestedFilename()).toBe('run-1-offline-regrade.json')
  await page.getByRole('button', { name: 'Export training matrix' }).click()
  await expect(page.getByRole('alert')).toContainText(
    'Holdout rows cannot be exported for training',
  )
})

test('manages long-lived runs and uses a dataset from its frozen inventory', async ({ page }) => {
  await mockBench(page)
  await page.goto('/evaluation')
  await expect(page.getByRole('heading', { name: 'Evaluation runs' })).toBeVisible()
  const bars = page
    .getByRole('region', { name: 'Evaluation runs', exact: true })
    .getByRole('progressbar')
  await expect(bars).toHaveCount(2)
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 844 })
    expect(
      await bars.evaluateAll((elements) =>
        elements.every((element) => {
          const bar = element.getBoundingClientRect()
          const cell = element.closest('td')!.getBoundingClientRect()
          return (
            bar.left >= cell.left &&
            bar.right <= cell.right + 0.5 &&
            bar.top >= cell.top &&
            bar.bottom <= cell.bottom + 0.5
          )
        }),
      ),
    ).toBe(true)
  }
  await page.setViewportSize({ width: 1280, height: 844 })
  await page.getByLabel('Search runs').fill('Candidate')
  await expect(page.getByRole('button', { name: 'Baseline test', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Candidate test', exact: true })).toBeVisible()
  await page.getByLabel('Run status').selectOption('failed')
  await expect(page.getByText('No runs match these filters.')).toBeVisible()
  await page.getByRole('button', { name: 'Datasets', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Dataset library' })).toBeVisible()
  await page.getByRole('button', { name: 'Evaluate' }).click()
  await expect(page.getByRole('radio', { name: /^Quick/ })).toBeChecked()
  await expect(page).toHaveURL(/dataset=quick-v1/)
})

test('automatically reconnects the task list without submitting evaluations', async ({ page }) => {
  const requests = await mockBench(page)
  let reads = 0
  let recovered = false
  await page.route('**/api/sr-bench/v1/runs', async (route) => {
    reads++
    await route.fulfill(
      !recovered
        ? { status: 503, json: { error: 'Temporary disconnect' } }
        : { json: { runs: [run] } },
    )
  })
  await page.goto('/evaluation')
  await expect(page.getByRole('alert')).toContainText('Temporary disconnect')
  recovered = true
  await expect(page.getByRole('button', { name: 'Baseline test', exact: true })).toBeVisible({
    timeout: 10000,
  })
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(requests).toHaveLength(0)
  expect(reads).toBeGreaterThan(1)
})

test('reopens arbitrary optimization comparisons from the saved URL', async ({
  page,
}, testInfo) => {
  await mockBench(page)
  const iterations = [
    'Current Balance',
    'Balance round 1',
    'Balance round 2',
    'Balance round 3',
  ].map((name, index) => ({
    ...run,
    id: `balance-${index}`,
    manifest: {
      ...manifest,
      name,
      targets: [{ ...target, id: 'balance', kind: 'mom', config_hash: String(index).repeat(64) }],
    },
  }))
  await page.route('**/api/sr-bench/v1/runs', (route) =>
    route.fulfill({ json: { runs: [run, ...iterations] } }),
  )
  await page.route('**/api/sr-bench/v1/runs/*/report', (route) => {
    const baseline = route.request().url().includes('/run-1/')
    return route.fulfill({
      json: {
        ...report,
        summary: {
          ...report.summary,
          targets: [
            {
              ...report.summary.targets[0],
              id: baseline ? 'single' : 'balance',
              macro_accuracy: baseline ? 0.5 : 0.6,
              cost_usd: baseline ? 1 : 0.8,
            },
          ],
        },
      },
    })
  })
  await page.goto('/evaluation?view=compare')
  await page.getByLabel('Baseline run').selectOption('run-1')
  await page.getByRole('button', { name: 'Select all', exact: true }).click()
  await expect(page.getByRole('checkbox', { checked: true })).toHaveCount(4)
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await expect(page.getByRole('article').getByText('20%', { exact: true })).toHaveCount(4)
  await expect(page).toHaveURL(/candidate=balance-3/)
  await page.reload()
  await expect(
    page.getByRole('article').getByRole('heading', { name: 'Balance round 3', exact: true }),
  ).toBeVisible()
  await page.getByText('Technical comparison evidence', { exact: true }).click()
  await page.getByText('Frozen configuration identity', { exact: true }).nth(2).click()
  await expect(page.getByText('2'.repeat(64), { exact: true })).toBeVisible()
  await page.getByText('Detailed iteration metrics and uncertainty', { exact: true }).click()
  await expect(page.getByRole('cell', { name: '20% saving', exact: false })).toHaveCount(4)
  const exported = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Export comparison CSV', exact: true }).click()
  expect((await exported).suggestedFilename()).toBe('sr-bench-comparison.csv')
  await page.getByText('Frozen configuration identity', { exact: true }).nth(2).click()
  await page.getByText('Technical comparison evidence', { exact: true }).click()
  await page.getByText('Detailed iteration metrics and uncertainty', { exact: true }).click()
  await page.screenshot({
    path: testInfo.outputPath('optimization-comparison-desktop.png'),
    fullPage: true,
  })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
    .toBe(true)
  await expect
    .poll(() =>
      page
        .getByRole('region', { name: 'sr-bench workspace' })
        .evaluate((element) => element.getBoundingClientRect().right <= window.innerWidth),
    )
    .toBe(true)
  await page.screenshot({
    path: testInfo.outputPath('optimization-comparison-mobile.png'),
    fullPage: true,
  })
})

test('shows available datasets while a run read stalls and recovers without false empty states', async ({
  page,
}) => {
  const submissions = await mockBench(page)
  await page.clock.install()
  let stalled = true
  await page.route('**/api/sr-bench/v1/runs', async (route) => {
    if (!stalled) await route.fulfill({ json: { runs: [run] } })
  })
  await page.goto('/evaluation?view=datasets')
  await expect(page.getByRole('heading', { name: 'Dataset library', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Evaluate', exact: true })).toBeVisible()
  await expect(page.getByText('Runs loading', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Runs (0)', exact: true })).toHaveCount(0)
  await page.clock.fastForward(30001)
  await expect(page.getByRole('alert')).toContainText(
    'runs: Reading saved sr-bench evidence timed out',
  )
  await expect(page.getByRole('button', { name: 'Evaluate', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Runs', exact: true }).click()
  await expect(page.getByText('Run inventory is unavailable.', { exact: true })).toBeVisible()
  await expect(page.getByText('No evaluation runs yet.', { exact: true })).toHaveCount(0)
  stalled = false
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Baseline test', exact: true })).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(submissions).toHaveLength(0)
})

test('requires explicit recovery scope and reuses a lost-response submission after reload', async ({
  page,
}) => {
  await mockBench(page)
  const failedRun = { ...run, status: 'failed', progress: { total: 2, completed: 1, failed: 1 } }
  await page.route('**/api/sr-bench/v1/runs/run-1', (route) => route.fulfill({ json: failedRun }))
  const submissions: Record<string, unknown>[] = []
  await page.route('**/api/sr-bench/v1/runs/run-1/recover-plan', (route) =>
    route.fulfill({
      json: {
        parent_run_id: 'run-1',
        mode: route.request().postDataJSON().mode,
        eligible_cells: [{ target_id: 'single', case_id: 'failed-case' }],
        excluded: [{ target_id: 'single', case_id: 'complete-case', reason: 'already completed' }],
        counts: { eligible: 1, excluded: 1 },
        parent: {
          status: 'failed',
          progress: failedRun.progress,
          known_spend_usd: 0.2,
          spend_complete: true,
        },
        plan_sha256: 'f'.repeat(64),
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/recover', async (route) => {
    submissions.push(route.request().postDataJSON())
    await route.fulfill(
      submissions.length === 1
        ? { status: 503, json: { error: 'Response unavailable' } }
        : { json: { ...run, id: 'child-1' } },
    )
  })
  await page.goto('/evaluation?view=runs&run=run-1')
  await page.getByLabel('Recovery scope').selectOption('failed')
  await page.getByRole('button', { name: 'Review recovery plan' }).click()
  await page.getByLabel('Recover single failed-case', { exact: true }).check()
  await expect(page.getByRole('button', { name: 'Create recovery run (1 cases)' })).toBeDisabled()
  await page.getByLabel('I acknowledge these are new model attempts', { exact: false }).check()
  await page.getByRole('button', { name: 'Create recovery run (1 cases)' }).click()
  await expect(page.getByText('Recovery submission needs reconciliation')).toBeVisible()
  await page.reload()
  await expect(page.getByText('Recovery submission needs reconciliation')).toBeVisible()
  await page.getByRole('button', { name: 'Check or submit same recovery' }).click()
  await expect(page).toHaveURL(/run=child-1/)
  expect(submissions).toHaveLength(2)
  expect(submissions[1]).toEqual(submissions[0])
  expect(submissions[0]).toMatchObject({
    mode: 'failed',
    acknowledge_new_attempt: true,
    cells: [{ case_id: 'failed-case', target_id: 'single' }],
  })
})

async function mockUndispatchedRecovery(page: Page) {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/runs/run-1', (route) =>
    route.fulfill({ json: { ...run, status: 'cancelled' } }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/recover-plan', (route) =>
    route.fulfill({
      json: {
        parent_run_id: 'run-1',
        mode: 'undispatched',
        eligible_cells: [{ case_id: 'unstarted', target_id: 'single' }],
        excluded: [],
        counts: { eligible: 1, excluded: 0 },
        parent: {
          status: 'cancelled',
          progress: run.progress,
          known_spend_usd: 0,
          spend_complete: true,
        },
        plan_sha256: 'f'.repeat(64),
      },
    }),
  )
}

async function openParentInSameDocument(page: Page) {
  await page.evaluate(() => {
    history.pushState({}, '', '/evaluation?view=runs&run=run-1')
    window.dispatchEvent(new PopStateEvent('popstate'))
  })
}

test('keeps a newer recovery intent when an unmounted recovery responds late', async ({ page }) => {
  await mockUndispatchedRecovery(page)
  const submissions: Array<{ idempotency_key: string }> = []
  let finishOld!: () => void
  const oldPending = new Promise<void>((resolve) => {
    finishOld = resolve
  })
  let finishNew!: () => void
  const newPending = new Promise<void>((resolve) => {
    finishNew = resolve
  })
  await page.route('**/api/sr-bench/v1/runs/run-1/recover', async (route) => {
    submissions.push(route.request().postDataJSON())
    const index = submissions.length
    if (index === 1) await oldPending
    if (index === 3) await newPending
    await route.fulfill({ json: { ...run, id: index === 3 ? 'child-2' : 'child-1' } })
  })
  await page.goto('/evaluation?view=runs&run=run-1')
  await page.getByRole('button', { name: 'Review recovery plan', exact: true }).click()
  await page.getByLabel('Recover single unstarted', { exact: true }).check()
  await page.getByRole('button', { name: 'Create recovery run (1 cases)', exact: true }).click()
  await expect.poll(() => submissions.length).toBe(1)
  await page.getByRole('button', { name: 'Datasets', exact: true }).click()
  await expect(
    page.getByRole('heading', { name: 'Recover unfinished work', exact: true }),
  ).toHaveCount(0)
  await openParentInSameDocument(page)
  await page.getByRole('button', { name: 'Check or submit same recovery', exact: true }).click()
  await expect(page).toHaveURL(/run=child-1/)
  expect(submissions[1]).toEqual(submissions[0])
  await openParentInSameDocument(page)
  await page.getByRole('button', { name: 'Review recovery plan', exact: true }).click()
  await page.getByLabel('Recover single unstarted', { exact: true }).check()
  await page.getByRole('button', { name: 'Create recovery run (1 cases)', exact: true }).click()
  await expect.poll(() => submissions.length).toBe(3)
  const oldResponse = page.waitForResponse(
    (response) =>
      response.request().method() === 'POST' &&
      response.request().postDataJSON().idempotency_key === submissions[0].idempotency_key,
  )
  finishOld()
  await (await oldResponse).finished()
  await page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  )
  await expect(page).toHaveURL(/run=run-1$/)
  const saved = await page.evaluate(() =>
    JSON.parse(sessionStorage.getItem('sr-bench-recovery:user-admin-1:run-1')!),
  )
  expect(saved.request.idempotency_key).toBe(submissions[2].idempotency_key)
  finishNew()
  await expect(page).toHaveURL(/run=child-2/)
})

test('isolates recovery intent across logout and ignores the old account response', async ({
  page,
}) => {
  await mockUndispatchedRecovery(page)
  let finishOld!: () => void
  const oldPending = new Promise<void>((resolve) => {
    finishOld = resolve
  })
  let submitted = false
  await page.route('**/api/sr-bench/v1/runs/run-1/recover', async (route) => {
    submitted = true
    await oldPending
    await route.fulfill({ json: { ...run, id: 'old-child' } })
  })
  await page.route('**/api/auth/logout', (route) => route.fulfill({ json: {} }))
  await page.route('**/api/auth/login', (route) =>
    route.fulfill({
      json: {
        token: 'second-test-session',
        user: {
          id: 'second-account',
          name: 'Second account',
          email: 'second@example.com',
          role: 'admin',
        },
      },
    }),
  )
  await page.goto('/evaluation?view=runs&run=run-1')
  await page.getByRole('button', { name: 'Review recovery plan', exact: true }).click()
  await page.getByLabel('Recover single unstarted', { exact: true }).check()
  await page.getByRole('button', { name: 'Create recovery run (1 cases)', exact: true }).click()
  await expect.poll(() => submitted).toBe(true)
  await page.getByRole('button', { name: 'Open account menu for Admin User', exact: true }).click()
  await page.getByRole('button', { name: 'Logout', exact: true }).click()
  await expect(page).toHaveURL(/\/login/)
  await page.getByLabel('Email', { exact: true }).fill('second@example.com')
  await page.getByLabel('Password', { exact: true }).fill('fixture-password')
  await page.getByRole('button', { name: 'Continue', exact: true }).click()
  await expect(
    page.getByRole('button', { name: 'Open account menu for Second account', exact: true }),
  ).toBeVisible()
  await openParentInSameDocument(page)
  await expect(
    page.getByRole('button', { name: 'Review recovery plan', exact: true }),
  ).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Check or submit same recovery', exact: true }),
  ).toHaveCount(0)
  const oldResponse = page.waitForResponse(
    (response) =>
      response.request().method() === 'POST' &&
      new URL(response.url()).pathname.endsWith('/recover'),
  )
  finishOld()
  await (await oldResponse).finished()
  await page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  )
  await expect(page).toHaveURL(/run=run-1$/)
  expect(
    await page.evaluate(() => sessionStorage.getItem('sr-bench-recovery:user-admin-1:run-1')),
  ).not.toBeNull()
  expect(
    await page.evaluate(() => sessionStorage.getItem('sr-bench-recovery:second-account:run-1')),
  ).toBeNull()
})

test('blocks recovery when its saved scope is corrupted', async ({ page }) => {
  await mockUndispatchedRecovery(page)
  await page.addInitScript(() =>
    sessionStorage.setItem('sr-bench-recovery:user-admin-1:run-1', '{broken'),
  )
  const submissions: string[] = []
  page.on('request', (request) => {
    if (request.method() === 'POST') submissions.push(request.url())
  })
  await page.goto('/evaluation?view=runs&run=run-1')
  await expect(page.getByRole('alert')).toContainText('saved recovery cannot be read safely')
  await expect(page.getByRole('button', { name: 'Review recovery plan', exact: true })).toHaveCount(
    0,
  )
  await expect(
    page.getByRole('button', { name: 'Check or submit same recovery', exact: true }),
  ).toHaveCount(0)
  expect(submissions).toEqual([])
})

test('displays and downloads a matching server-captured recipe', async ({ page }) => {
  await mockBench(page)
  const configHash = 'c'.repeat(64)
  let activeHash = configHash
  await page.route('**/api/sr-bench/v1/runs/run-1', (route) =>
    route.fulfill({
      json: {
        ...run,
        manifest: {
          ...manifest,
          targets: [{ ...target, id: 'balance', kind: 'mom', config_hash: configHash }],
        },
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/report', (route) =>
    route.fulfill({
      json: {
        ...report,
        provenance: {
          runner: {
            recipe_snapshots: {
              balance: {
                source: 'router_config_api_bracketed_hashes',
                source_config_hash: 'e'.repeat(64),
                generated_runtime_hash: configHash,
                active_runtime_hash: activeHash,
                config_hash: configHash,
                captured_at: '2026-09-18T00:00:00Z',
                recipe_sha256: 'd'.repeat(64),
                recipes: { balance: { decisions: [] } },
                redacted: true,
              },
            },
          },
        },
      },
    }),
  )
  await page.goto('/evaluation?view=runs&run=run-1')
  await section(page, 'Recipe')
  await expect(
    page.getByRole('heading', { name: 'balance · Verified config snapshot' }),
  ).toBeVisible()
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download balance recipe' }).click()
  expect((await download).suggestedFilename()).toBe('run-1-balance-recipe.json')
  activeHash = 'f'.repeat(64)
  await page.reload()
  await expect(page.getByRole('heading', { name: 'balance · Snapshot unavailable' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Download balance recipe' })).toHaveCount(0)
})

test('replans a definitely undispatched recovery after eligibility changes', async ({ page }) => {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/runs/run-1', (route) =>
    route.fulfill({ json: { ...run, status: 'failed' } }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/recover-plan', (route) =>
    route.fulfill({
      json: {
        parent_run_id: 'run-1',
        mode: 'undispatched',
        eligible_cells: [{ target_id: 'single', case_id: 'unstarted' }],
        excluded: [],
        counts: { eligible: 1, excluded: 0 },
        parent: {
          status: 'failed',
          progress: run.progress,
          known_spend_usd: 0,
          spend_complete: true,
        },
        plan_sha256: 'f'.repeat(64),
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/recover', (route) =>
    route.fulfill({
      status: 400,
      json: {
        error: 'Recovery eligibility changed; inspect a fresh recovery plan',
        code: 'recovery_plan_required',
        dispatch_started: false,
      },
    }),
  )
  await page.goto('/evaluation?view=runs&run=run-1')
  await page.getByRole('button', { name: 'Review recovery plan' }).click()
  await page.getByLabel('Recover single unstarted', { exact: true }).check()
  await page.getByRole('button', { name: 'Create recovery run (1 cases)' }).click()
  await expect(page.getByRole('alert')).toContainText('Recovery eligibility changed')
  await expect(page.getByRole('button', { name: 'Review recovery plan' })).toBeEnabled()
  await expect(page.getByRole('button', { name: 'Check or submit same recovery' })).toHaveCount(0)
  await page.reload()
  await expect(page.getByRole('button', { name: 'Review recovery plan' })).toBeEnabled()
})

test('keeps recovery lineage separate from the child denominator and spend', async ({ page }) => {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/runs/run-1/report', (route) =>
    route.fulfill({
      json: {
        ...report,
        recovery: {
          parent_run_id: 'parent-1',
          mode: 'undispatched',
          parent_snapshot: {
            progress: { completed: 30, total: 50 },
            known_spend_usd: 2.5,
            spend_complete: false,
          },
        },
        child_attempts: [
          { id: 'child-1', status: 'completed', progress: { completed: 1, total: 1, failed: 0 } },
        ],
      },
    }),
  )
  await page.goto('/evaluation?view=runs&run=run-1')
  await section(page, 'Evidence')
  await expect(page.getByRole('heading', { name: 'Recovery lineage' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Open parent run' })).toHaveAttribute(
    'href',
    '?view=runs&run=parent-1',
  )
  await expect(
    page.getByText('Parent snapshot: 30 / 50 completed', { exact: false }),
  ).toContainText('$2.50000 (incomplete accounting)')
  await expect(page.getByRole('link', { name: 'child-1', exact: true })).toBeVisible()
  await section(page, 'Results')
  await expect(page.getByRole('cell', { name: '1 / 2', exact: true })).toBeVisible()
  await section(page, 'Evidence')
  await expect(page.getByRole('cell', { name: '1 / 1', exact: true })).toBeVisible()
})

const accountingCorrection = {
  id: 'accounting-1',
  version: 'sr-bench-accounting-v1',
  created_at: '2026-09-18T12:00:00Z',
  evidence_sha256: 'e'.repeat(64),
  qualified: true,
  corrected_call_count: 1,
  verified_call_count: 2,
  unverifiable_call_count: 0,
  original_known_spend_usd: 1,
  corrected_known_spend_usd: 1.5,
  original_receipts_preserved: true,
  model_requests: 0,
}

test('distinguishes corrected report accounting from unchanged call and case receipts', async ({
  page,
}) => {
  const submissions = await mockBench(page)
  let qualified = true
  await page.route('**/api/sr-bench/v1/runs/run-1/report', (route) =>
    route.fulfill({
      json: {
        ...report,
        summary: {
          ...report.summary,
          targets: [{ ...report.summary.targets[0], cost_usd: qualified ? 1.5 : null }],
        },
        provenance: {
          accounting_correction: {
            ...accountingCorrection,
            qualified,
            unverifiable_call_count: qualified ? 0 : 1,
          },
        },
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/calls?*', (route) =>
    route.fulfill({
      json: {
        total: 1,
        limit: 100,
        next_cursor: null,
        calls: [
          {
            id: 'original-call',
            target_id: 'single',
            case_id: 'case-a',
            role: 'subject',
            status: 'completed',
            cost_usd: 1,
            latency_s: 1,
          },
        ],
      },
    }),
  )
  await page.goto('/evaluation?view=runs&run=run-1')
  const notice = page.getByRole('note', { name: 'Accounting correction' })
  await expect(notice.getByText('Accounting verified', { exact: true })).toBeVisible()
  await expect(notice.getByText('New model requests', { exact: true })).toBeVisible()
  await expect(page.getByRole('cell', { name: '$1.50000', exact: true })).toBeVisible()
  await notice.getByText('Accounting correction receipt', { exact: true }).click()
  await expect(notice.getByText('sr-bench-accounting-v1', { exact: true })).toBeVisible()
  await section(page, 'Questions')
  await expect(
    page.getByRole('columnheader', { name: 'Cost (original)', exact: true }),
  ).toBeVisible()
  await section(page, 'Calls')
  await expect(
    page.getByRole('columnheader', { name: 'Original cost / latency', exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('cell', { name: '$1.00000 / 1 s', exact: true })).toBeVisible()
  qualified = false
  await section(page, 'Results')
  await page.reload()
  await expect(notice.getByText('Partial accounting', { exact: true })).toBeVisible()
  await expect(
    notice.getByText('Unverifiable usage remains unknown', { exact: false }),
  ).toBeVisible()
  await expect(page.getByRole('cell', { name: '$1.50000', exact: true })).toHaveCount(0)
  expect(submissions).toHaveLength(0)
})

test('keeps accounting correction provenance visible for both sides of a comparison', async ({
  page,
}) => {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/runs/*/report', (route) =>
    route.fulfill({
      json: { ...report, provenance: { accounting_correction: accountingCorrection } },
    }),
  )
  await page.goto('/evaluation?view=compare&baseline=run-1&candidate=run-2')
  await expect(
    page.getByRole('article').getByText('Accounting verified', { exact: false }),
  ).toBeVisible()
  await page.getByText('Technical comparison evidence', { exact: true }).click()
  await expect(page.getByRole('note', { name: 'Accounting correction' })).toHaveCount(2)
  await expect(page.getByRole('heading', { name: 'Comparison evidence' })).toBeVisible()
})

test('uses conservative quality uncertainty and identifies a zero-width bootstrap diagnostic', async ({
  page,
}) => {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/comparisons', (route) =>
    route.fulfill({
      json: {
        baseline_selection: 'Best observed single model on identical cases.',
        comparisons: [
          {
            baseline_target_id: 'single',
            candidate_target_id: 'balance',
            paired_cases: 25,
            quality_delta: 0,
            quality_delta_ci95: [-0.47, 0.47],
            quality_delta_ci95_method: 'weighted-paired-hoeffding',
            quality_delta_ci95_qualification:
              'Case-independent bounded-difference interval with frozen benchmark weights; excludes strongest-baseline-selection, source contamination and tuning-selection uncertainty.',
            quality_delta_bootstrap_ci95: [0, 0],
            baseline_cost_usd: 1,
            candidate_cost_usd: 0.8,
            cost_saving_percent: 20,
          },
        ],
      },
    }),
  )
  await page.goto('/evaluation?view=compare&baseline=run-1&candidate=run-2')
  await page.getByText('Detailed iteration metrics and uncertainty', { exact: true }).click()
  const uncertainty = page.getByRole('cell').filter({ hasText: 'Conservative weighted Hoeffding' })
  await expect(uncertainty).toContainText('-47 pp to 47 pp')
  await expect(uncertainty).toContainText('25 paired cases')
  const diagnostic = uncertainty.getByText('Bootstrap diagnostic (95%):', { exact: false })
  await expect(diagnostic).not.toBeVisible()
  await uncertainty.getByText('Uncertainty details', { exact: true }).click()
  await expect(diagnostic).toBeVisible()
  await expect(diagnostic).toContainText('0 pp to 0 pp')
  await expect(diagnostic).toContainText('Use the conservative interval above for quality claims')
  await expect(uncertainty).toContainText('tuning-selection uncertainty')
})

test('separates observed savings from cache-neutral estimates and preserves unknown costs', async ({
  page,
}) => {
  const submissions = await mockBench(page)
  let complete = true
  await page.route('**/api/sr-bench/v1/runs/*/report', (route) =>
    route.fulfill({
      json: {
        ...report,
        summary: {
          ...report.summary,
          targets: [
            {
              ...report.summary.targets[0],
              cost_usd: 1,
              cache_neutral_cost_usd: complete ? 2 : null,
              cache_neutral_cost_basis: 'Frozen fresh-input and output rates.',
            },
          ],
        },
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/comparisons', (route) =>
    route.fulfill({
      json: {
        baseline_selection: 'Best observed single model on identical cases.',
        comparisons: [
          {
            baseline_target_id: 'single',
            candidate_target_id: 'balance',
            paired_cases: 2,
            quality_delta: 0,
            quality_delta_ci95: [-0.1, 0.1],
            baseline_cost_usd: 1,
            candidate_cost_usd: 0.8,
            cost_saving_percent: 20,
            cache_neutral_baseline_cost_usd: complete ? 2 : null,
            cache_neutral_candidate_cost_usd: complete ? 1.8 : null,
            cache_neutral_cost_saving_percent: complete ? 10 : null,
            cache_neutral_cost_basis: 'Counterfactual cost against the same baseline.',
          },
        ],
      },
    }),
  )
  await page.goto('/evaluation?view=runs&run=run-1')
  await expect(
    page.getByRole('columnheader', { name: 'Observed model cost', exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('cell', { name: '$1.00000', exact: true })).toBeVisible()
  await section(page, 'Evidence')
  await page.getByText('Token buckets and evaluation overhead', { exact: true }).click()
  await expect(page.getByRole('cell', { name: '$2.00000', exact: true })).toBeVisible()
  await section(page, 'Results')
  await page.getByText('Cost and timing interpretation', { exact: true }).click()
  await expect(
    page.getByText('neither billed spend nor a measured cache-free run.', { exact: false }),
  ).toBeVisible()
  await page.goto('/evaluation?view=compare&baseline=run-1&candidate=run-2')
  await page.getByText('Detailed iteration metrics and uncertainty', { exact: true }).click()
  await expect(page.getByRole('cell', { name: '$0.80000 20% saving', exact: true })).toBeVisible()
  await expect(
    page.getByRole('cell', {
      name: '$1.80000 10% estimated saving Baseline $2.00000',
      exact: true,
    }),
  ).toBeVisible()
  complete = false
  await page.reload()
  await page.getByText('Detailed iteration metrics and uncertainty', { exact: true }).click()
  await expect(page.getByRole('cell', { name: '$0.80000 20% saving', exact: true })).toBeVisible()
  await expect(
    page.getByRole('cell', { name: '— Saving unknown Baseline —', exact: true }),
  ).toBeVisible()
  await expect(page.getByText('10% estimated saving', { exact: true })).toHaveCount(0)
  expect(submissions).toHaveLength(0)
})

test('composes selected benchmarks under one size without resampling or generation', async ({
  page,
}, testInfo) => {
  const submissions = await mockBench(page)
  const prepared = [
    {
      id: 'quick-suite',
      profile: 'quick',
      seed: 42,
      split: 'dev',
      benchmarks: ['mmlu-pro', 'gpqa-diamond'],
      case_count: 10,
    },
    {
      id: 'smoke-suite',
      profile: 'smoke',
      seed: 42,
      split: 'dev',
      benchmarks: ['mmlu-pro', 'gpqa-diamond'],
      case_count: 2,
    },
    {
      id: 'wrong-seed',
      profile: 'quick',
      seed: 43,
      split: 'dev',
      benchmarks: ['hle'],
      case_count: 2,
    },
    {
      id: 'standard-suite',
      profile: 'standard',
      seed: 42,
      split: 'holdout',
      benchmarks: ['mmlu-pro', 'gpqa-diamond'],
      case_count: 100,
    },
  ].map((value) => ({
    ...value,
    name: `${value.profile} suite`,
    path: `/prepared/${value.id}.json`,
    sha256: 'a'.repeat(64),
  }))
  const compositions: unknown[] = []
  await page.route('**/api/sr-bench/v1/datasets', (route) =>
    route.fulfill({ json: { datasets: prepared } }),
  )
  await page.route('**/api/sr-bench/v1/datasets/compose', (route) => {
    compositions.push(route.request().postDataJSON())
    return route.fulfill({
      json: {
        dataset: { ...prepared[0], path: '/prepared/composed.json', sha256: 'd'.repeat(64) },
      },
    })
  })
  await page.goto('/evaluation?view=new')
  await expect(page.getByRole('radio')).toHaveCount(3)
  await page.getByRole('radio', { name: /^Smoke/ }).check()
  await expect(page.getByRole('checkbox', { name: /MMLU-Pro/ })).toBeChecked()
  await page.getByRole('radio', { name: /^Standard/ }).check()
  await page.getByRole('radio', { name: /^Quick/ }).check()
  await page.getByRole('button', { name: 'Clear benchmarks', exact: true }).click()
  await expect(page.getByRole('checkbox', { checked: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'Select all benchmarks', exact: true }).click()
  await expect(page.getByRole('checkbox', { checked: true })).toHaveCount(2)
  await page.getByRole('checkbox', { name: /GPQA Diamond/ }).uncheck()
  await expect(page.getByRole('checkbox', { name: /Humanity/ })).toBeDisabled()
  await page.getByLabel('Add configured target').selectOption('single')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  expect(compositions).toEqual([{ dataset_ids: ['quick-suite'], benchmarks: ['mmlu-pro'] }])
  expect(submissions).toHaveLength(1)
  expect(submissions[0]).toMatchObject({
    manifest: {
      profile: 'quick',
      dataset: { path: '/prepared/composed.json', sha256: 'd'.repeat(64) },
      targets: [target],
    },
  })
  await page.screenshot({
    path: testInfo.outputPath('create-evaluation-desktop.png'),
    fullPage: true,
  })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
    .toBe(true)
  await page.screenshot({
    path: testInfo.outputPath('create-evaluation-mobile.png'),
    fullPage: true,
  })
})

test('navigates paginated runs to a dedicated detail view with persistent tabs and bounded charts', async ({
  page,
}, testInfo) => {
  await mockBench(page)
  const runs = Array.from({ length: 15 }, (_, index) => ({
    ...run,
    id: `run-${index + 1}`,
    manifest: { ...manifest, name: `Evaluation ${index + 1}` },
  }))
  await page.route('**/api/sr-bench/v1/runs', (route) => route.fulfill({ json: { runs } }))
  await page.route('**/api/sr-bench/v1/runs/run-1/report', (route) =>
    route.fulfill({
      json: {
        ...report,
        summary: {
          ...report.summary,
          total_spend_usd: 0.012,
          targets: [{ ...report.summary.targets[0], cost_usd: 0.012 }],
        },
      },
    }),
  )
  await page.goto('/evaluation')
  await expect(page.getByRole('button', { name: 'Evaluation 11', exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'Next runs', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Evaluation 11', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Previous runs', exact: true }).click()
  await page.getByRole('button', { name: 'Evaluation 1', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Evaluation runs', exact: true })).toHaveCount(0)
  await expect(
    page.getByRole('region', { name: 'Quality and cost chart', exact: true }),
  ).toContainText('$0.01200')
  await section(page, 'Questions')
  await expect(page.getByRole('button', { name: 'case-a', exact: true })).toBeVisible()
  await page.reload()
  await expect(page.getByRole('tab', { name: 'Questions', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  )
  await page.getByRole('button', { name: 'case-a', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Final answer', exact: true })).toBeVisible()
  await expect(page.getByLabel('Filter loaded results')).not.toBeVisible()
  await page.getByRole('button', { name: 'Back to questions', exact: true }).click()
  await section(page, 'Results')
  await expect(
    page.getByRole('columnheader', { name: 'Macro accuracy', exact: true }),
  ).toBeVisible()
  await expect(
    page.getByRole('region', { name: 'Quality and cost chart', exact: true }).locator('svg'),
  ).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('run-summary-desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
    .toBe(true)
  await page.screenshot({ path: testInfo.outputPath('run-summary-mobile.png'), fullPage: true })
  await page.getByRole('button', { name: 'Back to runs', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Evaluation runs', exact: true })).toBeVisible()
})

test('withholds trend charts for incompatible comparisons and never plots unknown cost', async ({
  page,
}) => {
  await mockBench(page)
  let incompatible = false
  await page.route('**/api/sr-bench/v1/comparisons', (route) =>
    route.fulfill(
      incompatible
        ? { status: 400, json: { error: 'Dataset digest mismatch' } }
        : {
            json: {
              baseline_selection: 'Identical cases only',
              comparisons: [
                {
                  baseline_target_id: 'single',
                  candidate_target_id: 'single',
                  paired_cases: 2,
                  quality_delta: 0,
                  quality_delta_ci95: [-0.5, 0.5],
                  baseline_cost_usd: null,
                  candidate_cost_usd: null,
                  cost_saving_percent: null,
                },
              ],
            },
          },
    ),
  )
  await page.goto('/evaluation?view=compare&baseline=run-1&candidate=run-2')
  await expect(
    page.getByText('Complete quality and cost evidence is needed for this chart.', { exact: true }),
  ).toBeVisible()
  await expect(page.getByText('Unknown', { exact: true })).toBeVisible()
  incompatible = true
  await page.reload()
  await expect(page.getByText('A continuous trend is withheld', { exact: false })).toBeVisible()
  await expect(
    page.getByRole('region', { name: 'Quality and cost chart', exact: true }),
  ).toHaveCount(0)
  await expect(
    page.getByRole('region', { name: 'Iteration progress chart', exact: true }),
  ).toHaveCount(0)
})
