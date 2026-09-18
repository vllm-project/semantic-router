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
          },
        ],
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

test('plans and launches a reusable frozen dataset from the normal form', async ({ page }) => {
  const requests = await mockBench(page)
  await page.goto('/evaluation?view=new')
  await expect(page.getByRole('heading', { name: 'sr-bench 1.0' })).toBeVisible()
  await page.getByLabel('Prepared dataset').selectOption('quick-v1')
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

test('shows truthful metrics, routing distribution and case evidence', async ({ page }) => {
  await mockBench(page)
  await page.goto('/evaluation?view=runs&run=run-1')
  await expect(page.getByRole('heading', { name: 'Target comparison' })).toBeVisible()
  await expect(page.getByRole('cell', { name: '50%', exact: true })).toBeVisible()
  await expect(page.getByText('Subset evaluation; not a complete sr-bench score.')).toBeVisible()
  await expect(page.getByRole('cell', { name: '$0.00000' })).toHaveCount(0)
  await page.getByRole('button', { name: 'case-a' }).click()
  await expect(page.getByRole('heading', { name: 'Final answer' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Open report JSON' })).toHaveAttribute(
    'href',
    '/api/sr-bench/v1/runs/run-1/report',
  )
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
  await page.getByRole('button', { name: 'unresolved-case', exact: true }).click()
  await expect(evidence.getByText('Execution is still required', { exact: true })).toBeVisible()
  await expect(evidence.getByText('model-a', { exact: true })).toHaveCount(0)
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
  await expect(
    page.getByText('Loaded 100 of 250 persisted results.', { exact: false }),
  ).toBeVisible()
  await expect(
    page.getByText('Showing 100 of 201 persisted call summaries.', { exact: false }),
  ).toBeVisible()
  await expect(page.getByText('single: full-report-model', { exact: true })).toBeVisible()
  await expect(page.getByText('single: full-report-decision', { exact: true })).toBeVisible()
  await expect(page.getByRole('cell', { name: '125 / 250', exact: true })).toBeVisible()
  expect(resultReads).toEqual([0])
  expect(callReads).toEqual([0])
  expect(detailReads).toEqual([])
  await page.getByRole('button', { name: 'Load more results', exact: true }).click()
  await expect(
    page.getByText('Loaded 200 of 250 persisted results.', { exact: false }),
  ).toBeVisible()
  await page.getByLabel('Filter loaded results').fill('case-150')
  await page.getByRole('button', { name: 'case-150', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'case-150 · single' })).toBeVisible()
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
  await page.getByRole('button', { name: 'Load more call records', exact: true }).click()
  await expect(
    page.getByText('Showing 200 of 201 persisted call summaries.', { exact: false }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'call-150', exact: true }).click()
  await expect(page.getByText('Saved call prompt', { exact: false })).toBeVisible()
  expect(detailReads).toEqual(['call-150'])
  await expect(page.getByRole('cell', { name: '125 / 250', exact: true })).toBeVisible()
})

test('compares complete runs using paired results', async ({ page }) => {
  await mockBench(page)
  await page.goto('/evaluation?view=compare')
  await page.getByLabel('Baseline run').selectOption('run-1')
  await page.getByLabel('Current Balance run').selectOption('run-2')
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await expect(page.getByRole('heading', { name: 'Balance optimization trajectory' })).toBeVisible()
  await expect(
    page.getByText('Tied best single models: single, another-single.', { exact: false }),
  ).toBeVisible()
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
  await page.getByLabel('Search runs').fill('Candidate')
  await expect(page.getByRole('button', { name: 'Baseline test', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Candidate test', exact: true })).toBeVisible()
  await page.getByLabel('Run status').selectOption('failed')
  await expect(page.getByText('No runs match these filters.')).toBeVisible()
  await page.getByRole('button', { name: 'Datasets', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Prepared datasets' })).toBeVisible()
  await page.getByRole('button', { name: 'Evaluate this dataset' }).click()
  await expect(page.getByLabel('Prepared dataset')).toHaveValue('quick-v1')
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

test('reopens all three optimization comparisons from the saved URL', async ({
  page,
}, testInfo) => {
  await mockBench(page)
  const iterations = ['Current Balance', 'Balance round 1', 'Balance round 2'].map(
    (name, index) => ({
      ...run,
      id: `balance-${index}`,
      manifest: {
        ...manifest,
        name,
        targets: [{ ...target, id: 'balance', kind: 'mom', config_hash: String(index).repeat(64) }],
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs', (route) =>
    route.fulfill({ json: { runs: [run, ...iterations] } }),
  )
  await page.goto('/evaluation?view=compare')
  await page.getByLabel('Baseline run').selectOption('run-1')
  await page.getByLabel('Current Balance run').selectOption('balance-0')
  await page.getByLabel('Optimization 1 run').selectOption('balance-1')
  await page.getByLabel('Optimization 2 run').selectOption('balance-2')
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await expect(page.getByRole('cell', { name: '20% saving', exact: false })).toHaveCount(3)
  await expect(page).toHaveURL(/iteration2=balance-2/)
  await page.reload()
  await expect(
    page.getByRole('heading', { name: 'Optimization 2 · Balance round 2' }),
  ).toBeVisible()
  await expect(page.getByText('2'.repeat(64), { exact: true })).toBeVisible()
  await expect(page.getByRole('cell', { name: '20% saving', exact: false })).toHaveCount(3)
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
  await expect(page.getByRole('heading', { name: 'Recovery lineage' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Open parent run' })).toHaveAttribute(
    'href',
    '?view=runs&run=parent-1',
  )
  await expect(
    page.getByText('Parent snapshot: 30 / 50 completed', { exact: false }),
  ).toContainText('$2.50000 (incomplete accounting)')
  await expect(page.getByRole('link', { name: 'child-1', exact: true })).toBeVisible()
  await expect(page.getByRole('cell', { name: '1 / 2', exact: true })).toBeVisible()
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
  await expect(
    page.getByRole('columnheader', { name: 'Cost (original)', exact: true }),
  ).toBeVisible()
  await expect(
    page.getByRole('columnheader', { name: 'Original cost / latency', exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('cell', { name: '$1.50000', exact: true })).toBeVisible()
  await expect(page.getByRole('cell', { name: '$1.00000 / 1 s', exact: true })).toBeVisible()
  await notice.getByText('Accounting correction receipt', { exact: true }).click()
  await expect(notice.getByText('sr-bench-accounting-v1', { exact: true })).toBeVisible()
  qualified = false
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
  await page.goto('/evaluation?view=compare&baseline=run-1&iteration0=run-2')
  await expect(page.getByRole('note', { name: 'Accounting correction' })).toHaveCount(2)
  await expect(page.getByRole('heading', { name: 'Balance optimization trajectory' })).toBeVisible()
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
  await page.goto('/evaluation?view=compare&baseline=run-1&iteration0=run-2')
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
  await expect(page.getByRole('cell', { name: '$2.00000', exact: true })).toBeVisible()
  await expect(
    page.getByText('neither billed spend nor a measured cache-free run.', { exact: false }),
  ).toBeVisible()
  await page.goto('/evaluation?view=compare&baseline=run-1&iteration0=run-2')
  await expect(page.getByRole('cell', { name: '$0.80000 20% saving', exact: true })).toBeVisible()
  await expect(
    page.getByRole('cell', {
      name: '$1.80000 10% estimated saving Baseline $2.00000',
      exact: true,
    }),
  ).toBeVisible()
  complete = false
  await page.reload()
  await expect(page.getByRole('cell', { name: '$0.80000 20% saving', exact: true })).toBeVisible()
  await expect(
    page.getByRole('cell', { name: '— Saving unknown Baseline —', exact: true }),
  ).toBeVisible()
  await expect(page.getByText('10% estimated saving', { exact: true })).toHaveCount(0)
  expect(submissions).toHaveLength(0)
})
