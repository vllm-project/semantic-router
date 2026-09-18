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
    else if (path === '/runs/run-1/report') body = report
    else if (path === '/runs/run-1/results')
      body = {
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
    else if (path === '/runs/run-1/calls') body = { calls: [] }
    else if (path === '/runs/run-1/events') body = { events: [{ seq: 1, type: 'completed' }] }
    else if (path === '/comparisons')
      body = {
        baseline_selection: 'Best observed single model on identical cases.',
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
  await page.goto('/evaluation')
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

test('compares complete runs using paired results', async ({ page }) => {
  await mockBench(page)
  await page.goto('/evaluation?view=compare')
  await page.getByLabel('Baseline run').selectOption('run-1')
  await page.getByLabel('Candidate run').selectOption('run-2')
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await expect(page.getByRole('heading', { name: 'Paired comparison' })).toBeVisible()
  await expect(page.getByText('95% interval: -10% to 30%')).toBeVisible()
})

test('keeps actions disabled in a server readonly session', async ({ page }) => {
  await mockBench(page, { serverReadonly: true })
  await page.goto('/evaluation')
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
        ? { calls: [] }
        : path.endsWith('/results')
          ? { results: [] }
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
