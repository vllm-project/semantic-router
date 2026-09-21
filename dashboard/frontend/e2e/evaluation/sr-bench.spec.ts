import { readFile } from 'node:fs/promises'
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
        pending: 0,
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

const comparisonProtocol = {
  baseline_status: 'completed',
  candidate_status: 'completed',
  baseline_quality_complete: true,
  candidate_quality_complete: true,
  baseline_targets: report.summary.targets,
  candidate_targets: [{ ...report.summary.targets[0], id: 'balance' }],
  quality_denominator: 'all_planned_cases; explicit failed outcomes count as incorrect',
  baseline_selection_qualification: 'Delivered outcomes under the frozen limits.',
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
    else if (path === '/datasets/selection')
      body = {
        profile: new URL(route.request().url()).searchParams.get('profile'),
        seed: 42,
        split: 'dev',
        model_requests: 0,
        benchmarks: [
          {
            id: 'mmlu-pro',
            title: 'MMLU-Pro',
            eligible: true,
            case_count: 2,
            source_ids: ['quick-v1'],
            reason: null,
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
    else if (path === '/comparison-options') {
      const baseline = new URL(route.request().url()).searchParams.get('baseline_run_id')
      const source = { run_id: 'run-1', name: manifest.name, profile: 'quick', case_count: 2 }
      body = {
        baseline: baseline ? source : null,
        baselines: baseline ? [] : [source],
        options: baseline
          ? [{ run_id: 'run-2', name: 'Candidate test', profile: 'quick', case_count: 2 }]
          : [],
        next_cursor: null,
        has_more: false,
        scanned_pairs: 1,
        scan_limited: false,
        unverified_pairs: 0,
        unverified_baselines: 0,
        model_requests: 0,
      }
    } else if (path === '/replay-options')
      body = {
        baseline: null,
        baselines: [],
        options: [],
        next_cursor: null,
        has_more: false,
        scanned_pairs: 0,
        scan_limited: false,
        unverified_pairs: 0,
        unverified_baselines: 0,
        model_requests: 0,
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
        baseline_tie_policy: 'Lowest complete known total cost, then stable target ID.',
        ...comparisonProtocol,
        comparisons: [
          {
            baseline_target_id: 'single',
            candidate_target_id: 'balance',
            paired_cases: 2,
            wins: 1,
            losses: 0,
            ties: 1,
            baseline_subject_cost_usd: 1,
            baseline_total_cost_usd: 1,
            baseline_evaluation_cost_usd: 0,
            candidate_subject_cost_usd: 0.8,
            candidate_total_cost_usd: 0.8,
            candidate_evaluation_cost_usd: 0,
            quality_delta: 0.1,
            subject_cost_saving_percent: 20,
            total_cost_saving_percent: 20,
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

async function mockComparisonOptions(
  page: Page,
  baseline: { id: string; manifest: { name: string; profile: string } },
  candidates: Array<{ id: string; manifest: { name: string; profile: string } }>,
) {
  const choice = (item: typeof baseline) => ({
    run_id: item.id,
    name: item.manifest.name,
    profile: item.manifest.profile,
    case_count: 2,
  })
  await page.route('**/api/sr-bench/v1/comparison-options?**', (route) => {
    const query = new URL(route.request().url()).searchParams
    const child = query.get('baseline_run_id')
    const after = query.has('after') ? 10 : 0
    const rows = child ? candidates.slice(after, after + 10).map(choice) : [choice(baseline)]
    const more = !!child && candidates.length > after + 10
    return route.fulfill({
      json: {
        baseline: child ? choice(baseline) : null,
        baselines: child ? [] : rows,
        options: child ? rows : [],
        next_cursor: more ? 'next-page' : null,
        has_more: more,
        scanned_pairs: rows.length,
        scan_limited: false,
        unverified_pairs: 0,
        unverified_baselines: 0,
        model_requests: 0,
      },
    })
  })
}

async function chooseOption(page: Page, label: string, value: string) {
  await page.getByRole('combobox', { name: label, exact: true }).click()
  await page.getByRole('listbox').locator(`[data-value="${value}"]`).click()
}

async function chooseDataset(page: Page, id: string) {
  await page.getByRole('button', { name: 'Use a specific dataset', exact: true }).click()
  await chooseOption(page, 'Specific dataset', id)
  if (!(await page.getByRole('checkbox', { checked: true }).count()))
    await page.getByRole('button', { name: 'Select all benchmarks' }).click()
}

async function section(page: Page, name: string) {
  await page.getByRole('tab', { name, exact: true }).click()
}

test('plans and launches a reusable frozen dataset from the normal form', async ({ page }) => {
  const requests = await mockBench(page)
  await page.goto('/evaluation?view=new')
  await expect(page.getByRole('heading', { name: 'sr-bench 1.0' })).toBeVisible()
  await chooseDataset(page, 'quick-v1')
  await chooseOption(page, 'Add configured target', 'single')
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
  const requestParams = {
    max_tokens: 4096,
    temperature: 1,
    top_p: 0.95,
    seed: 42,
    top_k: 20,
    min_p: 0.1,
    chat_template_kwargs: { enable_thinking: true },
  }
  await page.route('**/api/sr-bench/v1/targets', (route) =>
    route.fulfill({ json: { targets: [{ ...target, request_params: requestParams }] } }),
  )
  await page.goto('/evaluation?view=new')
  await chooseDataset(page, 'quick-v1')
  await page.getByText('Sampling and advanced limits', { exact: true }).click()
  await page.getByLabel('Temperature', { exact: true }).fill('3')
  await chooseOption(page, 'Add configured target', 'single')
  const profile = page.getByRole('region', { name: 'model-a request profile', exact: true })
  await expect(profile.locator('dl > div').filter({ hasText: 'Output tokens' })).toContainText(
    '4,096',
  )
  await expect(profile.locator('dl > div').filter({ hasText: 'Temperature' })).toContainText(
    '1Fixed',
  )
  await expect(page.getByLabel('Temperature', { exact: true })).toBeDisabled()
  await expect(page.getByLabel('Temperature', { exact: true })).toHaveValue('1')
  await expect(page.getByLabel('Top P', { exact: true })).toHaveValue('0.95')
  await expect(page.getByLabel('Sampling seed', { exact: true })).toHaveValue('42')
  await profile.getByText('Other fixed settings', { exact: true }).click()
  await expect(profile.getByText('Top K', { exact: true })).toBeVisible()
  await expect(profile.getByText('Min P', { exact: true })).toBeVisible()
  await expect(profile.getByText('Thinking', { exact: true })).toBeVisible()
  await expect(profile.getByText('Enabled', { exact: true })).toBeVisible()
  await expect(page.getByLabel('Manifest JSON')).toHaveCount(0)
  await expect(page.getByLabel('Use edited manifest')).toHaveCount(0)
  await page.getByLabel('Max output tokens', { exact: true }).fill('512')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText(
    'Target model-a has a registered output limit of 4096 tokens, above the run cap of 512',
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

test('reviews typed sampling and case limits and invalidates a plan after settings change', async ({
  page,
}) => {
  const requests = await mockBench(page)
  await page.goto('/evaluation?view=new')
  await chooseOption(page, 'Add configured target', 'single')
  await page.getByText('Sampling and advanced limits', { exact: true }).click()
  await expect(page.locator('textarea')).toHaveCount(0)
  await expect(page.getByText('Advanced manifest', { exact: true })).toHaveCount(0)
  for (const [label, value] of [
    ['Temperature', '0.4'],
    ['Top P', '0.85'],
    ['Sampling seed', '73'],
    ['Request deadline (seconds)', '240'],
    ['Idle timeout (seconds)', '45'],
    ['Case deadline (seconds)', '900'],
    ['Max calls per case', '4'],
  ])
    await page.getByLabel(label, { exact: true }).fill(value)
  await page.getByLabel('Budget (USD)', { exact: true }).fill('')
  await chooseOption(page, 'Cost accounting', 'capability_only')
  await expect(page.getByLabel('Budget (USD)', { exact: true })).toBeDisabled()
  await expect(page.getByLabel('Budget (USD)', { exact: true })).toHaveAttribute(
    'placeholder',
    'Not applied',
  )
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Start evaluation', exact: true })).toBeEnabled()
  expect(requests[0]).toMatchObject({
    manifest: {
      targets: [target],
      cost_policy: 'capability_only',
      sampling: { temperature: 0.4, top_p: 0.85, seed: 73, max_tokens: 4096 },
      limits: {
        total_timeout_s: 240,
        idle_timeout_s: 45,
        case_timeout_s: 900,
        max_calls_per_case: 4,
        max_cost_usd: 5,
      },
    },
  })
  await page.getByLabel('Temperature', { exact: true }).fill('3')
  await expect(page.getByRole('button', { name: 'Start evaluation', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Temperature must be between 0 and 2')
  expect(requests).toHaveLength(1)
  await page.getByLabel('Temperature', { exact: true }).fill('0.5')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Start evaluation', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'Start evaluation', exact: true }).click()
  expect(requests[2]).toMatchObject({
    manifest: { sampling: { temperature: 0.5, seed: 73 }, targets: [target] },
  })
})

test('freezes optional learning-session context only for route preview', async ({ page }) => {
  const requests = await mockBench(page)
  const mom = {
    ...target,
    id: 'balance',
    kind: 'mom',
    config_hash: 'c'.repeat(64),
    preview_url: 'http://localhost:8001/v1',
  }
  await page.route('**/api/sr-bench/v1/targets', (route) =>
    route.fulfill({ json: { targets: [mom] } }),
  )
  await page.goto('/evaluation?view=new')
  await chooseOption(page, 'Add configured target', 'balance')
  await page.getByRole('button', { name: 'Preview routing', exact: true }).click()
  await chooseOption(page, 'Add configured target', 'balance')
  await page.getByText('Sampling and advanced limits', { exact: true }).click()
  await expect(page.getByLabel('Preview sampling seed', { exact: true })).toHaveValue('42')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  expect(requests[0]).toMatchObject({ manifest: { mode: 'preview', seed: 42 } })
  expect((requests[0] as { manifest: object }).manifest).not.toHaveProperty('preview_context')
  await page.getByLabel('Session ID', { exact: true }).fill('session-alpha')
  await page.getByLabel('Conversation ID', { exact: true }).fill('conversation-beta')
  await page.getByLabel('Preview sampling seed', { exact: true }).fill('73')
  await expect(
    page.getByRole('button', { name: 'Start route preview', exact: true }),
  ).toBeDisabled()
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  expect(requests[1]).toMatchObject({
    manifest: {
      mode: 'preview',
      sampling: { seed: 42 },
      preview_context: {
        session_id: 'session-alpha',
        conversation_id: 'conversation-beta',
        sampling_seed: 73,
      },
    },
  })
  await page.getByRole('button', { name: 'Create evaluation', exact: true }).click()
  await chooseOption(page, 'Add configured target', 'balance')
  await expect(page.getByLabel('Session ID', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Start evaluation', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  expect((requests[2] as { manifest: object }).manifest).not.toHaveProperty('preview_context')
})

test('supports keyboard selection, search and escape without reopening custom controls', async ({
  page,
}) => {
  await mockBench(page)
  await page.goto('/evaluation?view=new')
  await page.getByText('Sampling and advanced limits', { exact: true }).click()
  const mode = page.getByRole('combobox', { name: 'Cost accounting', exact: true })
  await mode.focus()
  await mode.press('ArrowDown')
  await expect(page.getByRole('option', { name: /^Quality and cost/ })).toBeFocused()
  await page.getByRole('option', { name: /^Quality and cost/ }).press('End')
  await expect(page.getByRole('option', { name: /^Quality only/ })).toBeFocused()
  await page.getByRole('option', { name: /^Quality only/ }).press('Enter')
  await expect(mode).toHaveText('Quality only')
  await expect(mode).toBeFocused()
  await mode.press('ArrowUp')
  await page.getByRole('option', { name: /^Quality only/ }).press('Home')
  await page.getByRole('option', { name: /^Quality and cost/ }).press('Escape')
  await expect(mode).toBeFocused()
  await expect(mode).toHaveAttribute('aria-expanded', 'false')
  await expect(page.getByRole('listbox')).toHaveCount(0)
  const targets = page.getByRole('combobox', { name: 'Add configured target', exact: true })
  await targets.click()
  await page.getByRole('searchbox', { name: 'Search add configured target' }).fill('no-such-target')
  await expect(page.getByText('No matching options.', { exact: true })).toBeVisible()
  await page.getByRole('searchbox', { name: 'Search add configured target' }).fill('model-a')
  await page.getByRole('searchbox', { name: 'Search add configured target' }).press('ArrowDown')
  await expect(page.getByRole('option')).toBeFocused()
  await page.getByRole('option').press('Escape')
  await expect(targets).toBeFocused()
  await expect(targets).toHaveAttribute('aria-expanded', 'false')
})

test('keeps mixed fixed and editable sampling profiles explicit', async ({ page }) => {
  const requests = await mockBench(page)
  const fixed = {
    ...target,
    id: 'fixed',
    model: 'model-fixed',
    request_params: { temperature: 1, top_p: 0.95 },
  }
  await page.route('**/api/sr-bench/v1/targets', (route) =>
    route.fulfill({ json: { targets: [target, fixed] } }),
  )
  await page.goto('/evaluation?view=new')
  await chooseOption(page, 'Add configured target', 'single')
  await chooseOption(page, 'Add configured target', 'fixed')
  await page.getByText('Sampling and advanced limits', { exact: true }).click()
  await expect(page.getByLabel('Temperature', { exact: true })).toBeEnabled()
  await page.getByLabel('Temperature', { exact: true }).fill('0.3')
  await expect(
    page.getByText('Applies to 1 of 2 targets; fixed profiles keep their own values.'),
  ).toHaveCount(2)
  await expect(
    page
      .getByRole('region', { name: 'model-fixed request profile', exact: true })
      .locator('dl > div')
      .filter({ hasText: 'Temperature' }),
  ).toContainText('1Fixed')
  await expect(
    page
      .getByRole('region', { name: 'model-a request profile', exact: true })
      .locator('dl > div')
      .filter({ hasText: 'Temperature' }),
  ).toContainText('0.3')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  expect(requests[0]).toMatchObject({
    manifest: { sampling: { temperature: 0.3 }, targets: [target, fixed] },
  })
})

test('keeps comparison and create controls compact on desktop and within mobile width', async ({
  page,
}, testInfo) => {
  await mockBench(page)
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/evaluation?view=compare')
  const baseline = page.getByRole('combobox', { name: 'Reference run', exact: true })
  await expect(baseline).toBeVisible()
  expect((await baseline.boundingBox())!.width).toBeLessThanOrEqual(400)
  await expect(page.getByRole('group', { name: 'Comparison runs', exact: true })).toHaveCount(0)
  await chooseOption(page, 'Reference run', 'run-1')
  const selectAll = page.getByRole('button', { name: 'Select all', exact: true })
  expect((await selectAll.boundingBox())!.width).toBeLessThan(130)
  await baseline.click()
  await expect(page.getByRole('listbox')).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('compare-dropdown-desktop.png') })
  await page.getByRole('searchbox', { name: 'Search reference run' }).press('Escape')
  await page.goto('/evaluation?view=new')
  await chooseOption(page, 'Add configured target', 'single')
  await page.getByText('Sampling and advanced limits', { exact: true }).click()
  await page
    .getByText('Sampling and advanced limits', { exact: true })
    .evaluate((element) => element.scrollIntoView({ block: 'start' }))
  await page.screenshot({ path: testInfo.outputPath('create-settings-desktop.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  await page
    .getByRole('combobox', { name: 'Cost accounting', exact: true })
    .scrollIntoViewIfNeeded()
  await page.getByRole('combobox', { name: 'Cost accounting', exact: true }).click()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
  const menuBounds = (await page.getByRole('listbox').boundingBox())!
  expect(menuBounds.y).toBeGreaterThanOrEqual(0)
  expect(menuBounds.y + menuBounds.height).toBeLessThanOrEqual(844)
  await page.screenshot({ path: testInfo.outputPath('create-accounting-mobile.png') })
  await page.goto('/evaluation?view=datasets')
  await expect(page.getByRole('heading', { name: 'Dataset library', exact: true })).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('library-mobile.png') })
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.screenshot({ path: testInfo.outputPath('library-desktop.png') })
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
  await expect(page.getByRole('heading', { name: 'Parsed answer' })).toBeVisible()
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
  await chooseOption(page, 'Add configured target', 'single')
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
  await chooseOption(page, 'Add configured target', 'single')
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
  await chooseOption(page, 'Add configured target', 'single')
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
  await chooseOption(page, 'Add configured target', 'single')
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
  await chooseOption(page, 'Add configured target', 'single')
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
  await chooseOption(page, 'Add configured target', 'single')
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

test('bounds event reads and recovers an explicitly requested page without advancing a failed cursor', async ({
  page,
}) => {
  await mockBench(page)
  let progressReads = 0
  await page.route('**/api/sr-bench/v1/runs/run-1', (route) => {
    progressReads += 1
    return route.fulfill({ json: { ...run, status: 'running', updated_at: String(progressReads) } })
  })
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
  await section(page, 'Evidence')
  await expect(
    page.getByRole('heading', { name: 'Run events (1000 loaded)', exact: true }),
  ).toBeVisible()
  await expect.poll(() => progressReads).toBeGreaterThan(1)
  expect(cursors).toEqual([0])
  await page.getByRole('button', { name: 'Next events', exact: true }).click()
  expect(cursors).toEqual([0])
  await page.getByRole('button', { name: 'Load more events', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Event page unavailable')
  await expect(
    page.getByRole('heading', { name: 'Run events (1000 loaded)', exact: true }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Load more events', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Run events (1001)', exact: true })).toBeVisible()
  expect(cursors).toEqual([0, 1000, 1000])
  await expect(page.getByRole('button', { name: 'Load more events', exact: true })).toHaveCount(0)
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
    page.getByText('Loaded 100 of 201 persisted call summaries.', { exact: false }),
  ).toBeVisible()
  const callTable = page.getByRole('region', { name: 'Persisted call records', exact: true })
  await expect(callTable.locator('tbody tr')).toHaveCount(25)
  await page.getByRole('button', { name: 'Next calls', exact: true }).click()
  await expect(page.getByRole('button', { name: 'call-25', exact: true })).toBeVisible()
  expect(callReads).toEqual([0])
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
  await expect(page.getByText('model-a: full-report-model', { exact: true })).toBeVisible()
  await expect(page.getByText('model-a: full-report-decision', { exact: true })).toBeVisible()
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
  await expect(page.getByRole('heading', { name: 'case-150 · model-a' })).toBeVisible()
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
    page.getByText('Loaded 200 of 201 persisted call summaries.', { exact: false }),
  ).toBeVisible()
  await page.getByLabel('Filter loaded calls').fill('call-150')
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
  await chooseOption(page, 'Reference run', 'run-1')
  await page.getByRole('checkbox', { name: /Candidate test/ }).check()
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await expect(page.getByRole('heading', { name: 'Comparison evidence' })).toBeVisible()
  await expect(
    page.getByText(
      'Total cost includes model answers and evaluation calls, priced from recorded usage at frozen rates. These are estimates, not invoices or hardware costs.',
      { exact: true },
    ),
  ).toBeVisible()
  await page.getByText('Technical comparison evidence', { exact: true }).click()
  await page.getByText('Baseline selection policy', { exact: true }).click()
  await expect(
    page.getByText('Tied best single models: model-a, Unknown model.', { exact: false }),
  ).toBeVisible()
  await page.getByText('Detailed iteration metrics and uncertainty', { exact: true }).click()
  await expect(page.getByRole('cell', { name: '−10 pp to +30 pp', exact: false })).toBeVisible()
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
    route.fulfill({
      json: {
        ...run,
        status: 'failed',
        error: null,
        progress: { total: 2, completed: 1, failed: 1 },
      },
    }),
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
    'Model Unknown model · case case-a · from saved case evidence',
  )
  await expect(
    page.getByText('This run is failed. 1 failed results remain part of the saved evidence.', {
      exact: false,
    }),
  ).toBeVisible()
  await expect(
    page.getByText(
      'Comparisons require an explicit terminal outcome for every planned result; missing or ungraded results are not eligible.',
      { exact: false },
    ),
  ).toBeVisible()
})

test('removes the Benchmarks tab while keeping benchmark selection in Create', async ({ page }) => {
  await mockBench(page)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/evaluation')
  const navigation = page.getByRole('navigation', { name: 'Evaluation views' })
  await expect(navigation.getByRole('button')).toHaveCount(4)
  await expect(navigation.getByRole('button', { name: 'Benchmarks', exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'Create evaluation', exact: true }).click()
  await expect(
    page.getByRole('group', { name: 'Included benchmarks' }).getByRole('checkbox'),
  ).toHaveCount(9)
  await expect(page.getByRole('checkbox', { name: /MMLU-Pro/ })).toBeEnabled()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
  await page.goto('/evaluation?view=unknown')
  await expect(page.getByRole('heading', { name: 'Evaluation runs' })).toBeVisible()
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
  await page.route('**/api/sr-bench/v1/replay-options?*', (route) => {
    const baseline = { run_id: 'run-1', name: 'Baseline test', profile: 'quick', case_count: 2 }
    const selected = new URL(route.request().url()).searchParams.has('baseline_run_id')
    return route.fulfill({
      json: {
        baseline: selected ? baseline : null,
        baselines: selected ? [] : [baseline],
        options: selected
          ? [{ run_id: 'preview-1', name: 'Route preview', profile: 'quick', case_count: 2 }]
          : [],
        next_cursor: null,
        has_more: false,
        scanned_pairs: 1,
        scan_limited: false,
        unverified_pairs: 0,
        unverified_baselines: 0,
        model_requests: 0,
      },
    })
  })
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
  await page
    .locator('summary')
    .filter({ hasText: /^Estimate a routing change$/ })
    .click()
  await chooseOption(page, 'Saved single-model baseline', 'run-1')
  await chooseOption(page, 'Routing preview', 'preview-1')
  await page.getByRole('button', { name: 'Create offline estimate' }).click()
  await expect(page.getByRole('heading', { name: 'Diagnostic replay estimates' })).toBeVisible()
  await expect(page.getByRole('cell', { name: '75%', exact: true })).toBeVisible()
  await expect(
    page.getByText('These estimates reuse saved answers.', { exact: false }),
  ).toBeVisible()
  expect(submitted).toEqual([
    { baseline_run_id: 'run-1', preview_run_id: 'preview-1', idempotency_key: expect.any(String) },
  ])
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
  const table = page
    .getByRole('region', { name: 'Evaluation runs', exact: true })
    .getByRole('table')
  await expect(table.getByRole('columnheader')).toHaveText([
    'Run / targets',
    'Mode',
    'Profile',
    'Status',
    'Progress',
    'Last update',
  ])
  const firstRow = table.getByRole('row').nth(1)
  await expect(firstRow.getByRole('cell').nth(1)).toHaveText('Live')
  await expect(firstRow.getByRole('cell').nth(2)).toHaveText('Quick')
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
  await chooseOption(page, 'Run status', 'failed')
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
  await mockComparisonOptions(page, run, iterations)
  await page.goto('/evaluation?view=compare')
  await chooseOption(page, 'Reference run', 'run-1')
  await page.getByRole('button', { name: 'Select all', exact: true }).click()
  await expect(page.getByRole('checkbox', { checked: true })).toHaveCount(4)
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await expect(page.getByRole('article').getByText('Lower cost', { exact: true })).toHaveCount(4)
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
        excluded: Array.from({ length: 31 }, (_, index) => ({
          target_id: 'single',
          case_id: `complete-case-${index}`,
          reason: 'already completed',
        })),
        counts: { eligible: 1, excluded: 31 },
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
  await chooseOption(page, 'Recovery scope', 'failed')
  await page.getByRole('button', { name: 'Review recovery plan' }).click()
  await page.getByText('Excluded cases and reasons', { exact: true }).click()
  const excluded = page.getByRole('table', { name: 'Excluded recovery cases' })
  await expect(excluded.locator('tbody tr')).toHaveCount(25)
  await expect(excluded.locator('pre')).toHaveCount(0)
  await page.getByRole('button', { name: 'Next excluded cases', exact: true }).click()
  await expect(excluded.locator('tbody tr')).toHaveCount(6)
  await expect(excluded.getByText('complete-case-30', { exact: true })).toBeVisible()
  await page.getByLabel('Recover model-a failed-case', { exact: true }).check()
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
  await page.getByLabel('Recover model-a unstarted', { exact: true }).check()
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
  await page.getByLabel('Recover model-a unstarted', { exact: true }).check()
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
  await page.getByLabel('Recover model-a unstarted', { exact: true }).check()
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
    page.getByRole('heading', { name: 'model-a · Verified config snapshot' }),
  ).toBeVisible()
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download model-a recipe' }).click()
  expect((await download).suggestedFilename()).toBe('run-1-balance-recipe.json')
  activeHash = 'f'.repeat(64)
  await page.reload()
  await expect(page.getByRole('heading', { name: 'model-a · Snapshot unavailable' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Download model-a recipe' })).toHaveCount(0)
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
  await page.getByLabel('Recover model-a unstarted', { exact: true }).check()
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
        child_attempts: Array.from({ length: 13 }, (_, index) => ({
          id: `child-${index + 1}`,
          status: 'completed',
          progress: { completed: 1, total: 1, failed: 0 },
        })),
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
  await expect(page.getByRole('link', { name: /^child-\d+$/ })).toHaveCount(10)
  await page.getByRole('button', { name: 'Next recovery attempts', exact: true }).click()
  await expect(page.getByRole('link', { name: /^child-\d+$/ })).toHaveCount(3)
  await expect(page.getByRole('link', { name: 'child-13', exact: true })).toHaveAttribute(
    'href',
    '?view=runs&run=child-13',
  )
  await section(page, 'Results')
  await expect(page.getByRole('cell', { name: '1 / 2', exact: true })).toBeVisible()
  await section(page, 'Evidence')
  await expect(page.getByRole('cell', { name: '1 / 1', exact: true })).toHaveCount(3)
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
        ...comparisonProtocol,
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
            baseline_subject_cost_usd: 1,
            baseline_total_cost_usd: 1,
            baseline_evaluation_cost_usd: 0,
            candidate_subject_cost_usd: 0.8,
            candidate_total_cost_usd: 0.8,
            candidate_evaluation_cost_usd: 0,
            subject_cost_saving_percent: 20,
            total_cost_saving_percent: 20,
          },
        ],
      },
    }),
  )
  await page.goto('/evaluation?view=compare&baseline=run-1&candidate=run-2')
  await page.getByText('Detailed iteration metrics and uncertainty', { exact: true }).click()
  const uncertainty = page.getByRole('cell').filter({ hasText: 'Conservative weighted Hoeffding' })
  await expect(uncertainty).toContainText('−47 pp to +47 pp')
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
        ...comparisonProtocol,
        comparisons: [
          {
            baseline_target_id: 'single',
            candidate_target_id: 'balance',
            paired_cases: 2,
            quality_delta: 0,
            quality_delta_ci95: [-0.1, 0.1],
            baseline_subject_cost_usd: 1,
            baseline_total_cost_usd: 1,
            baseline_evaluation_cost_usd: 0,
            candidate_subject_cost_usd: 0.8,
            candidate_total_cost_usd: 0.8,
            candidate_evaluation_cost_usd: 0,
            subject_cost_saving_percent: 20,
            total_cost_saving_percent: 20,
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
  await expect(
    page.getByRole('cell').filter({ hasText: '$0.80000' }).filter({ hasText: '+20% saving' }),
  ).toBeVisible()
  await expect(
    page.getByRole('cell', {
      name: /\$1\.80000.*\+10% estimated saving.*Baseline \$2\.00000/,
    }),
  ).toBeVisible()
  complete = false
  await page.reload()
  await page.getByText('Detailed iteration metrics and uncertainty', { exact: true }).click()
  await expect(
    page.getByRole('cell').filter({ hasText: '$0.80000' }).filter({ hasText: '+20% saving' }),
  ).toBeVisible()
  await expect(
    page.getByRole('cell', { name: '— Unknown Not available Baseline —', exact: true }),
  ).toBeVisible()
  await expect(page.getByText('10% estimated saving', { exact: true })).toHaveCount(0)
  expect(submissions).toHaveLength(0)
})

test('total-cost comparison includes evaluation overhead and withholds unknown auxiliary spend', async ({
  page,
}, testInfo) => {
  const submissions = await mockBench(page)
  let evaluationKnown = true
  const baseline = {
    ...report.summary.targets[0],
    cost_usd: 1,
    evaluation_cost_usd: 0.2,
    total_spend_usd: 1.2,
  }
  await page.route('**/api/sr-bench/v1/comparisons', (route) =>
    route.fulfill({
      json: {
        ...comparisonProtocol,
        baseline_selection: 'Strongest single model, then lowest known total cost.',
        baseline_targets: [baseline],
        candidate_targets: [
          {
            ...baseline,
            id: 'balance',
            cost_usd: 0.4,
            evaluation_cost_usd: evaluationKnown ? 1 : null,
            total_spend_usd: evaluationKnown ? 1.4 : null,
          },
        ],
        comparisons: [
          {
            baseline_target_id: 'single',
            candidate_target_id: 'balance',
            paired_cases: 2,
            quality_delta: 0,
            quality_delta_ci95: [-0.1, 0.1],
            baseline_subject_cost_usd: 1,
            candidate_subject_cost_usd: 0.4,
            subject_cost_saving_percent: 60,
            baseline_evaluation_cost_usd: 0.2,
            candidate_evaluation_cost_usd: evaluationKnown ? 1 : null,
            baseline_total_cost_usd: 1.2,
            candidate_total_cost_usd: evaluationKnown ? 1.4 : null,
            total_cost_saving_percent: evaluationKnown ? -100 / 6 : null,
            total_cost_comparison_reason: evaluationKnown
              ? null
              : 'Candidate evaluation cost is incomplete.',
          },
        ],
      },
    }),
  )
  await page.goto('/evaluation?view=compare&baseline=run-1&candidate=run-2')
  const card = page.getByRole('article')
  await expect(card.getByText('Total cost saving', { exact: true })).toBeVisible()
  const headline = card.locator('strong > [data-direction]').nth(1)
  await expect(headline).toContainText('−16.67%')
  await expect(headline).toHaveAttribute('data-direction', 'negative')
  await expect(card.getByText('Subject model saving:', { exact: false })).not.toBeVisible()
  await card.getByText('Cost breakdown', { exact: true }).click()
  await expect(card.getByText('Subject model saving:', { exact: false })).toContainText('+60%')
  const costs = card.getByRole('table', { name: 'Cost breakdown' })
  await expect(
    costs
      .getByRole('row')
      .filter({ has: page.getByRole('rowheader', { name: 'Subject model', exact: true }) }),
  ).toContainText('$1.00000$0.40000')
  await expect(
    costs
      .getByRole('row')
      .filter({ has: page.getByRole('rowheader', { name: 'Evaluation', exact: true }) }),
  ).toContainText('$0.20000$1.00000')
  await expect(
    costs
      .getByRole('row')
      .filter({ has: page.getByRole('rowheader', { name: 'Total', exact: true }) }),
  ).toContainText('$1.20000$1.40000')
  const chart = page.getByRole('region', { name: 'Quality and cost chart' })
  await expect(
    chart.getByRole('heading', { name: 'Quality and total cost', exact: true }),
  ).toBeVisible()
  await expect(chart.getByRole('listitem').first()).toContainText('$1.20000')
  await expect(chart.getByRole('listitem').last()).toContainText('$1.40000')
  const downloadEvent = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Export comparison CSV' }).click()
  const download = await downloadEvent
  const csv = await readFile((await download.path())!, 'utf8')
  const cells = csv
    .split('\n')
    .map((line) => line.split(',').map((cell) => JSON.parse(cell) as string))
  const exported = Object.fromEntries(cells[0].map((key, index) => [key, cells[1][index]]))
  expect(exported).toMatchObject({
    baseline_subject_cost_usd: '1',
    candidate_subject_cost_usd: '0.4',
    baseline_evaluation_cost_usd: '0.2',
    candidate_evaluation_cost_usd: '1',
    baseline_total_cost_usd: '1.2',
    candidate_total_cost_usd: '1.4',
    subject_cost_saving_percent: '60',
    total_cost_saving_percent: String(-100 / 6),
  })
  expect(exported).not.toHaveProperty('cost_saving_percent')
  expect(exported).not.toHaveProperty('model_cost_usd')
  await card.screenshot({ path: testInfo.outputPath('total-cost-regression.png') })
  evaluationKnown = false
  await page.reload()
  await expect(headline).toContainText('Unknown')
  await expect(headline).toHaveAttribute('data-direction', 'unknown')
  await expect(
    card.getByText('Candidate evaluation cost is incomplete.', { exact: true }),
  ).toBeVisible()
  await expect(chart.getByRole('listitem')).toHaveCount(1)
  await card.getByText('Cost breakdown', { exact: true }).click()
  await expect(card.getByText('Subject model saving:', { exact: false })).toContainText('+60%')
  await expect(
    costs
      .getByRole('row')
      .filter({ has: page.getByRole('rowheader', { name: 'Evaluation', exact: true }) }),
  ).toContainText('$0.20000—')
  await expect(
    costs
      .getByRole('row')
      .filter({ has: page.getByRole('rowheader', { name: 'Total', exact: true }) }),
  ).toContainText('$1.20000—')
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
  await page.route('**/api/sr-bench/v1/datasets/selection?*', (route) => {
    const profile = new URL(route.request().url()).searchParams.get('profile')!
    return route.fulfill({
      json: {
        profile,
        seed: 42,
        split: profile === 'standard' ? 'holdout' : 'dev',
        model_requests: 0,
        benchmarks: ['mmlu-pro', 'gpqa-diamond'].map((id) => ({
          id,
          title: id,
          eligible: true,
          case_count: 5,
          source_ids: [`${profile}-suite`],
          reason: null,
        })),
      },
    })
  })
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
  await chooseOption(page, 'Add configured target', 'single')
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
  await expect(page.getByRole('heading', { name: 'Parsed answer', exact: true })).toBeVisible()
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
              ...comparisonProtocol,
              comparisons: [
                {
                  baseline_target_id: 'single',
                  candidate_target_id: 'single',
                  paired_cases: 2,
                  quality_delta: 0,
                  quality_delta_ci95: [-0.5, 0.5],
                  baseline_subject_cost_usd: null,
                  baseline_total_cost_usd: null,
                  baseline_evaluation_cost_usd: 0,
                  candidate_subject_cost_usd: null,
                  candidate_total_cost_usd: null,
                  candidate_evaluation_cost_usd: 0,
                  subject_cost_saving_percent: null,
                  total_cost_saving_percent: null,
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
  await expect(
    page.getByRole('article').locator('strong > [data-direction="unknown"]'),
  ).toContainText('Unknown')
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

test('presents readable paginated events and removes the generic limitations disclosure', async ({
  page,
}, testInfo) => {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/runs/run-1/report', (route) =>
    route.fulfill({
      json: {
        ...report,
        limitations: [
          report.limitations[0],
          ...Array.from({ length: 11 }, (_, index) => `Generic limitation ${index}`),
        ],
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/events?*', (route) =>
    route.fulfill({
      json: {
        events: Array.from({ length: 419 }, (_, index) => ({
          seq: index + 1,
          at: '2026-09-18T00:00:00Z',
          kind: index === 0 ? 'created' : index === 418 ? 'failure_observed' : 'case_completed',
          data: {
            case_id: `case-${index}`,
            target_id: 'single',
            ...(index === 418 ? { reason: 'Request deadline exceeded' } : {}),
            opaque_payload: { secret: 'never render this event payload' },
          },
        })),
      },
    }),
  )
  await page.goto('/evaluation?view=runs&run=run-1')
  await expect(page.getByText(report.limitations[0], { exact: true })).toBeVisible()
  await expect(page.getByText('More limitations', { exact: false })).toHaveCount(0)
  await expect(page.getByText('Generic limitation', { exact: false })).toHaveCount(0)
  await section(page, 'Evidence')
  const events = page.getByRole('list', { name: 'Saved run events', exact: true })
  await expect(page.getByRole('heading', { name: 'Run events (419)', exact: true })).toBeVisible()
  await expect(events.locator('li')).toHaveCount(25)
  await expect(events.getByText('Evaluation created', { exact: true })).toBeVisible()
  await expect(events.locator('pre')).toHaveCount(0)
  await expect(page.getByText('never render this event payload', { exact: false })).toHaveCount(0)
  await page.getByRole('button', { name: 'Next events', exact: true }).click()
  await expect(events.getByText('Case: case-25 · Model: model-a', { exact: true })).toBeVisible()
  await expect(events.getByText('Evaluation created', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('navigation', { name: 'Events pages' })).toContainText('Page 2 of 17')
  await chooseOption(page, 'Event type', 'attention')
  await expect(events.locator('li')).toHaveCount(1)
  await expect(events.getByText('Failure recorded', { exact: true })).toBeVisible()
  await expect(events.getByText('Request deadline exceeded', { exact: true })).toBeVisible()
  await expect(page.getByRole('navigation', { name: 'Events pages' })).toHaveCount(0)
  await chooseOption(page, 'Event type', 'all')
  await page
    .getByRole('heading', { name: 'Run events (419)', exact: true })
    .scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('readable-events-desktop.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('readable-events-mobile.png') })
})

test('shows only server-qualified comparison baselines and paginates arbitrary iterations', async ({
  page,
}, testInfo) => {
  await mockBench(page)
  const protocol = { ...manifest, case_sha256: 'a'.repeat(64), limits: { max_output_tokens: 4096 } }
  const baseline = { ...run, manifest: { ...protocol, name: 'Single-model reference' } }
  const candidates = Array.from({ length: 12 }, (_, index) => ({
    ...run,
    id: `eligible-${index}`,
    created_at: `2026-09-18T00:${String(index).padStart(2, '0')}:00Z`,
    manifest: {
      ...protocol,
      name: `Balance iteration ${index + 1}`,
      targets: [{ ...target, id: 'balance', kind: 'mom' }],
    },
  }))
  const otherBaseline = {
    ...baseline,
    id: 'other-baseline',
    manifest: { ...protocol, name: 'Different cases baseline', case_sha256: 'b'.repeat(64) },
  }
  await page.route('**/api/sr-bench/v1/runs', (route) =>
    route.fulfill({
      json: {
        runs: [
          baseline,
          ...candidates,
          otherBaseline,
          {
            ...candidates[0],
            id: 'running',
            status: 'running',
            manifest: { ...protocol, name: 'Still running' },
          },
          {
            ...candidates[0],
            id: 'different-limits',
            manifest: { ...protocol, name: 'Different limits', limits: { max_output_tokens: 512 } },
          },
          {
            ...candidates[0],
            id: 'preview',
            manifest: { ...protocol, name: 'Preview evidence', mode: 'preview' },
          },
        ],
      },
    }),
  )
  const comparisons: Array<{ baseline_run_id: string; candidate_run_id: string }> = []
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.endsWith('/comparisons'))
      comparisons.push(request.postDataJSON())
  })
  await mockComparisonOptions(page, baseline, candidates)
  await page.goto('/evaluation?view=compare')
  await expect(page.getByRole('group', { name: 'Comparison runs', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Compare runs', exact: true })).toHaveCount(0)
  await page.getByRole('combobox', { name: 'Reference run', exact: true }).click()
  await expect(page.getByRole('listbox').getByRole('option')).toHaveCount(1)
  await expect(page.getByRole('listbox').locator('[data-value="other-baseline"]')).toHaveCount(0)
  await page.getByRole('listbox').locator('[data-value="run-1"]').click()
  const choices = page.getByRole('group', { name: 'Comparison runs', exact: true })
  await expect(choices.getByRole('checkbox')).toHaveCount(8)
  await page.getByRole('button', { name: 'Load more comparison runs', exact: true }).click()
  await expect(
    page.getByRole('button', { name: 'Load more comparison runs', exact: true }),
  ).toHaveCount(0)
  await page.getByRole('button', { name: 'Select all', exact: true }).click()
  await expect(page.getByText('12 selected', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Next comparison runs', exact: true }).click()
  await expect(choices.getByRole('checkbox')).toHaveCount(4)
  await expect(choices.getByRole('checkbox', { checked: true })).toHaveCount(4)
  await page.getByLabel('Find comparison runs').fill('Different')
  await expect(choices.getByRole('checkbox')).toHaveCount(0)
  await expect(page.getByText('No loaded runs match this search.')).toBeVisible()
  expect(comparisons).toEqual([])
  await page.getByLabel('Find comparison runs').fill('')
  await page.screenshot({ path: testInfo.outputPath('guided-comparison-desktop.png') })
  await page.getByRole('button', { name: 'Compare runs', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Comparison evidence' })).toBeVisible()
  expect(comparisons).toHaveLength(12)
  expect(comparisons.every((request) => request.candidate_run_id.startsWith('eligible-'))).toBe(
    true,
  )
  await expect(page.getByRole('article')).toHaveCount(6)
  await page.getByRole('button', { name: 'Next comparison results', exact: true }).click()
  await expect(
    page.getByRole('article').getByRole('heading', { name: 'Balance iteration 7', exact: true }),
  ).toBeVisible()
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('combobox', { name: 'Reference run', exact: true }).scrollIntoViewIfNeeded()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('guided-comparison-mobile.png') })
})

test('preserves selected benchmarks across profiles and blocks conflicting server resolutions', async ({
  page,
}, testInfo) => {
  const submissions = await mockBench(page)
  const compositions: unknown[] = []
  await page.route('**/api/sr-bench/v1/datasets/selection?*', (route) => {
    const profile = new URL(route.request().url()).searchParams.get('profile')!
    return route.fulfill({
      json: {
        profile,
        seed: 42,
        split: profile === 'standard' ? 'holdout' : 'dev',
        model_requests: 0,
        benchmarks: [
          {
            id: 'mmlu-pro',
            title: 'MMLU-Pro',
            eligible: profile !== 'standard',
            case_count: profile === 'standard' ? 0 : 2,
            source_ids: profile === 'standard' ? [] : ['canonical'],
            reason:
              profile === 'standard'
                ? 'Conflicting frozen revisions. Choose a specific dataset.'
                : null,
          },
          {
            id: 'gpqa-diamond',
            title: 'GPQA Diamond',
            eligible: true,
            case_count: 2,
            source_ids: ['canonical-gpqa'],
            reason: null,
          },
        ],
      },
    })
  })
  await page.route('**/api/sr-bench/v1/datasets/compose', (route) => {
    compositions.push(route.request().postDataJSON())
    return route.fulfill({
      json: { dataset: { path: '/prepared/canonical.json', sha256: 'a'.repeat(64) } },
    })
  })
  await page.goto('/evaluation?view=new')
  await expect(page.getByRole('combobox', { name: 'Mode', exact: true })).toHaveCount(0)
  await expect(page.getByRole('combobox', { name: 'Prepared dataset', exact: true })).toHaveCount(0)
  await expect(page.getByRole('combobox', { name: 'Cost accounting', exact: true })).toBeHidden()
  await expect(page.getByRole('checkbox', { name: /MMLU-Pro/ })).toBeChecked()
  await chooseOption(page, 'Add configured target', 'single')
  await page.getByRole('radio', { name: /^Standard/ }).check()
  await expect(
    page.getByText('Conflicting frozen revisions. Choose a specific dataset.', { exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('checkbox', { name: /MMLU-Pro/ })).toBeChecked()
  await expect(page.getByRole('button', { name: 'Review plan', exact: true })).toBeDisabled()
  await page.getByRole('checkbox', { name: /MMLU-Pro/ }).uncheck()
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  expect(compositions).toEqual([{ dataset_ids: ['canonical-gpqa'], benchmarks: ['gpqa-diamond'] }])
  expect(submissions).toHaveLength(1)
  expect(submissions[0]).toMatchObject({
    manifest: { profile: 'standard', mode: 'live', cost_policy: 'require_priced' },
  })
  await page
    .getByRole('radiogroup', { name: 'Evaluation size' })
    .evaluate((element) => element.scrollIntoView({ block: 'start' }))
  await page.screenshot({ path: testInfo.outputPath('canonical-scope-desktop.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('canonical-scope-mobile.png') })
})

test('does not fall back to inventory guesses when canonical dataset resolution fails', async ({
  page,
}) => {
  const submissions = await mockBench(page)
  let reads = 0
  let recovered = false
  await page.route('**/api/sr-bench/v1/datasets/selection?*', (route) => {
    reads += 1
    return !recovered
      ? route.fulfill({ status: 503, json: { error: 'Dataset resolution unavailable' } })
      : route.fulfill({
          json: {
            profile: 'quick',
            seed: 42,
            split: 'dev',
            model_requests: 0,
            benchmarks: [
              {
                id: 'mmlu-pro',
                title: 'MMLU-Pro',
                eligible: true,
                case_count: 2,
                source_ids: ['quick-v1'],
                reason: null,
              },
            ],
          },
        })
  })
  await page.goto('/evaluation?view=new')
  await expect(page.getByRole('alert')).toContainText('Dataset resolution unavailable')
  await expect(page.getByRole('button', { name: 'Review plan', exact: true })).toBeDisabled()
  await expect(page.getByRole('checkbox', { name: /MMLU-Pro/ })).toBeDisabled()
  recovered = true
  await page.getByRole('button', { name: 'Retry benchmark check' }).click()
  await expect(page.getByRole('checkbox', { name: /MMLU-Pro/ })).toBeChecked()
  expect(reads).toBeGreaterThanOrEqual(2)
  expect(submissions).toHaveLength(0)
})

test('persists profile, status and search filters across reload and run details', async ({
  page,
}) => {
  await mockBench(page)
  await page.route('**/api/sr-bench/v1/runs', (route) =>
    route.fulfill({
      json: {
        runs: [
          run,
          {
            ...run,
            id: 'smoke-run',
            manifest: { ...manifest, name: 'Smoke baseline', profile: 'smoke' },
          },
        ],
      },
    }),
  )
  await page.goto('/evaluation?view=runs')
  await chooseOption(page, 'Run profile', 'quick')
  await chooseOption(page, 'Run status', 'completed')
  await chooseOption(page, 'Run mode', 'live')
  await page.getByLabel('Search runs').fill('Baseline')
  await expect(page.getByRole('button', { name: 'Smoke baseline', exact: true })).toHaveCount(0)
  await expect(page).toHaveURL(
    (url) =>
      url.searchParams.get('profile') === 'quick' &&
      url.searchParams.get('status') === 'completed' &&
      url.searchParams.get('mode') === 'live' &&
      url.searchParams.get('q') === 'Baseline',
  )
  await page.reload()
  await expect(page.getByRole('combobox', { name: 'Run profile', exact: true })).toHaveText('Quick')
  await expect(page.getByLabel('Search runs')).toHaveValue('Baseline')
  await page.getByRole('button', { name: 'Baseline test', exact: true }).click()
  await page.getByRole('button', { name: 'Back to runs', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Run profile', exact: true })).toHaveText('Quick')
  await expect(page.getByLabel('Search runs')).toHaveValue('Baseline')
  await expect(page.getByRole('button', { name: 'Smoke baseline', exact: true })).toHaveCount(0)
})

test('keeps same-event filter changes and restores the URL after history navigation', async ({
  page,
}) => {
  await mockBench(page)
  await page.goto('/evaluation?view=runs&profile=quick&status=completed&context=kept')
  await page.getByRole('combobox', { name: 'Run mode', exact: true }).click()
  await page
    .getByRole('listbox')
    .locator('[data-value="live"]')
    .evaluate((option) => {
      ;(option as HTMLButtonElement).click()
      const input = document.querySelector<HTMLInputElement>('input[type="search"]')!
      const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!
      setValue.call(input, 'Baseline')
      input.dispatchEvent(new Event('input', { bubbles: true }))
    })
  const expected = (url: URL) =>
    url.searchParams.get('profile') === 'quick' &&
    url.searchParams.get('status') === 'completed' &&
    url.searchParams.get('mode') === 'live' &&
    url.searchParams.get('q') === 'Baseline' &&
    url.searchParams.get('context') === 'kept'
  await expect(page).toHaveURL(expected)
  await page.getByRole('button', { name: 'Datasets', exact: true }).click()
  await expect(page).toHaveURL(/view=datasets/)
  await page.goBack()
  await expect(page).toHaveURL(expected)
  await expect(page.getByLabel('Search runs')).toHaveValue('Baseline')
  await chooseOption(page, 'Run profile', 'smoke')
  await expect(page).toHaveURL(
    (url) =>
      url.searchParams.get('profile') === 'smoke' && url.searchParams.get('context') === 'kept',
  )
  await page.goForward()
  await expect(page).toHaveURL(/view=datasets/)
  await page.goBack()
  await expect(page.getByRole('combobox', { name: 'Run profile', exact: true })).toHaveText('Smoke')
  await page.getByLabel('Search runs').fill('model-a')
  await expect(page).toHaveURL(
    (url) =>
      url.searchParams.get('profile') === 'smoke' &&
      url.searchParams.get('mode') === 'live' &&
      url.searchParams.get('q') === 'model-a' &&
      url.searchParams.get('context') === 'kept',
  )
})

test('derives a MoM candidate from a failed 42-result frozen baseline without changing its protocol', async ({
  page,
}) => {
  await mockBench(page)
  const mom = { ...target, id: 'balance', kind: 'mom', config_hash: 'c'.repeat(64) }
  const frozen = {
    ...manifest,
    profile: 'standard',
    cost_policy: 'require_priced',
    seed: 77,
    cases: Array.from({ length: 14 }, (_, index) => ({ id: `frozen-case-${index}` })),
    targets: [target, { ...target, id: 'second' }, { ...target, id: 'third' }],
    sampling: { temperature: 0.2, max_tokens: 512 },
    limits: {
      concurrency: 1,
      max_output_tokens: 512,
      max_run_seconds: 120,
      max_cost_usd: 1,
      max_calls_per_case: 1,
      total_timeout_s: 60,
      idle_timeout_s: 30,
    },
  }
  const candidateRequests: unknown[] = []
  const submissions: Array<{ idempotency_key: string; manifest: Record<string, unknown> }> = []
  const forbidden: string[] = []
  await page.route('**/api/sr-bench/v1/targets', (route) =>
    route.fulfill({ json: { targets: [target, mom] } }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1', (route) =>
    route.fulfill({
      json: {
        ...run,
        status: 'failed',
        progress: { total: 42, completed: 41, failed: 1 },
        manifest: frozen,
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/candidate-plan', (route) => {
    const body = route.request().postDataJSON()
    candidateRequests.push(body)
    return route.fulfill({
      json: {
        status: 'validated',
        total: 14,
        plan_sha256: 'd'.repeat(64),
        manifest: {
          ...frozen,
          baseline_run_id: 'run-1',
          name: body.name,
          mode: body.mode,
          targets: [mom],
          experiment: body.experiment,
        },
      },
    })
  })
  for (const path of ['/datasets/compose', '/plans'])
    await page.route(`**/api/sr-bench/v1${path}`, async (route) => {
      if (route.request().method() === 'POST') {
        forbidden.push(path)
        return route.fulfill({
          status: 400,
          json: { error: 'Unexpected dispatch or new protocol' },
        })
      }
      return route.fallback()
    })
  await page.route('**/api/sr-bench/v1/runs', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback()
    const body = route.request().postDataJSON()
    submissions.push(body)
    if (submissions.length === 1) return route.abort('failed')
    return route.fulfill({ json: { ...run, id: 'run-derived', manifest: body.manifest } })
  })
  await page.route('**/api/sr-bench/v1/runs/run-derived', (route) =>
    route.fulfill({ json: { ...run, id: 'run-derived', manifest: submissions[0].manifest } }),
  )
  await page.goto(
    '/evaluation?view=new&baseline=run-1&experiment=exp-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa&role=candidate',
  )
  await expect(page.getByRole('region', { name: 'Frozen baseline protocol' })).toBeVisible()
  await expect(page.getByRole('region', { name: 'Frozen baseline protocol' })).toContainText(
    'Temperature 0.2 · Top P Not set · seed Not set',
  )
  await expect(page.getByRole('region', { name: 'Frozen baseline protocol' })).not.toContainText(
    'seed 77',
  )
  await expect(page.getByRole('region', { name: 'Frozen baseline protocol' })).toContainText(
    'Baseline status: failed · 1 failed results',
  )
  await expect(page.getByRole('radiogroup', { name: 'Evaluation size' })).toHaveCount(0)
  await expect(page.getByLabel('Budget (USD)', { exact: true })).toHaveCount(0)
  await expect(page.getByLabel('Temperature', { exact: true })).toHaveCount(0)
  await page.getByRole('combobox', { name: 'Add configured target', exact: true }).click()
  await expect(page.getByRole('listbox').locator('[data-value="single"]')).toHaveCount(0)
  await page.getByRole('listbox').locator('[data-value="balance"]').click()
  await page.getByLabel('Run name', { exact: true }).fill('Candidate with frozen protocol')
  await page
    .getByLabel('What changed? (optional)', { exact: true })
    .fill('Reduce cost without changing the question set.')
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Start evaluation', exact: true })).toBeEnabled()
  expect(candidateRequests).toEqual([
    {
      target_ids: ['balance'],
      mode: 'live',
      name: 'Candidate with frozen protocol',
      experiment: {
        id: 'exp-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        role: 'validation',
        hypothesis: 'Reduce cost without changing the question set.',
      },
    },
  ])
  expect(forbidden).toEqual([])
  await page.getByRole('button', { name: 'Start evaluation', exact: true }).click()
  await expect(
    page.getByRole('heading', { name: 'Evaluation submission needs reconciliation' }),
  ).toBeVisible()
  await page.reload()
  await expect(
    page.getByRole('button', { name: 'Check or submit same evaluation', exact: true }),
  ).toBeVisible()
  expect(submissions).toHaveLength(1)
  await page.getByRole('button', { name: 'Check or submit same evaluation', exact: true }).click()
  await expect(page).toHaveURL(/run=run-derived/)
  expect(submissions).toHaveLength(2)
  expect(submissions[1]).toEqual(submissions[0])
  expect(submissions[0].manifest.sampling).toEqual(frozen.sampling)
  expect(
    await page.evaluate(() => sessionStorage.getItem('sr-bench-submission:user-admin-1')),
  ).toBeNull()
  expect(forbidden).toEqual([])
})

test('uses authoritative failed-baseline comparisons without masking failed results or unknown savings', async ({
  page,
}, testInfo) => {
  await mockBench(page)
  const singles = [
    target,
    { ...target, id: 'flash', model: 'provider/flash' },
    { ...target, id: 'qwen', model: 'provider/qwen' },
  ]
  const failedBaseline = {
    ...run,
    status: 'failed',
    manifest: { ...manifest, targets: singles },
    progress: { total: 42, completed: 41, failed: 1 },
  }
  const candidate = {
    ...run,
    id: 'run-2',
    manifest: {
      ...manifest,
      name: 'Balance terminal comparison',
      targets: [{ ...target, id: 'balance', kind: 'mom' }],
    },
    progress: { total: 14, completed: 14, failed: 0 },
  }
  const baselineTargets = singles.map((item, index) => ({
    ...report.summary.targets[0],
    id: item.id,
    total: 14,
    completed: index ? 14 : 13,
    failed: index ? 0 : 1,
    scored: index ? 14 : 13,
    pending: 0,
    correct: index ? 9 : 13,
    macro_accuracy: index ? 9 / 14 : 13 / 14,
    complete: index > 0,
    cost_usd: index ? 1 : null,
    cost_complete: index > 0,
  }))
  const candidateTargets = [
    { ...baselineTargets[1], id: 'balance', correct: 12, macro_accuracy: 12 / 14, cost_usd: 0.5 },
  ]
  await page.route('**/api/sr-bench/v1/runs', (route) =>
    route.fulfill({ json: { runs: [failedBaseline, candidate] } }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1', (route) =>
    route.fulfill({ json: failedBaseline }),
  )
  await page.route('**/api/sr-bench/v1/runs/*/report', (route) =>
    route.fulfill({
      json: {
        ...report,
        status: route.request().url().includes('/run-1/') ? 'failed' : 'completed',
        summary: {
          targets: route.request().url().includes('/run-1/') ? baselineTargets : candidateTargets,
        },
        failure: null,
      },
    }),
  )
  await mockComparisonOptions(page, failedBaseline, [candidate])
  await page.route('**/api/sr-bench/v1/comparisons', (route) =>
    route.fulfill({
      json: {
        ...comparisonProtocol,
        baseline_run_id: 'run-1',
        candidate_run_id: 'run-2',
        baseline_status: 'failed',
        baseline_quality_complete: false,
        baseline_targets: baselineTargets,
        candidate_targets: candidateTargets,
        baseline_selection: 'Best observed single model under the frozen limits.',
        baseline_cost_comparison_eligible: false,
        baseline_cost_comparison_reason: 'Best single-model accounting is incomplete.',
        comparisons: [
          {
            baseline_target_id: 'single',
            candidate_target_id: 'balance',
            paired_cases: 14,
            quality_delta: -1 / 14,
            quality_delta_ci95: [-0.4, 0.3],
            wins: 0,
            losses: 1,
            ties: 13,
            baseline_subject_cost_usd: null,
            baseline_total_cost_usd: null,
            baseline_evaluation_cost_usd: 0,
            candidate_subject_cost_usd: 0.5,
            candidate_total_cost_usd: 0.5,
            candidate_evaluation_cost_usd: 0,
            subject_cost_saving_percent: null,
            total_cost_saving_percent: null,
          },
        ],
      },
    }),
  )
  await page.goto('/evaluation?view=runs&run=run-1')
  await expect(page.getByRole('button', { name: 'Evaluate candidate', exact: true })).toBeEnabled()
  await expect(page.getByText('This run is failed.', { exact: false })).toBeVisible()
  await page.getByRole('button', { name: 'Evaluate candidate', exact: true }).click()
  await expect(page).toHaveURL(/view=new&baseline=run-1/)
  await page.goto('/evaluation?view=compare')
  await chooseOption(page, 'Reference run', 'run-1')
  await page.getByRole('checkbox', { name: /Balance terminal comparison/ }).check()
  await page.getByRole('button', { name: 'Compare runs', exact: true }).click()
  await expect(
    page.getByText('1 failed baseline results · 0 failed candidate results.', { exact: false }),
  ).toBeVisible()
  const card = page
    .getByRole('article')
    .filter({ has: page.getByRole('heading', { name: candidate.manifest.name, exact: true }) })
  await expect(card).toContainText('−7.14 pp')
  await expect(card).toContainText('Unknown')
  await expect(card).not.toContainText('0%')
  await expect(page.getByText('A continuous trend is withheld', { exact: false })).toHaveCount(0)
  await page.getByText('Single-model baseline metrics', { exact: true }).click()
  await expect(
    page
      .getByRole('row')
      .filter({ has: page.getByRole('rowheader', { name: 'model-a', exact: true }) }),
  ).toContainText('13 completed · 1 failed')
  await expect(
    page
      .getByRole('row')
      .filter({ has: page.getByRole('rowheader', { name: 'model-a', exact: true }) }),
  ).not.toContainText('$0.00000')
  await page.screenshot({ path: testInfo.outputPath('failed-baseline-comparison.png') })
})

test('does not enable active baseline derivation or comparisons with missing terminal results', async ({
  page,
}) => {
  await mockBench(page)
  let candidatePlans = 0,
    comparisons = 0
  await page.route('**/api/sr-bench/v1/runs/run-1', (route) =>
    route.fulfill({
      json: { ...run, status: 'running', progress: { total: 42, completed: 40, failed: 1 } },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs/run-1/candidate-plan', (route) => {
    candidatePlans++
    return route.abort()
  })
  await page.goto('/evaluation?view=new&baseline=run-1')
  await expect(
    page.getByText('Choose a finished live single-model baseline with its full plan.', {
      exact: true,
    }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Review plan', exact: true })).toHaveCount(0)
  await page.route('**/api/sr-bench/v1/comparison-options?*', (route) =>
    route.fulfill({
      json: {
        baseline: null,
        baselines: [],
        options: [],
        next_cursor: null,
        has_more: false,
        scanned_pairs: 1,
        scan_limited: false,
        unverified_pairs: 0,
        unverified_baselines: 0,
        model_requests: 0,
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/comparisons', (route) => {
    comparisons++
    return route.fulfill({
      status: 400,
      json: { error: 'Baseline quality results are incomplete' },
    })
  })
  await page.goto('/evaluation?view=compare')
  await expect(page.getByText('No comparable results yet.', { exact: false })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Compare runs', exact: true })).toHaveCount(0)
  expect(comparisons).toBe(0)
  await page.goto('/evaluation?view=compare&baseline=run-1&candidate=run-2')
  await expect(page.getByText('No comparable results yet.', { exact: false })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Compare runs', exact: true })).toHaveCount(0)
  await expect(page.getByText('Quality Δ', { exact: true })).toHaveCount(0)
  expect(candidatePlans).toBe(0)
  expect(comparisons).toBe(0)
})

test('an explicit dataset can be selected while automatic resolution is stalled without adopting a late response', async ({
  page,
}) => {
  await mockBench(page)
  let release!: () => void
  const waiting = new Promise<void>((resolve) => {
    release = resolve
  })
  await page.route('**/api/sr-bench/v1/datasets/selection?*', async (route) => {
    await waiting
    await route
      .fulfill({
        json: { profile: 'quick', seed: 99, split: 'dev', model_requests: 0, benchmarks: [] },
      })
      .catch(() => {})
  })
  await page.goto('/evaluation?view=new')
  await expect(page.getByText('Checking prepared benchmarks…', { exact: true })).toBeVisible()
  await chooseDataset(page, 'quick-v1')
  await chooseOption(page, 'Add configured target', 'single')
  await expect(page.getByRole('checkbox', { name: /MMLU-Pro/ })).toBeChecked()
  await expect(page.getByRole('button', { name: 'Review plan', exact: true })).toBeEnabled()
  release()
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
})

test('renders signed quality and cost changes with truthful positive negative and zero colors', async ({
  page,
}, testInfo) => {
  await mockBench(page)
  const values = [
    { id: 'positive', quality: 0.0425, saving: 7.89 },
    { id: 'negative', quality: -0.0425, saving: -7.89 },
    { id: 'zero', quality: 0, saving: 0 },
    { id: 'tiny', quality: 0.000002, saving: -0.0002 },
  ]
  const candidates = values.map((value) => ({
    ...run,
    id: value.id,
    manifest: {
      ...manifest,
      name: value.id,
      targets: [{ ...target, id: 'balance-internal', model: 'balance-connected', kind: 'mom' }],
    },
  }))
  await page.route('**/api/sr-bench/v1/runs', (route) =>
    route.fulfill({ json: { runs: [run, ...candidates] } }),
  )
  await mockComparisonOptions(page, run, candidates)
  await page.route('**/api/sr-bench/v1/comparisons', (route) => {
    const value = values.find(
      (item) => item.id === route.request().postDataJSON().candidate_run_id,
    )!
    return route.fulfill({
      json: {
        baseline_selection: 'Best saved single model.',
        ...comparisonProtocol,
        comparisons: [
          {
            baseline_target_id: 'single',
            candidate_target_id: 'balance-internal',
            paired_cases: 25,
            quality_delta: value.quality,
            quality_delta_ci95: [-0.1, 0.1],
            baseline_subject_cost_usd: 1,
            baseline_total_cost_usd: 1,
            baseline_evaluation_cost_usd: 0,
            candidate_subject_cost_usd: 1 - value.saving / 100,
            candidate_total_cost_usd: 1 - value.saving / 100,
            candidate_evaluation_cost_usd: 0,
            subject_cost_saving_percent: value.saving,
            total_cost_saving_percent: value.saving,
            cache_neutral_cost_saving_percent: value.saving,
          },
        ],
      },
    })
  })
  await page.goto(
    '/evaluation?view=compare&baseline=run-1&candidate=positive&candidate=negative&candidate=zero&candidate=tiny',
  )
  await expect(page.getByRole('article')).toHaveCount(4)
  await expect(
    page
      .getByRole('region', { name: 'Iteration progress chart' })
      .getByText('Total cost saving', { exact: true }),
  ).toBeVisible()
  const colors = await page.evaluate(() => {
    const probe = document.createElement('span')
    document.body.appendChild(probe)
    const color = (value: string) => {
      probe.style.color = value
      return getComputedStyle(probe).color
    }
    const output = {
      positive: color('var(--color-success, #60c98c)'),
      negative: color('var(--color-danger, #ff7b7b)'),
      neutral: color('var(--text-primary)'),
    }
    probe.remove()
    return output
  })
  for (const [id, direction, quality, cost] of [
    ['positive', 'positive', '+4.25 pp', '+7.89%'],
    ['negative', 'negative', '−4.25 pp', '−7.89%'],
    ['zero', 'neutral', '0 pp', '0%'],
  ] as const) {
    const article = page
      .getByRole('article')
      .filter({ has: page.getByRole('heading', { name: id, exact: true }) })
    const metrics = article.locator('strong > [data-direction]')
    await expect(metrics).toHaveCount(2)
    await expect(metrics.nth(0)).toContainText(quality)
    await expect(metrics.nth(1)).toContainText(cost)
    for (const metric of await metrics.all())
      await expect(metric).toHaveCSS('color', colors[direction])
    if (id === 'zero') await expect(metrics.getByText('No change', { exact: true })).toHaveCount(2)
  }
  expect(colors.positive).not.toBe(colors.negative)
  expect(colors.neutral).not.toBe(colors.negative)
  const tiny = page
    .getByRole('article')
    .filter({ has: page.getByRole('heading', { name: 'tiny', exact: true }) })
  await expect(tiny).toContainText('+0.0002 pp')
  await expect(tiny).toContainText('−0.0002%')
  await page.getByText('Single-model baseline metrics', { exact: true }).click()
  await expect(page.getByRole('rowheader', { name: 'model-a', exact: true })).toBeVisible()
  await page.getByText('Detailed iteration metrics and uncertainty', { exact: true }).click()
  await expect(page.getByRole('rowheader').filter({ hasText: 'balance-connected' })).toHaveCount(4)
  await page
    .getByRole('article')
    .first()
    .locator('..')
    .screenshot({
      path: testInfo.outputPath('signed-comparison-desktop.png'),
    })
})

test('a deep link cannot promote a baseline until its first compatible child is verified', async ({
  page,
}) => {
  await mockBench(page)
  const baseline = {
    run_id: 'deep-baseline',
    name: 'Deep saved baseline',
    profile: 'quick',
    case_count: 2,
  }
  await page.route('**/api/sr-bench/v1/comparison-options?*', (route) => {
    const query = new URL(route.request().url()).searchParams
    const selected = query.has('baseline_run_id')
    const more = selected && !query.has('after')
    return route.fulfill({
      json: {
        baseline: selected ? baseline : null,
        baselines: selected
          ? []
          : [{ ...baseline, run_id: 'other-baseline', name: 'Verified reference' }],
        options:
          selected && !more
            ? [{ ...baseline, run_id: 'valid-child', name: 'Compatible result' }]
            : [],
        next_cursor: more ? 'next' : null,
        has_more: more,
        scanned_pairs: 1,
        scan_limited: false,
        unverified_pairs: 0,
        unverified_baselines: 0,
        model_requests: 0,
      },
    })
  })
  await page.goto('/evaluation?view=compare&baseline=deep-baseline')
  await expect(
    page.getByRole('button', { name: 'Load more comparison runs', exact: true }),
  ).toBeVisible()
  await page.getByRole('combobox', { name: 'Reference run', exact: true }).click()
  await expect(page.getByRole('option')).toHaveCount(1)
  await expect(page.getByRole('option', { name: /Deep saved baseline/ })).toHaveCount(0)
  await page.getByRole('combobox', { name: 'Reference run', exact: true }).press('Escape')
  await page.getByRole('button', { name: 'Load more comparison runs', exact: true }).click()
  await expect(page.getByRole('checkbox', { name: /Compatible result/ })).toBeVisible()
  await page.getByRole('combobox', { name: 'Reference run', exact: true }).click()
  await expect(page.getByRole('option', { name: /Deep saved baseline/ })).toBeVisible()
})
