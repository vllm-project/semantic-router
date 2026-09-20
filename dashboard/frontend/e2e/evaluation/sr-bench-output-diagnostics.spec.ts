import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'

const targets = ['large', 'fast', 'small'].map((id) => ({
  id,
  kind: 'single',
  model: `provider/model-${id}`,
  base_url: 'http://localhost:8000/v1',
}))

async function mockOutputEvidence(
  page: Page,
  {
    mode = 'live',
    partial = false,
    parsedFormatAnswer = false,
    stopped = false,
    explicitFailure = false,
  }: {
    mode?: string
    partial?: boolean
    parsedFormatAnswer?: boolean
    stopped?: boolean
    explicitFailure?: boolean
  } = {},
) {
  await mockAuthenticatedAppShell(page)
  const run = {
    id: 'output-run',
    status: stopped || explicitFailure ? 'failed' : partial ? 'running' : 'completed',
    created_at: '2026-09-20T00:00:00Z',
    updated_at: '2026-09-20T00:01:00Z',
    progress: {
      total: 84,
      completed: partial ? 58 : explicitFailure ? 83 : 84,
      failed: explicitFailure ? 1 : 0,
    },
    manifest: {
      version: 'sr-bench-1.0',
      name: 'Output evidence fixture',
      mode,
      profile: 'quick',
      seed: 42,
      targets,
      sampling: { temperature: 1, max_tokens: 8192 },
      limits: {},
      cases: Array.from({ length: 28 }, (_, index) => ({
        id: `case-${index}`,
        benchmark: 'mmlu-pro',
        messages: [{ role: 'user', content: 'Synthetic choice question.' }],
      })),
    },
  }
  const metrics = targets.map((target, index) => {
    const limited = [0, 1, 5][index]
    const diagnostic = {
      planned_cases: 28,
      result_cases: 28,
      output_limit_cases: limited,
      strict_format: {
        checked_cases: 28 - limited,
        failed_cases: index === 0 ? 9 : 0,
        unassessed_cases: limited,
      },
      subject_calls: {
        total: 28,
        finish_reasons: limited ? { stop: 28 - limited, length: limited } : { stop: 28 },
        unknown_finish_reason: 0,
      },
    }
    return {
      id: target.id,
      total: 28,
      completed: index === 0 ? (partial ? 2 : explicitFailure ? 27 : 28) : 28,
      failed: explicitFailure && index === 0 ? 1 : 0,
      scored: index === 0 ? (partial ? 2 : explicitFailure ? 27 : 28) : 28,
      pending: partial && index === 0 ? 25 : 0,
      correct: index === 0 && parsedFormatAnswer ? 1 : 0,
      accuracy: index === 0 && parsedFormatAnswer ? 1 / 28 : 0,
      macro_accuracy: index === 0 && parsedFormatAnswer ? 1 / 28 : 0,
      complete: !((partial || explicitFailure) && index === 0),
      cost_usd: null,
      tokens: null,
      output_diagnostics:
        partial && index === 0
          ? {
              planned_cases: 28,
              result_cases: 2,
              output_limit_cases: 0,
              strict_format: { checked_cases: 0, failed_cases: 0, unassessed_cases: 28 },
              subject_calls: {
                total: 3,
                finish_reasons: { tool_calls: 1 },
                unknown_finish_reason: 2,
              },
            }
          : diagnostic,
    }
  })
  const writes: string[] = []
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const request = route.request()
    const endpoint = new URL(request.url()).pathname.replace('/api/sr-bench/v1', '')
    if (request.method() !== 'GET') writes.push(endpoint)
    let body: unknown
    if (endpoint === '/catalog')
      body = {
        version: 'sr-bench-1.0',
        profiles: ['smoke', 'quick', 'standard'].map((id) => ({ id, purpose: id })),
        benchmarks: [{ id: 'mmlu-pro', title: 'MMLU-Pro', kind: 'capability' }],
      }
    else if (endpoint === '/targets') body = { targets }
    else if (endpoint === '/datasets') body = { datasets: [] }
    else if (endpoint === '/runs') body = { runs: [run] }
    else if (endpoint === `/runs/${run.id}`) body = run
    else if (endpoint === `/runs/${run.id}/report`)
      body = {
        version: 'sr-bench-1.0',
        run_id: run.id,
        status: run.status,
        summary: { targets: metrics, wall_time_s: 60 },
        benchmarks: metrics.map((metric) => ({
          ...metric,
          benchmark: 'mmlu-pro',
          target_id: metric.id,
        })),
        limitations: [],
        provenance: {},
      }
    else if (endpoint === `/runs/${run.id}/results`)
      body = {
        total: 84,
        limit: 2,
        next_cursor: 2,
        results: [
          {
            case_id: 'case-0',
            target_id: 'large',
            benchmark: 'mmlu-pro',
            status: 'completed',
            correct: parsedFormatAnswer,
            score: parsedFormatAnswer ? 1 : 0,
            answer: parsedFormatAnswer ? 'B' : null,
            details: { strict_format: false },
          },
          {
            case_id: 'case-1',
            target_id: 'fast',
            benchmark: 'mmlu-pro',
            status: 'completed',
            correct: false,
            score: 0,
            answer: null,
            details: { quality_failure: 'output_limit', output_complete: false },
          },
        ],
      }
    else if (endpoint === `/runs/${run.id}/calls`)
      body = {
        total: 84,
        limit: 1,
        next_cursor: 1,
        calls: [
          {
            id: 'format-call',
            case_id: 'case-0',
            target_id: 'large',
            role: 'subject',
            status: 'completed',
            finish_reason: 'stop',
            output_complete: true,
          },
        ],
      }
    else if (endpoint === `/runs/${run.id}/calls/format-call`)
      body = {
        id: 'format-call',
        case_id: 'case-0',
        target_id: 'large',
        role: 'subject',
        status: 'completed',
        finish_reason: 'stop',
        output_complete: true,
        final: '**B**',
      }
    else if (endpoint === `/runs/${run.id}/events`) body = { events: [] }
    else return route.fulfill({ status: 404, json: { error: 'Unexpected fixture request' } })
    await route.fulfill({ json: body })
  })
  await page.goto('/evaluation?view=runs&run=output-run')
  await expect(page.getByRole('heading', { name: run.manifest.name, exact: true })).toBeVisible()
  return writes
}

test('full report output counts remain distinct from completed status and loaded evidence', async ({
  page,
}, testInfo) => {
  const writes = await mockOutputEvidence(page)
  const output = page.getByRole('region', { name: 'Output diagnostics', exact: true })
  const large = output.getByRole('listitem', { name: targets[0].model, exact: true })
  const fast = output.getByRole('listitem', { name: targets[1].model, exact: true })
  const small = output.getByRole('listitem', { name: targets[2].model, exact: true })
  await expect(large).toContainText('0 / 28(0%)')
  await expect(large).toContainText('9 / 28(32.14%)')
  await expect(fast).toContainText('1 / 28(3.57%)')
  await expect(fast).toContainText('0 / 27(0%)')
  await expect(fast).toContainText('1 planned case not assessed')
  await expect(small).toContainText('5 / 28(17.86%)')
  await expect(page.getByText('completed', { exact: true }).first()).toBeVisible()
  const disclosure = output.locator('details')
  await expect(disclosure).not.toHaveAttribute('open', '')
  await expect(output.locator('pre')).toHaveCount(0)
  const icon = disclosure.locator('summary svg')
  await expect(icon).toHaveAttribute('width', '14')
  await expect(icon).toHaveAttribute('height', '14')

  await page.setViewportSize({ width: 1600, height: 1000 })
  await output.screenshot({ path: testInfo.outputPath('output-diagnostics-desktop.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  await output.scrollIntoViewIfNeeded()
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1))
    .toBe(true)
  const iconBox = (await icon.boundingBox())!
  expect(iconBox.width).toBeLessThanOrEqual(16)
  expect(iconBox.height).toBeLessThanOrEqual(16)
  await output.screenshot({ path: testInfo.outputPath('output-diagnostics-mobile.png') })
  await output.getByText('How subject calls ended', { exact: true }).click()
  const ended = disclosure.getByRole('listitem').filter({ hasText: targets[0].model })
  await expect(ended).toContainText('28 recorded subject calls')
  await expect(ended).toContainText('Stop28 / 28')
  await expect(ended).toContainText('Unknown0 / 28')
  await page.getByRole('tab', { name: 'Questions', exact: true }).click()
  await expect(page.getByText(/Loaded 2 of 84 persisted results/)).toBeVisible()
  expect(writes).toEqual([])
})

test('format noncompliance does not replace a correctly parsed answer or its grade', async ({
  page,
}) => {
  await mockOutputEvidence(page, { parsedFormatAnswer: true })
  await expect(
    page.getByText('Full-report counts. Format compliance is separate from answer correctness.'),
  ).toBeVisible()
  await page.getByRole('tab', { name: 'Questions', exact: true }).click()
  const row = page
    .getByRole('row')
    .filter({ has: page.getByRole('button', { name: 'case-0', exact: true }) })
  await expect(row.getByRole('cell', { name: '1', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'case-0', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Parsed answer', exact: true })).toBeVisible()
  await expect(
    page
      .getByRole('tabpanel', { name: 'Questions', exact: true })
      .locator('pre')
      .filter({ hasText: /^B$/ }),
  ).toHaveText('B')
  await expect(page.getByText(/final text did not match the required answer format/)).toBeVisible()
  await expect(page.getByText('No answer parsed.', { exact: true })).toHaveCount(0)
})

test('unassessed cases and unknown finish reasons are not zero failures or normal stops', async ({
  page,
}) => {
  await mockOutputEvidence(page, { partial: true })
  const output = page.getByRole('region', { name: 'Output diagnostics', exact: true })
  const row = output.getByRole('listitem', { name: targets[0].model, exact: true })
  await expect(row).toContainText('2 / 28 case results recorded')
  await expect(row).toContainText('Not assessed')
  await expect(row).toContainText('28 planned cases not assessed')
  await expect(row).not.toContainText('0 / 0')
  await output.getByText('How subject calls ended', { exact: true }).click()
  const ended = output
    .locator('details')
    .getByRole('listitem')
    .filter({ hasText: targets[0].model })
  await expect(ended).toContainText('3 recorded subject calls')
  await expect(ended).toContainText('Tool calls1 / 3')
  await expect(ended).toContainText('Unknown2 / 3')
  await expect(ended.getByText('Stop', { exact: true })).toHaveCount(0)
  await expect(output.getByText(/tool calls can continue a task/i)).toBeVisible()
})

test('unfinished targets show scored coverage instead of a provisional accuracy', async ({
  page,
}) => {
  const writes = await mockOutputEvidence(page, { partial: true, parsedFormatAnswer: true })
  const table = page
    .getByRole('table')
    .filter({ has: page.getByRole('columnheader', { name: 'Macro accuracy', exact: true }) })
  const row = table
    .getByRole('row')
    .filter({ has: page.getByRole('rowheader', { name: targets[0].model, exact: true }) })
  await expect(row.getByRole('cell', { name: 'Pending', exact: true })).toBeVisible()
  await expect(row.getByRole('cell', { name: '2 / 28', exact: true })).toBeVisible()
  await expect(row).not.toContainText('3.57%')
  const finished = table
    .getByRole('row')
    .filter({ has: page.getByRole('rowheader', { name: targets[1].model, exact: true }) })
  await expect(finished.getByRole('cell', { name: '0%', exact: true })).toBeVisible()
  await page.getByText('Benchmark results', { exact: true }).click()
  const benchmarkTable = page
    .getByRole('table')
    .filter({ has: page.getByRole('columnheader', { name: 'Accuracy', exact: true }) })
  const benchmarkRow = benchmarkTable
    .getByRole('row')
    .filter({ has: page.getByRole('rowheader', { name: targets[0].model, exact: true }) })
  await expect(benchmarkRow.getByRole('cell', { name: 'Pending', exact: true })).toBeVisible()
  await expect(benchmarkRow).not.toContainText('95% CI')
  expect(writes).toEqual([])
})

test('a stopped target with missing outcomes remains incomplete', async ({ page }) => {
  await mockOutputEvidence(page, { partial: true, parsedFormatAnswer: true, stopped: true })
  const table = page
    .getByRole('table')
    .filter({ has: page.getByRole('columnheader', { name: 'Macro accuracy', exact: true }) })
  const row = table
    .getByRole('row')
    .filter({ has: page.getByRole('rowheader', { name: targets[0].model, exact: true }) })
  await expect(row.getByRole('cell', { name: 'Incomplete', exact: true })).toBeVisible()
  await expect(row).not.toContainText('3.57%')
})

test('explicit failures retain the full denominator once all outcomes are known', async ({
  page,
}) => {
  await mockOutputEvidence(page, { parsedFormatAnswer: true, explicitFailure: true })
  const table = page
    .getByRole('table')
    .filter({ has: page.getByRole('columnheader', { name: 'Macro accuracy', exact: true }) })
  const row = table
    .getByRole('row')
    .filter({ has: page.getByRole('rowheader', { name: targets[0].model, exact: true }) })
  await expect(row.getByRole('cell', { name: '3.57%', exact: true })).toBeVisible()
  await expect(row.getByRole('cell', { name: '1 / 28', exact: true })).toBeVisible()
  await expect(row.getByRole('cell', { name: '27 / 28', exact: true })).toBeVisible()
  await expect(row).not.toContainText('Incomplete')
  await expect(row).not.toContainText('Pending')
})

test('a Markdown final with no parsed answer is explained separately from an output-limit result', async ({
  page,
}) => {
  const writes = await mockOutputEvidence(page)
  await page.getByRole('tab', { name: 'Questions', exact: true }).click()
  await page.getByRole('button', { name: 'case-0', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Parsed answer', exact: true })).toBeVisible()
  await expect(page.getByText('No answer parsed.', { exact: true })).toBeVisible()
  await expect(page.getByText(/final text did not match the required answer format/)).toBeVisible()
  await expect(page.getByText('No final answer recorded.', { exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'Back to questions', exact: true }).click()
  await page.getByRole('button', { name: 'case-1', exact: true }).click()
  await expect(page.getByText(/output limit was reached/)).toBeVisible()
  await expect(page.getByText(/final text did not match the required answer format/)).toHaveCount(0)
  await page.getByRole('tab', { name: 'Calls', exact: true }).click()
  await page.getByRole('button', { name: 'format-call', exact: true }).click()
  await page.getByText('Original call receipt', { exact: true }).click()
  await expect(
    page.getByRole('tabpanel', { name: 'Calls', exact: true }).locator('pre'),
  ).toContainText('**B**')
  expect(writes).toEqual([])
})

for (const mode of ['preview', 'replay']) {
  test(`${mode} does not present saved output diagnostics as measured live evidence`, async ({
    page,
  }) => {
    await mockOutputEvidence(page, { mode })
    await expect(
      page.getByRole('heading', { name: 'Target comparison', exact: true }),
    ).toBeVisible()
    await expect(page.getByRole('region', { name: 'Output diagnostics', exact: true })).toHaveCount(
      0,
    )
  })
}
