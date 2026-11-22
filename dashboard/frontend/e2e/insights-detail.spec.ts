import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const record = {
  id: 'insight-ui-example',
  request_id: 'request-ui-example',
  timestamp: '2026-09-19T00:00:00Z',
  turn_index: 0,
  recipe: 'balanced',
  decision: 'technical_request',
  decision_tier: 1,
  decision_priority: 10,
  original_model: 'balance',
  selected_model: 'efficient-model',
  selection_method: 'multi_factor',
  lifecycle_state: 'completed',
  response_status: 200,
  duration_ms: 1420,
  context_token_count: 90,
  prompt_tokens: 80,
  completion_tokens: 20,
  total_tokens: 100,
  currency: 'USD',
  actual_cost: 0.001,
  baseline_cost: 0.004,
  cost_savings: 0.003,
  baseline_model: 'baseline-model',
  signals: { domain: ['technical', 'code'] },
  route_diagnostics: { selection_reasoning: 'Selected the highest ranked eligible model.' },
  projections: ['specialist'],
  projection_scores: { complexity: 0.72, action_margin: -0.125 },
  signal_values: { 'domain:code': 0.9, 'systemone:zero': 0 },
  signal_confidences: { 'systemone:zero': 0, 'domain:technical': 0.82 },
  signal_error_matches: { 'systemone:zero': false },
  projection_trace: {
    schema_version: '1',
    partitions: [
      {
        group_name: 'intent',
        signal_type: 'domain',
        semantics: 'softmax',
        temperature: 0.7,
        winner: 'code',
        winner_score: 0.8,
        raw_winner_score: 0.9,
        margin: 0.6,
        contenders: [
          { name: 'code', raw_score: 0.9, normalized_score: 0.8 },
          { name: 'general', raw_score: 0.3, normalized_score: 0.2 },
        ],
      },
      {
        group_name: 'fallback_group',
        signal_type: 'domain',
        winner: 'general',
        default_used: true,
        winner_score: 0,
        margin: 0,
      },
    ],
    scores: [
      {
        name: 'complexity',
        total: 0.72,
        inputs: [
          { type: 'signal', name: 'domain:code', value: 0.9, weight: 0.8, contribution: 0.72 },
          { type: 'signal', name: 'systemone:zero', value: 0, weight: 0.2, contribution: 0 },
        ],
      },
    ],
    mappings: [
      {
        mapping_name: 'capability',
        source_score: 'complexity',
        score_value: 0.72,
        selected_output: 'specialist',
        confidence: 0.88,
        boundary_distance: 0.12,
        outputs: [
          { name: 'specialist', matched: true, boundary_distance: 0.12 },
          { name: 'general', matched: false, boundary_distance: -0.12 },
        ],
      },
    ],
  },
}

async function setup(page: Page, data: unknown = record) {
  await mockAuthenticatedAppShell(page)
  const writes: string[] = []
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.startsWith('/api/') && request.method() !== 'GET')
      writes.push(request.method())
  })
  await page.route('**/api/router/api/v1/observability/replays/insight-ui-example', (route) =>
    route.fulfill({ json: data }),
  )
  return writes
}

test('insight detail leads with routing outcomes and progressively reveals projection and metadata evidence', async ({
  page,
}, info) => {
  const writes = await setup(page)
  await page.goto('/observe/requests/insight-ui-example')
  const trace = page.getByRole('region', { name: 'Projection Trace', exact: true })
  await expect(trace).toBeVisible()
  await expect(page.getByRole('heading', { level: 1 })).toContainText('request-ui-ex')
  await expect(page.getByText('Routed to', { exact: false }).first()).toContainText(
    'efficient-model',
  )
  const steps = page.locator('[data-trace-step]')
  await expect(steps).toHaveText([
    'Received',
    'Signals matched',
    'Projections',
    'Decision matched',
    'Model selected',
    'Plugins',
    'Response',
  ])
  await expect(trace.getByRole('region', { name: 'Signal groups', exact: true })).toBeVisible()
  await expect(trace.getByRole('region', { name: 'Weighted scores', exact: true })).toBeVisible()
  await expect(trace.getByRole('region', { name: 'Routing outputs', exact: true })).toBeVisible()
  await expect(trace.getByText('Default fallback', { exact: true })).toBeVisible()
  await expect(trace.getByRole('table')).toBeHidden()
  await expect(trace.locator('pre')).toBeHidden()
  await expect(
    page.getByRole('button', { name: 'Expand Routing Metadata', exact: true }),
  ).toHaveAttribute('aria-expanded', 'false')
  await page.screenshot({ path: info.outputPath('insight-desktop-overview.png'), fullPage: true })
  await trace.getByText('Inspect intent candidates', { exact: true }).click()
  await expect(trace.getByRole('table', { name: 'intent candidates', exact: true })).toBeVisible()
  await expect(trace.getByRole('row', { name: /code Winner/ })).toContainText('0.9000')
  await trace.getByText('Inspect complexity calculation', { exact: true }).click()
  await expect(
    trace
      .getByRole('table', { name: 'complexity calculation', exact: true })
      .getByRole('row', { name: /systemone:zero/ }),
  ).toContainText('0.0000')
  await trace.getByText('Inspect capability thresholds', { exact: true }).click()
  const source = trace.getByRole('link', { name: 'complexity', exact: true })
  const id = (await source.getAttribute('href'))!.slice(1)
  await expect(page.locator(`[id="${id}"]`)).toContainText('complexity')
  await expect(
    trace
      .getByRole('table', { name: 'capability thresholds', exact: true })
      .getByRole('row', { name: /general/ }),
  ).toContainText('No')
  await trace.screenshot({ path: info.outputPath('insight-projection-expanded.png') })
  await page.getByRole('button', { name: 'Expand Routing Metadata', exact: true }).click()
  const metadata = page.getByRole('region', { name: 'Routing Metadata', exact: true })
  await expect(
    metadata.getByRole('region', { name: 'Projection outputs', exact: true }),
  ).toContainText('specialist')
  await expect(
    metadata.getByRole('region', { name: 'Projection scores', exact: true }),
  ).toContainText('-0.1250')
  // System One's raw values fold away under its answers until opened.
  await metadata.getByText('Raw System One values', { exact: false }).click()
  const signals = metadata.getByRole('table', {
    name: 'Recorded signal values and confidence',
    exact: true,
  })
  await expect(signals.getByRole('row', { name: /systemone:zero/ })).toHaveText(
    'systemone:zero0.00000.0000No',
  )
  await expect(signals.getByRole('row', { name: /domain:technical/ })).toContainText('Not recorded')
  await metadata.screenshot({ path: info.outputPath('insight-metadata-expanded.png') })
  expect(writes).toEqual([])
})

test('projection and metadata stay usable on mobile with keyboard disclosure and contained tables', async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 390, height: 844 })
  const writes = await setup(page)
  await page.goto('/observe/requests/insight-ui-example')
  const trace = page.getByRole('region', { name: 'Projection Trace', exact: true })
  await expect(trace).toBeVisible()
  const summary = trace.getByText('Inspect complexity calculation', { exact: true }).locator('..')
  await summary.focus()
  await summary.press('Enter')
  await expect(
    trace.getByRole('table', { name: 'complexity calculation', exact: true }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Expand Routing Metadata', exact: true }).click()
  const metadata = page.getByRole('region', { name: 'Routing Metadata', exact: true })
  await expect(metadata.getByRole('table')).toBeVisible()
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1))
    .toBe(true)
  for (const region of [trace, metadata]) {
    const box = await region.boundingBox()
    expect(box!.x).toBeGreaterThanOrEqual(0)
    expect(box!.x + box!.width).toBeLessThanOrEqual(390)
  }
  await page.screenshot({ path: info.outputPath('insight-mobile.png'), fullPage: true })
  expect(writes).toEqual([])
})

test('legacy and partial records do not manufacture a projection result', async ({ page }) => {
  await setup(page, {
    ...record,
    projection_trace: {
      schema_version: '1',
      mappings: [
        { mapping_name: 'uncaptured_mapping', source_score: 'unknown_score', score_value: 0 },
      ],
    },
    projections: undefined,
    projection_scores: undefined,
    signal_values: { 'systemone:zero': 0 },
    signal_confidences: undefined,
  })
  await page.goto('/observe/requests/insight-ui-example')
  const trace = page.getByRole('region', { name: 'Projection Trace', exact: true })
  await expect(trace.getByText('No output selected', { exact: true })).toBeVisible()
  await expect(trace.getByRole('region', { name: 'Signal groups', exact: true })).toHaveCount(0)
  await trace.getByText('Inspect uncaptured_mapping thresholds', { exact: true }).click()
  await expect(
    trace.getByText('Threshold evaluations were not recorded.', { exact: true }),
  ).toBeVisible()
  await expect(trace.getByRole('link')).toHaveCount(0)
  await page.getByRole('button', { name: 'Expand Routing Metadata', exact: true }).click()
  await page.getByText('Raw System One values', { exact: false }).click()
  await expect(
    page
      .getByRole('table', { name: 'Recorded signal values and confidence', exact: true })
      .getByRole('row', { name: /systemone:zero/ }),
  ).toContainText('Not recorded')
})

test('usage, signals and plugin status keep their recorded evidence behind progressive disclosure', async ({
  page,
}, info) => {
  const writes = await setup(page, {
    ...record,
    signals: { ...record.signals, future_signal: ['future_rule_v1'] },
    jailbreak_detected: true,
    jailbreak_score_available: true,
    jailbreak_confidence: 0,
    hallucination_enabled: true,
    hallucination_score_available: false,
    outcomes: [
      {
        source: 'router',
        target: 'response-check-unavailable',
        verdict: 'unavailable',
        reason: 'Detector unavailable',
        metadata: { stage: 'response', signal: 'hallucination', score_available: 'false' },
      },
      {
        source: 'router',
        target: 'response-check-not-applicable',
        verdict: 'not_applicable',
        reason: 'No applicable context',
        metadata: { stage: 'response', signal: 'hallucination', score_available: 'false' },
      },
    ],
  })
  await page.goto('/observe/requests/insight-ui-example')
  await page.getByRole('button', { name: 'Expand Routing Metadata', exact: true }).click()
  const metadata = page.getByRole('region', { name: 'Routing Metadata', exact: true })
  await expect(
    metadata.getByRole('region', { name: 'Projection scores', exact: true }),
  ).toBeVisible()
  await expect(metadata.getByRole('region', { name: 'Signal evidence', exact: true })).toBeVisible()
  await metadata.screenshot({ path: info.outputPath('routing-metadata.png') })

  const usage = page.getByRole('region', { name: 'Usage & Cost', exact: true })
  const signals = page
    .getByRole('list', { name: 'Routing trace' })
    .locator('[data-trace-step="Signals matched"]')
    .locator('xpath=ancestor::li[1]')
  const plugins = page.getByRole('region', { name: 'Plugin Status', exact: true })
  await expect(signals.getByTitle('future_rule_v1', { exact: true })).toBeVisible()
  await expect(plugins.getByText('Request jailbreak:', { exact: false })).toContainText('0.0%')
  const outcomeSummary = plugins
    .locator('summary')
    .filter({ hasText: '1 unavailable · 1 not applicable' })
  await expect(outcomeSummary).toBeVisible()
  await expect(plugins.getByText('Clean', { exact: true })).toHaveCount(0)
  await expect(plugins.getByText('Detector unavailable', { exact: true })).toBeHidden()
  await expect(
    usage.getByText('not a GPU bill or provider invoice.', { exact: false }),
  ).toBeVisible()
  const explanation = usage.getByText('How these estimates are calculated', { exact: true })
  const baselineBasis = usage.getByText('New records use the highest estimate', { exact: false })
  await expect(baselineBasis).toBeHidden()
  await usage.evaluate((element) => element.scrollIntoView({ block: 'start' }))
  await page.screenshot({ path: info.outputPath('usage-and-status-desktop.png') })
  await explanation.click()
  await expect(baselineBasis).toBeVisible()
  await expect(
    usage.getByRole('link', { name: 'View current configured model rates' }),
  ).toHaveAttribute('href', '/models/backends')
  await explanation.click()
  await outcomeSummary.focus()
  await outcomeSummary.press('Enter')
  await expect(plugins.getByText('Detector unavailable', { exact: true })).toBeVisible()
  await expect(plugins.getByText('Not applicable', { exact: true })).toBeVisible()
  await expect(plugins.getByText('Score unavailable', { exact: true })).toHaveCount(2)
  await outcomeSummary.press('Enter')
  await page.setViewportSize({ width: 390, height: 844 })
  for (const region of [usage, plugins]) {
    const box = (await region.boundingBox())!
    expect(box.x).toBeGreaterThanOrEqual(0)
    expect(box.x + box.width).toBeLessThanOrEqual(390)
  }
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1))
    .toBe(true)
  await usage.screenshot({ path: info.outputPath('usage-mobile.png') })
  expect(writes).toEqual([])
})
