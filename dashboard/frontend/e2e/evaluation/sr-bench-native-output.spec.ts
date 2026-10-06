import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'
import { DEFAULT_LIMITS, makeManifest } from '../../src/components/sr-bench/model'
import type { Manifest, Target } from '../../src/components/sr-bench/types'

const single: Target = {
  id: 'single',
  kind: 'single',
  model: 'provider/physical-model',
  base_url: 'http://localhost:8000/v1',
  request_params: { temperature: 1, top_p: 0.95 },
  native_limits: {
    'provider/physical-model': { context_window: 131072, max_output_tokens: 65536 },
  },
}
const mom: Target = {
  ...single,
  id: 'mom',
  kind: 'mom',
  model: 'recipe',
  capture_recipe: true,
  max_inference_calls: 1,
}
const unsupported: Target = {
  id: 'unsupported',
  kind: 'single',
  model: 'unconfigured-model',
  base_url: single.base_url,
}
const dataset = {
  id: 'quick',
  path: '/prepared/native.jsonl',
  sha256: 'a'.repeat(64),
  profile: 'quick',
  case_count: 1,
  seed: 42,
  benchmarks: ['mmlu-pro'],
}
const baselineManifest = makeManifest('Native reference', 'live', 'quick', dataset, [single], {
  ...DEFAULT_LIMITS,
  max_output_tokens: 65536,
})
baselineManifest.output_policy = 'native'
delete baselineManifest.sampling.max_tokens
const baseline = {
  id: 'native-baseline',
  status: 'completed',
  manifest: { ...baselineManifest, cases: [{ id: 'a', benchmark: 'mmlu-pro' }] },
  progress: { total: 1, completed: 1, failed: 0 },
}

async function choose(page: Page, label: string, value: string) {
  await page.getByRole('combobox', { name: label, exact: true }).click()
  await page.locator(`[role="option"][data-value="${value}"]`).click()
}

async function setup(page: Page, loseFirstStart = false) {
  await mockAuthenticatedAppShell(page)
  const plans: Manifest[] = []
  const starts: Array<{ manifest: Manifest; idempotency_key: string }> = []
  const candidates: unknown[] = []
  let saved: typeof baseline | null = null
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname.replace('/api/sr-bench/v1', '')
    let body: unknown
    if (path === '/catalog')
      body = {
        version: 'sr-bench-1.0',
        profiles: ['smoke', 'quick', 'standard'].map((id) => ({ id, purpose: id })),
        benchmarks: [{ id: 'mmlu-pro', title: 'MMLU-Pro', kind: 'capability' }],
      }
    else if (path === '/datasets') body = { datasets: [dataset] }
    else if (path === '/datasets/selection')
      body = {
        profile: url.searchParams.get('profile'),
        seed: 42,
        split: 'dev',
        model_requests: 0,
        benchmarks: [
          {
            id: 'mmlu-pro',
            title: 'MMLU-Pro',
            eligible: true,
            case_count: 1,
            source_ids: [dataset.id],
            reason: null,
          },
        ],
      }
    else if (path === '/datasets/compose') body = { dataset }
    else if (path === '/targets') body = { targets: [single, mom, unsupported] }
    else if (path === '/plans') {
      const manifest = request.postDataJSON().manifest as Manifest
      plans.push(manifest)
      const frozen = {
        ...manifest,
        limits: {
          ...manifest.limits,
          max_output_tokens:
            manifest.output_policy === 'native' ? 65536 : manifest.limits.max_output_tokens,
        },
      }
      body = { status: 'validated', total: 1, plan_sha256: 'b'.repeat(64), manifest: frozen }
    } else if (path === `/runs/${baseline.id}/candidate-plan`) {
      const input = request.postDataJSON()
      candidates.push(input)
      body = {
        status: 'validated',
        total: 1,
        plan_sha256: 'b'.repeat(64),
        manifest: { ...baselineManifest, name: input.name, targets: [mom] },
      }
    } else if (path === '/runs' && request.method() === 'POST') {
      starts.push(request.postDataJSON())
      saved = {
        ...baseline,
        id: 'native-started',
        manifest: { ...request.postDataJSON().manifest, cases: baseline.manifest.cases },
      }
      if (loseFirstStart && starts.length === 1) return route.abort('failed')
      body = saved
    } else if (path === '/runs') body = { runs: [baseline, ...(saved ? [saved] : [])] }
    else if (path === `/runs/${baseline.id}`) body = baseline
    else if (path === '/runs/native-started') body = saved
    else if (path.endsWith('/report'))
      body = { summary: { targets: [] }, benchmarks: [], provenance: {}, limitations: [] }
    else if (path.endsWith('/results')) body = { results: [], total: 0, next_cursor: null }
    else if (path.endsWith('/calls')) body = { calls: [], total: 0, next_cursor: null }
    else if (path.endsWith('/events')) body = { events: [], next_cursor: null }
    else return route.fulfill({ status: 404, json: { error: 'Unexpected fixture request' } })
    await route.fulfill({ json: body })
  })
  return { plans, starts, candidates }
}

test('native single-model plans omit artificial caps and reconcile a lost start with the same frozen identity', async ({
  page,
}, testInfo) => {
  const state = await setup(page, true)
  await page.goto('/evaluation?view=new')
  await choose(page, 'Add configured target', single.id)
  await expect(
    page.getByRole('spinbutton', { name: 'Max output tokens', exact: true }),
  ).toHaveValue('4096')
  await choose(page, 'Output policy', 'native')
  await expect(
    page.getByRole('spinbutton', { name: 'Max output tokens', exact: true }),
  ).toHaveCount(0)
  await expect(page.getByText(/No fixed shared token cap is sent/)).toBeVisible()
  await page.getByText('Registered model capacities', { exact: true }).click()
  await expect(
    page.getByText('131,072 context tokens · 65,536 maximum output tokens', { exact: true }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(
    page.getByRole('heading', { name: 'Plan ready for review', exact: true }),
  ).toBeVisible()
  expect(state.plans).toHaveLength(1)
  expect(state.plans[0].output_policy).toBe('native')
  expect(state.plans[0].sampling).not.toHaveProperty('max_tokens')
  expect(state.plans[0].limits).not.toHaveProperty('max_output_tokens')
  await page.screenshot({
    path: testInfo.outputPath('native-output-plan-desktop.png'),
    fullPage: true,
  })
  await page.getByRole('button', { name: 'Start evaluation', exact: true }).click()
  await expect(page.getByRole('alert')).toBeVisible()
  await page.reload()
  await expect(
    page.getByRole('heading', { name: 'Evaluation submission needs reconciliation', exact: true }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Check or submit same evaluation', exact: true }).click()
  await expect.poll(() => state.starts.length).toBe(2)
  expect(state.starts[1]).toEqual(state.starts[0])
  expect(state.starts[1].manifest.limits.max_output_tokens).toBe(65536)
  expect(state.starts[1].manifest.sampling).not.toHaveProperty('max_tokens')
  await expect(
    page.getByRole('heading', { name: 'Evaluation submission needs reconciliation', exact: true }),
  ).toHaveCount(0)
  expect(
    await page.evaluate(() => sessionStorage.getItem('sr-bench-submission:user-admin-1')),
  ).toBeNull()
})

test('native selection blocks unsupported existing targets and filters new target choices', async ({
  page,
}) => {
  const state = await setup(page)
  await page.goto('/evaluation?view=new')
  await choose(page, 'Add configured target', unsupported.id)
  await choose(page, 'Output policy', 'native')
  await expect(
    page.getByText(/Native capacity is unavailable for the selected targets/),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Review plan', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: 'Remove target 1', exact: true }).click()
  await page.getByRole('combobox', { name: 'Add configured target', exact: true }).click()
  await expect(page.locator('[role="option"][data-value="unsupported"]')).toHaveCount(0)
  await page.locator('[role="option"][data-value="mom"]').click()
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect.poll(() => state.plans.length).toBe(1)
  expect(state.plans[0].targets[0].kind).toBe('mom')
  expect(state.plans[0].sampling).not.toHaveProperty('max_tokens')
  await choose(page, 'Output policy', 'bounded')
  await expect(
    page.getByRole('spinbutton', { name: 'Max output tokens', exact: true }),
  ).toHaveValue('4096')
  await expect(page.getByRole('button', { name: 'Start evaluation', exact: true })).toBeDisabled()
  expect(state.starts).toHaveLength(0)
})

test('candidate plans inherit native policy and preview uses the same cap-free request defaults', async ({
  page,
}) => {
  const state = await setup(page)
  await page.goto(`/evaluation?view=new&baseline=${baseline.id}`)
  await expect(
    page.getByRole('region', { name: 'Frozen baseline protocol', exact: true }),
  ).toContainText('Native capacity · no shared token cap')
  await expect(page.getByRole('combobox', { name: 'Output policy', exact: true })).toHaveCount(0)
  await choose(page, 'Add configured target', mom.id)
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect.poll(() => state.candidates.length).toBe(1)
  expect(state.candidates[0]).toMatchObject({ target_ids: ['mom'], mode: 'live' })
  await page.goto('/evaluation?view=preview')
  await choose(page, 'Output policy', 'native')
  await choose(page, 'Add configured target', mom.id)
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect.poll(() => state.plans.length).toBe(1)
  expect(state.plans[0]).toMatchObject({ mode: 'preview', output_policy: 'native' })
  expect(state.plans[0].sampling).not.toHaveProperty('max_tokens')
  expect(state.starts).toHaveLength(0)
})
