import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'
import type { Manifest, Target } from '../../src/components/sr-bench/types'

const target: Target = {
  id: 'unpriced-model',
  kind: 'single',
  model: 'Qwen/Qwen3.8-27B',
  base_url: 'http://localhost:8000/v1',
}
const freePrice = { input: 0, cached_input: 0, cache_write: 0, output: 0 }
const dataset = {
  id: 'd'.repeat(64),
  path: '/fixture/prepared/questions.jsonl',
  sha256: 'd'.repeat(64),
  profile: 'quick',
  seed: 42,
  case_count: 1,
  benchmarks: ['simpleqa-verified'],
}

async function setup(page: Page, registered = target) {
  await mockAuthenticatedAppShell(page)
  const state = { plans: [] as Manifest[], runs: 0, unexpected: [] as string[] }
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname.replace('/api/sr-bench/v1', '')
    let body: unknown
    if (path === '/catalog')
      body = {
        version: 'sr-bench-1.0',
        profiles: ['smoke', 'quick', 'standard'].map((id) => ({ id, purpose: id })),
        benchmarks: [{ id: 'simpleqa-verified', title: 'SimpleQA Verified', kind: 'capability' }],
      }
    else if (path === '/targets') body = { targets: [registered] }
    else if (path === '/datasets') body = { datasets: [dataset] }
    else if (path === '/datasets/selection')
      body = {
        profile: url.searchParams.get('profile'),
        seed: 42,
        split: 'dev',
        model_requests: 0,
        benchmarks: [
          {
            id: 'simpleqa-verified',
            eligible: true,
            case_count: 1,
            source_ids: [dataset.id],
            reason: null,
          },
        ],
      }
    else if (path === '/datasets/compose') body = { dataset }
    else if (path === '/plans') {
      const manifest = request.postDataJSON().manifest as Manifest
      state.plans.push(manifest)
      if (manifest.cost_policy === 'require_priced' && !registered.prices?.[registered.model]) {
        await route.fulfill({
          status: 400,
          json: { error: 'Known prices are required for this target.' },
        })
        return
      }
      body = { status: 'validated', total: 1, plan_sha256: 'e'.repeat(64), manifest }
    } else if (path === '/runs' && request.method() === 'GET') body = { runs: [] }
    else {
      if (path === '/runs' && request.method() === 'POST') state.runs += 1
      state.unexpected.push(`${request.method()} ${path}`)
      await route.fulfill({
        status: 404,
        json: { error: 'Unexpected missing-price fixture request' },
      })
      return
    }
    await route.fulfill({ json: body })
  })
  return state
}

async function chooseTarget(page: Page, view = 'new') {
  await page.goto(`/evaluation?view=${view}`)
  await page.getByRole('combobox', { name: 'Add configured target' }).click()
  await page.locator(`[role="option"][data-value="${target.id}"]`).click()
}

for (const width of [1280, 390]) {
  test(`changes unpriced evaluation to Quality only only after an explicit click at ${width}px`, async ({
    page,
  }, testInfo) => {
    const state = await setup(page)
    await page.setViewportSize({ width, height: 1000 })
    await chooseTarget(page)
    const notice = page.getByRole('note', { name: 'Missing target prices' })
    await expect(notice).toContainText(
      'Quality only removes the USD budget; time, token and call limits remain.',
    )
    const budget = page.getByRole('spinbutton', { name: 'Budget (USD)', exact: true })
    await expect(budget).toBeEnabled()
    await budget.fill('12')
    await page.getByLabel('Run deadline (seconds)', { exact: true }).fill('900')
    await page.getByLabel('Max output tokens', { exact: true }).fill('2048')
    await page.getByLabel('Concurrency', { exact: true }).fill('2')
    const advanced = page.getByText('Sampling and advanced limits', { exact: true })
    await advanced.click()
    await expect(
      page.getByRole('combobox', { name: 'Cost accounting', exact: true }),
    ).toContainText('Quality and cost')
    for (const [label, value] of [
      ['Request deadline (seconds)', '120'],
      ['Idle timeout (seconds)', '15'],
      ['Case deadline (seconds)', '300'],
      ['Max calls per case', '6'],
    ])
      await page.getByLabel(label, { exact: true }).fill(value)
    await advanced.click()
    const action = notice.getByRole('button', { name: 'Use Quality only', exact: true })
    expect((await action.boundingBox())!.height).toBeLessThanOrEqual(40)
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    await notice.screenshot({ path: testInfo.outputPath(`missing-prices-${width}.png`) })
    expect(state.plans).toEqual([])
    await page.getByRole('button', { name: 'Review plan', exact: true }).click()
    await expect(page.getByRole('alert')).toHaveText('Known prices are required for this target.')
    expect(state.plans).toHaveLength(1)
    expect(state.plans[0].cost_policy).toBe('require_priced')
    expect(state.plans[0].limits.max_cost_usd).toBe(12)
    await expect(budget).toBeEnabled()
    await expect(budget).toHaveValue('12')
    await action.click()
    await expect(notice).toHaveCount(0)
    await expect(budget).toBeDisabled()
    await expect(budget).toHaveValue('')
    await expect(budget).toHaveAttribute('placeholder', 'Not applied')
    await advanced.click()
    await expect(
      page.getByRole('combobox', { name: 'Cost accounting', exact: true }),
    ).toContainText('Quality only')
    await expect(page.getByText('No USD budget is applied.', { exact: false })).toBeVisible()
    await page.getByRole('button', { name: 'Review plan', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
    expect(state.plans).toHaveLength(2)
    expect(state.plans[1].cost_policy).toBe('capability_only')
    // The manifest schema retains the saved USD value; this explicit policy disables its enforcement.
    expect(state.plans[1].limits).toEqual(state.plans[0].limits)
    expect(state.plans[1].limits).toMatchObject({
      max_run_seconds: 900,
      total_timeout_s: 120,
      idle_timeout_s: 15,
      case_timeout_s: 300,
      max_output_tokens: 2048,
      max_calls_per_case: 6,
      concurrency: 2,
    })
    expect(state.plans[1].sampling.max_tokens).toBe(2048)
    expect(state.runs).toBe(0)
    expect(state.unexpected).toEqual([])
  })
}

for (const [name, prices] of [
  ['empty price map', {}],
  ['prices only for another model', { 'another-model': freePrice }],
] as const) {
  test(`offers an explicit policy choice for ${name}`, async ({ page }) => {
    const state = await setup(page, { ...target, prices })
    await chooseTarget(page)
    await expect(page.getByRole('button', { name: 'Use Quality only', exact: true })).toBeVisible()
    await expect(page.getByLabel('Budget (USD)', { exact: true })).toBeEnabled()
    expect(state.plans).toEqual([])
    expect(state.runs).toBe(0)
  })
}

test('keeps explicitly zero-priced targets in the priced workflow', async ({ page }) => {
  const state = await setup(page, { ...target, prices: { [target.model]: freePrice } })
  await chooseTarget(page)
  await expect(page.getByRole('button', { name: 'Use Quality only', exact: true })).toHaveCount(0)
  await expect(page.getByLabel('Budget (USD)', { exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  expect(state.plans[0].cost_policy).toBe('require_priced')
  expect(state.runs).toBe(0)
})

test('does not offer the live cost-policy switch in routing preview', async ({ page }) => {
  const state = await setup(page, { ...target, kind: 'mom' })
  await chooseTarget(page, 'preview')
  await expect(page.getByRole('button', { name: 'Use Quality only', exact: true })).toHaveCount(0)
  expect(state.plans).toEqual([])
  expect(state.runs).toBe(0)
})
