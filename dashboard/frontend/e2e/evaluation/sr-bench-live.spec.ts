import { readFileSync } from 'node:fs'
import { expect, test, type Page, type TestInfo } from '@playwright/test'

interface LiveAcceptancePlan {
  base_url: string
  baseline_run_id?: string
  balance_run_ids?: [string, string, string]
  final_target_id?: string
  final_config_hash?: string
  dataset_id?: string
  failed_run_id?: string
}

const planPath = process.env.SR_BENCH_LIVE_PLAN
const plan: LiveAcceptancePlan | null = planPath
  ? (JSON.parse(readFileSync(planPath, 'utf8')) as LiveAcceptancePlan)
  : null
if (plan) {
  const origin = new URL(plan.base_url)
  if (
    origin.protocol !== 'http:' ||
    !['127.0.0.1', 'localhost', '[::1]'].includes(origin.hostname) ||
    origin.username ||
    origin.password ||
    origin.pathname !== '/' ||
    origin.search ||
    origin.hash
  )
    throw new Error(
      'Live sr-bench acceptance requires an explicit loopback Dashboard origin without credentials.',
    )
  plan.base_url = origin.origin
}

test.skip(!plan, 'Opt in with SR_BENCH_LIVE_PLAN; never run against a real deployment implicitly.')

async function openAcceptance(page: Page) {
  const blocked: string[] = []
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    // A comparison reads saved evidence. All generation, recovery, cancellation,
    // replay, regrade and export mutations remain blocked in this acceptance pass.
    if (
      !['GET', 'HEAD'].includes(request.method()) &&
      !(request.method() === 'POST' && path === '/api/sr-bench/v1/comparisons')
    ) {
      blocked.push(`${request.method()} ${path}`)
      await route.abort('blockedbyclient')
      return
    }
    await route.continue()
  })
  await page.goto(`${plan!.base_url}/__acceptance/login`)
  await expect(page.getByRole('heading', { name: 'sr-bench 1.0 acceptance' })).toBeVisible()
  // This form carries no credentials. The already-running isolated relay handles
  // authentication privately; tests never inspect cookies or operator secrets.
  await page.getByRole('button', { name: 'Start acceptance session', exact: true }).click()
  await page.waitForURL((url) => url.pathname !== '/__acceptance/login')
  await page.goto(`${plan!.base_url}/evaluation?view=runs`)
  await expect(page.getByRole('heading', { name: 'sr-bench 1.0' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Evaluation runs' })).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  return blocked
}

async function screenshot(page: Page, testInfo: TestInfo, name: string) {
  await page.screenshot({ path: testInfo.outputPath(name), fullPage: true })
  await testInfo.attach(name, { path: testInfo.outputPath(name), contentType: 'image/png' })
}

async function openRun(page: Page, id: string) {
  await page.goto(`${plan!.base_url}/evaluation?view=runs&run=${encodeURIComponent(id)}`)
  await expect(page.getByRole('heading', { name: 'Target comparison', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Refresh evidence', exact: true })).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
}

test('live inventory, frozen dataset and persisted CLI run are visible', async ({
  page,
}, testInfo) => {
  const blocked = await openAcceptance(page)
  const progressBars = page
    .getByRole('region', { name: 'Evaluation runs', exact: true })
    .getByRole('progressbar')
  expect(await progressBars.count()).toBeGreaterThan(0)
  expect(
    await progressBars.evaluateAll((elements) =>
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
  await screenshot(page, testInfo, 'live-run-management.png')
  await page.getByRole('button', { name: 'Datasets', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Prepared datasets', exact: true })).toBeVisible()
  if (plan!.dataset_id) {
    await page.getByLabel('Search datasets', { exact: true }).fill(plan!.dataset_id)
    await expect(
      page.getByRole('button', { name: 'Evaluate this dataset', exact: true }),
    ).toHaveCount(1)
    await screenshot(page, testInfo, 'live-frozen-dataset.png')
    await page.getByRole('button', { name: 'Evaluate this dataset', exact: true }).click()
    await expect(page.getByLabel('Prepared dataset', { exact: true })).toHaveValue(plan!.dataset_id)
    await expect(page.getByRole('button', { name: 'Start evaluation', exact: true })).toBeDisabled()
  }
  if (plan!.baseline_run_id) {
    await openRun(page, plan!.baseline_run_id)
    await screenshot(page, testInfo, 'live-single-model-baseline.png')
    await page.reload()
    await expect(
      page.getByRole('heading', { name: 'Target comparison', exact: true }),
    ).toBeVisible()
    await expect(page).toHaveURL(new RegExp(`run=${plan!.baseline_run_id}`))
  }
  expect(
    blocked,
    'Read-only UI acceptance must not attempt to launch or mutate evaluations',
  ).toEqual([])
})

test('live comparison retains current Balance and two optimization revisions', async ({
  page,
}, testInfo) => {
  test.skip(
    !plan?.baseline_run_id || plan.balance_run_ids?.length !== 3,
    'Requires one real single-model baseline and three completed Balance revisions.',
  )
  const blocked = await openAcceptance(page)
  await page.getByRole('button', { name: 'Compare iterations', exact: true }).click()
  await page.getByLabel('Baseline run', { exact: true }).selectOption(plan!.baseline_run_id!)
  for (const [index, label] of [
    'Current Balance run',
    'Optimization 1 run',
    'Optimization 2 run',
  ].entries())
    await page.getByLabel(label, { exact: true }).selectOption(plan!.balance_run_ids![index])
  await page.getByRole('button', { name: 'Compare runs', exact: true }).click()
  await expect(
    page.getByRole('heading', { name: 'Balance optimization trajectory', exact: true }),
  ).toBeVisible()
  await expect(page.getByText('Comparison withheld:', { exact: false })).toHaveCount(0)
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByRole('heading', { name: /^Optimization 2 ·/ })).toBeVisible()
  await screenshot(page, testInfo, 'live-balance-two-optimization-loops.png')
  const comparisonURL = page.url()
  await page.reload()
  await expect(page.getByRole('heading', { name: /^Optimization 2 ·/ })).toBeVisible()
  await expect(page.getByText('Comparison withheld:', { exact: false })).toHaveCount(0)
  await expect(page).toHaveURL(comparisonURL)
  await page.setViewportSize({ width: 390, height: 844 })
  await expect
    .poll(() =>
      page
        .getByRole('region', { name: 'sr-bench workspace' })
        .evaluate((element) => element.getBoundingClientRect().right <= window.innerWidth),
    )
    .toBe(true)
  await screenshot(page, testInfo, 'live-balance-comparison-mobile.png')
  await testInfo.attach('comparison-selection.json', {
    body: JSON.stringify(
      {
        baseline_run_id: plan!.baseline_run_id,
        balance_run_ids: plan!.balance_run_ids,
        comparison_url: comparisonURL,
        model_jobs_started: 0,
      },
      null,
      2,
    ),
    contentType: 'application/json',
  })
  expect(blocked).toEqual([])
})

test('live final recipe is verified and downloadable without launching a job', async ({
  page,
}, testInfo) => {
  test.skip(
    plan?.balance_run_ids?.length !== 3 || !plan.final_target_id || !plan.final_config_hash,
    'Requires the completed final Balance run and its independently recorded configuration hash.',
  )
  const blocked = await openAcceptance(page)
  await openRun(page, plan!.balance_run_ids![2])
  const recipes = page.locator('#run-recipe')
  await expect(
    recipes.getByRole('heading', {
      name: `${plan!.final_target_id} · Verified config snapshot`,
      exact: true,
    }),
  ).toBeVisible()
  await expect(recipes.getByText(plan!.final_config_hash!, { exact: true })).toBeVisible()
  const downloading = page.waitForEvent('download')
  await recipes
    .getByRole('button', { name: `Download ${plan!.final_target_id} recipe`, exact: true })
    .click()
  const download = await downloading
  await download.saveAs(testInfo.outputPath('final-recipe.json'))
  const recipe = JSON.parse(readFileSync(testInfo.outputPath('final-recipe.json'), 'utf8'))
  expect(recipe.config_hash).toBe(plan!.final_config_hash)
  expect(recipe.generated_runtime_hash).toBe(plan!.final_config_hash)
  expect(recipe.active_runtime_hash).toBe(plan!.final_config_hash)
  expect(recipe.redacted).toBe(true)
  await screenshot(page, testInfo, 'live-final-recipe-evidence.png')
  expect(blocked).toEqual([])
})
