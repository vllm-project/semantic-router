import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'
import type { DatasetPreparationJob } from '../../src/components/sr-bench/datasetPreparationApi'
import type { Dataset, Manifest, Target } from '../../src/components/sr-bench/types'

const benchmarks = ['mmlu-pro', 'simpleqa-verified']
const seed = 20260918
const dataset: Dataset = {
  id: 'd'.repeat(64),
  path: '/fixture/prepared/collection.jsonl',
  sha256: 'd'.repeat(64),
  profile: 'quick',
  seed,
  case_count: 600,
  benchmarks,
}
const target: Target = {
  id: 'fixture-model',
  kind: 'single',
  model: 'Fixture model',
  base_url: 'http://localhost:8000/v1',
  prices: { input_per_million: 1, output_per_million: 2 },
}

async function setup(page: Page, options: { prepared?: boolean; readonly?: boolean } = {}) {
  await mockAuthenticatedAppShell(
    page,
    options.readonly
      ? {
          user: {
            id: 'dataset-reader',
            email: 'reader@example.com',
            name: 'Dataset Reader',
            permissions: ['evaluation.read'],
          },
        }
      : {},
  )
  const state = {
    preparations: [] as DatasetPreparationJob[],
    writes: [] as Array<{ path: string; body: Record<string, unknown> }>,
    reads: [] as string[],
    plans: [] as Manifest[],
    created: 0,
    loseResponse: false,
    failNext: '',
    unexpected: [] as string[],
  }
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname.replace('/api/sr-bench/v1', '')
    const method = request.method()
    if (method === 'GET') state.reads.push(path)
    else state.writes.push({ path, body: request.postDataJSON() })
    let body: unknown
    if (path === '/catalog')
      body = {
        version: 'sr-bench-1.0',
        profiles: ['smoke', 'quick', 'standard'].map((id) => ({ id, purpose: id })),
        benchmarks: benchmarks.map((id) => ({ id, title: id, kind: 'capability' })),
      }
    else if (path === '/targets') body = { targets: [target] }
    else if (path === '/datasets') body = { datasets: options.prepared ? [dataset] : [] }
    else if (path === '/datasets/selection')
      body = {
        profile: url.searchParams.get('profile'),
        seed: options.prepared ? seed : null,
        split: 'dev',
        model_requests: 0,
        benchmarks: benchmarks.map((id) => ({
          id,
          title: id,
          eligible: !!options.prepared,
          case_count: options.prepared ? 300 : 0,
          source_ids: options.prepared ? [dataset.id] : [],
          reason_code: options.prepared ? null : 'not_prepared',
          reason: options.prepared ? null : 'Prepare this benchmark for the selected profile.',
        })),
      }
    else if (path === '/dataset-preparations' && method === 'GET')
      body = { preparations: state.preparations }
    else if (path === '/dataset-preparations' && method === 'POST') {
      const input = request.postDataJSON()
      const existing = state.preparations.find(
        (job) =>
          (job.status === 'queued' || job.status === 'running') &&
          job.profile === input.profile &&
          job.seed === input.seed &&
          JSON.stringify(job.benchmarks) === JSON.stringify(input.benchmarks),
      )
      let preparation = existing
      if (!preparation) {
        state.created += 1
        preparation = {
          ...input,
          id: `prep-${String(state.created).padStart(32, '0')}`,
          status: 'running',
          phase: 'checking_dependencies',
          created_at: '2026-09-21T00:00:00Z',
          updated_at: '2026-09-21T00:00:01Z',
          items: benchmarks.map((benchmark) => ({
            benchmark,
            status: 'queued',
            phase: 'queued',
            reused: false,
            source_ids: [],
          })),
        }
        state.preparations.unshift(preparation!)
      }
      if (state.failNext) {
        Object.assign(preparation!, { status: 'failed', phase: 'failed', error: state.failNext })
        state.failNext = ''
      }
      if (state.loseResponse) {
        state.loseResponse = false
        return route.abort('failed')
      }
      await route.fulfill({ status: 202, json: { preparation } })
      return
    } else if (path.startsWith('/dataset-preparations/prep-'))
      body = {
        preparation: state.preparations.find((job) => path.endsWith(job.id)),
      }
    else if (path === '/datasets/compose') body = { dataset }
    else if (path === '/plans') {
      const manifest = request.postDataJSON().manifest as Manifest
      state.plans.push(manifest)
      body = {
        status: 'validated',
        total: dataset.case_count,
        plan_sha256: 'e'.repeat(64),
        manifest,
      }
    } else if (path === '/runs' && method === 'GET') body = { runs: [] }
    else {
      state.unexpected.push(`${method} ${path}`)
      await route.fulfill({
        status: 404,
        json: { error: 'Unexpected auto-preparation fixture request' },
      })
      return
    }
    await route.fulfill({ json: body })
  })
  return state
}

type State = Awaited<ReturnType<typeof setup>>

async function selectScope(page: Page) {
  await page.goto('/evaluation?view=new')
  const choices = page.getByRole('group', { name: 'Included benchmarks' }).getByRole('checkbox')
  await expect(choices).toHaveCount(benchmarks.length)
  for (const choice of await choices.all()) {
    await expect(choice).toBeEnabled()
    await choice.check()
  }
  await page.getByRole('combobox', { name: 'Add configured target' }).click()
  await page.locator(`[role="option"][data-value="${target.id}"]`).click()
}

function phase(state: State, name: string) {
  Object.assign(state.preparations[0], { phase: name, updated_at: '2026-09-21T00:00:02Z' })
  const first = state.preparations[0].items?.[0]
  if (first) Object.assign(first, { status: 'running', phase: name })
}

function complete(state: State) {
  Object.assign(state.preparations[0], { status: 'completed', phase: 'completed', dataset })
}

const review = (page: Page) => page.getByRole('button', { name: 'Review plan', exact: true })
const start = (page: Page) => page.getByRole('button', { name: 'Start evaluation', exact: true })
const preparationWrites = (state: State) =>
  state.writes.filter((write) => write.path === '/dataset-preparations')
const noRuns = (state: State) =>
  expect(state.writes.filter((write) => write.path === '/runs')).toEqual([])

for (const width of [1280, 390]) {
  test(`prepares missing questions and dependencies before planning at ${width}px without starting a run`, async ({
    page,
  }, testInfo) => {
    const state = await setup(page)
    await page.setViewportSize({ width, height: 1000 })
    await selectScope(page)
    await expect(
      page.getByText(
        'Missing datasets and required dependencies are prepared automatically when you review.',
      ),
    ).toBeVisible()
    await expect(start(page)).toBeDisabled()
    await review(page).click()
    const progress = page.getByRole('region', { name: 'Preparing evaluation' })
    await expect(
      progress.getByText('Checking required dependencies', { exact: true }),
    ).toBeVisible()
    expect(preparationWrites(state)).toEqual([
      {
        path: '/dataset-preparations',
        body: { benchmarks, profile: 'quick', seed },
      },
    ])
    expect(state.plans).toEqual([])
    noRuns(state)
    phase(state, 'installing_dependencies')
    await expect(
      progress.getByText('Installing required dependencies', { exact: true }),
    ).toBeVisible()
    await expect(
      page.getByRole('group', { name: 'Included benchmarks' }).getByRole('checkbox').first(),
    ).toBeDisabled()
    phase(state, 'downloading')
    await expect(progress.getByText('Downloading source data', { exact: true })).toBeVisible()
    const icon = await progress.locator('svg').boundingBox()
    expect(icon!.width).toBeLessThanOrEqual(20)
    expect(icon!.height).toBeLessThanOrEqual(20)
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    await page.screenshot({
      path: testInfo.outputPath(`automatic-preparation-${width}.png`),
      fullPage: true,
    })
    phase(state, 'composing')
    await expect(
      progress.getByText('Finalizing the evaluation questions', { exact: true }),
    ).toBeVisible()
    expect(state.plans).toEqual([])
    complete(state)
    await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
    await expect(start(page)).toBeEnabled()
    expect(state.plans).toHaveLength(1)
    expect(state.plans[0].dataset).toEqual({ path: dataset.path, sha256: dataset.sha256 })
    expect(state.writes.map((write) => write.path)).toEqual(['/dataset-preparations', '/plans'])
    noRuns(state)
    expect(state.unexpected).toEqual([])
  })
}

test('composes already prepared questions without submitting a download', async ({ page }) => {
  const state = await setup(page, { prepared: true })
  await selectScope(page)
  await review(page).click()
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  expect(state.writes.map((write) => write.path)).toEqual(['/datasets/compose', '/plans'])
  expect(state.writes[0].body).toEqual({ dataset_ids: [dataset.id], benchmarks })
  expect(preparationWrites(state)).toEqual([])
  noRuns(state)
})

test('allows read-only users to select missing benchmarks without writes', async ({ page }) => {
  const state = await setup(page, { readonly: true })
  await selectScope(page)
  await expect(review(page)).toBeDisabled()
  await expect(start(page)).toBeDisabled()
  expect(state.writes).toEqual([])
})

test('returns to the same service-owned job after leaving the composer', async ({ page }) => {
  const state = await setup(page)
  await selectScope(page)
  await review(page).click()
  await expect(page.getByText('Checking required dependencies', { exact: true })).toBeVisible()
  const id = state.preparations[0].id
  await page.goto('/evaluation?view=datasets')
  await expect(page.getByRole('heading', { name: 'Dataset library', exact: true })).toBeVisible()
  phase(state, 'downloading')
  expect(state.preparations[0].status).toBe('running')
  expect(state.plans).toEqual([])
  await selectScope(page)
  await review(page).click()
  await expect(page.getByText('Downloading source data', { exact: true })).toBeVisible()
  expect(state.created).toBe(1)
  expect(state.preparations[0].id).toBe(id)
  expect(preparationWrites(state)).toHaveLength(2)
  expect(preparationWrites(state)[1]).toEqual(preparationWrites(state)[0])
  complete(state)
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  noRuns(state)
})

test('keeps failures before planning and retries only after explicit review', async ({ page }) => {
  const state = await setup(page)
  await selectScope(page)
  await review(page).click()
  await expect(page.getByText('Checking required dependencies', { exact: true })).toBeVisible()
  Object.assign(state.preparations[0], {
    status: 'failed',
    phase: 'failed',
    error: 'Dataset access was denied. Check source credentials.',
  })
  await expect(page.getByRole('alert')).toHaveText(
    'Dataset access was denied. Check source credentials.',
  )
  await expect(review(page)).toBeEnabled()
  await expect(start(page)).toBeDisabled()
  expect(preparationWrites(state)).toHaveLength(1)
  expect(state.plans).toEqual([])
  noRuns(state)
  await review(page).click()
  await expect(page.getByText('Checking required dependencies', { exact: true })).toBeVisible()
  expect(state.created).toBe(2)
  expect(preparationWrites(state)).toHaveLength(2)
  complete(state)
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  noRuns(state)
})

test('recovers a lost preparation response through history without repeating the POST', async ({
  page,
}) => {
  const state = await setup(page)
  state.loseResponse = true
  await selectScope(page)
  await review(page).click()
  await expect(page.getByText('Checking required dependencies', { exact: true })).toBeVisible()
  expect(state.reads).toContain('/dataset-preparations')
  expect(preparationWrites(state)).toHaveLength(1)
  phase(state, 'downloading')
  await expect(page.getByText('Downloading source data', { exact: true })).toBeVisible()
  expect(state.reads).toContain(`/dataset-preparations/${state.preparations[0].id}`)
  expect(preparationWrites(state)).toHaveLength(1)
  complete(state)
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toBeVisible()
  expect(preparationWrites(state)).toHaveLength(1)
  expect(state.plans).toHaveLength(1)
  noRuns(state)
})

test('surfaces the new failed job after a lost response instead of accepting an older completed collection', async ({
  page,
}) => {
  const state = await setup(page)
  const older: DatasetPreparationJob = {
    id: `prep-${'f'.repeat(32)}`,
    benchmarks,
    profile: 'quick',
    seed,
    status: 'completed',
    phase: 'completed',
    created_at: '2026-09-20T00:00:00Z',
    updated_at: '2026-09-20T00:00:01Z',
    dataset,
  }
  state.preparations = [older]
  state.loseResponse = true
  state.failNext = 'Selected benchmarks have conflicting frozen sources.'
  await selectScope(page)
  await review(page).click()
  await expect(page.getByRole('alert')).toHaveText(
    'Selected benchmarks have conflicting frozen sources.',
  )
  await expect(review(page)).toBeEnabled()
  await expect(start(page)).toBeDisabled()
  await expect(page.getByRole('heading', { name: 'Plan ready for review' })).toHaveCount(0)
  expect(state.preparations).toHaveLength(2)
  expect(state.preparations[0].status).toBe('failed')
  expect(state.preparations[0].id).not.toBe(older.id)
  expect(state.preparations[1]).toEqual(older)
  expect(
    state.reads.filter((path) => path === '/dataset-preparations').length,
  ).toBeGreaterThanOrEqual(2)
  expect(preparationWrites(state)).toHaveLength(1)
  expect(state.plans).toEqual([])
  noRuns(state)
  expect(state.unexpected).toEqual([])
})
