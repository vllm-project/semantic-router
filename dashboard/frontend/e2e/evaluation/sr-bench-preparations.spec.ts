import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'
import type { DatasetPreparationJob } from '../../src/components/sr-bench/datasetPreparationApi'

const dataset = {
  id: 'd'.repeat(64),
  name: 'simpleqa-verified/smoke',
  path: '/fixture/store/questions.jsonl',
  sha256: 'd'.repeat(64),
  profile: 'smoke',
  benchmarks: ['simpleqa-verified'],
  case_count: 5,
  seed: 42,
}

const job = (patch: Partial<DatasetPreparationJob> = {}): DatasetPreparationJob => ({
  id: `prep-${'a'.repeat(32)}`,
  benchmark: 'simpleqa-verified',
  profile: 'smoke',
  seed: 42,
  status: 'running',
  phase: 'downloading',
  created_at: '2026-09-21T00:00:00Z',
  updated_at: '2026-09-21T00:00:02Z',
  ...patch,
})

async function setup(page: Page, permission: 'write' | 'read' | 'readonly' = 'write') {
  await mockAuthenticatedAppShell(page, {
    user: {
      id: 'dataset-user',
      email: 'dataset@example.com',
      name: 'Dataset User',
      permissions:
        permission === 'read' ? ['evaluation.read'] : ['evaluation.read', 'evaluation.write'],
    },
    settings: { serverReadonly: permission === 'readonly' },
  })
  const state = {
    preparations: [] as DatasetPreparationJob[],
    writes: [] as Array<{ path: string; body: Record<string, unknown> }>,
    reads: [] as string[],
    failNext: false,
    loseResponse: false,
  }
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const url = new URL(route.request().url())
    const path = url.pathname.replace('/api/sr-bench/v1', '')
    const method = route.request().method()
    if (method === 'GET') state.reads.push(path)
    else state.writes.push({ path, body: route.request().postDataJSON() })
    let body: unknown
    if (path === '/catalog')
      body = {
        version: 'sr-bench-1.0',
        profiles: [
          { id: 'smoke', purpose: 'Smoke' },
          { id: 'quick', purpose: 'Quick' },
        ],
        benchmarks: [],
      }
    else if (path === '/datasets')
      body = {
        datasets: state.preparations.flatMap((item) =>
          item.status === 'completed' && item.dataset ? [item.dataset] : [],
        ),
      }
    else if (path === '/targets') body = { targets: [] }
    else if (path === '/runs') body = { runs: [] }
    else if (path === '/datasets/selection')
      body = {
        profile: url.searchParams.get('profile'),
        seed: null,
        split: 'dev',
        benchmarks: [],
        model_requests: 0,
      }
    else if (path === '/dataset-preparations/options')
      body = {
        benchmarks: [
          {
            id: 'simpleqa-verified',
            name: 'SimpleQA Verified',
            profiles: { smoke: 5, quick: 100, standard: 500 },
            source_url: 'https://example.com/simpleqa',
            access_note: 'Public fixture source.',
            dependencies: [],
          },
          {
            id: 'mmlu-pro',
            name: 'MMLU-Pro',
            profiles: { smoke: 14, quick: 500, standard: 2000 },
            source_url: 'https://example.com/mmlu',
            access_note: 'Public fixture source.',
            dependencies: ['pyarrow'],
          },
        ],
      }
    else if (path === '/dataset-preparations' && method === 'GET')
      body = { preparations: state.preparations }
    else if (path === '/dataset-preparations' && method === 'POST') {
      if (state.failNext) {
        state.failNext = false
        await route.fulfill({ status: 503, json: { error: 'Download service unavailable' } })
        return
      }
      const request = route.request().postDataJSON()
      const preparation = job({
        ...request,
        id: `prep-${String(state.writes.length).padStart(32, '0')}`,
        phase: 'checking_dependencies',
      })
      state.preparations.unshift(preparation)
      if (state.loseResponse) {
        state.loseResponse = false
        await route.abort('failed')
        return
      }
      await route.fulfill({ status: 202, json: { preparation } })
      return
    } else {
      await route.fulfill({ status: 404, json: { error: 'Missing preparation fixture' } })
      return
    }
    await route.fulfill({ json: body })
  })
  return state
}

test('downloads from the dataset library and refreshes completed datasets without model calls', async ({
  page,
}, testInfo) => {
  const state = await setup(page)
  await page.goto('/evaluation?view=datasets')
  await expect(page.getByRole('heading', { name: 'Dataset library', exact: true })).toBeVisible()
  expect(state.reads).not.toContain('/dataset-preparations/options')
  await page.getByRole('button', { name: 'Prepare dataset', exact: true }).click()
  await expect(page).toHaveURL(/prepare=1/)
  const panel = page.getByRole('region', { name: 'Prepare datasets' })
  await panel.getByRole('combobox', { name: 'Benchmark to download' }).click()
  await panel.getByRole('option', { name: 'MMLU-Pro', exact: true }).click()
  await expect(panel.getByText('14 questions', { exact: true })).toBeVisible()
  await expect(panel.getByText('Dependencies: pyarrow · installed if needed')).toBeVisible()
  await expect(panel.getByRole('link', { name: 'View source dataset' })).toHaveAttribute(
    'href',
    'https://example.com/mmlu',
  )
  await panel.getByRole('combobox', { name: 'Dataset size' }).click()
  await panel.getByRole('option', { name: 'Quick 500 questions' }).click()
  await panel.getByRole('button', { name: 'Download and prepare' }).click()
  await expect(
    panel.getByRole('list', { name: 'Dataset preparations' }).getByRole('status'),
  ).toHaveText('Checking required dependencies')
  expect(state.writes).toEqual([
    { path: '/dataset-preparations', body: { benchmark: 'mmlu-pro', profile: 'quick' } },
  ])
  state.preparations[0] = { ...state.preparations[0], phase: 'installing_dependencies' }
  await panel.getByRole('button', { name: 'Refresh status' }).click()
  await expect(
    panel.getByRole('list', { name: 'Dataset preparations' }).getByRole('status'),
  ).toHaveText('Installing required dependencies')
  await page.screenshot({ path: testInfo.outputPath('dataset-preparation.png'), fullPage: true })
  state.preparations[0] = {
    ...state.preparations[0],
    status: 'completed',
    phase: 'completed',
    dataset: {
      ...dataset,
      name: 'mmlu-pro/quick',
      profile: 'quick',
      benchmarks: ['mmlu-pro'],
      case_count: 500,
    },
  }
  await panel.getByRole('button', { name: 'Refresh status' }).click()
  await expect(
    panel.getByRole('list', { name: 'Dataset preparations' }).getByRole('status'),
  ).toHaveText('Ready to evaluate')
  await expect(panel.getByText('500 questions ready')).toBeVisible()
  await expect(
    page.getByRole('region', { name: 'Dataset library', exact: true }).getByRole('article'),
  ).toHaveCount(1)
  expect(state.writes).toHaveLength(1)
})

test('restores a running service preparation after refresh and does not resubmit it', async ({
  page,
}) => {
  const state = await setup(page)
  state.preparations = [job()]
  await page.goto('/evaluation?view=datasets&prepare=1')
  await expect(page.getByText('Downloading source data', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Preparation in progress' })).toBeDisabled()
  await page.reload()
  await expect(page.getByText('Downloading source data', { exact: true })).toBeVisible()
  await page.getByRole('combobox', { name: 'Benchmark to download' }).click()
  await page.getByRole('option', { name: 'MMLU-Pro', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Another preparation is running' })).toBeDisabled()
  expect(state.writes).toEqual([])
  state.preparations[0] = job({ phase: 'freezing' })
  await expect(page.getByText('Freezing the selected questions')).toBeVisible()
})

test('shows a failed preparation and retries only after the user acts', async ({ page }) => {
  const state = await setup(page)
  state.preparations = [
    job({
      status: 'failed',
      phase: 'failed',
      error: 'Source download failed. Check network access.',
      limit: 3,
      seed: 123,
    }),
  ]
  await page.goto('/evaluation?view=datasets&prepare=1')
  await expect(page.getByText('Source download failed. Check network access.')).toBeVisible()
  expect(state.writes).toEqual([])
  await page.getByRole('button', { name: 'Retry preparation' }).click()
  await expect(page.getByText('Checking required dependencies', { exact: true })).toBeVisible()
  expect(state.writes[0].body).toEqual({
    benchmark: 'simpleqa-verified',
    profile: 'smoke',
    seed: 123,
    limit: 3,
  })
  await expect(page.getByRole('button', { name: 'Retry preparation' })).toBeDisabled()
})

test('recovers a lost submission response by reading persisted jobs without an automatic retry', async ({
  page,
}) => {
  const state = await setup(page)
  state.loseResponse = true
  await page.goto('/evaluation?view=datasets&prepare=1')
  await page.getByRole('button', { name: 'Download and prepare' }).click()
  await expect(page.getByRole('alert')).toContainText(
    'Check the preparation history before trying again.',
  )
  await expect(page.getByText('Checking required dependencies', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Preparation in progress' })).toBeDisabled()
  expect(state.writes).toHaveLength(1)
})

test('shows a submission error without automatically starting another download', async ({
  page,
}) => {
  const state = await setup(page)
  state.failNext = true
  await page.goto('/evaluation?view=datasets&prepare=1')
  await page.getByRole('button', { name: 'Download and prepare' }).click()
  await expect(page.getByRole('alert')).toContainText('Download service unavailable')
  await expect(page.getByRole('button', { name: 'Download and prepare' })).toBeEnabled()
  expect(state.writes).toHaveLength(1)
  await page.getByRole('button', { name: 'Download and prepare' }).click()
  await expect(page.getByText('Checking required dependencies', { exact: true })).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(state.writes).toHaveLength(2)
})

test('keeps preparation controls and saved failures usable on a narrow screen', async ({
  page,
}) => {
  const state = await setup(page)
  state.preparations = [
    job({
      status: 'failed',
      phase: 'failed',
      error: 'Source download failed. Check access to the source dataset and try again.',
    }),
  ]
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/evaluation?view=datasets&prepare=1')
  const panel = page.getByRole('region', { name: 'Prepare datasets' })
  await expect(panel.getByRole('button', { name: 'Download and prepare' })).toBeVisible()
  await expect(panel.getByRole('button', { name: 'Retry preparation' })).toBeVisible()
  const horizontalOverflow = await panel.evaluate(
    (element) => element.scrollWidth > element.clientWidth,
  )
  expect(horizontalOverflow).toBe(false)
})

for (const permission of ['read', 'readonly'] as const) {
  test(`keeps preparation and retry disabled in a ${permission} session`, async ({ page }) => {
    const state = await setup(page, permission)
    state.preparations = [job({ status: 'failed', phase: 'failed', error: 'Download failed' })]
    await page.goto('/evaluation?view=datasets&prepare=1')
    await expect(page.getByRole('button', { name: 'Download and prepare' })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Retry preparation' })).toBeDisabled()
    await expect(
      page.getByText(
        'Preparing datasets requires evaluation write permission and a writable Dashboard session.',
      ),
    ).toBeVisible()
    expect(state.writes).toEqual([])
  })
}

test('offers preparation directly from the empty evaluation composer', async ({ page }) => {
  const state = await setup(page)
  await page.goto('/evaluation?view=new')
  await page.getByRole('button', { name: 'Prepare dataset', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Prepare dataset', exact: true })).toBeVisible()
  await expect(page.getByText('5 questions', { exact: true })).toBeVisible()
  expect(state.writes).toEqual([])
})
