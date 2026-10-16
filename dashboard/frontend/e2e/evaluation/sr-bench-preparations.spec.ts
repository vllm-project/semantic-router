import { expect, test, type Page } from '@playwright/test'
import { dashboardSettingsResponse, mockAuthenticatedAppShell } from '../support/auth'
import type {
  DatasetPreparationJob,
  PreparationBenchmark,
} from '../../src/components/sr-bench/datasetPreparationApi'

const benchmarks: PreparationBenchmark[] = [
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
  ...[
    ['gpqa-diamond', 'GPQA Diamond'],
    ['hle', "Humanity's Last Exam (text)"],
    ['livecodebench', 'LiveCodeBench'],
    ['scicode', 'SciCode'],
    ['terminal-bench-2.1', 'Terminal-Bench 2.1'],
    ['arc-agi-2', 'ARC-AGI-2'],
    ['tau3', 'τ³'],
  ].map(([id, name]) => ({
    id,
    name,
    profiles: { smoke: 1, quick: 3, standard: 10 },
    source_url: `https://example.com/${id}`,
    access_note: 'Public fixture source.',
    dependencies: [],
  })),
]

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
          { id: 'standard', purpose: 'Standard' },
        ],
        benchmarks: benchmarks.map((benchmark) => ({
          id: benchmark.id,
          title: benchmark.name,
          kind: 'fixture',
        })),
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
        benchmarks: benchmarks.map((benchmark) => ({
          id: benchmark.id,
          title: benchmark.name,
          eligible: false,
          case_count: 0,
          source_ids: [],
          reason_code: 'not_prepared',
          reason: 'Prepare this benchmark for the selected profile.',
        })),
        model_requests: 0,
      }
    else if (path === '/dataset-preparations/options') body = { benchmarks }
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
  await expect(panel.getByText('Dependencies: pyarrow · installed automatically')).toBeHidden()
  await panel.getByText('Source and requirements', { exact: true }).click()
  await expect(panel.getByText('Dependencies: pyarrow · installed automatically')).toBeVisible()
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
  await panel.getByText('Completed (1)', { exact: true }).click()
  await expect(
    panel.getByRole('list', { name: 'Completed preparations' }).getByRole('status'),
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

test('shows combined preparation history and retries the same benchmark collection', async ({
  page,
}) => {
  const state = await setup(page)
  state.preparations = [
    job({
      benchmark: undefined,
      benchmarks: ['simpleqa-verified', 'mmlu-pro'],
      status: 'failed',
      phase: 'failed',
      error: 'MMLU-Pro source was unavailable.',
      seed: 123,
    }),
  ]
  await page.goto('/evaluation?view=datasets&prepare=1')
  const panel = page.getByRole('region', { name: 'Prepare datasets' })
  await expect(panel.getByText('SimpleQA Verified, MMLU-Pro', { exact: true })).toBeVisible()
  await expect(panel.getByText('MMLU-Pro source was unavailable.')).toBeVisible()
  expect(state.writes).toEqual([])
  await panel.getByRole('button', { name: 'Retry preparation' }).click()
  await expect(panel.getByRole('button', { name: 'Preparation in progress' })).toBeDisabled()
  expect(state.writes).toEqual([
    {
      path: '/dataset-preparations',
      body: { benchmarks: ['simpleqa-verified', 'mmlu-pro'], profile: 'smoke', seed: 123 },
    },
  ])
  await panel.getByRole('combobox', { name: 'Benchmark to download' }).click()
  await panel.getByRole('option', { name: 'MMLU-Pro', exact: true }).click()
  await expect(panel.getByRole('button', { name: 'Preparation in progress' })).toBeDisabled()
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
  test(`allows browsing all benchmarks and sizes without writes in a ${permission} session`, async ({
    page,
  }) => {
    const state = await setup(page, permission)
    state.preparations = [job({ status: 'failed', phase: 'failed', error: 'Download failed' })]
    await page.goto('/evaluation?view=datasets&prepare=1')
    const panel = page.getByRole('region', { name: 'Prepare datasets' })
    await expect(panel.getByRole('heading', { name: 'Manage datasets' })).toBeVisible()
    const benchmark = panel.getByRole('combobox', { name: 'Benchmark to download' })
    const size = panel.getByRole('combobox', { name: 'Dataset size' })
    await expect(benchmark).toBeEnabled()
    await expect(size).toBeEnabled()
    await benchmark.click()
    await expect(panel.getByRole('option')).toHaveCount(9)
    for (const item of benchmarks)
      await expect(panel.getByRole('option', { name: item.name, exact: true })).toBeVisible()
    await panel.getByRole('option', { name: 'MMLU-Pro', exact: true }).click()
    for (const [profile, count] of [
      ['Quick', '500'],
      ['Standard', '2,000'],
      ['Smoke', '14'],
    ]) {
      await size.click()
      await expect(panel.getByRole('option')).toHaveCount(3)
      await panel
        .getByRole('option', { name: `${profile} ${count} questions`, exact: true })
        .click()
      await expect(size).toContainText(profile)
      await expect(panel.getByText(`${count} questions`, { exact: true })).toBeVisible()
    }
    await panel.getByText('Source and requirements', { exact: true }).click()
    await expect(panel.getByRole('link', { name: 'View source dataset' })).toHaveAttribute(
      'href',
      'https://example.com/mmlu',
    )
    await expect(page.getByRole('button', { name: 'Download and prepare' })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Retry preparation' })).toBeDisabled()
    await expect(
      page.getByText(
        permission === 'readonly'
          ? 'Preparation is disabled in read-only mode.'
          : 'View only. Ask an administrator for dataset preparation access.',
      ),
    ).toBeVisible()
    expect(state.writes).toEqual([])
  })
}

test('waits for verified settings before opening the interactive preparation catalog', async ({
  page,
}) => {
  const state = await setup(page)
  let releaseSettings!: () => void
  const settingsReady = new Promise<void>((resolve) => {
    releaseSettings = resolve
  })
  await page.route('**/api/settings', async (route) => {
    await settingsReady
    await route.fulfill({ json: dashboardSettingsResponse() })
  })
  try {
    await page.goto('/evaluation?view=datasets&prepare=1', { waitUntil: 'domcontentloaded' })
    const panel = page.getByRole('region', { name: 'Prepare datasets' })
    await expect(
      page.getByRole('status').getByRole('heading', { name: 'Checking Evaluation' }),
    ).toBeVisible()
    await expect(panel).toHaveCount(0)
    expect(state.writes).toEqual([])
    releaseSettings()
    await expect(panel.getByRole('button', { name: 'Download and prepare' })).toBeEnabled()
    await expect(panel.getByRole('combobox', { name: 'Benchmark to download' })).toBeEnabled()
    await expect(panel.getByRole('combobox', { name: 'Dataset size' })).toBeEnabled()
    await expect(page.getByRole('heading', { name: 'Checking Evaluation' })).toHaveCount(0)
    expect(state.writes).toEqual([])
  } finally {
    releaseSettings()
  }
})

for (const status of [403, 503]) {
  test(`recovers a ${status} settings response through Refresh access without submitting a download`, async ({
    page,
  }) => {
    const state = await setup(page)
    let settingsReads = 0
    let sessionReads = 0
    await page.route('**/api/auth/me', async (route) => {
      sessionReads += 1
      await route.fulfill({
        json: {
          user: {
            id: 'dataset-user',
            email: 'dataset@example.com',
            name: 'Dataset User',
            permissions: ['evaluation.read', 'evaluation.write'],
          },
        },
      })
    })
    await page.route('**/api/settings', async (route) => {
      settingsReads += 1
      await route.fulfill(
        settingsReads === 1
          ? { status, json: { error: 'Settings unavailable' } }
          : { json: dashboardSettingsResponse() },
      )
    })
    await page.goto('/evaluation?view=datasets&prepare=1')
    const panel = page.getByRole('region', { name: 'Prepare datasets' })
    await expect(
      page.getByRole('heading', { name: 'Unable to check Evaluation access' }),
    ).toBeVisible()
    await expect(
      page.getByText(
        status === 403
          ? 'Access to Dashboard settings was denied. Refresh access or contact an administrator.'
          : 'Dashboard access settings are unavailable. Refresh access to retry.',
      ),
    ).toBeVisible()
    await expect(panel).toHaveCount(0)
    expect(state.writes).toEqual([])
    const previousSessionReads = sessionReads
    await page.getByRole('button', { name: 'Refresh access' }).click()
    await expect(panel.getByRole('button', { name: 'Download and prepare' })).toBeEnabled()
    await expect(panel.getByRole('combobox', { name: 'Benchmark to download' })).toBeEnabled()
    await expect(panel.getByRole('combobox', { name: 'Dataset size' })).toBeEnabled()
    await expect(panel.getByRole('button', { name: 'Refresh access' })).toHaveCount(0)
    expect(settingsReads).toBeGreaterThanOrEqual(2)
    expect(sessionReads).toBeGreaterThan(previousSessionReads)
    expect(state.writes).toEqual([])
  })
}

test('Refresh access reloads current account permissions and settings', async ({ page }) => {
  const state = await setup(page, 'read')
  let granted = false
  let settingsReads = 0
  await page.route('**/api/auth/me', async (route) => {
    await route.fulfill({
      json: {
        user: {
          id: granted ? 'dataset-writer' : 'dataset-reader',
          email: 'dataset@example.com',
          name: 'Dataset User',
          permissions: granted ? ['evaluation.read', 'evaluation.write'] : ['evaluation.read'],
        },
      },
    })
  })
  await page.route('**/api/settings', async (route) => {
    settingsReads += 1
    await route.fulfill({ json: dashboardSettingsResponse() })
  })
  await page.goto('/evaluation?view=datasets&prepare=1')
  const panel = page.getByRole('region', { name: 'Prepare datasets' })
  await expect(
    panel.getByText('View only. Ask an administrator for dataset preparation access.', {
      exact: true,
    }),
  ).toBeVisible()
  await expect(panel.getByRole('button', { name: 'Download and prepare' })).toBeDisabled()
  granted = true
  await panel.getByRole('button', { name: 'Refresh access' }).click()
  await expect(panel.getByRole('button', { name: 'Download and prepare' })).toBeEnabled()
  await expect(panel.getByRole('button', { name: 'Refresh access' })).toHaveCount(0)
  expect(settingsReads).toBeGreaterThanOrEqual(2)
  expect(state.writes).toEqual([])
})

test('allows missing benchmarks in the evaluation composer without a separate download step', async ({
  page,
}) => {
  const state = await setup(page)
  await page.goto('/evaluation?view=new')
  await expect(
    page.getByText(
      'Missing datasets and required dependencies are prepared automatically when you review.',
    ),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Prepare dataset', exact: true })).toHaveCount(0)
  const choices = page.getByRole('group', { name: 'Included benchmarks' }).getByRole('checkbox')
  await expect(choices).toHaveCount(9)
  for (const choice of await choices.all()) await expect(choice).toBeEnabled()
  await choices.first().check()
  await expect(choices.first()).toBeChecked()
  expect(state.writes).toEqual([])
})

for (const viewport of [
  { width: 1280, height: 1000 },
  { width: 390, height: 844 },
]) {
  test(`keeps dataset management compact at ${viewport.width}px with visible failures and folded history`, async ({
    page,
  }, testInfo) => {
    const state = await setup(page, 'read')
    state.preparations = [
      job(),
      job({
        id: `prep-${'b'.repeat(32)}`,
        benchmark: 'mmlu-pro',
        status: 'failed',
        phase: 'failed',
        error: 'Download failed. Check source access and retry.',
      }),
      ...['c', 'd'].map((id) =>
        job({
          id: `prep-${id.repeat(32)}`,
          status: 'completed',
          phase: 'completed',
          dataset: { ...dataset, id: id.repeat(64), sha256: id.repeat(64) },
        }),
      ),
    ]
    await page.setViewportSize(viewport)
    await page.goto('/evaluation?view=datasets&prepare=1')
    const panel = page.getByRole('region', { name: 'Prepare datasets' })
    await expect(panel.getByRole('heading', { name: 'Manage datasets' })).toBeVisible()
    await expect(
      panel.getByText('Create evaluation prepares datasets automatically. Manage downloads here.'),
    ).toBeVisible()
    await expect(panel.getByRole('combobox', { name: 'Benchmark to download' })).toBeEnabled()
    await expect(panel.getByRole('combobox', { name: 'Dataset size' })).toBeEnabled()
    await expect(panel.getByText('Downloading source data', { exact: true })).toBeVisible()
    await expect(panel.getByText('Download failed. Check source access and retry.')).toBeVisible()
    await expect(panel.getByRole('button', { name: 'Retry preparation' })).toBeDisabled()
    await expect(panel.getByRole('button', { name: 'Preparation in progress' })).toBeDisabled()
    await expect(panel.getByRole('button', { name: 'View prepared dataset' })).toHaveCount(0)
    const refresh = panel.getByRole('button', { name: 'Refresh access' })
    await expect(refresh).toBeEnabled()
    const iconSize = await refresh.locator('svg').boundingBox()
    expect(iconSize).not.toBeNull()
    expect(iconSize!.width).toBeGreaterThan(0)
    expect(iconSize!.width).toBeLessThanOrEqual(20)
    expect(iconSize!.height).toBeLessThanOrEqual(20)
    const buttonSize = await refresh.boundingBox()
    expect(buttonSize!.height).toBeLessThanOrEqual(40)
    expect(await panel.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true)
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    await panel.screenshot({
      path: testInfo.outputPath(`dataset-management-${viewport.width}.png`),
    })
    await panel.getByText('Completed (2)', { exact: true }).click()
    await expect(panel.getByRole('list', { name: 'Completed preparations' })).toBeVisible()
    await expect(panel.getByRole('button', { name: 'View prepared dataset' })).toHaveCount(2)
    expect(state.writes).toEqual([])
  })
}
