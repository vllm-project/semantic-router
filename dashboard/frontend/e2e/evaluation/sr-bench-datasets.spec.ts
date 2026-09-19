import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'

const id = 'a'.repeat(64)
const dataset = {
  id,
  name: 'gpqa-diamond+mmlu-pro/quick',
  profile: 'quick',
  benchmarks: ['gpqa-diamond', 'mmlu-pro'],
  case_count: 28,
  path: '/private/store/questions.jsonl',
  sha256: 'b'.repeat(64),
}
const detail = {
  ...dataset,
  categories: [
    { benchmark: 'mmlu-pro', name: 'Physics', count: 26 },
    { benchmark: 'gpqa-diamond', name: 'Chemistry', count: 2 },
  ],
  benchmarks: [
    { id: 'mmlu-pro', title: 'MMLU-Pro', count: 26, categories: [{ name: 'Physics', count: 26 }] },
    {
      id: 'gpqa-diamond',
      title: 'GPQA Diamond',
      count: 2,
      categories: [{ name: 'Chemistry', count: 2 }],
    },
  ],
  provenance: {
    sha256: dataset.sha256,
    seed: 42,
    sources: [{ benchmark: 'mmlu-pro', url: 'https://example.com/source' }],
  },
}

async function setup(page: Page, count = 1) {
  await mockAuthenticatedAppShell(page)
  const reads: URL[] = []
  const writes: string[] = []
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const url = new URL(route.request().url())
    const path = url.pathname.replace('/api/sr-bench/v1', '')
    if (route.request().method() !== 'GET') writes.push(path)
    let body: unknown
    if (path === '/catalog')
      body = {
        version: 'sr-bench-1.0',
        profiles: [{ id: 'quick', purpose: 'Quick comparison' }],
        benchmarks: [],
      }
    else if (path === '/datasets')
      body = {
        datasets: Array.from({ length: count }, (_, index) => ({
          ...dataset,
          id: index ? index.toString(16).padStart(64, '0') : id,
          name: index ? `Study ${index + 1}` : dataset.name,
        })),
      }
    else if (path === '/targets') body = { targets: [] }
    else if (path === '/runs') body = { runs: [] }
    else if (path === `/datasets/${id}`) body = detail
    else if (path === `/datasets/${id}/cases`) {
      reads.push(url)
      const cursor = Number(url.searchParams.get('cursor') || 0)
      const filtered =
        url.searchParams.has('benchmark') ||
        url.searchParams.has('category') ||
        url.searchParams.has('q')
      const total = filtered ? 1 : 28
      body = {
        dataset_id: id,
        total,
        dataset_total: 28,
        limit: 25,
        next_cursor: !filtered && cursor === 0 ? '25' : null,
        cases: Array.from({ length: filtered ? 1 : Math.min(25, total - cursor) }, (_, index) => ({
          id: `case-${cursor + index + 1}`,
          benchmark: 'mmlu-pro',
          category: 'Physics',
          question: `Question ${cursor + index + 1}: Which quantity is conserved in this isolated system?`,
          messages: [],
          choices: ['Momentum', 'Temperature', 'Pressure'],
          input_status: 'available',
        })),
      }
    } else {
      await route.fulfill({ status: 404, json: { error: 'Missing fixture' } })
      return
    }
    await route.fulfill({ json: body })
  })
  return { reads, writes }
}

test('dataset library groups by mode, pages cards and hides storage identities', async ({
  page,
}) => {
  const { writes } = await setup(page, 14)
  await page.goto('/evaluation?view=datasets')
  const library = page.getByRole('region', { name: 'Dataset library' })
  await expect(
    library.getByRole('heading', { name: 'Capability suite', exact: true }),
  ).toBeVisible()
  await expect(library.getByRole('article')).toHaveCount(12)
  await expect(library.getByText('1–12 of 14 datasets')).toBeVisible()
  await expect(library.getByRole('button', { name: /^All datasets/ })).toContainText(
    '392 total questions',
  )
  await expect(library.getByText(dataset.name, { exact: true })).toHaveCount(0)
  await expect(library.getByText(dataset.sha256)).toHaveCount(0)
  await expect(library.getByText(dataset.path)).toHaveCount(0)
  await library.getByRole('button', { name: 'Next dataset page' }).click()
  await expect(library.getByRole('article')).toHaveCount(2)
  await expect(library.getByText('13–14 of 14 datasets')).toBeVisible()
  await library.getByLabel('Search datasets').fill('Study 2')
  await expect(library.getByRole('article')).toHaveCount(1)
  expect(writes).toEqual([])
})

test('library mode cards summarize question counts across all prepared sets independently of filters', async ({
  page,
}) => {
  const { writes } = await setup(page)
  await page.route('**/api/sr-bench/v1/datasets', (route) =>
    route.fulfill({
      json: {
        datasets: [
          { ...dataset, id: 'smoke', profile: 'smoke', case_count: 12 },
          { ...dataset, id: 'quick', profile: 'quick', case_count: 1000 },
          { ...dataset, id: 'standard', profile: 'standard', case_count: 288 },
        ],
      },
    }),
  )
  await page.goto('/evaluation?view=datasets')
  const modes = page.getByRole('group', { name: 'Filter by evaluation mode' })
  await expect(modes.getByRole('button', { name: /^All datasets/ })).toContainText(
    '1,300 total questions',
  )
  await expect(modes.getByRole('button', { name: /^Smoke/ })).toContainText('12 total questions')
  await expect(modes.getByRole('button', { name: /^Quick/ })).toContainText('1,000 total questions')
  await expect(modes.getByRole('button', { name: /^Standard/ })).toContainText(
    '288 total questions',
  )
  await modes.getByRole('button', { name: /^Quick/ }).click()
  await page.getByLabel('Search datasets').fill('not a dataset')
  await expect(page.getByText('No matching datasets')).toBeVisible()
  await expect(modes.getByRole('button', { name: /^All datasets/ })).toContainText(
    '1,300 total questions',
  )
  await expect(
    page.getByText('Question totals are across prepared sets. Sets may overlap.'),
  ).toBeVisible()
  expect(writes).toEqual([])
})

test('dataset details show real questions, options and server pagination; URL survives reload', async ({
  page,
}) => {
  const { reads, writes } = await setup(page)
  await page.goto('/evaluation?view=datasets')
  await page.getByRole('button', { name: 'Explore Capability suite, 28 questions' }).click()
  await expect(page).toHaveURL(new RegExp(`dataset=${id}`))
  await expect(page.getByRole('button', { name: 'Open question 1', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Next questions', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Open question 26', exact: true })).toBeVisible()
  expect(reads.at(-1)?.searchParams.get('cursor')).toBe('25')
  await page.getByRole('button', { name: 'Open question 26', exact: true }).click()
  await expect(page.getByText('Momentum', { exact: true })).toBeVisible()
  await expect(page.getByText('Temperature', { exact: true })).toBeVisible()
  await expect(page.getByText('case-26', { exact: true })).not.toBeVisible()
  await page.getByRole('button', { name: 'Back to questions', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Open question 26', exact: true })).toBeVisible()
  await page.reload()
  await expect(page.getByRole('heading', { name: 'Capability suite', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Open question 1', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Back to datasets', exact: true }).click()
  await expect(page.getByRole('region', { name: 'Dataset library' })).toBeVisible()
  expect(writes).toEqual([])
})

test('benchmark and subject filters request server strata; coverage shows aggregation', async ({
  page,
}) => {
  const { reads } = await setup(page)
  await page.goto(`/evaluation?view=datasets&dataset=${id}`)
  const questions = page.getByRole('region', { name: 'Dataset questions' })
  await questions.getByRole('combobox', { name: 'Benchmark', exact: true }).click()
  await page.getByRole('option', { name: 'MMLU-Pro (26)', exact: true }).click()
  await questions.getByRole('combobox', { name: 'Subject group', exact: true }).click()
  await page.getByRole('option', { name: 'Physics', exact: true }).click()
  await expect(
    questions.getByRole('button', { name: 'Open question 1', exact: true }),
  ).toBeVisible()
  expect(reads.at(-1)?.searchParams.get('category')).toBe('Physics')
  await questions
    .getByRole('searchbox', { name: 'Search questions', exact: true })
    .fill('conserved')
  await questions.getByRole('button', { name: 'Submit question search', exact: true }).click()
  await expect(questions.getByText('1 question matching “conserved”')).toBeVisible()
  expect(reads.at(-1)?.searchParams.get('q')).toBe('conserved')
  await page.getByRole('button', { name: 'Coverage', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Benchmark coverage' })).toBeVisible()
  await page.getByRole('button', { name: 'MMLU-Pro 26', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'MMLU-Pro · Subject groups' })).toBeVisible()
  await expect(page.getByText('Physics', { exact: true })).toBeVisible()
  await expect(page.getByText('Chemistry', { exact: true })).toHaveCount(0)
})

test('dataset reading failures offer explicit reload without dispatching and narrow layout stays usable', async ({
  page,
}, testInfo) => {
  const { writes } = await setup(page)
  let fail = true
  await page.route(`**/api/sr-bench/v1/datasets/${id}`, async (route) => {
    if (fail)
      await route.fulfill({
        status: 503,
        json: { error: 'Dataset service temporarily unavailable' },
      })
    else await route.fallback()
  })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto(`/evaluation?view=datasets&dataset=${id}`)
  await expect(page.getByRole('alert')).toContainText('Dataset service temporarily unavailable')
  fail = false
  await page.getByRole('button', { name: 'Retry loading dataset' }).click()
  await expect(page.getByRole('button', { name: 'Open question 1', exact: true })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
  await page.screenshot({
    path: testInfo.outputPath('dataset-mobile.png'),
    fullPage: true,
  })
  expect(writes).toEqual([])
})
