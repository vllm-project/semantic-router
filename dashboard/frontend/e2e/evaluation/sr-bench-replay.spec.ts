import { expect, test, type Page, type Route } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'
const baseline = (id: string) => ({
  id,
  status: 'completed',
  created_at: '2026-09-18T00:00:00Z',
  updated_at: '2026-09-18T00:00:10Z',
  progress: { total: 2, completed: 2, failed: 0 },
  manifest: {
    name: id,
    mode: 'live',
    profile: 'quick',
    targets: [{ id: 'single', kind: 'single', model: 'model-a' }],
  },
})
const choice = (id: string) => ({ run_id: id, name: id, profile: 'quick', case_count: 2 })
const response = (
  id?: string,
  rows = id ? [choice('compatible')] : [choice('baseline-a'), choice('baseline-b')],
  next: string | null = null,
) => ({
  baseline: id ? choice(id) : null,
  baselines: id ? [] : rows,
  options: id ? rows : [],
  next_cursor: next,
  has_more: !!next,
  scanned_pairs: rows.length,
  scan_limited: false,
  unverified_pairs: 0,
  unverified_baselines: 0,
  model_requests: 0,
})
const estimate = {
  ...baseline('estimate'),
  manifest: { ...baseline('estimate').manifest, name: 'Offline estimate', mode: 'replay' },
}
async function choose(page: Page, label: string, value: string) {
  await page.getByRole('combobox', { name: label, exact: true }).click()
  await page.getByRole('listbox').locator(`[data-value="${value}"]`).click()
}
async function setup(page: Page, handler: (route: Route) => Promise<void>) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/catalog')) return route.fulfill({ json: { profiles: [], benchmarks: [] } })
    if (path.endsWith('/datasets')) return route.fulfill({ json: { datasets: [] } })
    if (path.endsWith('/targets')) return route.fulfill({ json: { targets: [] } })
    if (path.endsWith('/runs'))
      return route.fulfill({
        json: {
          runs: [baseline('baseline-a'), baseline('baseline-b'), baseline('unusable-baseline')],
        },
      })
    if (path.endsWith('/comparison-options'))
      return route.fulfill({ json: response(undefined, []) })
    if (path.endsWith('/replay-options') || path.endsWith('/replays')) return handler(route)
    if (path.endsWith('/report'))
      return route.fulfill({
        json: { summary: { targets: [] }, benchmarks: [], limitations: [], provenance: {} },
      })
    if (path.endsWith('/events')) return route.fulfill({ json: { events: [] } })
    if (path.endsWith('/calls')) return route.fulfill({ json: { calls: [], total: 0 } })
    if (path.endsWith('/results')) return route.fulfill({ json: { results: [], total: 0 } })
    return route.fulfill({ json: estimate })
  })
  await page.goto('/evaluation?view=compare')
  await page
    .locator('summary')
    .filter({ hasText: /^Estimate a routing change$/ })
    .click()
}
test('shows only eligible baselines and previews with explicit bounded page loading', async ({
  page,
}, testInfo) => {
  const posts: unknown[] = [],
    reads: string[] = []
  await setup(page, async (route) => {
    const url = new URL(route.request().url())
    if (url.pathname.endsWith('/replays')) {
      posts.push(route.request().postDataJSON())
      return route.fulfill({ json: estimate })
    }
    reads.push(url.pathname + url.search)
    expect(url.pathname).toBe('/api/sr-bench/v1/replay-options')
    expect(url.searchParams.get('limit')).toBe('10')
    return route.fulfill({
      json: !url.searchParams.has('baseline_run_id')
        ? response(
            undefined,
            url.searchParams.has('after') ? [choice('baseline-a')] : [],
            url.searchParams.has('after') ? null : 'baseline-page',
          )
        : response(
            'baseline-a',
            url.searchParams.has('after') ? [choice('compatible')] : [],
            url.searchParams.has('after') ? null : 'next-page',
          ),
    })
  })
  await expect(page.getByRole('button', { name: 'Load more saved baselines' })).toBeVisible()
  expect(reads.some((url) => url.includes('after='))).toBe(false)
  await page.getByRole('button', { name: 'Load more saved baselines' }).click()
  await page.getByRole('combobox', { name: 'Saved single-model baseline' }).click()
  await expect(page.getByRole('listbox').getByRole('option')).toHaveCount(1)
  await expect(page.getByRole('listbox').locator('[data-value="unusable-baseline"]')).toHaveCount(0)
  await page.getByRole('listbox').locator('[data-value="baseline-a"]').click()
  await expect(page.getByRole('button', { name: 'Load more previews' })).toBeVisible()
  expect(
    reads.filter((url) => url.includes('baseline_run_id=')).some((url) => url.includes('after=')),
  ).toBe(false)
  await page.getByRole('button', { name: 'Load more previews' }).click()
  await choose(page, 'Routing preview', 'compatible')
  const replay = page.getByRole('region', { name: 'Estimate a routing change', exact: true })
  await replay.scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('eligible-replay-desktop.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await replay.scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('eligible-replay-mobile.png') })
  await page.getByRole('button', { name: 'Create offline estimate' }).click()
  await expect(page.getByRole('heading', { name: 'Offline estimate', exact: true })).toBeVisible()
  expect(posts).toEqual([
    {
      baseline_run_id: 'baseline-a',
      preview_run_id: 'compatible',
      idempotency_key: expect.any(String),
    },
  ])
})
test('empty authoritative combinations have one clear state and no selection controls', async ({
  page,
}, testInfo) => {
  await setup(page, (route) => route.fulfill({ json: response(undefined, []) }))
  const replay = page.getByRole('region', { name: 'Estimate a routing change', exact: true })
  await expect(replay.getByText(/^No compatible saved pairs\./)).toBeVisible()
  await expect(replay.getByRole('combobox')).toHaveCount(0)
  await expect(replay.getByRole('button')).toHaveCount(0)
  await expect(page.getByText(/^No comparable results yet\./)).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('empty-combinations-desktop.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('empty-combinations-mobile.png') })
})
test('availability failures stay blocked and retry the same service scope without local fallback', async ({
  page,
}, testInfo) => {
  let failing = true
  await setup(page, async (route) => {
    expect(route.request().method()).toBe('GET')
    const id = new URL(route.request().url()).searchParams.get('baseline_run_id') ?? undefined
    if (failing)
      return route.fulfill({ status: 503, json: { error: 'Selection service unavailable' } })
    return route.fulfill({ json: response(id) })
  })
  await expect(page.getByRole('alert')).toContainText('Selection service unavailable')
  await expect(page.getByRole('combobox', { name: 'Saved single-model baseline' })).toHaveCount(0)
  const retry = page.getByRole('button', { name: 'Retry saved baselines' })
  await expect(retry.locator('svg')).toHaveCSS('width', '15px')
  expect((await retry.boundingBox())!.height).toBeLessThan(48)
  await page
    .getByRole('alert')
    .screenshot({ path: testInfo.outputPath('selection-error-desktop.png') })
  failing = false
  await page.getByRole('button', { name: 'Retry saved baselines' }).click()
  failing = true
  await choose(page, 'Saved single-model baseline', 'baseline-a')
  await expect(page.getByRole('alert')).toContainText('Selection service unavailable')
  await expect(page.getByRole('button', { name: 'Create offline estimate' })).toHaveCount(0)
  failing = false
  await page.getByRole('button', { name: 'Retry previews' }).click()
  await choose(page, 'Routing preview', 'compatible')
  await expect(page.getByRole('button', { name: 'Create offline estimate' })).toBeEnabled()
})
test('late previous-baseline options never replace the current eligible pair', async ({ page }) => {
  let release!: () => void, seen!: () => void
  const held = new Promise<void>((resolve) => {
    release = resolve
  })
  const started = new Promise<void>((resolve) => {
    seen = resolve
  })
  await setup(page, async (route) => {
    const id = new URL(route.request().url()).searchParams.get('baseline_run_id') ?? undefined
    if (id === 'baseline-a') {
      seen()
      await held
      return route.fulfill({ json: response(id, [choice('old-preview')]) }).catch(() => {})
    }
    return route.fulfill({ json: response(id, id ? [choice('current-preview')] : undefined) })
  })
  await choose(page, 'Saved single-model baseline', 'baseline-a')
  await started
  await choose(page, 'Saved single-model baseline', 'baseline-b')
  await choose(page, 'Routing preview', 'current-preview')
  release()
  await expect(page.getByRole('combobox', { name: 'Routing preview' })).toHaveText(
    'current-preview',
  )
  await page.getByRole('combobox', { name: 'Routing preview' }).click()
  await expect(page.getByRole('listbox').locator('[data-value="old-preview"]')).toHaveCount(0)
})
test('lost creation response survives reload and reconciles the exact saved identity', async ({
  page,
}) => {
  const posts: Array<Record<string, string>> = []
  await setup(page, async (route) => {
    const url = new URL(route.request().url())
    if (!url.pathname.endsWith('/replays'))
      return route.fulfill({ json: response(url.searchParams.get('baseline_run_id') ?? undefined) })
    posts.push(route.request().postDataJSON())
    return posts.length === 1 ? route.abort('failed') : route.fulfill({ json: estimate })
  })
  await choose(page, 'Saved single-model baseline', 'baseline-a')
  await choose(page, 'Routing preview', 'compatible')
  await page.getByRole('button', { name: 'Create offline estimate' }).click()
  await expect(page.getByRole('button', { name: 'Check or submit same estimate' })).toBeEnabled()
  await page.reload()
  await page
    .locator('summary')
    .filter({ hasText: /^Estimate a routing change$/ })
    .click()
  await expect(
    page.getByRole('heading', { name: 'Estimate submission needs reconciliation' }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Check or submit same estimate' }).click()
  await expect(page.getByRole('heading', { name: 'Offline estimate', exact: true })).toBeVisible()
  expect(posts).toHaveLength(2)
  expect(posts[1]).toEqual(posts[0])
})
test('submission revalidation clears only a proven uncreated estimate and refreshes availability', async ({
  page,
}) => {
  let posts = 0
  await setup(page, async (route) => {
    const url = new URL(route.request().url())
    if (route.request().method() === 'GET')
      return route.fulfill({
        json: posts
          ? response(undefined, [])
          : response(url.searchParams.get('baseline_run_id') ?? undefined),
      })
    posts++
    return route.fulfill({
      status: 400,
      json: {
        error: 'Saved routing state is no longer eligible.',
        code: 'replay_ineligible',
        dispatch_started: false,
        model_requests: 0,
      },
    })
  })
  await choose(page, 'Saved single-model baseline', 'baseline-a')
  await choose(page, 'Routing preview', 'compatible')
  await page.getByRole('button', { name: 'Create offline estimate' }).click()
  await expect(
    page.getByText('Saved routing state is no longer eligible.', { exact: true }),
  ).toBeVisible()
  await expect(page.getByText(/^No compatible saved pairs\./)).toBeVisible()
  expect(posts).toBe(1)
  expect(
    await page.evaluate(() => sessionStorage.getItem('sr-bench-replay:user-admin-1')),
  ).toBeNull()
})

test('scan limits are not misreported as proof that no compatible pairs exist', async ({
  page,
}) => {
  await setup(page, (route) =>
    route.fulfill({
      json: { ...response(undefined, []), scan_limited: true, unverified_pairs: 1 },
    }),
  )
  const replay = page.getByRole('region', { name: 'Estimate a routing change', exact: true })
  await expect(
    replay.getByText(
      'No verified replay pairs available. Some saved evidence exceeded the verification limit.',
    ),
  ).toBeVisible()
  await expect(replay.getByText(/^No compatible saved pairs\./)).toHaveCount(0)
  await expect(replay.getByRole('combobox')).toHaveCount(0)
})

test('an expired saved-run cursor requires a deliberate refresh from the first page', async ({
  page,
}) => {
  let refreshed = false
  const cursors: Array<string | null> = []
  await setup(page, (route) => {
    const url = new URL(route.request().url())
    cursors.push(url.searchParams.get('after'))
    if (url.searchParams.has('after')) {
      refreshed = true
      return route.fulfill({
        status: 400,
        json: { error: 'Saved runs changed; refresh options from the first page' },
      })
    }
    return route.fulfill({
      json: refreshed
        ? response(undefined, [])
        : response(undefined, [choice('baseline-a')], 'expired'),
    })
  })
  await page.getByRole('button', { name: 'Load more saved baselines' }).click()
  await expect(page.getByRole('alert')).toContainText('Saved runs changed')
  await page.getByRole('button', { name: 'Retry saved baselines' }).click()
  await expect(page.getByText(/^No compatible saved pairs\./)).toBeVisible()
  expect(cursors.slice(-2)).toEqual(['expired', null])
})
