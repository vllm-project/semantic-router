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
    targets: [{ id: 'single', kind: 'single', model: 'a' }],
  },
})
const option = (id: string, eligible = true, message = '') => ({
  preview_run_id: id,
  name: id,
  profile: 'quick',
  case_count: 2,
  eligible,
  reasons: message ? [{ code: 'ineligible', message }] : [],
})
const response = (id: string, options = [option('compatible')], next: string | null = null) => ({
  baseline: {
    run_id: id,
    name: id,
    mode: 'live',
    status: 'completed',
    profile: 'quick',
    case_count: 2,
  },
  options,
  next_cursor: next,
  has_more: !!next,
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
      return route.fulfill({ json: { runs: [baseline('baseline-a'), baseline('baseline-b')] } })
    if (path.includes('replay-options') || path.endsWith('/replays')) return handler(route)
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
  await page.getByText('Estimate a routing change', { exact: true }).first().click()
}
test('only server-qualified previews can be selected; unavailable reasons and pagination stay explicit', async ({
  page,
}) => {
  const requests: unknown[] = []
  await setup(page, async (route) => {
    const url = new URL(route.request().url())
    if (url.pathname.endsWith('/replays')) {
      requests.push(route.request().postDataJSON())
      return route.fulfill({ json: estimate })
    }
    expect(url.searchParams.get('limit')).toBe('10')
    return route.fulfill({
      json: url.searchParams.has('after')
        ? response('baseline-a')
        : response(
            'baseline-a',
            [
              option('Learning preview', false, 'Learning depends on mutable routing state.'),
              option('Other questions', false, 'Frozen cases do not match.'),
            ],
            'next',
          ),
    })
  })
  await expect(page.getByRole('button', { name: 'Create offline estimate' })).toBeDisabled()
  await choose(page, 'Saved single-model baseline', 'baseline-a')
  await expect(page.getByRole('combobox', { name: 'Routing preview', exact: true })).toBeDisabled()
  await page.getByText('Unavailable previews (2 loaded)', { exact: true }).click()
  await expect(page.getByText('Learning depends on mutable routing state.')).toBeVisible()
  await expect(page.getByText('Frozen cases do not match.')).toBeVisible()
  await page.getByRole('button', { name: 'Load more previews' }).click()
  await choose(page, 'Routing preview', 'compatible')
  await page.getByRole('button', { name: 'Create offline estimate' }).click()
  await expect(page.getByRole('heading', { name: 'Offline estimate', exact: true })).toBeVisible()
  expect(requests).toEqual([
    {
      baseline_run_id: 'baseline-a',
      preview_run_id: 'compatible',
      idempotency_key: expect.any(String),
    },
  ])
})
test('failed compatibility reads block submission and retry the same bounded page', async ({
  page,
}) => {
  let reads = 0
  await setup(page, async (route) => {
    expect(route.request().method()).toBe('GET')
    reads += 1
    return reads === 1
      ? route.fulfill({ status: 503, json: { error: 'Compatibility service unavailable' } })
      : route.fulfill({ json: response('baseline-a', []) })
  })
  await choose(page, 'Saved single-model baseline', 'baseline-a')
  await expect(page.getByRole('alert')).toContainText('Compatibility service unavailable')
  await expect(page.getByRole('button', { name: 'Create offline estimate' })).toBeDisabled()
  await page.getByRole('button', { name: 'Retry compatibility check' }).click()
  await expect(page.getByText('No compatible previews.', { exact: false })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Create offline estimate' })).toBeDisabled()
  expect(reads).toBe(2)
})
test('a late compatibility response cannot enable a pair from the previous baseline', async ({
  page,
}) => {
  let release!: () => void
  const wait = new Promise<void>((resolve) => {
    release = resolve
  })
  let firstSeen!: () => void
  const seen = new Promise<void>((resolve) => {
    firstSeen = resolve
  })
  await setup(page, async (route) => {
    if (route.request().url().includes('baseline-a/')) {
      firstSeen()
      await wait
      await route.fulfill({ json: response('baseline-a') }).catch(() => {})
    } else await route.fulfill({ json: response('baseline-b', []) })
  })
  await choose(page, 'Saved single-model baseline', 'baseline-a')
  await seen
  await choose(page, 'Saved single-model baseline', 'baseline-b')
  await expect(page.getByText('No compatible previews.', { exact: false })).toBeVisible()
  release()
  await expect(page.getByRole('combobox', { name: 'Routing preview', exact: true })).toBeDisabled()
  await expect(page.getByRole('button', { name: 'Create offline estimate' })).toBeDisabled()
})
test('a lost creation response survives reload and reconciles the same submission identity', async ({
  page,
}) => {
  const requests: Array<Record<string, string>> = []
  await setup(page, async (route) => {
    if (!route.request().url().endsWith('/replays'))
      return route.fulfill({ json: response('baseline-a') })
    requests.push(route.request().postDataJSON())
    if (requests.length === 1) return route.abort('failed')
    return route.fulfill({ json: estimate })
  })
  await choose(page, 'Saved single-model baseline', 'baseline-a')
  await choose(page, 'Routing preview', 'compatible')
  await page.getByRole('button', { name: 'Create offline estimate' }).click()
  await expect(page.getByRole('button', { name: 'Check or submit same estimate' })).toBeEnabled()
  await page.reload()
  await page.getByText('Estimate a routing change', { exact: true }).first().click()
  await expect(
    page.getByRole('heading', { name: 'Estimate submission needs reconciliation' }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Check or submit same estimate' }).click()
  await expect(page.getByRole('heading', { name: 'Offline estimate', exact: true })).toBeVisible()
  expect(requests).toHaveLength(2)
  expect(requests[1]).toEqual(requests[0])
})

test('submission revalidation rejects a changed pair and requires a fresh compatibility check', async ({
  page,
}) => {
  let reads = 0
  let posts = 0
  await setup(page, async (route) => {
    if (route.request().method() === 'GET') {
      reads += 1
      return route.fulfill({
        json: response('baseline-a', reads === 1 ? [option('compatible')] : []),
      })
    }
    posts += 1
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
  await expect(page.getByRole('button', { name: 'Create offline estimate' })).toBeDisabled()
  await expect(
    page.getByText('Saved routing state is no longer eligible.', { exact: true }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Retry compatibility check' }).click()
  await expect(page.getByText('No compatible previews.', { exact: false })).toBeVisible()
  expect(posts).toBe(1)
  expect(
    await page.evaluate(() => sessionStorage.getItem('sr-bench-replay:user-admin-1')),
  ).toBeNull()
})
