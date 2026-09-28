import { expect, test, type Page, type Route } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'

const start = '2026-01-01T00:00:00Z'
const now = '2026-01-01T00:10:00Z'
const target = {
  id: 'single',
  kind: 'single',
  model: 'provider/model',
  base_url: 'http://localhost:8000/v1',
}

async function fixture(page: Page) {
  await mockAuthenticatedAppShell(page)
  await page.clock.install({ time: new Date(now) })
  await page.clock.pauseAt(new Date(now))
  const runs = ['first', 'second'].map((id) => ({
    id,
    status: 'running',
    created_at: start,
    updated_at: start,
    progress: { total: 2, completed: 0, failed: 0, running: 1 },
    manifest: {
      version: 'sr-bench-1.0',
      name: `${id} long response`,
      profile: 'quick',
      mode: 'live',
      seed: 42,
      targets: [target],
      sampling: { temperature: 1, max_tokens: 100 },
      limits: {},
      cases: [{ id: `${id}-question-0` }, { id: `${id}-question-1` }],
    },
  }))
  const state = {
    runs,
    activityReads: { first: 0, second: 0 },
    reportReads: 0,
    receivedBytes: 1024,
    phase: 'streaming',
    role: 'subject',
    callModel: '',
    lastActivityAt: now as string | null,
    writes: [] as string[],
    failActivity: false,
    missingActivity: false,
    paginate: false,
    badCursor: false,
    blockFirst: null as null | ((route: Route) => Promise<void>),
  }
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const request = route.request(),
      url = new URL(request.url())
    const endpoint = url.pathname.replace('/api/sr-bench/v1', '')
    if (request.method() !== 'GET') state.writes.push(endpoint)
    if (endpoint === '/catalog')
      return route.fulfill({ json: { version: 'sr-bench-1.0', profiles: [], benchmarks: [] } })
    if (endpoint === '/targets') return route.fulfill({ json: { targets: [target] } })
    if (endpoint === '/datasets') return route.fulfill({ json: { datasets: [] } })
    if (endpoint === '/runs') return route.fulfill({ json: { runs } })
    const run = runs.find(
      (item) => endpoint === `/runs/${item.id}` || endpoint.startsWith(`/runs/${item.id}/`),
    )
    if (!run) return route.fulfill({ status: 404, json: { error: 'Unexpected fixture read' } })
    if (endpoint === `/runs/${run.id}`) return route.fulfill({ json: run })
    if (endpoint.endsWith('/report')) {
      state.reportReads++
      return route.fulfill({
        json: {
          version: 'sr-bench-1.0',
          run_id: run.id,
          summary: { targets: [], wall_time_s: 60, total_spend_usd: null },
          benchmarks: [],
          limitations: [],
          provenance: {},
        },
      })
    }
    if (endpoint.endsWith('/events')) return route.fulfill({ json: { events: [] } })
    if (endpoint.endsWith('/results'))
      return route.fulfill({ json: { total: 0, next_cursor: null, limit: 100, results: [] } })
    if (endpoint.endsWith('/calls')) {
      if (url.searchParams.get('active') !== 'true')
        return route.fulfill({ json: { total: 0, next_cursor: null, limit: 100, calls: [] } })
      const id = run.id as 'first' | 'second'
      state.activityReads[id]++
      if (id === 'first' && state.blockFirst) return state.blockFirst(route)
      if (state.failActivity)
        return route.fulfill({ status: 503, json: { error: 'Activity read unavailable' } })
      const after = Number(url.searchParams.get('after'))
      return route.fulfill({
        json: {
          total: state.paginate ? 7 : 1,
          limit: 100,
          next_cursor: state.badCursor ? after : state.paginate && after === 0 ? 1 : null,
          calls: Array.from({ length: state.paginate ? (after === 0 ? 3 : 4) : 1 }, (_, index) => ({
            id: `${id}-call-${after}-${index}`,
            case_id: `${id}-question-${after}`,
            target_id: 'single',
            role: state.role,
            status: 'sent',
            model: state.callModel || `${target.model}/${id}`,
            started_at: start,
            ...(state.missingActivity
              ? {}
              : {
                  activity: {
                    phase: state.phase,
                    phase_started_at: start,
                    last_activity_at: state.lastActivityAt,
                    received_bytes: state.receivedBytes,
                    updated_at: now,
                  },
                }),
          })),
        },
      })
    }
    return route.fulfill({ status: 404, json: { error: 'Unexpected fixture read' } })
  })
  await page.goto('/evaluation?view=runs&run=first')
  await expect(
    page.getByRole('heading', { name: 'first long response', exact: true }),
  ).toBeVisible()
  return state
}

const activity = (page: Page) => page.getByRole('region', { name: 'Current activity', exact: true })
const elapsed = (page: Page) => page.locator('[aria-label="Elapsed wall time"]')

test('long responses update elapsed and wire activity with unchanged result timestamp, then stop at terminal', async ({
  page,
}, testInfo) => {
  const state = await fixture(page)
  await expect(activity(page)).toContainText('1,024 bytes')
  await expect(elapsed(page)).toContainText('10m 0s')
  await expect(activity(page)).toContainText('0s ago')
  const initialReads = state.activityReads.first
  const initialReports = state.reportReads
  state.receivedBytes = 2048
  await page.clock.fastForward(5 * 60_000)
  await expect.poll(() => state.activityReads.first).toBe(initialReads + 1)
  await expect(activity(page)).toContainText('2,048 bytes')
  await expect(elapsed(page)).toContainText('15m 0s')
  await expect(activity(page)).toContainText('5m 0s ago')
  expect(state.runs[0].updated_at).toBe(start)
  expect(state.reportReads).toBe(initialReports)
  await expect(page.getByText('Total recorded cost').locator('..').locator('strong')).toHaveText(
    '—',
  )
  await expect(
    page.getByText('Model tokens', { exact: true }).locator('..').locator('strong'),
  ).toHaveText('—')
  await expect(activity(page)).toContainText(
    'tokens and cost remain unknown until usage is recorded',
  )
  const icon = activity(page).getByRole('heading').locator('svg')
  expect((await icon.boundingBox())!.width).toBeLessThanOrEqual(16)
  expect((await icon.boundingBox())!.height).toBeLessThanOrEqual(16)
  await activity(page).screenshot({ path: testInfo.outputPath('active-response-desktop.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  expect((await icon.boundingBox())!.width).toBeLessThanOrEqual(16)
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1))
    .toBe(true)
  await activity(page).screenshot({ path: testInfo.outputPath('active-response-mobile.png') })
  state.runs[0].status = 'completed'
  state.runs[0].updated_at = '2026-01-01T00:15:00Z'
  await page.clock.runFor(2600)
  await expect(activity(page)).toHaveCount(0)
  await expect(elapsed(page)).toContainText('60 s')
  const reads = state.activityReads.first
  await page.clock.fastForward(60_000)
  expect(state.activityReads.first).toBe(reads)
  await expect(elapsed(page)).toContainText('60 s')
  expect(state.writes).toEqual([])
})

test('unrecorded activity remains unknown; failed refresh keeps explicitly stale observations', async ({
  page,
}) => {
  const state = await fixture(page)
  await expect(activity(page)).toContainText('Receiving response')
  state.missingActivity = true
  await page.clock.runFor(2600)
  await expect(activity(page)).toContainText('Activity not recorded')
  await expect(
    activity(page)
      .locator('dd')
      .filter({ hasText: /^Not recorded$/ }),
  ).toHaveCount(2)
  await expect(activity(page)).not.toContainText('0 bytes')
  await expect(activity(page)).not.toContainText('No response bytes observed')
  state.failActivity = true
  await page.clock.runFor(2600)
  await expect(activity(page).getByRole('alert')).toContainText(
    'Showing the last observed calls below',
  )
  await expect(activity(page)).toContainText('Activity not recorded')
  expect(state.writes).toEqual([])
})

test('active pages are bounded and a failed page never becomes a misleading empty result', async ({
  page,
}) => {
  const state = await fixture(page)
  await expect(activity(page)).toContainText('Receiving response')
  state.paginate = true
  await page.clock.runFor(2600)
  await expect(activity(page).getByRole('listitem')).toHaveCount(5)
  await expect(activity(page)).toContainText('7 in progress')
  await activity(page).getByRole('button', { name: 'Next active calls', exact: true }).click()
  await expect(activity(page).getByRole('listitem')).toHaveCount(2)
  state.badCursor = true
  await page.clock.runFor(2600)
  await expect(activity(page).getByRole('alert')).toContainText('cursor did not advance')
  await expect(activity(page).getByRole('listitem')).toHaveCount(2)
  await expect(activity(page)).not.toContainText('No model call was in progress')
})

test('unmount aborts activity polling and a delayed prior-run response cannot leak into the next run', async ({
  page,
}) => {
  const state = await fixture(page)
  await expect(activity(page)).toContainText('provider/model/first')
  let release: () => void = () => {}
  let entered = false
  state.blockFirst = async (route) => {
    entered = true
    await new Promise<void>((resolve) => {
      release = resolve
    })
    await route
      .fulfill({
        json: {
          total: 1,
          next_cursor: null,
          calls: [
            {
              id: 'late',
              case_id: 'old-question',
              model: 'STALE-FIRST-RESPONSE',
              target_id: 'single',
              status: 'sent',
              role: 'subject',
            },
          ],
        },
      })
      .catch(() => {})
  }
  await page.clock.runFor(2600)
  await expect.poll(() => entered).toBe(true)
  await page.getByRole('button', { name: 'Back to runs', exact: true }).click()
  const firstReads = state.activityReads.first
  await page.clock.fastForward(60_000)
  expect(state.activityReads.first).toBe(firstReads)
  await page.getByRole('button', { name: 'second long response', exact: true }).click()
  await expect(activity(page)).toContainText('provider/model/second')
  release()
  await page.clock.runFor(2600)
  await expect(activity(page)).not.toContainText('STALE-FIRST-RESPONSE')
  await expect(activity(page)).not.toContainText('provider/model/first')
  await expect(activity(page)).toContainText('provider/model/second')
  expect(state.activityReads.first).toBe(firstReads)
  expect(state.writes).toEqual([])
})

test('real zero-byte phases stay distinct from missing activity and auxiliary model identity', async ({
  page,
}) => {
  const state = await fixture(page)
  await expect(activity(page)).toContainText('Receiving response')
  state.phase = 'preparing'
  state.receivedBytes = 0
  state.lastActivityAt = null
  state.role = 'judge'
  state.callModel = 'provider/judge-model'
  await page.clock.runFor(2600)
  await expect(activity(page)).toContainText('Preparing request')
  await expect(activity(page)).toContainText('0 bytes')
  await expect(activity(page)).toContainText('No response bytes observed')
  await expect(activity(page)).toContainText('provider/judge-model')
  await expect(activity(page)).toContainText('Judge · Question 1')
  await expect(activity(page)).not.toContainText('provider/model/first')
  state.phase = 'waiting'
  await page.clock.runFor(2600)
  await expect(activity(page)).toContainText('Waiting for response')
  await expect(activity(page)).toContainText('No response bytes observed')
  expect(state.writes).toEqual([])
})
