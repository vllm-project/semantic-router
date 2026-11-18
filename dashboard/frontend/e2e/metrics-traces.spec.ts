import { readFileSync } from 'node:fs'
import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const provisionedDashboard = JSON.parse(
  readFileSync(
    new URL('../../../src/vllm-sr/cli/templates/llm-router-dashboard.serve.json', import.meta.url),
    'utf8',
  ),
) as { uid: string }

/** Prometheus HTTP API answers for the queries the Metrics page sends. */
function prometheusAnswer(url: URL) {
  const query = url.searchParams.get('query') ?? ''
  if (url.pathname.endsWith('/query_range')) {
    const start = Number(url.searchParams.get('start'))
    const end = Number(url.searchParams.get('end'))
    const step = Number(url.searchParams.get('step'))
    const values = (value: string) => {
      const samples: [number, string][] = []
      for (let time = start; time <= end; time += step) samples.push([time, value])
      return samples
    }
    const result = query.startsWith('up{')
      ? [{ metric: { job: 'semantic-router', instance: 'router:9190' }, values: values('1') }]
      : query.includes('llm_request_outcomes_total')
        ? [
            { metric: { outcome: 'success' }, values: values('1.5') },
            { metric: { outcome: 'server_error' }, values: values('0.05') },
          ]
        : query.includes('llm_model_requests_total')
          ? [{ metric: { model: 'qwen3-8b' }, values: values('1.2') }]
          : []
    return { status: 'success', data: { resultType: 'matrix', result } }
  }
  const result = query.includes('llm_request_outcomes_total')
    ? [
        { metric: { outcome: 'success' }, value: [0, '950'] },
        { metric: { outcome: 'server_error' }, value: [0, '50'] },
      ]
    : query.includes('"stat", "requests"')
      ? [{ metric: { model: 'qwen3-8b', stat: 'requests' }, value: [0, '800'] }]
      : query.includes('llm_recipe_selections_total')
        ? [
            {
              metric: { recipe: 'default', decision: 'coding', model: 'qwen3-8b' },
              value: [0, '600'],
            },
          ]
        : []
  return { status: 'success', data: { resultType: 'vector', result } }
}

async function mockPrometheus(page: Page, queries: URL[] = []) {
  await page.route('**/embedded/prometheus/api/v1/**', async (route) => {
    const url = new URL(route.request().url())
    queries.push(url)
    await route.fulfill({ json: prometheusAnswer(url) })
  })
  return queries
}

test.describe('Metrics', () => {
  test('charts the Router metrics from Prometheus and links the provisioned Grafana dashboard', async ({
    page,
  }) => {
    await mockAuthenticatedAppShell(page)
    const methods: string[] = []
    page.on('request', (request) => {
      if (request.url().includes('/embedded/prometheus/')) methods.push(request.method())
    })
    const queries = await mockPrometheus(page)
    await page.route('**/embedded/grafana/api/health', (route) =>
      route.fulfill({ json: { database: 'ok', version: '11.0.0' } }),
    )

    await page.goto('/observe/metrics')
    await expect(page.getByRole('heading', { name: 'Metrics', exact: true })).toBeVisible()
    await expect(page.getByText('50 of 1K failed')).toBeVisible()
    await expect(page.getByRole('region', { name: 'Requests', exact: true })).toBeVisible()
    await expect(
      page
        .getByRole('region', { name: 'By model', exact: true })
        .getByRole('cell', { name: 'qwen3-8b' }),
    ).toBeVisible()
    await expect(page.getByRole('region', { name: 'By decision', exact: true })).toContainText(
      'coding',
    )
    await expect(page.getByRole('list', { name: 'Router instances' })).toContainText('router:9190')

    const grafana = page.getByRole('link', { name: 'Open Grafana' })
    await expect(grafana).toHaveAttribute(
      'href',
      new RegExp(`^/embedded/grafana/d/${provisionedDashboard.uid}\\?`),
    )
    await expect(grafana).toHaveAttribute('target', '_blank')
    expect(queries.length).toBeGreaterThan(0)
    expect(methods.every((method) => method === 'GET')).toBe(true)
    expect(
      queries.every((url) => url.searchParams.get('query')?.includes('job="semantic-router"')),
    ).toBe(true)

    await page.getByRole('radio', { name: '24h' }).click()
    await expect(page).toHaveURL(/range=24h/)
    await expect
      .poll(() => queries.some((url) => url.searchParams.get('step') === '600'))
      .toBe(true)
    await expect(grafana).toHaveAttribute('href', /from=now-24h/)
  })

  test('asks for a Prometheus connection and hides Grafana when neither is configured', async ({
    page,
  }) => {
    await mockAuthenticatedAppShell(page)
    await page.route('**/embedded/prometheus/api/v1/**', (route) =>
      route.fulfill({ status: 503, contentType: 'text/html', body: '<h1>Not Configured</h1>' }),
    )
    await page.route('**/embedded/grafana/api/health', (route) =>
      route.fulfill({ status: 503, contentType: 'text/html', body: '<h1>Not Configured</h1>' }),
    )
    await page.goto('/observe/metrics')
    await expect(page.getByRole('heading', { name: 'Connect Prometheus' })).toBeVisible()
    await expect(page.getByText('TARGET_PROMETHEUS_URL').first()).toBeVisible()
    await expect(page.getByRole('link', { name: 'Open Grafana' })).toHaveCount(0)
    await expect(page.getByRole('radiogroup', { name: 'Time range' })).toHaveCount(0)
  })

  test('an access failure stays unavailable until a successful retry', async ({ page }) => {
    await mockAuthenticatedAppShell(page)
    let denied = true
    await page.route('**/embedded/prometheus/api/v1/**', (route) =>
      denied
        ? route.fulfill({ status: 403, contentType: 'text/plain', body: 'Forbidden' })
        : route.fulfill({ json: prometheusAnswer(new URL(route.request().url())) }),
    )
    await page.goto('/observe/metrics')
    await expect(page.getByRole('alert')).toContainText('Access to Prometheus was denied')
    await expect(page.getByRole('region', { name: 'By model', exact: true })).toHaveCount(0)
    denied = false
    await page.getByRole('button', { name: 'Try again', exact: true }).click()
    await expect(page.getByRole('region', { name: 'By model', exact: true })).toBeVisible()
    await expect(page.getByRole('alert')).toHaveCount(0)
  })

  test('a stalled Prometheus read times out and can be retried', async ({ page }) => {
    await mockAuthenticatedAppShell(page)
    await page.clock.install()
    let stalled = true
    await page.route('**/embedded/prometheus/api/v1/**', async (route) => {
      if (stalled) return
      await route.fulfill({ json: prometheusAnswer(new URL(route.request().url())) })
    })
    await page.goto('/observe/metrics')
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeDisabled()
    await page.clock.runFor(15_001)
    await expect(page.getByRole('alert')).toContainText('Prometheus took too long to respond')
    await expect(page.getByRole('region', { name: 'Requests', exact: true })).toHaveCount(0)
    stalled = false
    await page.getByRole('button', { name: 'Try again', exact: true }).click()
    await expect(page.getByRole('region', { name: 'Requests', exact: true })).toBeVisible()
    await expect(page.getByRole('alert')).toHaveCount(0)
  })
})

const TRACE_ID = '4bf92f3577b34da6a3ce929d0e0e4736'

function jaegerTrace() {
  const start = Date.now() * 1_000 - 60_000_000
  return {
    traceID: TRACE_ID,
    spans: [
      {
        traceID: TRACE_ID,
        spanID: 'a1',
        operationName: 'semantic_router.request',
        references: [],
        startTime: start,
        duration: 1_250_000,
        tags: [
          { key: 'span.kind', type: 'string', value: 'server' },
          { key: 'request.outcome', type: 'string', value: 'server_error' },
        ],
        logs: [
          {
            timestamp: start + 10_000,
            fields: [
              { key: 'event', type: 'string', value: 'routing.decision' },
              { key: 'decision', type: 'string', value: 'coding' },
            ],
          },
        ],
        processID: 'p1',
      },
      {
        traceID: TRACE_ID,
        spanID: 'b2',
        operationName: 'semantic_router.decision.evaluation',
        references: [{ refType: 'CHILD_OF', traceID: TRACE_ID, spanID: 'a1' }],
        startTime: start + 20_000,
        duration: 80_000,
        tags: [],
        logs: [],
        processID: 'p1',
      },
      {
        traceID: TRACE_ID,
        spanID: 'c3',
        operationName: 'POST /v1/chat/completions',
        references: [{ refType: 'CHILD_OF', traceID: TRACE_ID, spanID: 'a1' }],
        startTime: start + 150_000,
        duration: 1_000_000,
        tags: [
          { key: 'error', type: 'bool', value: true },
          { key: 'otel.status_code', type: 'string', value: 'ERROR' },
        ],
        logs: [],
        processID: 'p2',
      },
    ],
    processes: {
      p1: { serviceName: 'vllm-sr', tags: [] },
      p2: { serviceName: 'qwen3-backend', tags: [] },
    },
  }
}

/** The Dashboard's summary of {@link jaegerTrace}, as its trace search answers. */
function traceSummary() {
  return {
    traceID: TRACE_ID,
    rootOperation: 'semantic_router.request',
    rootService: 'vllm-sr',
    services: [
      { name: 'vllm-sr', spanCount: 2 },
      { name: 'qwen3-backend', spanCount: 1 },
    ],
    spanCount: 3,
    errorCount: 1,
    startTimeMicros: jaegerTrace().spans[0].startTime,
    durationMicros: 1_250_000,
  }
}

const TRACE_SUMMARIES = '**/api/observability/traces/summaries**'

async function mockRouterTracing(page: Page) {
  await page.route('**/api/router/config/all', (route) =>
    route.fulfill({
      json: {
        global: {
          services: {
            observability: { tracing: { enabled: true, resource: { service_name: 'vllm-sr' } } },
          },
        },
      },
    }),
  )
}

/** Jaeger's query API through the Dashboard proxy; returns the paths it was asked for. */
async function mockJaegerProxy(page: Page) {
  const reads: string[] = []
  await page.route('**/embedded/jaeger/api/**', async (route) => {
    const url = new URL(route.request().url())
    reads.push(url.pathname)
    if (url.pathname === '/embedded/jaeger/api/services') {
      return route.fulfill({ json: { data: ['jaeger-all-in-one', 'vllm-sr'], total: 2 } })
    }
    if (url.pathname === '/embedded/jaeger/api/services/vllm-sr/operations') {
      return route.fulfill({ json: { data: ['semantic_router.request'], total: 1 } })
    }
    if (url.pathname === `/embedded/jaeger/api/traces/${TRACE_ID}`) {
      return route.fulfill({ json: { data: [jaegerTrace()], total: 1 } })
    }
    return route.fulfill({
      status: 404,
      json: { data: null, errors: [{ code: 404, msg: 'not found' }] },
    })
  })
  return reads
}

test.describe('Traces', () => {
  test('searches Jaeger for the Router service and opens a span waterfall', async ({ page }) => {
    await mockAuthenticatedAppShell(page)
    await mockRouterTracing(page)
    const searches: URL[] = []
    await page.route(TRACE_SUMMARIES, async (route) => {
      searches.push(new URL(route.request().url()))
      await route.fulfill({ json: { data: [traceSummary()], limit: 20, total: 1 } })
    })
    const jaegerReads = await mockJaegerProxy(page)

    await page.goto('/observe/traces')
    const row = page.getByRole('row').filter({ hasText: 'semantic_router.request' })
    await expect(row).toBeVisible()
    await expect(row).toContainText('qwen3-backend')
    await expect(row).toContainText('1.25 s')
    // The page leaves every filter but the Router service to the person searching.
    await expect(page).toHaveURL(/\/observe\/traces$/)
    expect(Object.fromEntries(searches[0].searchParams)).toEqual({
      service: 'vllm-sr',
      lookback: '1h',
      limit: '20',
    })
    // The list reads summaries; only an opened trace is downloaded, by its full ID.
    const traceReads = () =>
      jaegerReads.filter((path) => path.startsWith('/embedded/jaeger/api/traces'))
    expect(traceReads()).toEqual([])

    await row.getByRole('button', { name: 'semantic_router.request' }).click()
    await expect(page).toHaveURL(new RegExp(`trace=${TRACE_ID}`))
    await expect.poll(traceReads).toContain(`/embedded/jaeger/api/traces/${TRACE_ID}`)
    expect(new Set(traceReads())).toEqual(new Set([`/embedded/jaeger/api/traces/${TRACE_ID}`]))
    const sheet = page.getByRole('dialog')
    await expect(
      sheet.getByRole('heading', { level: 2, name: 'semantic_router.request' }),
    ).toBeVisible()
    const spans = sheet.getByRole('tree', { name: 'Spans' })
    await expect(spans.getByRole('treeitem')).toHaveCount(3)
    const details = sheet.getByRole('region', { name: 'Span details' })
    await expect(details).toContainText('routing.decision')
    await spans.getByRole('treeitem').filter({ hasText: 'POST /v1/chat/completions' }).click()
    await expect(details).toContainText('otel.status_code')
    await expect(details).toContainText('qwen3-backend')
    await page.keyboard.press('Escape')
    await expect(page).toHaveURL(/\/observe\/traces$/)

    await page.getByRole('combobox', { name: 'Operation' }).click()
    await page.getByRole('option', { name: 'semantic_router.request' }).click()
    await expect
      .poll(() => searches[searches.length - 1].searchParams.get('operation'))
      .toBe('semantic_router.request')
  })

  test('asks for a Jaeger connection when the Dashboard has none', async ({ page }) => {
    await mockAuthenticatedAppShell(page)
    await page.route('**/embedded/jaeger/api/**', (route) =>
      route.fulfill({ status: 503, contentType: 'text/html', body: '<h1>Not Configured</h1>' }),
    )
    await page.goto('/observe/traces')
    await expect(page.getByRole('heading', { name: 'Connect Jaeger' })).toBeVisible()
    await expect(page.getByText('TARGET_JAEGER_URL').first()).toBeVisible()
  })

  test('a denied Jaeger read stays unavailable until a successful retry', async ({ page }) => {
    await mockAuthenticatedAppShell(page)
    let denied = true
    const forbidden = { status: 403, contentType: 'text/plain', body: 'Forbidden' }
    await page.route('**/embedded/jaeger/api/**', (route) => {
      if (denied) return route.fulfill(forbidden)
      const url = new URL(route.request().url())
      return route.fulfill({
        json: { data: url.pathname.endsWith('/services') ? ['vllm-sr'] : [] },
      })
    })
    await page.route(TRACE_SUMMARIES, (route) =>
      denied
        ? route.fulfill(forbidden)
        : route.fulfill({ json: { data: [], limit: 20, total: 0 } }),
    )
    await page.goto('/observe/traces')
    await expect(page.getByRole('alert')).toContainText('Access to Jaeger was denied')
    denied = false
    await page.getByRole('button', { name: 'Try again', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'No traces match' })).toBeVisible()
    await expect(page.getByRole('alert')).toHaveCount(0)
  })

  test('a search Jaeger is too slow for says so and can be retried', async ({ page }) => {
    await mockAuthenticatedAppShell(page)
    await mockRouterTracing(page)
    await mockJaegerProxy(page)
    const slow =
      'Jaeger did not finish the trace search within 20s. Show fewer traces, pick a shorter time range, or set a minimum duration.'
    let timedOut = true
    await page.route(TRACE_SUMMARIES, (route) =>
      timedOut
        ? route.fulfill({ status: 504, json: { error: 'jaeger_timeout', message: slow } })
        : route.fulfill({ json: { data: [traceSummary()], limit: 20, total: 1 } }),
    )
    await page.goto('/observe/traces')
    const alert = page.getByRole('alert')
    await expect(alert).toContainText('The trace search failed')
    await expect(alert).toContainText(slow)
    await expect(alert).not.toContainText('instead of data')
    timedOut = false
    await alert.getByRole('button', { name: 'Try again', exact: true }).click()
    await expect(page.getByRole('row').filter({ hasText: 'semantic_router.request' })).toBeVisible()
    await expect(page.getByRole('alert')).toHaveCount(0)
  })

  test('an opened trace that fails to load can be retried from its sheet', async ({ page }) => {
    await mockAuthenticatedAppShell(page)
    await mockRouterTracing(page)
    await page.route(TRACE_SUMMARIES, (route) =>
      route.fulfill({ json: { data: [traceSummary()], limit: 20, total: 1 } }),
    )
    await mockJaegerProxy(page)
    let failing = true
    await page.route(`**/embedded/jaeger/api/traces/${TRACE_ID}`, (route) =>
      failing
        ? route.fulfill({ status: 502, contentType: 'text/plain', body: 'Bad Gateway' })
        : route.fallback(),
    )
    await page.goto(`/observe/traces?trace=${TRACE_ID}`)
    const sheet = page.getByRole('dialog')
    await expect(sheet).toContainText('Jaeger did not return this trace')
    await expect(sheet).toContainText('Jaeger is unavailable (HTTP 502)')
    failing = false
    await sheet.getByRole('button', { name: 'Try again', exact: true }).click()
    await expect(
      sheet.getByRole('heading', { level: 2, name: 'semantic_router.request' }),
    ).toBeVisible()
    await expect(sheet.getByRole('tree', { name: 'Spans' }).getByRole('treeitem')).toHaveCount(3)
  })
})
