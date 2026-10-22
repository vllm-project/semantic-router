import { readFileSync } from 'node:fs'
import { test, expect } from '@playwright/test'

const adapter = readFileSync('../backend/proxy/grafana_auth.js', 'utf8')
const documentPath = '/embedded/grafana/d/test'

test.beforeEach(async ({ page, context, baseURL }) => {
  await context.addCookies([{ name: 'vsr_csrf', value: 'fixture-csrf', url: baseURL! }])
  await page.route(`**${documentPath}`, (route) =>
    route.fulfill({
      contentType: 'text/html',
      body: '<html><head><script src="/embedded/grafana/_dashboard/auth.js"></script></head><body>Grafana fixture</body></html>',
    }),
  )
  await page.route('**/embedded/grafana/_dashboard/auth.js', (route) =>
    route.fulfill({
      contentType: 'application/javascript',
      body: adapter,
    }),
  )
  await page.goto(documentPath)
})

test('Grafana iframe echoes the current CSRF cookie and preserves a query Request', async ({
  page,
}) => {
  const calls: { method: string; body: string | null; headers: Record<string, string> }[] = []
  await page.route('**/embedded/grafana/api/ds/query', async (route) => {
    const request = route.request()
    calls.push({
      method: request.method(),
      body: request.postData(),
      headers: await request.allHeaders(),
    })
    await route.fulfill({ status: 200, json: { results: {} } })
  })
  const results = await page.evaluate(async () => {
    const request = new Request('/embedded/grafana/api/ds/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Request-Fixture': 'preserved' },
      body: JSON.stringify({ queries: [{ expr: 'sum(llm_model_requests_total)' }] }),
    })
    const first = await window.fetch(request)
    document.cookie = 'vsr_csrf=rotated-csrf; Path=/'
    const second = await window.fetch('/embedded/grafana/api/ds/query', {
      method: 'POST',
      headers: new Headers({ 'Content-Type': 'application/json' }),
      body: '{}',
    })
    return [first.status, second.status]
  })
  expect(results).toEqual([200, 200])
  expect(calls[0].method).toBe('POST')
  expect(JSON.parse(calls[0].body!)).toEqual({
    queries: [{ expr: 'sum(llm_model_requests_total)' }],
  })
  expect(calls[0].headers['content-type']).toBe('application/json')
  expect(calls[0].headers['x-request-fixture']).toBe('preserved')
  expect(calls[0].headers['x-csrf-token']).toBe('fixture-csrf')
  expect(calls[1].headers['x-csrf-token']).toBe('rotated-csrf')
})

test('the adapter respects explicit headers, method overrides, and cancellation', async ({
  page,
}) => {
  const calls: Record<string, string>[] = []
  await page.route('**/embedded/grafana/api/query-control', async (route) => {
    calls.push(await route.request().allHeaders())
    await route.fulfill({ status: 200, json: {} })
  })
  const aborted = await page.evaluate(async () => {
    const url = '/embedded/grafana/api/query-control'
    await window.fetch(new Request(url), {
      method: 'POST',
      headers: { 'X-CSRF-Token': 'explicit-csrf' },
    })
    await window.fetch(new Request(url, { method: 'POST', headers: { 'X-Old': 'discarded' } }), {
      headers: { 'X-New': 'preserved' },
    })
    const controller = new AbortController()
    controller.abort()
    try {
      await window.fetch(url, { method: 'POST', signal: controller.signal })
    } catch (error) {
      return (error as Error).name
    }
    return 'not aborted'
  })
  expect(aborted).toBe('AbortError')
  expect(calls).toHaveLength(2)
  expect(calls[0]['x-csrf-token']).toBe('explicit-csrf')
  expect(calls[1]['x-new']).toBe('preserved')
  expect(calls[1]['x-old']).toBeUndefined()
  expect(calls[1]['x-csrf-token']).toBe('fixture-csrf')
})

test('safe requests and paths outside the Grafana proxy keep their headers', async ({
  page,
  context,
}) => {
  const calls: Record<string, string>[] = []
  await page.route('**/query-control', async (route) => {
    calls.push(await route.request().allHeaders())
    await route.fulfill({ status: 200, headers: { 'Access-Control-Allow-Origin': '*' }, json: {} })
  })
  await page.evaluate(async () => {
    await window.fetch('/embedded/grafana/query-control')
    await window.fetch('/api/query-control', { method: 'POST' })
    await window.fetch('http://example.invalid/embedded/grafana/query-control', { method: 'POST' })
  })
  await context.clearCookies()
  await page.evaluate(() => window.fetch('/embedded/grafana/query-control', { method: 'POST' }))
  expect(calls).toHaveLength(4)
  expect(calls.every((headers) => headers['x-csrf-token'] === undefined)).toBe(true)
})
