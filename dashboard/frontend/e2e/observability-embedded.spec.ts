import { readFileSync } from 'node:fs'
import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const provisionedDashboard = JSON.parse(
  readFileSync(
    new URL('../../../src/vllm-sr/cli/templates/llm-router-dashboard.serve.json', import.meta.url),
    'utf8',
  ),
)

test('monitoring opens the provisioned dashboard rather than a short-link redirect', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/embedded/grafana/**', (route) =>
    route.fulfill({
      contentType: 'text/html',
      body: '<h1>Routing metrics</h1>',
    }),
  )
  await page.goto('/monitoring')
  const frame = page.getByTitle('Grafana monitoring dashboard')
  await expect(frame).toBeVisible()
  const target = new URL((await frame.getAttribute('src')) ?? '', page.url())
  expect(target.pathname).toBe(`/embedded/grafana/d/${provisionedDashboard.uid}`)
  await expect(
    page.frameLocator('iframe').getByRole('heading', { name: 'Routing metrics' }),
  ).toBeVisible()
  await expect(page.getByRole('link', { name: 'Open full view' })).toHaveAttribute(
    'href',
    (await frame.getAttribute('src')) ?? '',
  )
})

test('an access failure stays unavailable until a successful retry', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  let denied = true
  await page.route('**/embedded/grafana/**', (route) =>
    route.fulfill({
      status: denied ? 403 : 200,
      contentType: 'text/html',
      body: denied ? 'Access denied' : '<h1>Routing metrics</h1>',
    }),
  )
  await page.goto('/monitoring')
  await expect(page.getByRole('alert')).toContainText('Access to Grafana was denied')
  await expect(page.getByText('Connected through dashboard')).toHaveCount(0)
  await expect(page.getByTitle('Grafana monitoring dashboard')).toHaveCount(0)
  denied = false
  await page.getByRole('button', { name: 'Try again', exact: true }).click()
  await expect(page.getByTitle('Grafana monitoring dashboard')).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('tracing leaves service and result filters to the trace explorer', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/embedded/jaeger/**', (route) =>
    route.fulfill({
      contentType: 'text/html',
      body: '<h1>Trace search</h1>',
    }),
  )
  await page.goto('/tracing')
  const frame = page.getByTitle('Jaeger distributed tracing')
  await expect(frame).toBeVisible()
  const target = new URL((await frame.getAttribute('src')) ?? '', page.url())
  expect(target.pathname).toBe('/embedded/jaeger/search')
  expect([...target.searchParams.keys()]).toEqual([])
})

test('a stalled availability probe times out and can be retried', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.clock.install()
  let stalled = true
  await page.route('**/embedded/grafana/**', async (route) => {
    if (stalled) return
    await route.fulfill({ contentType: 'text/html', body: '<h1>Routing metrics</h1>' })
  })
  await page.goto('/monitoring')
  await expect(page.getByText('Checking connection')).toBeVisible()
  await page.clock.runFor(12001)
  await expect(page.getByRole('alert')).toContainText('took too long to respond')
  await expect(page.getByRole('button', { name: 'Reload', exact: true })).toBeEnabled()
  await expect(page.getByTitle('Grafana monitoring dashboard')).toHaveCount(0)
  stalled = false
  await page.getByRole('button', { name: 'Try again', exact: true }).click()
  await expect(page.getByTitle('Grafana monitoring dashboard')).toBeVisible()
})

test('services without HEAD support are checked with GET before embedding', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  const methods: string[] = []
  await page.route('**/embedded/grafana/**', (route) => {
    methods.push(route.request().method())
    return route.fulfill({
      status: route.request().method() === 'HEAD' ? 405 : 200,
      contentType: 'text/html',
      body: '<h1>Routing metrics</h1>',
    })
  })
  await page.goto('/monitoring')
  await expect(page.getByTitle('Grafana monitoring dashboard')).toBeVisible()
  expect(methods).toContain('HEAD')
  expect(methods).toContain('GET')
  expect(methods.every((method) => ['HEAD', 'GET'].includes(method))).toBe(true)
})
