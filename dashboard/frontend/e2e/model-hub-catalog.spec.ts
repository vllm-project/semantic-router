import { expect, test } from '@playwright/test'
import { readFileSync } from 'node:fs'
import type { BuiltInModelCatalog } from '../src/types/modelCatalog'
import { mockAuthenticatedAppShell } from './support/auth'

const bundledCatalog: BuiltInModelCatalog = JSON.parse(
  readFileSync(
    new URL('../../../website/static/model-catalog/catalog.json', import.meta.url),
    'utf8',
  ),
)

test('Model Hub keeps the bundled catalog usable and retries the server explicitly', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page)
  let available = false
  let requests = 0
  const serverCatalog = structuredClone(bundledCatalog)
  serverCatalog.models[0].display_name = 'Server catalog recovery marker'
  await page.route('**/api/models/catalog', async (route) => {
    requests += 1
    await route.fulfill({
      status: available ? 200 : 503,
      contentType: 'application/json',
      body: JSON.stringify(available ? serverCatalog : { error: 'catalog_unavailable' }),
    })
  })

  await page.goto('/models')
  const notice = page.getByRole('status').filter({ hasText: 'Showing the catalog bundled' })
  await expect(notice).toBeVisible()
  await expect(notice).toContainText('HTTP 503')
  await expect(notice).not.toContainText('identical')
  await expect(page.getByRole('searchbox', { name: 'Search models' })).toBeEnabled()
  const beforeRetry = requests
  available = true
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await expect(notice).toHaveCount(0)
  expect(requests).toBe(beforeRetry + 1)
  await page
    .getByRole('searchbox', { name: 'Search models' })
    .fill('Server catalog recovery marker')
  await expect(
    page.getByText('Server catalog recovery marker', { exact: true }).first(),
  ).toBeVisible()
})

test('Model Hub retains bundled results when the server returns an invalid catalog', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/models/catalog', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ models: [], catalogs: [] }),
    })
  })
  await page.goto('/models')
  const notice = page.getByRole('status').filter({ hasText: 'Showing the catalog bundled' })
  await expect(notice).toContainText('invalid contract')
  await expect(page.getByRole('button', { name: 'Retry', exact: true })).toBeEnabled()
  await page
    .getByRole('searchbox', { name: 'Search models' })
    .fill(bundledCatalog.models[0].display_name)
  await expect(
    page.getByText(bundledCatalog.models[0].display_name, { exact: true }).first(),
  ).toBeVisible()
})

test('Model Hub exposes model search before evaluations finish and shares the response with model config', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) =>
    route.fulfill({
      json: {
        version: 'v0.3',
        providers: { models: [] },
        routing: { signals: {}, decisions: [] },
        global: {},
      },
    }),
  )
  await page.route('**/api/router/config/global', (route) => route.fulfill({ json: {} }))
  await page.route('**/api/router/config/global/raw', (route) =>
    route.fulfill({ contentType: 'text/yaml', body: '{}\n' }),
  )
  let requests = 0
  let release!: () => void
  const responseReady = new Promise<void>((resolve) => {
    release = resolve
  })
  await page.route('**/api/models/catalog', async (route) => {
    requests += 1
    await responseReady
    await route.fulfill({ json: bundledCatalog })
  })
  await page.goto('/models')
  const search = page.getByRole('searchbox', { name: 'Search models' })
  await expect(search).toBeEnabled()
  await search.fill(bundledCatalog.models[0].display_name)
  await expect(
    page.getByText(bundledCatalog.models[0].display_name, { exact: true }).first(),
  ).toBeVisible()
  await expect(
    page.getByRole('status').filter({ hasText: 'Loading model evaluations' }),
  ).toBeVisible()
  release()
  await expect(
    page.getByRole('status').filter({ hasText: 'Loading model evaluations' }),
  ).toHaveCount(0)
  const beforeNavigation = requests
  await page.getByRole('link', { name: 'Add model', exact: false }).click()
  await expect(page).toHaveURL(/\/config\/models/)
  await expect(page.getByRole('heading', { name: 'Models', exact: true, level: 1 })).toBeVisible()
  expect(requests).toBe(beforeNavigation)
})

test('Global Config does not load unrelated manager sections or the full model catalog', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) =>
    route.fulfill({
      json: {
        version: 'v0.3',
        providers: { models: [] },
        routing: { signals: {}, decisions: [] },
        global: {},
      },
    }),
  )
  await page.route('**/api/router/config/global', (route) => route.fulfill({ json: {} }))
  await page.route('**/api/router/config/global/raw', (route) =>
    route.fulfill({ contentType: 'text/yaml', body: '{}\n' }),
  )
  const unrelated: string[] = []
  page.on('request', (request) => {
    const path = new URL(request.url()).pathname
    if (
      path.endsWith('/ConfigPageModelsSection.tsx') ||
      path.endsWith('/ConfigPageMCPSection.tsx') ||
      path.endsWith('/api/models/catalog')
    )
      unrelated.push(path)
  })
  await page.goto('/config/global-config')
  await expect(page.getByRole('heading', { name: 'Global Config', exact: true })).toBeVisible()
  expect(unrelated).toEqual([])
})
