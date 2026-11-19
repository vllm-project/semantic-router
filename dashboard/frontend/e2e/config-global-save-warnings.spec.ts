import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

// A Router settings save answers with the Router's warnings, and the page shows
// them after it refreshes the configuration it just saved.
const resetWarning = {
  code: 'replay_store_reset',
  field: 'global.services.router_replay.ttl_seconds',
  message:
    'Router Replay keeps its records in memory, and a changed ttl_seconds starts that store empty: the records kept so far are lost.',
}

async function mockGlobalConfig(page: Page): Promise<void> {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) =>
    route.fulfill({
      json: {
        version: 'v0.5',
        listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
        providers: { defaults: { model: 'test-model' }, models: [] },
        recipes: [{ name: 'default' }],
        global: { services: { router_replay: { store_backend: 'memory' } } },
      },
    }),
  )
  await page.route('**/api/router/config/global', (route) =>
    route.fulfill({
      json: { services: { router_replay: { store_backend: 'memory', ttl_seconds: 2592000 } } },
    }),
  )
  await page.route('**/api/router/config/global/raw', (route) =>
    route.fulfill({
      contentType: 'text/yaml',
      body: 'services:\n  router_replay:\n    store_backend: memory\n',
    }),
  )
  for (const path of [
    '**/api/router/config/global/update',
    '**/api/router/config/global/raw/update',
  ]) {
    await page.route(path, (route) =>
      route.fulfill({ json: { status: 'success', warnings: [resetWarning] } }),
    )
  }
}

async function openRouterSettings(page: Page, path: string) {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await mockGlobalConfig(page)
  await page.goto(path)
  await expect(
    page.getByRole('heading', { name: 'Router settings', exact: true }).first(),
  ).toBeVisible({ timeout: 20_000 })
}

const warningText = 'ttl_seconds starts that store empty'

test.describe('Router settings save warnings', () => {
  test('a form save keeps the warning on the page after the refresh', async ({ page }) => {
    await openRouterSettings(page, '/settings/router#global-section-router_replay')

    await page.locator('#setting-services-router_replay-ttl_seconds').fill('86400')
    await page.getByRole('button', { name: 'Review & save' }).click()
    const review = page.getByRole('dialog', { name: 'Review changes' })
    await review.getByRole('button', { name: 'Save and reload' }).click()
    await expect(review).toBeHidden()

    const status = page.getByRole('status').filter({ hasText: warningText })
    await expect(status).toBeVisible()
    await expect(status).toBeInViewport()
  })

  test('a YAML save keeps the warning after the refresh', async ({ page }) => {
    await openRouterSettings(page, '/settings/router?view=yaml')

    await page.getByRole('button', { name: 'Edit effective YAML' }).click()
    const editor = page
      .getByRole('dialog', { name: 'Effective global YAML' })
      .getByLabel('Effective global YAML', { exact: true })
    await expect(editor).toHaveValue(/store_backend: memory/)
    await editor.fill(
      'services:\n  router_replay:\n    store_backend: memory\n    ttl_seconds: 86400\n',
    )
    await page.getByRole('button', { name: 'Save YAML' }).click()

    await expect(page.getByRole('status').filter({ hasText: warningText })).toBeVisible()
  })
})
