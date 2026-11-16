import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

// A Global Config save answers with the Router's warnings, and the page shows
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

test.describe('Global Config save warnings', () => {
  test('a card save keeps the warning on the page after the refresh', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    await mockGlobalConfig(page)
    await page.goto('/config/global-config')

    await page
      .locator('#global-section-router_replay')
      .getByRole('button', { name: 'Edit' })
      .click()
    const modal = page.getByRole('dialog', { name: 'Edit Router Replay' })
    await modal.getByLabel('TTL (seconds)', { exact: true }).fill('86400')
    await modal.getByRole('button', { name: 'Save', exact: true }).click()
    await expect(modal).toBeHidden()

    const status = page.getByRole('status')
    await expect(status).toContainText('ttl_seconds starts that store empty')
    await expect(status).toBeInViewport()
  })

  test('a raw YAML save keeps the warning and the YAML editor after the refresh', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    await mockGlobalConfig(page)
    await page.goto('/config/global-config')

    await page.getByRole('button', { name: 'Raw YAML' }).click()
    const editor = page.locator('textarea')
    await expect(editor).toHaveValue(/store_backend: memory/)
    await editor.fill(
      'services:\n  router_replay:\n    store_backend: memory\n    ttl_seconds: 86400\n',
    )
    await page.getByRole('button', { name: 'Save YAML' }).click()

    await expect(page.getByRole('status')).toContainText('ttl_seconds starts that store empty')
    await expect(page.getByRole('button', { name: 'Save YAML' })).toBeVisible()
  })
})
