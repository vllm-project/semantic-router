import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

// config.yaml sets only part of each section; the Router fills in the rest.
const savedGlobal = {
  stores: { response_cache: { enabled: true } },
}

const effectiveGlobal = {
  router: { strategy: 'priority' },
  stores: {
    response_cache: {
      enabled: true,
      backend_type: 'memory',
      similarity_threshold: 0.8,
      max_entries: 1000,
      ttl_seconds: 3600,
      eviction_policy: 'fifo',
    },
  },
  model_catalog: {
    admission: {
      hazard: { max_concurrency: 8, max_queue: 8, on_overflow: 'wait' },
      safety: { max_concurrency: 8, max_queue: 8, on_overflow: 'wait' },
    },
  },
}

async function mockGlobalConfig(page: Page): Promise<unknown[]> {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) =>
    route.fulfill({
      json: {
        version: 'v0.5',
        listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
        providers: { defaults: { model: 'test-model' }, models: [] },
        recipes: [{ name: 'default' }],
        global: savedGlobal,
      },
    }),
  )
  await page.route('**/api/router/config/global', (route) => route.fulfill({ json: effectiveGlobal }))
  await page.route('**/api/router/config/global/raw', (route) =>
    route.fulfill({ contentType: 'text/yaml', body: 'router: {}\n' }),
  )

  const updates: unknown[] = []
  await page.route('**/api/router/config/global/update', async (route) => {
    updates.push(route.request().postDataJSON())
    await route.fulfill({ json: { status: 'success' } })
  })
  return updates
}

test.describe('Global Config card saves', () => {
  test('writes nothing when a card is saved unchanged', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    const updates = await mockGlobalConfig(page)
    await page.goto('/config/global-config')

    for (const [key, title] of [
      ['response_cache', 'Response Cache'],
      ['admission', 'Model Admission'],
    ]) {
      await page.locator(`#global-section-${key}`).getByRole('button', { name: 'Edit' }).click()
      const modal = page.getByRole('dialog', { name: `Edit ${title}` })
      await modal.getByRole('button', { name: 'Save', exact: true }).click()
      await expect(modal).toBeHidden()
    }
    expect(updates).toEqual([])
  })

  test('writes only the field the user changed', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    const updates = await mockGlobalConfig(page)
    await page.goto('/config/global-config')

    await page
      .locator('#global-section-response_cache')
      .getByRole('button', { name: 'Edit' })
      .click()
    const modal = page.getByRole('dialog', { name: 'Edit Response Cache' })
    await expect(modal.getByLabel('Max Entries', { exact: true })).toHaveValue('1000')
    await modal.getByLabel('Max Entries', { exact: true }).fill('2000')
    await modal.getByRole('button', { name: 'Save', exact: true }).click()
    await expect(modal).toBeHidden()

    expect(updates).toEqual([{ stores: { response_cache: { max_entries: 2000 } } }])
  })
})
