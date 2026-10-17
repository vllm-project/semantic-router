import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

// config.yaml sets only part of each section; the Router fills in the rest.
const savedGlobal = {
  model_catalog: {
    modules: { prompt_guard: { enabled: true, model_ref: 'prompt_guard' } },
  },
}

const effectiveGlobal = {
  router: { strategy: 'priority' },
  model_catalog: {
    modules: {
      prompt_guard: {
        enabled: true,
        model_ref: 'prompt_guard',
        model_id: 'models/Vela-2.0-0.3B',
        threshold: 0.75,
        use_cpu: true,
      },
    },
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
        version: 'v0.3',
        listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
        providers: { defaults: { default_model: 'test-model' }, models: [] },
        routing: { signals: {}, decisions: [] },
        global: savedGlobal,
      },
    }),
  )
  await page.route('**/api/router/config/global', (route) => route.fulfill({ json: effectiveGlobal }))
  await page.route('**/api/router/config/global/raw', (route) =>
    route.fulfill({ contentType: 'text/yaml', body: 'router: {}\n' }),
  )
  await page.route('**/api/tools-db', (route) => route.fulfill({ json: [] }))

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
      ['prompt_guard', 'Prompt Guard'],
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
      .locator('#global-section-prompt_guard')
      .getByRole('button', { name: 'Edit' })
      .click()
    const modal = page.getByRole('dialog', { name: 'Edit Prompt Guard' })
    await expect(modal.getByLabel('Threshold', { exact: true })).toHaveValue('75')
    await modal.getByLabel('Threshold', { exact: true }).fill('80')
    await modal.getByRole('button', { name: 'Save', exact: true }).click()
    await expect(modal).toBeHidden()

    expect(updates).toEqual([{ model_catalog: { modules: { prompt_guard: { threshold: 0.8 } } } }])
  })
})
