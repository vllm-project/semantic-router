import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

// config.yaml sets only part of each section; the Router fills in the rest.
const savedGlobal = {
  stores: { response_cache: { backend_type: 'memory' } },
}

const effectiveGlobal = {
  router: { strategy: 'priority', drain_timeout_seconds: 25 },
  stores: {
    response_cache: {
      backend_type: 'memory',
      max_entries: 1000,
      ttl_seconds: 3600,
      eviction_policy: 'fifo',
    },
  },
  model_catalog: {
    admission: {
      hazard: { max_concurrency: 8, max_queue: 8, on_overflow: 'wait' },
    },
  },
}

async function mockGlobalConfig(
  page: Page,
  extra: Record<string, unknown> = {},
): Promise<unknown[]> {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) =>
    route.fulfill({
      json: {
        version: 'v0.5',
        listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
        providers: { defaults: { model: 'test-model' }, models: [] },
        recipes: [{ name: 'default' }],
        global: { ...savedGlobal, ...extra },
      },
    }),
  )
  await page.route('**/api/router/config/global', (route) =>
    route.fulfill({ json: { ...effectiveGlobal, ...extra } }),
  )
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

test.describe('Router settings saves', () => {
  test('shows inherited defaults and writes nothing when an edit is undone', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    const updates = await mockGlobalConfig(page)
    await page.goto('/settings/router')

    const maxEntries = page.locator('#setting-stores-response_cache-max_entries')
    await expect(maxEntries).toHaveValue('')
    await expect(maxEntries).toHaveAttribute('placeholder', '1000')

    await maxEntries.fill('2000')
    const bar = page.getByRole('region', { name: 'Unsaved changes' })
    await expect(bar).toContainText('1 unsaved change')
    await page.getByRole('button', { name: 'Undo Max entries' }).click()
    await expect(bar).toBeHidden()
    expect(updates).toEqual([])
  })

  test('writes only the field the user changed', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    const updates = await mockGlobalConfig(page)
    await page.goto('/settings/router')

    await page.locator('#setting-stores-response_cache-max_entries').fill('2000')
    await page.getByRole('button', { name: 'Review & save' }).click()
    const review = page.getByRole('dialog', { name: 'Review changes' })
    await expect(review).toContainText('max_entries: 2000')
    await expect(review).toContainText('Applies with hot reload')
    await review.getByRole('button', { name: 'Save and reload' }).click()
    await expect(review).toBeHidden()

    expect(updates).toEqual([{ stores: { response_cache: { max_entries: 2000 } } }])
  })

  test('writes a whole map entry when an inherited one is edited', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    const updates = await mockGlobalConfig(page)
    await page.goto('/settings/router#global-section-admission')

    const admission = page.locator('#global-section-admission')
    await admission.getByRole('button', { name: /hazard/ }).click()
    await admission.getByLabel('Max concurrency').fill('12')
    await page.keyboard.press('ControlOrMeta+s')
    const review = page.getByRole('dialog', { name: 'Review changes' })
    await review.getByRole('button', { name: 'Save and reload' }).click()
    await expect(review).toBeHidden()

    expect(updates).toEqual([
      {
        model_catalog: {
          admission: { hazard: { max_concurrency: 12, max_queue: 8, on_overflow: 'wait' } },
        },
      },
    ])
  })

  test('switches embeddings from local inference to a remote provider', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    const updates = await mockGlobalConfig(page, {
      model_catalog: { embeddings: { default: { deployment: 'vela-embedding' } } },
    })
    await page.goto('/settings/router#global-section-embedding_models')

    const card = page.locator('#global-section-embedding_models')
    const defaultDeployment = card.getByRole('textbox', { name: 'Default deployment' })
    await expect(defaultDeployment).toHaveValue('vela-embedding')
    await expect(card.getByRole('textbox', { name: /^Base URL/ })).toHaveCount(0)
    await card.getByRole('radio', { name: 'Remote API' }).click()
    await expect(defaultDeployment).toHaveCount(0)
    await expect(card.getByRole('combobox', { name: 'API protocol' })).toHaveText(
      'openai_compatible',
    )
    await card.getByRole('textbox', { name: /^Base URL/ }).fill('https://embedding.example.com/v1')
    await card.getByRole('textbox', { name: /^Model/ }).fill('text-embedding-3-small')
    await card
      .getByRole('textbox', { name: /^API key environment variable/i })
      .fill('OPENAI_API_KEY')
    await card.getByRole('textbox', { name: 'Dimensions' }).fill('1536')

    await page.getByRole('button', { name: 'Review & save' }).click()
    const review = page.getByRole('dialog', { name: 'Review changes' })
    await review.getByRole('button', { name: 'Save and reload' }).click()
    await expect(review).toBeHidden()

    expect(updates).toHaveLength(1)
    expect(updates[0]).toMatchObject({
      model_catalog: {
        deployments: {
          'remote-embed': {
            provider: 'openai_embeddings',
            endpoint: 'https://embedding.example.com/v1',
            model: 'text-embedding-3-small',
            auth: { api_key_env: 'OPENAI_API_KEY' },
            capabilities: { dimensions: 1536 },
          },
        },
        embeddings: { default: { deployment: 'remote-embed' } },
      },
    })
  })

  test('opens decision model settings from a deep link and saves the model and a binding', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    const modelCatalog = {
      system: { decision_model: { deployment: 'main-judge' } },
      deployments: {
        'main-judge': { provider: 'model_runtime', artifact: 'vllm-sr/Vela-2.0-4B' },
        'alternate-judge': { provider: 'model_runtime', artifact: 'vllm-sr/Vela-2.0-0.8B' },
      },
    }
    const updates = await mockGlobalConfig(page, { model_catalog: modelCatalog })
    await page.goto('/settings/router#global-section-system_models')

    const card = page.locator('#global-section-system_models')
    await expect(card).toBeInViewport()
    const decisionModel = card.getByRole('combobox', { name: 'Decision model' })
    await expect(decisionModel).toHaveText('main-judge')
    await decisionModel.click()
    await page.getByRole('option', { name: 'alternate-judge' }).click()
    const domain = card.getByRole('combobox', { name: 'Domain binding' })
    await expect(domain).toHaveText('Follows alternate-judge')
    await domain.click()
    await page.getByRole('option', { name: 'alternate-judge', exact: true }).click()

    await page.getByRole('button', { name: 'Review & save' }).click()
    const review = page.getByRole('dialog', { name: 'Review changes' })
    await review.getByRole('button', { name: 'Save and reload' }).click()
    await expect(review).toBeHidden()

    expect(updates).toEqual([
      {
        model_catalog: {
          system: { decision_model: { deployment: 'alternate-judge' } },
          bindings: { domain: 'alternate-judge' },
        },
      },
    ])
  })
})
