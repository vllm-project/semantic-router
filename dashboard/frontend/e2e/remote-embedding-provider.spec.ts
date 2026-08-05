import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const globalConfig = {
  router: { strategy: 'priority' },
  model_catalog: {
    embeddings: {
      semantic: {
        mmbert_model_path: 'models/mmbert-embed-32k-2d-matryoshka',
        use_cpu: true,
        embedding_config: {
          backend: 'candle',
          model_type: 'mmbert',
          target_dimension: 768,
          preload_embeddings: true,
        },
      },
    },
  },
}

const configResponse = {
  version: 'v0.3',
  listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
  providers: { defaults: { default_model: 'test-model' }, models: [] },
  routing: { signals: {}, decisions: [] },
  global: globalConfig,
}

async function mockRemoteEmbeddingDashboard(page: Page) {
  await mockAuthenticatedAppShell(page)

  await page.route('**/api/auth/bootstrap/can-register', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: '{"canRegister":false}',
    })
  })
  await page.route('**/api/router/config/all', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(configResponse),
    })
  })
  await page.route('**/api/router/config/global', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(globalConfig),
    })
  })
  await page.route('**/api/router/config/global/raw', async (route) => {
    await route.fulfill({ status: 200, contentType: 'text/yaml', body: 'router: {}\n' })
  })
  await page.route('**/api/router/config/kbs', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{"items":[]}' })
  })
  await page.route('**/api/tools-db', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  })
  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        overall: 'healthy',
        deployment_type: 'local',
        services: [{ name: 'Router', status: 'running', healthy: true }],
        router_runtime: {
          phase: 'ready',
          ready: true,
          embedding_provider: {
            mode: 'remote',
            backend: 'openai_compatible',
            model: 'text-embedding-3-small',
            dimension: 1536,
            api_key_env: 'OPENAI_API_KEY',
            api_key_env_set: true,
            healthy: true,
            last_checked_at: '2026-07-13T12:00:00Z',
          },
        },
        models: {
          models: [],
          summary: { ready: true, phase: 'ready', loaded_models: 0, total_models: 0 },
        },
      }),
    })
  })
}

test.describe('Remote embedding provider Dashboard workflow', () => {
  test('switches from local inference and posts the canonical remote provider patch', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 1100 })
    await mockRemoteEmbeddingDashboard(page)

    let updateBody: Record<string, unknown> | null = null
    await page.route('**/api/router/config/global/update', async (route) => {
      updateBody = route.request().postDataJSON() as Record<string, unknown>
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{"status":"ok"}' })
    })

    await page.goto('/config/global-config')
    const card = page.locator('article').filter({
      has: page.getByRole('heading', { name: 'Embedding Models' }),
    })
    await expect(card).toContainText('Local / candle')
    await card.getByRole('button', { name: 'Edit Section' }).click()

    const modal = page.getByRole('dialog', { name: 'Edit Embedding Models' })
    await expect(modal.getByLabel('mmBERT Model Path')).toBeVisible()
    await expect(modal.getByLabel('Base URL')).toHaveCount(0)

    await expect(modal.getByLabel('Local Backend')).toBeVisible()
    await expect(modal.getByLabel('API Protocol')).toHaveCount(0)
    await modal.getByLabel('Provider Type').selectOption('remote')
    await expect(modal.getByLabel('mmBERT Model Path')).toHaveCount(0)
    await expect(modal.getByLabel('Local Backend')).toHaveCount(0)
    await modal.getByLabel('API Protocol').selectOption('openai_compatible')
    await modal.getByLabel('Base URL').fill('https://embedding.example.com/v1')
    await modal.getByLabel('Model').fill('text-embedding-3-small')
    await modal.getByLabel('API Key Environment Variable').fill('OPENAI_API_KEY')
    await modal.getByLabel('Dimensions').fill('1536')
    await modal.getByLabel('Target Dimension').fill('1536')
    await modal.getByRole('button', { name: 'Save' }).click()

    await expect(modal).toBeHidden()
    await expect.poll(() => updateBody).not.toBeNull()
    expect(updateBody).toMatchObject({
      model_catalog: {
        embeddings: {
          semantic: {
            embedding_config: {
              backend: 'openai_compatible',
              model_type: 'remote',
              target_dimension: 1536,
            },
            endpoint: {
              base_url: 'https://embedding.example.com/v1',
              model: 'text-embedding-3-small',
              api_key_env: 'OPENAI_API_KEY',
              dimensions: 1536,
            },
          },
        },
      },
    })
  })

  test('shows provider health without overflowing desktop or mobile layouts', async ({ page }) => {
    await mockRemoteEmbeddingDashboard(page)

    for (const viewport of [
      { width: 1440, height: 1000 },
      { width: 390, height: 844 },
    ]) {
      await page.setViewportSize(viewport)
      await page.goto('/status')

      const panel = page.getByTestId('embedding-provider-status')
      await expect(panel).toBeVisible()
      await expect(panel).toContainText('Healthy')
      await expect(panel).toContainText('text-embedding-3-small')
      await expect(panel).toContainText('OPENAI_API_KEY')
      const panelBox = await panel.boundingBox()
      expect(panelBox).not.toBeNull()
      expect(panelBox!.x).toBeGreaterThanOrEqual(0)
      expect(panelBox!.x + panelBox!.width).toBeLessThanOrEqual(viewport.width + 1)
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
        viewport.width,
      )
    }
  })
})
