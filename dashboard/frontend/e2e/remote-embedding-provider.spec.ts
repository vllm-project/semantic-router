import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const globalConfig = {
  router: { strategy: 'priority' },
  model_catalog: {
    embeddings: { default: { deployment: 'vela-embedding' } },
  },
}

const configResponse = {
  version: 'v0.5',
  listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
  providers: { defaults: { model: 'test-model' }, models: [] },
  recipes: [{ name: 'default' }],
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
  test('keeps service availability inside desktop and mobile layouts', async ({ page }) => {
    await mockRemoteEmbeddingDashboard(page)

    for (const viewport of [
      { width: 1440, height: 1000 },
      { width: 390, height: 844 },
    ]) {
      await page.setViewportSize(viewport)
      await page.goto('/observe/status')

      const panel = page.getByTestId('status-availability')
      await expect(panel).toBeVisible()
      await expect(panel).toContainText('All systems operational')
      await expect(panel).toContainText('Router')
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
