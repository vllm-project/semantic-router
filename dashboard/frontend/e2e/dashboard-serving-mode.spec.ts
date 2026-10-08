import { expect, test } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

for (const scenario of [
  { name: 'Router with a managed native engine', mode: 'router', models: [], label: 'Router mode' },
  {
    name: 'engine with configured backend models',
    mode: 'engine',
    models: [{ name: 'backend' }],
    label: 'Engine mode',
  },
  {
    name: 'unidentified service with loaded models',
    mode: 'unknown',
    models: [{ name: 'backend' }],
    label: 'Mode unavailable',
  },
  { name: 'missing service identity', mode: undefined, models: [], label: 'Mode unavailable' },
]) {
  test(`homepage reports ${scenario.name} from serving_mode`, async ({ page }) => {
    await mockAuthenticatedAppShell(page)
    await page.route('**/api/router/config/all', (route) =>
      route.fulfill({
        json: {
          version: 'v0.3',
          providers: { models: scenario.models },
          routing: { signals: {}, decisions: [] },
        },
      }),
    )
    await page.route('**/api/status', (route) =>
      route.fulfill({
        json: {
          serving_mode: scenario.mode,
          overall: 'healthy',
          deployment_type: 'docker',
          services: [],
          models: {
            models: [
              {
                name: 'pii_classifier',
                type: 'pii_detection',
                loaded: true,
                metadata: {
                  provider: 'model_runtime',
                  engine: 'native',
                  deployment: '@Vela-2.0-4B/auto',
                },
              },
            ],
          },
        },
      }),
    )

    await page.goto('/dashboard')
    const mode = page.getByTestId('dashboard-serving-mode')
    await expect(mode.getByText(scenario.label, { exact: true })).toBeVisible()
    if (scenario.mode === 'router') {
      await expect(mode).toContainText('Routes requests through configured recipes.')
    } else if (scenario.mode === 'engine') {
      await expect(mode).toContainText('Serves model inference APIs directly.')
    }
  })
}

test('homepage makes serving mode unavailable when the status request fails', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: {} }))
  await page.route('**/api/status', (route) => route.fulfill({ status: 503, body: 'Unavailable' }))
  await page.goto('/dashboard')
  await expect(page.getByTestId('dashboard-serving-mode')).toContainText('Mode unavailable')
})
