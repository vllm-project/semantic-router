import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const gatewayBackend = {
  name: 'remote-gateway-primary',
  base_url: 'https://gateway.example.com/v1',
  provider: 'vllm',
  openai_compat_gateway: true,
  api_key_env: 'GATEWAY_API_KEY',
  weight: 100,
}

const config = {
  version: 'v0.3',
  listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
  providers: {
    defaults: { model: 'gateway-model' },
    models: [
      {
        name: 'gateway-model',
        provider_model_id: 'gateway-model',
        api_format: 'openai',
        backend_refs: [gatewayBackend],
      },
    ],
  },
  routing: {
    modelCards: [{ name: 'gateway-model', modality: 'text' }],
    signals: {},
    decisions: [],
  },
  global: {},
}

test('keeps the gateway response contract when a model is saved unchanged', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  const writes: (typeof config)[] = []
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: config }))
  await page.route('**/api/router/config/global', (route) => route.fulfill({ json: config.global }))
  await page.route('**/api/router/config/update', async (route) => {
    writes.push(route.request().postDataJSON())
    await route.fulfill({ json: { status: 'success' } })
  })

  await page.goto('/config/models')
  await page.getByRole('button', { name: 'Edit gateway-model', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Edit Model: gateway-model' })
  await dialog.getByRole('button', { name: 'Save', exact: true }).click()

  await expect(dialog).toBeHidden()
  await expect.poll(() => writes.length).toBe(1)
  expect(writes[0].providers.models[0].backend_refs).toEqual([gatewayBackend])
})
