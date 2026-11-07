import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

// A native alias that serves a shared model_runtime deployment, as in config/config.yaml.
const nativeAlias = { name: 'native-kai', api_format: 'systemone', deployment: 'decision-kai' }

const config = {
  version: 'v0.3',
  listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
  providers: {
    defaults: { model: 'native-kai' },
    models: [nativeAlias],
  },
  routing: { modelCards: [], signals: {}, decisions: [] },
  global: {},
}

test('saves a deployment-bound native alias unchanged', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  const writes: (typeof config)[] = []
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: config }))
  await page.route('**/api/router/config/global', (route) => route.fulfill({ json: config.global }))
  await page.route('**/api/router/config/update', async (route) => {
    writes.push(route.request().postDataJSON())
    await route.fulfill({ json: { status: 'success' } })
  })

  await page.goto('/config/models')
  await page.getByRole('button', { name: 'Edit native-kai', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Edit Model: native-kai' })
  await dialog.getByRole('button', { name: 'Save', exact: true }).click()

  await expect(dialog).toBeHidden()
  await expect.poll(() => writes.length).toBe(1)
  const saved = writes[0].providers.models[0]
  expect(saved).toMatchObject(nativeAlias)
  expect(saved).not.toHaveProperty('provider_model_id')
  expect(saved).not.toHaveProperty('external_model_ids')
})
