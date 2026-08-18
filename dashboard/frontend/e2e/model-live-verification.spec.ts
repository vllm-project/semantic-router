import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

test('runs a real provider-model query and renders pending, verified, and retry states', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page, {
    settings: {
      readonlyMode: true,
      serverReadonly: true,
      runtimeConfigWritable: false,
    },
  })
  await page.route('**/api/router/config/all', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        version: 'v0.3',
        providers: {
          defaults: { default_model: 'logical-model' },
          models: [
            {
              name: 'logical-model',
              provider_model_id: 'provider/real-model',
              api_format: 'openai',
              backend_refs: [
                {
                  name: 'primary',
                  endpoint: 'provider.internal:8000',
                  protocol: 'http',
                },
              ],
            },
          ],
        },
        routing: {
          modelCards: [{ name: 'logical-model', description: 'Physical provider model' }],
          decisions: [],
        },
      }),
    })
  })
  await page.route('**/api/router/config/global', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' })
  })
  await page.route('**/api/recipe', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ managed: false }),
    })
  })

  let releaseFirstVerification!: () => void
  const firstVerificationReleased = new Promise<void>((resolve) => {
    releaseFirstVerification = resolve
  })
  let firstVerificationStarted!: () => void
  const firstVerificationRequest = new Promise<void>((resolve) => {
    firstVerificationStarted = resolve
  })
  const requestBodies: Array<Record<string, unknown>> = []
  await page.route('**/api/models/verify', async (route) => {
    requestBodies.push(route.request().postDataJSON() as Record<string, unknown>)
    if (requestBodies.length === 1) {
      firstVerificationStarted()
      await firstVerificationReleased
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          verified: true,
          model: 'logical-model',
          providerModel: 'provider/real-model',
          backend: 'logical-model/primary',
          provider: 'openai',
          summary: 'OK from provider',
          verifiedAt: '2026-08-15T05:06:07Z',
          latencyMs: 18,
        }),
      })
      return
    }
    await route.fulfill({
      status: 502,
      contentType: 'application/json',
      body: JSON.stringify({
        error: 'provider_rejected',
        message: 'Provider inference returned HTTP 503.',
      }),
    })
  })

  await page.goto('/config/models')
  const verifyButton = page.getByRole('button', {
    name: 'Verify logical-model with a real inference query',
  })
  await expect(verifyButton).toBeEnabled()
  await verifyButton.click()
  await firstVerificationRequest
  await expect(page.getByText('Sending test query')).toBeVisible()
  await expect(page.getByRole('button', { name: /Verifying… logical-model/ })).toBeDisabled()

  releaseFirstVerification()
  await expect(page.getByText('Live verified')).toBeVisible()
  await expect(page.getByText('OK from provider')).toBeVisible()
  await expect(page.getByText(/openai · 18 ms/)).toBeVisible()
  expect(requestBodies).toEqual([{ model: 'logical-model' }])

  await page
    .getByRole('button', { name: 'Verify again logical-model with a real inference query' })
    .click()
  await expect(page.getByText('Verification failed')).toBeVisible()
  await expect(page.getByText('Provider inference returned HTTP 503.')).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Retry logical-model with a real inference query' }),
  ).toBeEnabled()
  expect(requestBodies).toEqual([{ model: 'logical-model' }, { model: 'logical-model' }])
})
