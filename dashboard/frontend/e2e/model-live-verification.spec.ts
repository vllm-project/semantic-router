import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

test('lets an operator run a real provider-model query and renders pending, verified, and retry states', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page)
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
    name: 'Check logical-model with a real inference query',
  })
  await expect(verifyButton).toBeEnabled()
  await verifyButton.click()
  await firstVerificationRequest
  await expect(page.getByText('Checking', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: /Checking… logical-model/ })).toBeDisabled()

  releaseFirstVerification()
  await expect(page.getByText('Live', { exact: true })).toBeVisible()
  expect(requestBodies).toEqual([{ model: 'logical-model' }])

  await page
    .getByRole('button', { name: 'Check again logical-model with a real inference query' })
    .click()
  await expect(page.getByText('Unavailable', { exact: true })).toBeVisible()
  await expect(page.getByText('Provider inference returned HTTP 503.')).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Check again logical-model with a real inference query' }),
  ).toBeEnabled()
  expect(requestBodies).toEqual([{ model: 'logical-model' }, { model: 'logical-model' }])
})
test('keeps the Pricing column horizontally stable while the Live status changes', async ({
  page,
}) => {
  // Wide enough that the Models table (760px min-width, plus selection/expand/actions
  // columns) never needs horizontal scroll. Otherwise Playwright's own scroll-into-view
  // before the verify button's click moves the whole table, which would swamp the
  // sub-column-width reflow this test is isolating.
  await page.setViewportSize({ width: 1600, height: 1000 })
  await mockAuthenticatedAppShell(page)
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
              // Pricing must be present for this test: an unpriced model renders a static
              // "N/A" cell that can't reflow, which would hide the bug this test guards.
              pricing: { currency: 'USD', prompt_per_1m: 3.5 },
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
  const pricingCell = page.getByRole('cell', { name: '$3.50 / 1M' })
  const verifyButton = page.getByRole('button', {
    name: 'Check logical-model with a real inference query',
  })
  await expect(verifyButton).toBeEnabled()

  // idle: baseline position for the column immediately to the right of Live.
  const idleX = (await pricingCell.boundingBox())?.x
  expect(idleX).toBeDefined()

  await verifyButton.click()
  await firstVerificationRequest
  await expect(page.getByRole('button', { name: /Checking… logical-model/ })).toBeDisabled()

  // pending: the button label ("Checking…") is the widest of the three normal states.
  const pendingX = (await pricingCell.boundingBox())?.x
  expect(pendingX).toBe(idleX)

  releaseFirstVerification()
  await expect(page.getByText('Live', { exact: true })).toBeVisible()

  // verified: label shrinks back to "Check again" — must not un-shift the column either.
  const verifiedX = (await pricingCell.boundingBox())?.x
  expect(verifiedX).toBe(idleX)

  await page
    .getByRole('button', { name: 'Check again logical-model with a real inference query' })
    .click()
  await expect(page.getByText('Unavailable', { exact: true })).toBeVisible()
  await expect(page.getByText('Provider inference returned HTTP 503.')).toBeVisible()

  // failed: an inline error message is inserted into the Live cell alongside the dot/button.
  const failedX = (await pricingCell.boundingBox())?.x
  expect(failedX).toBe(idleX)
})
