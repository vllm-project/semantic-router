import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

interface HealthRequest {
  models: string[]
  refresh: boolean
}

async function mockSingleModel(page: Page, pricing?: Record<string, unknown>) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        version: 'v0.5',
        providers: {
          defaults: { model: 'logical-model' },
          models: [
            {
              name: 'logical-model',
              provider_model_id: 'provider/real-model',
              api_format: 'openai',
              backend_refs: [
                { name: 'primary', endpoint: 'provider.internal:8000', protocol: 'http' },
              ],
              pricing,
              description: 'Physical provider model',
            },
          ],
        },
        recipes: [{ name: 'default', decisions: [] }],
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
  await page.route('**/api/router/api/v1/observability/replays/aggregate*', (route) =>
    route.fulfill({ status: 404, body: 'Router Replay is off' }),
  )
}

const healthy = {
  model: 'logical-model',
  status: 'healthy',
  latencyMs: 412,
  p95Ms: 640,
  recentMs: [380, 412, 640],
  endpoints: [{ name: 'primary', status: 'healthy', latencyMs: 412, httpStatus: 200 }],
}
const down = {
  model: 'logical-model',
  status: 'down',
  reason: 'The backend could not be reached.',
  recentMs: [],
  endpoints: [{ name: 'primary', status: 'down', reason: 'The backend could not be reached.' }],
}

/**
 * Answers polls with `polled` and on-demand checks with `checked`, holding a check until
 * released. Development mode runs effects twice, so polls are not counted in order.
 */
async function mockHealth(
  page: Page,
  polled: Record<string, unknown>,
  checked: Record<string, unknown>,
) {
  const requests: HealthRequest[] = []
  let release = () => {}
  let hold = false
  await page.route('**/api/models/health*', async (route) => {
    const url = new URL(route.request().url())
    requests.push({
      models: url.searchParams.getAll('model'),
      refresh: url.searchParams.has('refresh'),
    })
    const answer = url.searchParams.has('refresh') ? checked : polled
    if (hold) {
      await new Promise<void>((resolve) => {
        release = () => resolve()
      })
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        checkedAt: new Date().toISOString(),
        intervalSeconds: 30,
        models: [answer],
      }),
    })
  })
  return {
    requests,
    holdNext: () => {
      hold = true
    },
    releaseHeld: () => {
      hold = false
      release()
    },
  }
}

test('checks health on load and on demand, and keeps the real test request in the model sheet', async ({
  page,
}) => {
  await mockSingleModel(page)
  const health = await mockHealth(page, healthy, down)

  let releaseFirstVerification!: () => void
  const firstVerificationReleased = new Promise<void>((resolve) => {
    releaseFirstVerification = resolve
  })
  const requestBodies: Array<Record<string, unknown>> = []
  await page.route('**/api/models/verify', async (route) => {
    requestBodies.push(route.request().postDataJSON() as Record<string, unknown>)
    if (requestBodies.length === 1) {
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

  await page.goto('/models/backends')
  const row = page.getByRole('row').filter({ hasText: 'provider/real-model' })
  await expect(row.getByText('Healthy', { exact: true })).toBeVisible({ timeout: 30_000 })
  await expect(row.getByText('p95 640ms')).toBeVisible()
  expect(health.requests.every((request) => !request.refresh && !request.models.length)).toBe(true)
  const polls = health.requests.length
  // No row carries its own check button; checks live in the row menu and the toolbar.
  await expect(page.getByRole('button', { name: /real inference query|^Check$/ })).toHaveCount(0)

  await page.getByRole('button', { name: 'Actions for logical-model', exact: true }).click()
  await page.getByRole('menuitem', { name: 'Check health', exact: true }).click()
  await expect(row.getByText('Down', { exact: true })).toBeVisible()
  await expect(page.getByText('logical-model is down')).toBeVisible()
  expect(health.requests[polls]).toEqual({ models: ['logical-model'], refresh: true })

  await page.getByRole('button', { name: 'Check health', exact: true }).click()
  await expect(page.getByText('Checked 1 model', { exact: true })).toBeVisible()
  expect(health.requests[polls + 1]).toEqual({ models: [], refresh: true })

  // The sheet reports the same health and runs the real inference request on demand.
  await row.getByText('logical-model', { exact: true }).click()
  const sheet = page.getByRole('dialog', { name: 'logical-model' })
  await expect(sheet.getByText('The backend could not be reached.')).toBeVisible()
  await expect(sheet.getByRole('button', { name: 'Check health', exact: true })).toBeVisible()

  await sheet.getByRole('button', { name: 'Model actions' }).click()
  await page.getByRole('menuitem', { name: 'Send a test request' }).click()
  await expect.poll(() => requestBodies.length).toBe(1)
  releaseFirstVerification()
  await expect(sheet.getByText('Answered a test request')).toBeVisible()
  await expect(sheet.getByText('in 18 ms')).toBeVisible()
  expect(requestBodies).toEqual([{ model: 'logical-model' }])

  await sheet.getByRole('button', { name: 'Model actions' }).click()
  await page.getByRole('menuitem', { name: 'Send a test request' }).click()
  await expect(sheet.getByText('The test request failed')).toBeVisible()
  await expect(sheet.getByText('Provider inference returned HTTP 503.')).toBeVisible()
  expect(requestBodies).toEqual([{ model: 'logical-model' }, { model: 'logical-model' }])
})

test('keeps the Pricing column horizontally stable while health changes', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1000 })
  await mockSingleModel(page, { currency: 'USD', prompt_per_1m: 3.5, completion_per_1m: 7 })
  const health = await mockHealth(page, healthy, down)

  await page.goto('/models/backends')
  const row = page.getByRole('row').filter({ hasText: 'provider/real-model' })
  const pricingCell = row.getByRole('cell').filter({ hasText: /\$3\.50\s*\/\s*\$7\.00/ })
  await expect(row.getByText('Healthy', { exact: true })).toBeVisible({ timeout: 30_000 })
  const healthyX = (await pricingCell.boundingBox())?.x
  expect(healthyX).toBeDefined()

  health.holdNext()
  await page.getByRole('button', { name: 'Actions for logical-model', exact: true }).click()
  await page.getByRole('menuitem', { name: 'Check health', exact: true }).click()
  await expect(row.getByText('Checking…', { exact: true })).toBeVisible()
  expect((await pricingCell.boundingBox())?.x).toBe(healthyX)

  health.releaseHeld()
  await expect(row.getByText('Down', { exact: true })).toBeVisible()
  expect((await pricingCell.boundingBox())?.x).toBe(healthyX)
})
