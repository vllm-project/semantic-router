import { expect, test } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const decisions = ['chosen', 'rejected'].map((name, index) => ({
  name,
  priority: 100 - index,
  rules: { operator: 'OR', conditions: [{ type: 'keyword', name: 'greeting' }] },
  modelRefs: [{ model: 'backend-a' }, { model: 'backend-b' }],
}))
const config = {
  version: 'v0.3',
  providers: { models: [{ name: 'backend-a' }, { name: 'backend-b' }] },
  routing: { decisions: [] },
  entrypoints: [{ model_names: ['public/balanced'], recipe: 'balanced' }],
  recipes: [
    {
      name: 'balanced',
      routing: {
        signals: { keywords: [{ name: 'greeting', operator: 'OR', keywords: ['hello'] }] },
        decisions,
      },
    },
    { name: 'unbound', routing: { decisions: [] } },
  ],
}
const success = {
  query: 'hello',
  mode: 'dry-run',
  isAccurate: true,
  requestedModel: 'public/balanced',
  recipe: 'balanced',
  algorithm: 'static',
  matchedSignals: [],
  matchedDecision: 'chosen',
  matchedModels: ['backend-b'],
  selectedModel: 'backend-b',
  recommendedModels: ['backend-a', 'backend-b'],
  selectionStatus: 'selected',
  highlightedPath: ['client', 'decision-chosen', 'model-backend-b'],
  evalTrace: [
    {
      decision_name: 'chosen',
      state: 'true',
      matched: true,
      root_trace: {
        node_type: 'OR',
        state: 'true',
        children: [
          {
            node_type: 'leaf',
            signal_type: 'keyword',
            signal_name: 'greeting',
            state: 'true',
            matched: true,
            confidence: 1,
          },
        ],
      },
    },
    {
      decision_name: 'rejected',
      state: 'false',
      matched: false,
      root_trace: {
        node_type: 'leaf',
        signal_type: 'keyword',
        signal_name: 'greeting',
        state: 'false',
        matched: false,
        confidence: 0,
      },
    },
  ],
}

test.beforeEach(async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: config }))
  await page.route('**/api/router/config/global', (route) => route.fulfill({ json: {} }))
  await page.goto('/topology?scope=balanced')
  await expect(page.getByLabel('Entrypoint / recipe')).toHaveValue('balanced')
})

test('forwards request context and displays the exact decision trace and selected model', async ({
  page,
}) => {
  let body: Record<string, unknown> | undefined
  await page.route('**/api/topology/test-query', async (route) => {
    body = route.request().postDataJSON()
    await route.fulfill({ json: success })
  })
  const context = {
    messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
    metadata: { tenant: 'example' },
    tools: [{ type: 'function', function: { name: 'lookup' } }],
  }
  await page.getByText('Request context (optional)', { exact: true }).click()
  await page.getByLabel('Messages, metadata and tools (JSON)').fill(JSON.stringify(context))
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  const trace = page.getByRole('region', { name: 'Decision trace' })
  await expect(trace).toContainText('chosen: Match')
  await expect(trace).toContainText('rejected: No match')
  await expect(trace).toContainText('keyword:greeting')
  await expect(page.getByText('Algorithm: static', { exact: true })).toBeVisible()
  await expect(page.getByText('Selected model:', { exact: true }).locator('..')).toContainText(
    'backend-b',
  )
  await expect(page.getByText('Candidate pool:', { exact: true }).locator('..')).toContainText(
    'backend-a',
  )
  expect(body).toEqual({
    ...context,
    query: '',
    mode: 'dry-run',
    model: 'public/balanced',
    recipe: 'balanced',
  })
  await expect(
    page.locator('[data-id="decision-chosen"] [class*="decisionNode"]').first(),
  ).toHaveClass(/highlighted/)
  await expect(
    page.locator('[data-id="decision-rejected"] [class*="decisionNode"]').first(),
  ).not.toHaveClass(/highlighted/)
})

test('shows classifier timeout and unknown policy without a false zero-percent match', async ({
  page,
}) => {
  await page.route('**/api/topology/test-query', (route) =>
    route.fulfill({
      status: 503,
      json: {
        ...success,
        isAccurate: false,
        matchedDecision: null,
        matchedModels: [],
        selectedModel: '',
        selectionStatus: 'unavailable',
        recommendedModels: [],
        highlightedPath: ['client'],
        warning: 'decision unresolved',
        decisionError: 'decision unresolved',
        signalErrors: { 'classifier:intent': 'classifier timed out' },
        appliedUnknownPolicies: { chosen: 'fail_request' },
        evalTrace: [
          {
            decision_name: 'chosen',
            state: 'unknown',
            matched: false,
            confidence: 0,
            on_unknown: 'fail_request',
            root_trace: {
              node_type: 'leaf',
              signal_type: 'classifier',
              signal_name: 'intent',
              state: 'unknown',
              matched: false,
              confidence: 0,
              signal_error: 'classifier timed out',
            },
          },
        ],
      },
    }),
  )
  await page.getByPlaceholder('Message...').fill('hello')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  const trace = page.getByRole('region', { name: 'Decision trace' })
  await expect(trace).toContainText('chosen: Unknown')
  await expect(trace).toContainText('Unknown policy: fail_request')
  await expect(trace).toContainText('Error: classifier timed out')
  await expect(trace).not.toContainText('0%')
  await expect(trace).not.toContainText('No match')
})

test('blocks unbound recipes and invalid context without sending a preview', async ({ page }) => {
  const requests: string[] = []
  page.on('request', (request) => {
    if (request.url().includes('/api/topology/test-query')) requests.push(request.url())
  })
  await page.getByLabel('Entrypoint / recipe').selectOption('unbound')
  await page.getByPlaceholder('Message...').fill('hello')
  await expect(page.getByRole('status')).toContainText('no Chat entrypoint')
  await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeDisabled()
  await page.getByPlaceholder('Message...').press('Meta+Enter')
  await page.getByLabel('Entrypoint / recipe').selectOption('balanced')
  await page.getByText('Request context (optional)', { exact: true }).click()
  await page.getByLabel('Messages, metadata and tools (JSON)').fill('{"model":"other"}')
  await expect(page.getByRole('status')).toContainText('Unsupported request context field: model')
  await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeDisabled()
  expect(requests).toEqual([])
})

test('discards an in-flight response after changing scope', async ({ page }) => {
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  let completed!: () => void
  const handled = new Promise<void>((resolve) => {
    completed = resolve
  })
  await page.route('**/api/topology/test-query', async (route) => {
    await gate
    await route.fulfill({ json: success })
    completed()
  })
  await page.getByPlaceholder('Message...').fill('hello')
  const requested = page.waitForRequest('**/api/topology/test-query')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await requested
  await page.getByLabel('Entrypoint / recipe').selectOption('unbound')
  release()
  await handled
  await expect(page.getByRole('region', { name: 'Decision trace' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeDisabled()
  await page.getByLabel('Entrypoint / recipe').selectOption('balanced')
  await expect(page.getByRole('region', { name: 'Decision trace' })).toHaveCount(0)
})

test('clears a completed result when the request changes', async ({ page }) => {
  await page.route('**/api/topology/test-query', (route) => route.fulfill({ json: success }))
  await page.getByPlaceholder('Message...').fill('hello')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.getByRole('region', { name: 'Decision trace' })).toBeVisible()
  await page.getByPlaceholder('Message...').fill('different request')
  await expect(page.getByRole('region', { name: 'Decision trace' })).toHaveCount(0)
})

test('keeps the expanded request context usable on a narrow screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.route('**/api/topology/test-query', (route) => route.fulfill({ json: success }))
  await page.getByText('Request context (optional)', { exact: true }).click()
  await page
    .getByLabel('Messages, metadata and tools (JSON)')
    .fill('{"metadata":{"tenant":"example"}}')
  await page.getByPlaceholder('Message...').fill('hello')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.getByRole('region', { name: 'Decision trace' })).toBeVisible()
})
