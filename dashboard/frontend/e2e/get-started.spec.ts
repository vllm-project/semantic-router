import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const config = {
  version: 'v0.5',
  listeners: [{ name: 'http-8899', address: '0.0.0.0', port: 8899 }],
  providers: {
    defaults: { model: 'qwen3-8b' },
    models: [{ name: 'qwen3-8b', backend_refs: [{ name: 'local', provider: 'vllm' }] }],
  },
  entrypoints: [
    { model_names: ['vllm-sr/balanced'], recipe: 'default', default: true },
    { model_names: ['vllm-sr/private'], recipe: 'privacy' },
    { api: 'systemone', model_names: ['vllm-sr/auto'], recipe: 'decide' },
  ],
  recipes: [
    {
      name: 'default',
      description: 'Balanced routing for everyday traffic',
      signals: { keywords: [{ name: 'code' }] },
      decisions: [{ name: 'balanced-route', modelRefs: [{ model: 'qwen3-8b' }] }],
    },
    {
      name: 'privacy',
      decisions: [{ name: 'private-route', modelRefs: [{ model: 'qwen3-8b' }] }],
    },
    { name: 'decide', decisions: [] },
  ],
}

const healthyStatus = {
  overall: 'healthy',
  serving_mode: 'router',
  deployment_type: 'docker',
  version: 'v0.5.0',
  services: [{ name: 'Router', status: 'running', healthy: true }],
  models: {
    models: [
      {
        name: 'decision_model',
        type: 'decision',
        loaded: true,
        state: 'ready',
        metadata: { provider: 'model_runtime', deployment: 'primary' },
      },
    ],
  },
}

async function mockGetStarted(
  page: Page,
  { replayTotal = 0, configStatus = 200 }: { replayTotal?: number; configStatus?: number } = {},
) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) =>
    configStatus === 200
      ? route.fulfill({ json: config })
      : route.fulfill({ status: configStatus, body: 'forbidden' }),
  )
  await page.route('**/api/status', (route) => route.fulfill({ json: healthyStatus }))
  await page.route('**/api/router/api/v1/observability/replays?**', (route) =>
    route.fulfill({
      json: {
        object: 'list',
        count: replayTotal ? 1 : 0,
        total: replayTotal,
        data: replayTotal
          ? [
              {
                id: 'replay-1',
                timestamp: new Date().toISOString(),
                turn_index: 0,
                decision_tier: 0,
                decision_priority: 100,
                signals: {},
                original_model: 'vllm-sr/balanced',
                decision: 'balanced_route',
                selected_model: 'qwen3-8b',
                duration_ms: 420,
              },
            ]
          : [],
      },
    }),
  )
}

test('shows the endpoint, public models and client snippets from the configuration', async ({
  page,
}) => {
  await mockGetStarted(page)
  await page.goto('/get-started')

  await expect(page.getByRole('heading', { name: 'Get started', level: 1 })).toBeVisible()
  const balanced = page.getByRole('button', { name: /vllm-sr\/balanced/ })
  await expect(balanced).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByText('http://localhost:8899/v1').first()).toBeVisible()
  await expect(page.getByText('http-8899', { exact: true })).toBeVisible()
  await expect(page.getByText('"model": "vllm-sr/balanced"').first()).toBeVisible()

  await page.getByRole('button', { name: /vllm-sr\/private/ }).click()
  await expect(page).toHaveURL(/model=vllm-sr%2Fprivate/)
  await expect(page.getByText('"model": "vllm-sr/private"').first()).toBeVisible()

  await expect(page.getByText('vllm-sr/auto', { exact: true })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Try System One' })).toHaveAttribute(
    'href',
    '/models/decision?tab=playground',
  )

  const agents = page.getByRole('tablist').filter({ hasText: 'DeepSeek Harness' })
  await expect(agents.getByRole('img', { name: 'Claude Code' })).toBeVisible()
  await expect(page.getByText('vllm-sr connect claude-code \\').first()).toBeVisible()
  await expect(page.getByText('--model vllm-sr/private').first()).toBeVisible()

  await agents.getByRole('tab', { name: /Codex/ }).click()
  await expect(page.getByText('codex --profile vllm-sr').first()).toBeVisible()
  await expect(page).toHaveURL(/agent=codex/)

  await agents.getByRole('tab', { name: /OpenCode/ }).click()
  await expect(page.getByText('"npm": "@ai-sdk/openai-compatible",').first()).toBeVisible()

  await agents.getByRole('tab', { name: /DeepSeek Harness/ }).click()
  await expect(page.getByText('~/.dsh/profiles/<profile>/cordis.patch.yml')).toBeVisible()
  await expect(page.getByText('baseURL: http://localhost:8899/v1').first()).toBeVisible()
  await expect(page.getByText('model: vllm-sr/private').first()).toBeVisible()
})

test('reflects live state in the checklist and routes a test request', async ({ page }) => {
  await mockGetStarted(page)
  let testRequest: Record<string, unknown> | null = null
  await page.route('**/api/router/v1/chat/completions', async (route) => {
    testRequest = route.request().postDataJSON()
    await route.fulfill({
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'x-vsr-selected-model': 'qwen3-8b',
        'x-vsr-selected-decision': 'computer_science_route',
        'x-vsr-replay-id': 'replay-9',
        'x-vsr-trace': 'signals=("domain:computer science")',
      },
      body: JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'Hi there!' } }] }),
    })
  })
  await page.goto('/get-started')

  const checklist = page.getByTestId('start-checklist')
  await expect(checklist.locator('[data-step="router"]')).toHaveAttribute('data-state', 'done')
  await expect(checklist.locator('[data-step="backend"]')).toHaveAttribute('data-state', 'done')
  await expect(checklist.locator('[data-step="decision-model"]')).toHaveAttribute(
    'data-state',
    'done',
  )
  await expect(checklist.locator('[data-step="first-request"]')).toHaveAttribute(
    'data-state',
    'todo',
  )

  const watch = page.getByTestId('get-started-watch')
  await expect(watch).toContainText('Listening for your first request')
  await watch.getByRole('button', { name: 'Send test request' }).click()
  await expect(watch).toContainText('Routed in')
  await expect(watch).toContainText('Computer Science')
  await expect(watch).toContainText('qwen3-8b')
  await expect(watch).toContainText('Hi there!')
  await expect(watch.getByRole('link', { name: 'Open the full trace' })).toHaveAttribute(
    'href',
    '/observe/requests/replay-9',
  )
  expect(testRequest).toMatchObject({ model: 'vllm-sr/balanced', stream: false })
})

test('shows a failed test request with what to do next', async ({ page }) => {
  await mockGetStarted(page, { replayTotal: 3 })
  await page.route('**/api/router/v1/chat/completions', (route) =>
    route.fulfill({ status: 503, body: 'upstream connect error' }),
  )
  await page.goto('/get-started')

  const watch = page.getByTestId('get-started-watch')
  await expect(watch).toContainText('Last request')
  await expect(
    page.getByTestId('start-checklist').locator('[data-step="first-request"]'),
  ).toHaveAttribute('data-state', 'done')
  await watch.getByRole('button', { name: 'Send test request' }).click()
  await expect(watch).toContainText('The test request failed')
  await expect(watch).toContainText('The model service is temporarily unavailable')
  await expect(watch).toContainText('upstream connect error')
})

test('keeps a typed gateway address and falls back without the configuration', async ({ page }) => {
  await mockGetStarted(page, { configStatus: 403 })
  await page.route('**/api/router/v1/models*', (route) =>
    route.fulfill({
      json: {
        object: 'list',
        data: [
          {
            id: 'vllm-sr/live',
            routing: { resolution: 'virtual', selectable: true, default_route: true },
          },
        ],
      },
    }),
  )
  await page.goto('/get-started')

  await expect(page.getByText('The router configuration could not load')).toBeVisible()
  await expect(page.getByRole('button', { name: /vllm-sr\/live/ })).toHaveAttribute(
    'aria-pressed',
    'true',
  )
  // setup reports the listener port when the configuration is out of reach
  await expect(page.getByText('http://localhost:8700/v1').first()).toBeVisible()

  await page.getByRole('button', { name: 'Behind another address?' }).click()
  const address = page.getByRole('textbox', { name: 'Gateway base URL' })
  await address.fill('https://gateway.example.com/v1/')
  await page.getByRole('button', { name: 'Use', exact: true }).click()
  await expect(page.getByText('https://gateway.example.com/v1').first()).toBeVisible()
  await expect(
    page.getByText('curl -sS -i https://gateway.example.com/v1/chat/completions \\').first(),
  ).toBeVisible()

  await page.reload({ waitUntil: 'domcontentloaded' })
  await expect(page.getByText('https://gateway.example.com/v1').first()).toBeVisible()
  await page.getByRole('button', { name: 'Use the listener' }).click()
  await expect(page.getByText('http://localhost:8700/v1').first()).toBeVisible()
})
