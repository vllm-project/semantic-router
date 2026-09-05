import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const config = {
  version: 'v0.3',
  providers: {
    defaults: { default_model: 'gpt-4' },
    models: [{ name: 'gpt-4' }, { name: 'gpt-4-mini' }],
  },
  routing: {
    modelCards: [{ name: 'gpt-4' }, { name: 'gpt-4-mini' }],
    decisions: [],
  },
  entrypoints: [{ model_names: ['vllm-sr/balanced'], recipe: 'balanced' }],
  recipes: [
    {
      name: 'balanced',
      description: 'Balanced objective',
      routing: {
        signals: {
          keywords: [
            { name: 'coding-keyword', operator: 'OR', keywords: ['debug'], case_sensitive: false },
          ],
          domains: [{ name: 'code' }],
        },
        decisions: [
          {
            name: 'coding_decision',
            description: 'Coding route',
            priority: 100,
            rules: {
              operator: 'OR',
              conditions: [
                { type: 'keyword', name: 'coding-keyword' },
                { type: 'domain', name: 'code' },
              ],
            },
            modelRefs: [
              { model: 'gpt-4', use_reasoning: false },
              { model: 'gpt-4-mini', use_reasoning: false },
            ],
          },
        ],
      },
    },
  ],
}

const traceEvalResponse = {
  query: 'help me debug this function',
  mode: 'dry-run',
  requestedModel: 'vllm-sr/balanced',
  recipe: 'balanced',
  matchedSignals: [
    { type: 'keyword', name: 'coding-keyword', confidence: 1, reason: 'Keyword rule matched' },
  ],
  matchedDecision: 'coding_decision',
  algorithm: 'priority',
  matchedModels: ['gpt-4', 'gpt-4-mini'],
  highlightedPath: [
    'client',
    'decision-coding_decision',
    'signal-keyword-coding-keyword',
    'model-gpt-4',
    'model-gpt-4-mini',
  ],
  isAccurate: true,
  evalTrace: [
    {
      decision_name: 'coding_decision',
      matched: true,
      confidence: 0.95,
      root_trace: {
        node_type: 'OR',
        matched: true,
        confidence: 0.95,
        children: [
          {
            node_type: 'leaf',
            signal_type: 'keyword',
            signal_name: 'coding-keyword',
            matched: true,
            confidence: 1,
          },
          {
            node_type: 'leaf',
            signal_type: 'domain',
            signal_name: 'code',
            matched: false,
            confidence: 0,
          },
        ],
      },
    },
  ],
  evaluatedRules: [],
  routingLatency: 12,
}

test.beforeEach(async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(config),
    })
  })
  await page.route('**/api/router/config/global', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) })
  })
  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        overall: 'healthy',
        deployment_type: 'local',
        services: [
          { name: 'router', status: 'healthy', healthy: true },
          { name: 'envoy', status: 'healthy', healthy: true },
          { name: 'dashboard', status: 'healthy', healthy: true },
        ],
        models: { models: [], summary: { loaded_models: 0, total_models: 2 } },
      }),
    })
  })
  await page.route('**/api/topology/test-query', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(traceEvalResponse),
    })
  })
})

test('topology test-query renders the exact recipe-scoped eval trace', async ({ page }) => {
  await page.goto('/topology')

  // Scope selection: the balanced recipe's entrypoint graph is shown.
  const scope = page.getByLabel('Entrypoint / recipe')
  await expect(scope).toHaveValue('balanced')
  await expect(page.getByTestId('rf__node-decision-coding_decision')).toBeVisible()

  // Query submission.
  await page.getByPlaceholder('Message...').fill('help me debug this function')
  await page.getByRole('button', { name: 'Send' }).click()

  // Result card: Candidate pool (not "Model"), full candidate list, Algorithm shown separately.
  await expect(page.getByText('Candidate pool (2):')).toBeVisible()
  const candidateBadges = page.locator('[class*="candidateBadge"]')
  await expect(candidateBadges).toHaveCount(2)
  await expect(candidateBadges.filter({ hasText: 'gpt-4-mini' })).toBeVisible()
  await expect(candidateBadges.filter({ hasText: /^gpt-4$/ })).toBeVisible()
  await expect(page.getByText('Algorithm:')).toBeVisible()
  await expect(page.getByText('priority')).toBeVisible()

  // Trace visibility: the recursive decision trace renders nested operators and leaves.
  await expect(page.getByText('Decision trace:')).toBeVisible()
  await expect(
    page.locator('[class*="decisionName"]', { hasText: 'coding_decision' }),
  ).toBeVisible()
  await expect(page.locator('[class*="nodeOperator"]', { hasText: 'OR' })).toBeVisible()
  await expect(
    page.locator('[class*="nodeLabel"]', { hasText: 'keyword(coding-keyword)' }),
  ).toBeVisible()
  await expect(page.locator('[class*="nodeLabel"]', { hasText: 'domain(code)' })).toBeVisible()

  // Decision highlighting: the matched decision node is part of the highlighted path.
  await expect(page.getByTestId('rf__node-decision-coding_decision')).toBeVisible()
})

test('a recipe with no entrypoint disables the send action instead of silently testing the default recipe', async ({
  page,
}) => {
  const noEntrypointConfig = {
    ...config,
    entrypoints: [],
  }
  await page.route('**/api/router/config/all', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(noEntrypointConfig),
    })
  })

  await page.goto('/topology?scope=balanced')

  await page.getByPlaceholder('Message...').fill('help me debug this function')
  await expect(page.getByRole('button', { name: 'Send' })).toBeDisabled()
  await expect(page.getByText(/has no entrypoint/)).toBeVisible()
})
