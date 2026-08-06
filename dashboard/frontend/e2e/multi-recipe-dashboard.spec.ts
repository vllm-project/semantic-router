import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const config = {
  version: 'v0.3',
  providers: {
    defaults: { default_model: 'model-a' },
    models: [{ name: 'model-a' }],
  },
  routing: {
    modelCards: [{ name: 'model-a' }],
    decisions: [],
  },
  entrypoints: [
    { model_names: ['vllm-sr/balanced'], recipe: 'balanced' },
    { model_names: ['vllm-sr/private'], recipe: 'privacy' },
  ],
  recipes: [
    {
      name: 'balanced',
      description: 'Balanced objective',
      routing: {
        signals: {
          keywords: [
            {
              name: 'balanced-keyword',
              operator: 'OR',
              keywords: ['balanced'],
              case_sensitive: false,
            },
          ],
        },
        projections: {
          scores: [{ name: 'balanced-score', method: 'weighted_sum', inputs: [] }],
          mappings: [
            {
              name: 'balanced-map',
              source: 'balanced-score',
              method: 'threshold_bands',
              outputs: [{ name: 'balanced-standard' }],
            },
          ],
        },
        decisions: [
          {
            name: 'balanced-route',
            description: 'Balanced route',
            priority: 100,
            rules: {
              operator: 'AND',
              conditions: [{ type: 'keyword', name: 'balanced-keyword' }],
            },
            modelRefs: [{ model: 'model-a', use_reasoning: false }],
          },
        ],
      },
    },
    {
      name: 'privacy',
      description: 'Private objective',
      routing: {
        projections: {
          scores: [{ name: 'private-score', method: 'weighted_sum', inputs: [] }],
          mappings: [],
        },
        signals: {
          pii: [{ name: 'private-pii', threshold: 0.8 }],
        },
        decisions: [
          {
            name: 'private-route',
            description: 'Private route',
            priority: 100,
            rules: {
              operator: 'AND',
              conditions: [{ type: 'pii', name: 'private-pii' }],
            },
            modelRefs: [{ model: 'model-a', use_reasoning: false }],
          },
        ],
      },
    },
  ],
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
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
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
        models: { models: [], summary: { loaded_models: 0, total_models: 1 } },
      }),
    })
  })
})

test('dashboard and managers expose recipe-owned routing state', async ({ page }) => {
  await page.goto('/dashboard')

  await expect(
    page.getByRole('button').filter({ hasText: 'Decisions' }).first(),
  ).toContainText('2')
  await expect(
    page.getByRole('button').filter({ hasText: 'Signals' }).first(),
  ).toContainText('2')
  await expect(page.getByText('balanced-route')).toBeVisible()
  await expect(page.getByText('private-route')).toBeVisible()

  await page.goto('/config/signals')
  const signalScope = page.getByLabel('Routing profile')
  await expect(signalScope).toHaveValue('balanced')
  await expect(page.getByText('balanced-keyword')).toBeVisible()
  await signalScope.selectOption('privacy')
  await expect(page.getByText('private-pii')).toBeVisible()
  await expect(page.getByText('balanced-keyword')).toHaveCount(0)

  await page.goto('/config/projections')
  const projectionScope = page.getByLabel('Routing profile')
  await expect(projectionScope).toHaveValue('balanced')
  await expect(page.getByText('balanced-score').first()).toBeVisible()
  await projectionScope.selectOption('privacy')
  await expect(page.getByText('private-score').first()).toBeVisible()
  await expect(page.getByText('balanced-score')).toHaveCount(0)

  await page.goto('/config/decisions')
  const decisionScope = page.getByLabel('Routing profile')
  await expect(decisionScope).toHaveValue('balanced')
  await expect(page.getByText('balanced-route')).toBeVisible()
  await decisionScope.selectOption('privacy')
  await expect(page.getByText('private-route')).toBeVisible()
})

test('topology switches the complete graph and test model by entrypoint recipe', async ({
  page,
}) => {
  await page.goto('/topology')

  const scope = page.getByLabel('Entrypoint / recipe')
  await expect(scope).toHaveValue('balanced')
  await expect(page.getByTestId('rf__node-decision-balanced-route')).toBeVisible()
  await expect(
    page.getByTestId('rf__node-signal-group-keyword').getByText('balanced-keyword'),
  ).toBeVisible()
  await expect(
    page.getByTestId('rf__node-projection-group-balanced-map').getByText('balanced-standard'),
  ).toBeVisible()

  await scope.selectOption('privacy')
  await expect(page.getByTestId('rf__node-decision-private-route')).toBeVisible()
  await expect(
    page.getByTestId('rf__node-signal-group-pii').getByText('private-pii'),
  ).toBeVisible()
  await expect(page.getByTestId('rf__node-decision-balanced-route')).toHaveCount(0)
})
