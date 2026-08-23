import { expect, test, type Page, type Route } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const mediaType = 'application/vnd.vllm-semantic-router.management.v1+json'
const now = '2026-08-23T00:00:00Z'

const models = ['model_fast', 'model_frontier'].map((id) => ({
  id,
  name: id === 'model_fast' ? 'local/fast' : 'remote/frontier',
  status: 'active',
  revision: 1,
  modelRevision: 1,
  catalogRevision: `sha256:${'a'.repeat(64)}`,
  aliases: [],
  capabilities: ['text'],
  loras: [],
  execution: { maxRetries: 0, requestTimeout: '30s', streamTimeout: '60s' },
  pricing: {
    inputCostPerMillionTokens: null,
    outputCostPerMillionTokens: null,
    cacheReadCostPerMillionTokens: null,
    cacheWriteCostPerMillionTokens: null,
  },
  backends: [
    {
      providerId: 'private',
      providerModelId: id,
      credentialConfigured: false,
      weight: '1',
    },
  ],
  createdAt: now,
  updatedAt: now,
}))

const recipe = {
  id: 'recipe_balanced',
  name: 'balanced',
  description: 'Balanced test recipe',
  status: 'active',
  revision: 1,
  recipeRevision: 1,
  origin: 'custom',
  immutable: false,
  decisions: [
    { id: 'decision_simple', name: 'Simple', dispatchCardinality: 'single' },
    { id: 'decision_complex', name: 'Complex', dispatchCardinality: 'single' },
  ],
  document: {
    signals: {},
    projections: {},
    decisions: [
      { id: 'decision_simple', name: 'Simple', rules: {} },
      { id: 'decision_complex', name: 'Complex', rules: {} },
    ],
  },
  createdAt: now,
  updatedAt: now,
}

const entrypoint = {
  id: 'entrypoint_balanced',
  name: 'balanced',
  status: 'active',
  revision: 2,
  entrypointRevision: 1,
  aliases: ['vllm-sr/balanced'],
  ruleCount: 1,
  assignedModelCount: 2,
  createdAt: now,
  updatedAt: now,
}

const topology = {
  ...entrypoint,
  rules: [
    {
      id: 'rule_default',
      name: 'Default',
      recipeId: recipe.id,
      recipeRevision: 1,
      assignments: {
        decision_simple: {
          models: [
            {
              modelId: 'model_fast',
              modelRevision: 1,
              priority: 0,
              weight: '1',
            },
          ],
        },
        decision_complex: {
          models: [
            {
              modelId: 'model_frontier',
              modelRevision: 1,
              priority: 0,
              weight: '1',
            },
          ],
        },
      },
    },
  ],
}

const fulfill = (route: Route, payload: unknown, status = 200) =>
  route.fulfill({
    status,
    headers: { 'Content-Type': mediaType },
    body: JSON.stringify(payload),
  })

async function mockRouting(
  page: Page,
  permissions: string[],
  dashboardRole: 'admin' | 'write' | 'read' = 'read',
) {
  await mockAuthenticatedAppShell(page, {
    user: {
      id: 'mom-user',
      email: 'mom@example.com',
      name: 'MoM User',
      role: dashboardRole,
    },
    managementPermissions: permissions,
  })
  const writes: Array<{ url: string; body: unknown }> = []
  const reads: string[] = []
  await page.route('**/api/router/management/v1/routing/models?*', (route) => {
    reads.push(route.request().url())
    return fulfill(route, { data: models, page: { hasMore: false, pageSize: 100 } })
  })
  await page.route('**/api/router/management/v1/routing/recipes?*', (route) => {
    reads.push(route.request().url())
    return fulfill(route, { data: [recipe], page: { hasMore: false, pageSize: 100 } })
  })
  await page.route('**/api/router/management/v1/routing/entrypoints?*', (route) => {
    reads.push(route.request().url())
    return fulfill(route, {
      data: [entrypoint],
      page: { hasMore: false, pageSize: 100 },
    })
  })
  await page.route(
    '**/api/router/management/v1/routing/entrypoints/entrypoint_balanced?includeTopology=true',
    (route) => {
      reads.push(route.request().url())
      return fulfill(route, { data: topology })
    },
  )
  await page.route('**/api/router/management/v1/routing/entrypoints', async (route) => {
    writes.push({
      url: route.request().url(),
      body: route.request().postDataJSON(),
    })
    await fulfill(
      route,
      {
        resource: {
          kind: 'routing_entrypoint',
          id: 'entrypoint_new',
          revision: 1,
        },
        idempotency: { replayed: false },
      },
      201,
    )
  })
  return { writes, reads }
}

test('routing reader sees authorized topology without Dashboard config access', async ({
  page,
}) => {
  const { writes, reads } = await mockRouting(page, ['routing.read'])
  await page.goto('/config/entrypoints-recipes')
  await page.getByRole('tab', { name: 'Models' }).click()
  await expect(page.getByRole('button', { name: 'Create model' })).toHaveCount(0)
  await expect(page.getByText('2 models', { exact: true })).toBeVisible()
  await expect(page.getByText('Topology', { exact: true })).toBeVisible()
  expect(reads.some((url) => url.includes('includeTopology=true'))).toBe(false)
  await page.getByRole('button', { name: 'Open balanced' }).click()
  const dialog = page.getByRole('dialog', {
    name: 'Topology for vllm-sr/balanced',
  })
  await expect(dialog.getByText('local/fast', { exact: true }).first()).toBeVisible()
  await expect(dialog.getByText('remote/frontier', { exact: true }).first()).toBeVisible()
  expect(reads.filter((url) => url.includes('includeTopology=true'))).toHaveLength(1)
  await expect(dialog.getByRole('button', { name: 'Edit' })).toHaveCount(0)
  expect(writes).toHaveLength(0)
})

test('direct routing route denies a user without routing.read', async ({ page }) => {
  const { reads, writes } = await mockRouting(page, [])
  await page.goto('/config/entrypoints-recipes')
  await expect(page).toHaveURL(/\/dashboard$/)
  expect(reads).toHaveLength(0)
  expect(writes).toHaveLength(0)
})

test('a broken Management identity is surfaced instead of showing empty routing', async ({
  page,
}) => {
  const { reads } = await mockRouting(page, ['routing.read', 'routing.manage'])
  await page.route('**/api/router/management/v1/me', (route) =>
    fulfill(
      route,
      {
        error: {
          code: 'identity_link_missing',
          message: 'Dashboard identity is not linked to this namespace.',
        },
      },
      503,
    ),
  )
  await page.goto('/config/entrypoints-recipes')
  await expect(page).toHaveURL(/\/dashboard$/)
  await expect(page.getByRole('alert')).toContainText('Routing access unavailable')
  await expect(page.getByRole('alert')).toContainText(
    'Dashboard identity is not linked to this namespace.',
  )
  expect(reads).toHaveLength(0)
})

test('routing manager creates a complete per-decision assignment', async ({ page }) => {
  const { writes } = await mockRouting(
    page,
    ['routing.read', 'routing.manage'],
    'admin',
  )
  await page.goto('/config/entrypoints-recipes')
  await page.getByRole('tab', { name: 'Models' }).click()
  await page.getByRole('button', { name: 'Create model' }).click()
  const dialog = page.getByRole('dialog', { name: 'Create a mixture' })
  await dialog.getByRole('textbox', { name: 'Model name', exact: true }).fill('vllm-sr/new')
  await dialog.locator('details').nth(0).getByText('local/fast', { exact: true }).click()
  await dialog.locator('details').nth(1).getByText('remote/frontier', { exact: true }).click()
  await dialog.getByRole('button', { name: 'Create mixture' }).click()
  await expect.poll(() => writes.length).toBe(1)
  const body = writes[0].body as {
    rules: Array<{
      assignments: Record<string, { models: Array<{ modelId: string }> }>
    }>
  }
  expect(Object.keys(body.rules[0].assignments).sort()).toEqual([
    'decision_complex',
    'decision_simple',
  ])
  expect(body.rules[0].assignments.decision_simple.models[0].modelId).toBeTruthy()
})
