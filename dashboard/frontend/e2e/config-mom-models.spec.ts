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

const modelCards = models.map((model) => ({
  id: model.id,
  name: model.name,
  card: {
    aliases: model.aliases,
    capabilities: model.capabilities,
    loras: model.loras,
    tags: [],
  },
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

const longRecipe = {
  ...recipe,
  id: 'recipe_long',
  name: 'long-form',
  description: 'A complete recipe with a long decision list',
  decisions: Array.from({ length: 24 }, (_, index) => ({
    id: `decision_${index + 1}`,
    name: `Decision ${index + 1}`,
    dispatchCardinality: 'single',
  })),
  document: {
    ...recipe.document,
    decisions: Array.from({ length: 24 }, (_, index) => ({
      id: `decision_${index + 1}`,
      name: `Decision ${index + 1}`,
      rules: {},
    })),
  },
}

const entrypoint = {
  id: 'entrypoint_balanced',
  name: 'balanced',
  status: 'active',
  revision: 2,
  entrypointRevision: 1,
  aliases: ['vllm-sr/balanced'],
  recipeIds: [recipe.id],
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
  recipeFixture = recipe,
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
  await page.route('**/api/router/management/v1/routing/model-cards?*', (route) => {
    reads.push(route.request().url())
    return fulfill(route, { data: modelCards, page: { hasMore: false, pageSize: 100 } })
  })
  await page.route('**/api/router/management/v1/routing/recipes?*', (route) => {
    reads.push(route.request().url())
    return fulfill(route, { data: [recipeFixture], page: { hasMore: false, pageSize: 100 } })
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

test('key-scoped consumer sees only the owned read-only Models and topology projection', async ({
  page,
}) => {
  const keyId = '10000000-0000-4000-8000-000000000003'
  await mockAuthenticatedAppShell(page, {
    user: {
      id: 'consumer-user',
      email: 'consumer@example.com',
      name: 'Consumer',
      role: 'read',
      permissions: [],
    },
    managementPermissions: [
      'access_policy.read',
      'agent.read',
      'agent.use',
      'delegation.use',
      'key.read',
      'routing_context.read',
    ],
  })
  const globalReads: string[] = []
  await page.route('**/api/router/management/v1/routing/**', async (route) => {
    globalReads.push(route.request().url())
    await fulfill(route, { error: { code: 'forbidden', message: 'Forbidden' } }, 403)
  })
  await page.route('**/api/router/management/v1/api-keys/*/routing-catalog', async (route) => {
    await fulfill(route, {
      keyId,
      policyRevision: 1,
      policyDigest: 'a'.repeat(64),
      routingRevision: 2,
      routingDigest: 'b'.repeat(64),
      models: models.map((model) => ({
        id: model.id,
        revision: model.revision,
        name: model.name,
        aliases: model.aliases,
        capabilities: model.capabilities,
        loras: model.loras,
        tags: [],
        pricing: model.pricing,
      })),
      recipes: [
        {
          id: recipe.id,
          revision: recipe.revision,
          name: recipe.name,
          description: recipe.description,
          decisions: recipe.decisions,
        },
      ],
      entrypoints: [
        {
          id: topology.id,
          revision: topology.revision,
          name: topology.name,
          aliases: topology.aliases,
          rules: topology.rules,
        },
      ],
    })
  })

  await page.goto('/config/entrypoints-recipes')
  await page.getByRole('tab', { name: 'Models' }).click()
  await expect(page.getByText('2 models', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Create model' })).toHaveCount(0)
  await page.getByRole('button', { name: 'Open balanced' }).click()
  const detail = page.getByRole('dialog', { name: 'Topology for vllm-sr/balanced' })
  await expect(detail.getByText('local/fast', { exact: true }).first()).toBeVisible()
  await expect(detail.getByRole('button', { name: 'Edit' })).toHaveCount(0)
  await page.keyboard.press('Escape')

  await page.goto('/topology')
  await expect(page.getByText('Routing Topology', { exact: true })).toBeVisible()
  await expect(page.getByRole('textbox', { name: /Test query/i })).toHaveCount(0)
  expect(globalReads).toHaveLength(0)
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
  await expect(page.getByTestId('routing-access-status-only')).toBeVisible()
  await expect(page.getByTestId('status-availability')).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(reads).toHaveLength(0)
})

test('routing manager creates a complete per-decision assignment', async ({ page }) => {
  const { writes } = await mockRouting(page, ['routing.read', 'routing.manage'], 'admin')
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

test('a long decision list uses one scroll surface and keeps dialog actions reachable', async ({
  page,
}) => {
  await page.setViewportSize({ width: 900, height: 640 })
  await mockRouting(page, ['routing.read', 'routing.manage'], 'admin', longRecipe)
  await page.goto('/config/entrypoints-recipes')
  await page.getByRole('tab', { name: 'Models' }).click()
  await page.getByRole('button', { name: 'Create model' }).click()

  const dialog = page.getByRole('dialog', { name: 'Create a mixture' })
  const lastDecision = dialog.locator('details', { hasText: /^Decision 24/ })
  const assignments = lastDecision.locator('..')
  const decisionSection = assignments.locator('..')
  const dialogBody = decisionSection.locator('..')
  const createButton = dialog.getByRole('button', { name: 'Create mixture' })
  const footer = createButton.locator('xpath=ancestor::footer')

  await expect(dialog.getByText('24 decisions', { exact: true })).toBeVisible()
  await expect(createButton).toBeInViewport()
  await expect(footer).toBeInViewport()

  const layout = await Promise.all([
    dialogBody.evaluate((element) => ({
      clientHeight: element.clientHeight,
      overflowY: window.getComputedStyle(element).overflowY,
      scrollHeight: element.scrollHeight,
    })),
    assignments.evaluate((element) => ({
      clientHeight: element.clientHeight,
      overflowY: window.getComputedStyle(element).overflowY,
      scrollHeight: element.scrollHeight,
    })),
  ])
  expect(layout[0].scrollHeight).toBeGreaterThan(layout[0].clientHeight)
  expect(layout[0].overflowY).toBe('auto')
  expect(layout[1].scrollHeight).toBeLessThanOrEqual(layout[1].clientHeight + 1)
  expect(['auto', 'scroll']).not.toContain(layout[1].overflowY)

  await dialogBody.evaluate((element) => element.scrollTo({ top: element.scrollHeight }))
  await expect.poll(() => dialogBody.evaluate((element) => element.scrollTop)).toBeGreaterThan(0)
  await expect(lastDecision).toBeInViewport()
  await expect(createButton).toBeInViewport()
  await expect(footer).toBeInViewport()
  await createButton.click({ trial: true })
})
