import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const config = {
  version: 'v0.5',
  providers: {
    defaults: { model: 'model-a' },
    models: [{ name: 'model-a' }],
  },
  entrypoints: [
    { model_names: ['vllm-sr/balanced'], recipe: 'balanced' },
    { model_names: ['vllm-sr/private'], recipe: 'privacy' },
  ],
  recipes: [
    { name: 'default', decisions: [] },
    {
      name: 'balanced',
      description: 'Balanced objective',
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
    {
      name: 'privacy',
      description: 'Private objective',
      projections: {
        scores: [{ name: 'private-score', method: 'weighted_sum', inputs: [] }],
        mappings: [],
      },
      signals: {
        pii: [{ name: 'private-pii', predicate: { gte: 0.8 } }],
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
  ],
}

const builtInRecipes = {
  catalog_version: 'latest',
  bundles: [
    {
      name: 'mom-v1',
      sha256: 'sha256:aaaa',
      recipes: [
        {
          name: 'vault',
          description: 'Keep private traffic inside the deployment boundary.',
          decisions: [
            {
              name: 'guard',
              description: 'Contain prompt attacks.',
              algorithm: null,
              minimum_candidates: 0,
            },
            {
              name: 'private',
              description: 'Serve private requests.',
              algorithm: 'multi_factor',
              minimum_candidates: 1,
            },
          ],
        },
      ],
    },
  ],
}

let savedConfig: typeof config | undefined

test.beforeEach(async ({ page }) => {
  savedConfig = undefined
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/recipes/builtin', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(builtInRecipes),
    })
  })
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
  await page.route('**/api/router/config/update', async (route) => {
    savedConfig = route.request().postDataJSON() as typeof config
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ success: true }),
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

test('creates a public model without clipped aliases or assignments', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/routing/recipes')

  await expect(page.getByRole('tab', { name: /Public models/ })).toHaveAttribute(
    'aria-selected',
    'true',
  )
  await page.getByRole('button', { name: 'New public model', exact: true }).click()

  const dialog = page.getByRole('dialog', { name: 'New public model' })
  await expect(dialog).toBeVisible()
  await dialog.getByLabel('Model name').fill('vllm-sr/new-mixture')

  const aliasInput = dialog.getByLabel('Aliases')
  await expect(aliasInput).toHaveAttribute('placeholder', 'Add an alias')
  await aliasInput.fill('new-mixture')
  await aliasInput.press('Enter')
  await expect(dialog.getByText('new-mixture', { exact: true })).toBeVisible()

  const assignment = dialog.getByText('balanced-route', { exact: true }).first()
  const decisionTab = dialog.getByRole('tab', { name: /balanced-route/ })
  const assignmentPanel = dialog.getByRole('tabpanel', {
    name: 'balanced-route model assignment',
  })
  const aliasBox = await aliasInput.boundingBox()
  const assignmentBox = await assignment.boundingBox()
  expect(aliasBox).not.toBeNull()
  expect(assignmentBox).not.toBeNull()
  expect((aliasBox?.y ?? 0) + (aliasBox?.height ?? 0)).toBeLessThan(assignmentBox?.y ?? 0)
  expect(await dialog.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true)
  await expect(decisionTab).toHaveAttribute('aria-selected', 'true')
  await expect(assignmentPanel).toBeVisible()
  await expect(assignmentPanel).toContainText('1 selected')

  await expect(assignmentPanel.getByRole('checkbox')).toBeChecked()
  await dialog.getByRole('button', { name: 'Create public model' }).click()
  await expect(dialog).toBeHidden()

  await expect.poll(() => savedConfig).toBeDefined()
  expect(savedConfig?.entrypoints).toContainEqual({
    model_names: ['vllm-sr/new-mixture', 'new-mixture'],
    recipe: 'balanced',
  })
  expect(savedConfig?.entrypoints).not.toContainEqual({
    model_names: ['vllm-sr/auto'],
    recipe: 'default',
  })
  const balanced = savedConfig?.recipes.find((recipe) => recipe.name === 'balanced')
  expect(balanced?.decisions[0].modelRefs).toEqual([{ model: 'model-a', use_reasoning: false }])
})

test('publishes a built-in recipe from the config the packaged CLI composes', async ({ page }) => {
  let composeRequest: unknown
  const vault = { name: 'vault', description: 'Composed vault', decisions: [] }
  await page.route('**/api/recipes/builtin/compose', async (route) => {
    composeRequest = route.request().postDataJSON()
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...config,
        entrypoints: [
          ...config.entrypoints,
          { model_names: ['acme/vault', 'vault'], recipe: 'vault' },
        ],
        recipes: [...config.recipes, vault],
      }),
    })
  })
  await page.goto('/routing/recipes')
  await page.getByRole('tab', { name: /^Recipes/ }).click()
  await page.getByRole('button', { name: 'Actions for vault', exact: true }).click()
  await page.getByRole('menuitem', { name: 'Create public model' }).click()

  const dialog = page.getByRole('dialog', { name: 'New public model' })
  await expect(dialog.getByLabel('Recipe', { exact: true }).locator('option:checked')).toHaveText(
    'vault',
  )
  await dialog.getByLabel('Model name').fill('acme/vault')
  const aliasInput = dialog.getByLabel('Aliases')
  await aliasInput.fill('vault')
  await aliasInput.press('Enter')
  await expect(dialog.getByRole('tabpanel', { name: 'guard model assignment' })).toContainText(
    'No model needed',
  )
  await dialog.getByRole('tab', { name: /private/ }).click()
  await dialog
    .getByRole('tabpanel', { name: 'private model assignment' })
    .getByText('model-a', { exact: true })
    .click()
  await dialog.getByRole('button', { name: 'Create public model' }).click()
  await expect(dialog).toBeHidden()

  expect(composeRequest).toEqual({
    bundle: 'mom-v1',
    recipe: 'vault',
    bindings: { private: [{ model: 'model-a', use_reasoning: false }] },
    model_names: ['acme/vault', 'vault'],
  })
  await expect.poll(() => savedConfig).toBeDefined()
  expect(savedConfig?.entrypoints).toContainEqual({
    model_names: ['acme/vault', 'vault'],
    recipe: 'vault',
  })
  expect(savedConfig?.recipes).toContainEqual(vault)
})

test('dashboard and managers expose recipe-owned routing state', async ({ page }) => {
  await page.goto('/dashboard')

  await expect(page.getByTestId('home-count-decisions')).toContainText('2')
  await expect(page.getByTestId('home-count-signals')).toContainText('2')
  const routingMix = page.getByTestId('home-routing-mix')
  await expect(routingMix.getByText('balanced-route')).toBeVisible()
  await expect(routingMix.getByText('private-route')).toBeVisible()
  const publicModels = page.getByTestId('home-public-models')
  await expect(publicModels.getByRole('heading', { name: 'Public models' })).toBeVisible()
  await expect(publicModels).toContainText('3 recipes')
  await expect(page.getByRole('button', { name: 'View Default routing topology' })).toBeVisible()
  await page.getByRole('button', { name: 'View privacy topology' }).click()
  await expect(page).toHaveURL(/\/routing\/topology\?scope=privacy$/)

  await page.goto('/routing/signals')
  const signalScope = page.getByLabel('Recipe', { exact: true })
  await expect(signalScope).toHaveValue('balanced')
  await expect(page.getByRole('row', { name: /balanced-keyword/ })).toBeVisible()
  await signalScope.selectOption('privacy')
  await expect(page).toHaveURL(/recipe=privacy/)
  await expect(page.getByRole('row', { name: /private-pii/ })).toBeVisible()
  await expect(page.getByRole('row', { name: /balanced-keyword/ })).toHaveCount(0)

  await page.goto('/routing/projections')
  const projectionScope = page.getByLabel('Recipe', { exact: true })
  await expect(projectionScope).toHaveValue('balanced')
  await expect(page.getByText('balanced-score').first()).toBeVisible()
  await projectionScope.selectOption('privacy')
  await expect(page.getByText('private-score').first()).toBeVisible()
  await expect(page.getByText('balanced-score')).toHaveCount(0)

  await page.goto('/routing/decisions')
  const decisionScope = page.getByLabel('Recipe', { exact: true })
  await expect(decisionScope).toHaveValue('balanced')
  await expect(page.getByRole('row', { name: /balanced-route/ })).toBeVisible()
  await decisionScope.selectOption('privacy')
  await expect(page.getByRole('row', { name: /private-route/ })).toBeVisible()
})

test('topology switches the complete graph and test model by entrypoint recipe', async ({
  page,
}) => {
  await page.goto('/topology')

  const scope = page.getByLabel('Recipe', { exact: true })
  await expect(scope).toHaveValue('default')
  await expect(page.getByTestId('rf__node-decision-balanced-route')).toHaveCount(0)
  await expect(page.getByTestId('rf__node-decision-private-route')).toHaveCount(0)
  await scope.selectOption('balanced')
  await expect(scope).toHaveValue('balanced')
  await expect(page).toHaveURL(/\/routing\/topology\?scope=balanced$/)
  await expect(page.getByTestId('rf__node-decision-balanced-route')).toBeVisible()
  await expect(page.getByTestId('rf__node-signal-group-keyword')).toContainText('balanced-keyword')
  await expect(page.getByTestId('rf__node-projection-group-balanced-map')).toContainText(
    'balanced-standard',
  )

  // The Router preview tests a prompt against the recipe's public model.
  const preview = page.waitForRequest('**/api/topology/test-query')
  await page.getByRole('textbox', { name: 'Test a prompt' }).fill('balanced answers please')
  await page.getByRole('button', { name: 'Run', exact: true }).click()
  expect((await preview).postDataJSON()).toMatchObject({
    query: 'balanced answers please',
    model: 'vllm-sr/balanced',
  })

  await scope.selectOption('privacy')
  await expect(page.getByTestId('rf__node-decision-private-route')).toBeVisible()
  await expect(page.getByTestId('rf__node-signal-group-pii')).toContainText('private-pii')
  await expect(page.getByTestId('rf__node-decision-balanced-route')).toHaveCount(0)
})
