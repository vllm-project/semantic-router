import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const config = {
  version: 'v0.5',
  providers: { defaults: { model: 'model-a' }, models: [{ name: 'model-a' }] },
  entrypoints: [{ model_names: ['vllm-sr/balanced'], recipe: 'balanced' }],
  recipes: [
    { name: 'default', decisions: [] },
    {
      name: 'balanced',
      description: 'Balanced objective',
      decisions: [
        {
          name: 'balanced-route',
          description: 'Balanced route',
          priority: 100,
          rules: { operator: 'AND', conditions: [] },
          modelRefs: [{ model: 'model-a', use_reasoning: false }],
        },
      ],
    },
  ],
}

const packageSummary = (active: boolean, warnings = true) => ({
  id: 'acme/secure',
  name: 'secure-package',
  version: '1.2.0',
  description: 'Keeps private traffic local.',
  recipe_digest: `sha256:${'a'.repeat(64)}`,
  config_digest: `sha256:${'b'.repeat(64)}`,
  archive_sha256: 'c'.repeat(64),
  archive_verified: true,
  installed_at: '2026-11-17T10:00:00Z',
  active,
  counts: { unified_models: 2, recipes: 2, decisions: 7, probes: 12 },
  warnings: warnings
    ? [
        {
          code: 'unpinned_model',
          path: 'recipes[0].decisions[1]',
          message: 'A model is not pinned.',
        },
      ]
    : [],
})

async function mockRecipesPage(page: Page, packages: ReturnType<typeof packageSummary>[]) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: config }))
  await page.route('**/api/router/config/global', (route) => route.fulfill({ json: {} }))
  await page.route('**/api/recipes/builtin', (route) =>
    route.fulfill({ json: { catalog_version: 'latest', bundles: [] } }),
  )
  await page.route(
    (url) => url.pathname === '/api/recipe/packages',
    (route) =>
      route.fulfill({
        json: {
          items: packages,
          page: 1,
          page_size: 100,
          total: packages.length,
          total_pages: 1,
          activation_state: packages.some((item) => item.active) ? 'active' : 'none',
        },
      }),
  )
  await page.route('**/api/status', (route) =>
    route.fulfill({
      json: {
        overall: 'healthy',
        deployment_type: 'local',
        services: [{ name: 'router', status: 'healthy', healthy: true }],
        models: { models: [], summary: { loaded_models: 0, total_models: 1 } },
      },
    }),
  )
}

async function gotoRecipes(page: Page, path: string) {
  await page.goto(path)
  // The first visit compiles the lazy config page in the dev server.
  await expect(page.getByRole('heading', { name: 'Recipes', exact: true })).toBeVisible({
    timeout: 30_000,
  })
}

test('activates a recipe package from a previewed plan and reports the pending restart', async ({
  page,
}) => {
  await mockRecipesPage(page, [packageSummary(false)])
  const previews: unknown[] = []
  const activations: unknown[] = []
  const planDigest = `sha256:${'f'.repeat(64)}`
  await page.route('**/api/recipe/activate/preview', async (route) => {
    previews.push(route.request().postDataJSON())
    await route.fulfill({
      json: {
        recipe_digest: packageSummary(false).recipe_digest,
        plan_digest: planDigest,
        mode: 'stack_recreation',
        requires_confirmation: true,
        listeners_before: [],
        listeners_after: [],
        storage: { before: [], after: ['redis'], add: ['redis'], remove: [], repair: [] },
        management_auth: { mode: 'disabled', service_permissions: [] },
        effects: ['recreate the managed stack'],
      },
    })
  })
  await page.route('**/api/recipe/activate', async (route) => {
    activations.push(route.request().postDataJSON())
    await route.fulfill({
      status: 202,
      json: {
        status: 'restart_required',
        message: 'Restart required: run `vllm-sr serve` to apply.',
        recipe_digest: packageSummary(false).recipe_digest,
        plan_digest: planDigest,
        mode: 'stack_recreation',
      },
    })
  })

  await gotoRecipes(page, '/routing/recipes?tab=recipes')
  const row = page.getByRole('row').filter({ hasText: 'secure-package' })
  await expect(row).toContainText('Inactive')
  await row.getByRole('switch', { name: 'Activate secure-package' }).click()

  const dialog = page.getByRole('dialog', { name: /Activate secure-package/ })
  await expect(dialog).toContainText('A model is not pinned.')
  await expect(dialog.getByRole('button', { name: /activate/i })).toBeDisabled()
  expect(previews).toHaveLength(0)

  await dialog.getByLabel('I reviewed these warnings').click()
  await expect(dialog).toContainText('Stack recreation')
  await expect(dialog).toContainText('Managed storage redis')
  await expect(dialog).toContainText('recreate the managed stack')
  await expect(dialog).toContainText('Restart required: run vllm-sr serve to apply')
  expect(previews).toEqual([
    { recipe_digest: packageSummary(false).recipe_digest, acknowledge_warnings: true },
  ])

  await dialog.getByRole('button', { name: 'Save and activate on restart' }).click()
  await expect(dialog).toBeHidden()
  expect(activations).toEqual([
    {
      recipe_digest: packageSummary(false).recipe_digest,
      acknowledge_warnings: true,
      expected_plan_digest: planDigest,
      confirm_stack_recreation: true,
    },
  ])
  await expect(page.getByText('Restart required: run vllm-sr serve to apply').first()).toBeVisible()
  await expect(row).toContainText('Restart pending')
  await expect(row.getByRole('switch')).toBeDisabled()
})

test('deactivates an active package with a hot switch', async ({ page }) => {
  await mockRecipesPage(page, [packageSummary(true, false)])
  const deactivations: string[] = []
  await page.route('**/api/recipe/deactivate/preview', (route) =>
    route.fulfill({
      json: {
        recipe_digest: packageSummary(true).recipe_digest,
        plan_digest: `sha256:${'e'.repeat(64)}`,
        mode: 'hot_switch',
        requires_confirmation: false,
        listeners_before: [],
        listeners_after: [],
        storage: { before: [], after: [], add: [], remove: [], repair: [] },
        management_auth: { mode: 'disabled', service_permissions: [] },
        effects: [],
      },
    }),
  )
  await page.route('**/api/recipe/deactivate', async (route) => {
    deactivations.push(route.request().postData() ?? '')
    await route.fulfill({ json: { status: 'inactive' } })
  })

  await gotoRecipes(page, '/routing/recipes?tab=recipes')
  const row = page.getByRole('row').filter({ hasText: 'secure-package' })
  await expect(row).toContainText('Active')
  await row.getByRole('switch', { name: 'Deactivate secure-package' }).click()

  const dialog = page.getByRole('dialog', { name: /Deactivate secure-package/ })
  await expect(dialog).toContainText('Hot switch, no restart')
  await dialog.getByRole('button', { name: 'Deactivate', exact: true }).click()
  await expect(dialog).toBeHidden()
  expect(deactivations).toEqual(['{}'])
  await expect(page.getByText('secure-package deactivated')).toBeVisible()
})

test('validates probes on the real endpoints and launches one in Playground', async ({ page }) => {
  await mockRecipesPage(page, [])
  const digest = `sha256:${'d'.repeat(64)}`
  const probe = (variant: string, decision: string) => ({
    id: `${decision}/${variant}`,
    decision_id: decision,
    variant_id: variant,
    query_preview: `Prompt for ${variant}`,
    tags: ['smoke'],
    request_shapes: ['text'],
    expected: {
      decision,
      recipe: 'balanced',
      plugins: [],
      forbidden_plugins: [],
      signals: { keyword: ['balanced-keyword'] },
      forbidden_signals: {},
    },
    playground: { enabled: true },
    editable: true,
  })
  const items = [probe('plain', 'balanced-route'), probe('tricky', 'balanced-route')]
  await page.route(
    (url) => url.pathname === '/api/recipe/probes',
    (route) =>
      route.fulfill({
        json: {
          items,
          page: 1,
          page_size: 50,
          total: 2,
          total_pages: 1,
          recipe_digest: digest,
          facets: {
            recipes: { balanced: 2 },
            decisions: { 'balanced-route': 2 },
            tags: { smoke: 2 },
            models: {},
            shapes: { text: 2 },
          },
        },
      }),
  )
  await page.route('**/api/recipe/probes/balanced-route/*', async (route) => {
    const url = new URL(route.request().url())
    const variant = decodeURIComponent(url.pathname.split('/').at(-1) ?? '')
    await route.fulfill({
      json: { ...probe(variant, 'balanced-route'), query: `Prompt for ${variant}`, repeat: 1 },
    })
  })
  const validations: string[] = []
  await page.route('**/api/recipe/probes/balanced-route/*/validate', async (route) => {
    const url = new URL(route.request().url())
    const variant = decodeURIComponent(url.pathname.split('/').at(-2) ?? '')
    validations.push(`${variant} ${route.request().headers()['if-match']}`)
    const passed = variant === 'plain'
    await route.fulfill({
      json: {
        probe_id: `balanced-route/${variant}`,
        recipe_digest: digest,
        passed,
        expected: probe(variant, 'balanced-route').expected,
        actual: {
          decision: passed ? 'balanced-route' : 'fallback',
          model: 'model-a',
          plugins: [],
          recommended_models: ['model-a'],
          matched_signals: {},
          trace_decisions: [],
        },
        checks: {},
        failures: passed ? [] : ['decision: expected balanced-route, got fallback'],
        latency_ms: 12,
      },
    })
  })
  let runPlans = 0
  await page.route('**/api/recipe/probes/balanced-route/*/run-plan', async (route) => {
    runPlans += 1
    await route.fulfill({
      json: {
        probe_id: 'balanced-route/plain',
        recipe_digest: digest,
        recipe: 'balanced',
        messages: [{ role: 'user', content: 'Prompt for plain' }],
        request: { model: 'vllm-sr/balanced' },
        editable: true,
      },
    })
  })

  await gotoRecipes(page, '/routing/recipes?tab=probes')
  await expect(page.getByText('2 probes', { exact: true })).toBeVisible()
  await page.getByRole('row').filter({ hasText: 'tricky' }).click()

  const sheet = page.getByRole('dialog', { name: 'tricky' })
  await expect(sheet).toContainText('keyword:balanced-keyword')
  await expect(sheet).toContainText('Prompt for tricky')
  await sheet.getByRole('button', { name: 'Validate route' }).click()
  await expect(sheet).toContainText('Route mismatch')
  await expect(sheet).toContainText('decision: expected balanced-route, got fallback')
  await page.keyboard.press('Escape')

  await page.getByRole('button', { name: 'Validate this page' }).click()
  await expect(page.getByText('1 of 2 probes pass')).toBeVisible()
  expect(validations).toEqual([`tricky "${digest}"`, `plain "${digest}"`, `tricky "${digest}"`])
  await expect(page.getByRole('tab', { name: /Probes/ })).toContainText('1 failing')

  await page.getByRole('button', { name: 'Row actions' }).first().click()
  await page.getByRole('menuitem', { name: 'Run in Playground' }).click()
  await expect(page).toHaveURL(/\/playground$/)
  expect(runPlans).toBe(1)
})
