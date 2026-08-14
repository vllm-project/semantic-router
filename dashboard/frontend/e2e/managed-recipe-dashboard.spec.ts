import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const config = {
  version: 'v0.3',
  providers: { defaults: { default_model: 'leaf-a' }, models: [{ name: 'leaf-a' }] },
  routing: { modelCards: [{ name: 'leaf-a' }], decisions: [] },
  entrypoints: [{ model_names: ['team/custom-balanced'], recipe: 'balanced' }],
  recipes: [
    {
      name: 'balanced',
      description: 'Balanced objective',
      routing: {
        decisions: [
          {
            name: 'decision-a',
            priority: 100,
            rules: { operator: 'AND', conditions: [] },
            modelRefs: [{ model: 'leaf-a', use_reasoning: false }],
          },
        ],
      },
    },
  ],
}

const expected = {
  decision: 'decision-a',
  recipe: 'balanced',
  algorithm: 'static',
  alias: 'leaf-a',
  plugins: [],
  forbidden_plugins: [],
  plugin_match: 'contains',
  signals: {},
  forbidden_signals: {},
  signal_match: 'contains',
}

const summary = {
  id: 'decision-a:variant-a',
  decision_id: 'decision-a',
  variant_id: 'variant-a',
  query_preview: 'Please correct the earlier answer.',
  model: 'team/custom-balanced',
  tags: ['messages', 'verified'],
  request_shapes: ['messages', 'tools'],
  expected,
  playground: { enabled: true },
  editable: true,
}

const probeMessages = [
  { role: 'user', content: 'Give the first answer.' },
  { role: 'assistant', content: 'This is the first answer.' },
  { role: 'user', content: 'Please correct the earlier answer.' },
]

const probeImageDataURL = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=='

const probeTools = [
  {
    type: 'function',
    function: {
      name: 'lookup_source',
      description: 'Look up a source.',
      parameters: { type: 'object', properties: {}, required: [] },
    },
  },
]

const managedDescriptor = {
  managed: true,
  metadata: {
    schema_version: 'vllm-sr/recipe-metadata/v1',
    id: 'multi-objective',
    name: 'Multi-Objective Mixture-of-Models',
    version: '0.1.0',
    description: 'One unified model surface over isolated routing objectives.',
    authors: [{ name: 'vLLM Semantic Router Contributors' }],
    license: 'Apache-2.0',
    tags: ['mixture-of-models'],
    links: { source: 'https://example.com/recipe' },
  },
  readme: '# Managed Recipe\n\nInspectable and testable.',
  source_health: {
    status: 'ready',
    files: Object.fromEntries(
      ['metadata', 'config', 'probes', 'dsl', 'readme'].map((name) => [
        name,
        { present: true, digest: `sha256:${name}` },
      ]),
    ),
    issues: [],
  },
  digests: { recipe: 'sha256:recipe' },
  counts: { unified_models: 1, recipes: 1, decisions: 1, probes: 51 },
  activation_state: 'active',
  active_recipe_digest: 'sha256:recipe',
}

const activationPlan = (recipeDigest: string, planDigest: string) => ({
  recipe_digest: recipeDigest,
  plan_digest: planDigest,
  mode: 'hot_switch',
  requires_confirmation: true,
  listeners_before: [],
  listeners_after: [],
  storage: {
    before: [],
    after: [],
    add: [],
    remove: [],
    repair: [],
  },
  management_auth: {
    mode: 'service',
    service_permissions: ['config.deploy'],
  },
  effects: ['Validate and hot-switch the managed Router configuration.'],
})

const ACTIVATION_PLAN_DIGEST = `sha256:${'1'.repeat(64)}`
const DEACTIVATION_PLAN_DIGEST = `sha256:${'2'.repeat(64)}`

const installedModelCatalog = {
  catalogs: [
    {
      catalog_version: 'latest',
      channel: 'latest',
      default_model: 'vllm-sr/chorus-v1',
      enabled_models: ['vllm-sr/chorus-v1'],
    },
    {
      catalog_version: 'v0.3',
      channel: 'release',
      default_model: 'vllm-sr/chorus-v1',
      enabled_models: ['vllm-sr/chorus-v1'],
    },
  ],
  models: ['latest', 'v0.3'].map((catalogVersion) => ({
    id: 'vllm-sr/chorus-v1',
    display_name: 'Chorus V1',
    description: 'Balanced out-of-box routing for chat and reasoning workloads.',
    kind: 'virtual',
    family: 'chorus',
    generation: 1,
    policy_version: '1.0.0',
    entrypoint: 'vllm-sr/chorus-v1',
    recipe: 'balance',
    protocols: ['openai_chat'],
    traits: ['balanced', 'chat', 'reasoning'],
    roles: [
      {
        name: 'balanced',
        required: true,
        minimum_candidates: 2,
        traits: ['chat', 'reasoning'],
        recommended_pool: ['Qwen/Qwen3-8B', 'Qwen/Qwen3-32B'],
      },
    ],
    catalog_version: catalogVersion,
    channel: catalogVersion === 'latest' ? 'latest' : 'release',
    compatible: true,
    compatibility_reason: 'Compatible with this vllm-sr build.',
    enabled_by_default: true,
    default: true,
    verification: {
      status: 'verified',
      authority: 'vllm-sr-maintainers',
      asset_sha256: `sha256:${'a'.repeat(64)}`,
    },
  })),
}

async function mockManagedRecipeWorkspace(page: Page) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/models/catalog*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(installedModelCatalog),
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
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' })
  })
  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ overall: 'healthy', services: [], models: { models: [] } }),
    })
  })
  await page.route('**/api/recipe', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(managedDescriptor),
    })
  })
  await page.route('**/api/recipe/packages**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: [],
        page: 1,
        page_size: 25,
        total: 0,
        total_pages: 0,
        active_digest: 'sha256:recipe',
        activation_state: 'active',
      }),
    })
  })
  await page.route('**/api/recipe/activate/preview', async (route) => {
    const body = route.request().postDataJSON() as { recipe_digest?: string }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(
        activationPlan(body.recipe_digest || 'sha256:recipe', ACTIVATION_PLAN_DIGEST),
      ),
    })
  })
  await page.route('**/api/recipe/deactivate/preview', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(activationPlan('sha256:source', DEACTIVATION_PLAN_DIGEST)),
    })
  })
  await page.route('**/api/router/v1/models*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        object: 'list',
        data: [
          {
            id: 'team/custom-balanced',
            owned_by: 'vllm-semantic-router',
            routing: { resolution: 'virtual', selectable: true, recipe: 'balanced' },
          },
        ],
      }),
    })
  })
}

async function mockProbeAPI(
  page: Page,
  requestLog: string[],
  runPlanMessages: Array<Record<string, unknown>> = probeMessages,
) {
  await page.route('**/api/recipe/probes**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    requestLog.push(`${request.method()} ${url.pathname}${url.search}`)
    const path = url.pathname

    if (path.endsWith('/validate')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          probe_id: summary.id,
          recipe_digest: 'sha256:recipe',
          passed: true,
          expected,
          actual: {
            decision: 'decision-a',
            model: 'leaf-a',
            requested_model: 'team/custom-balanced',
            selection_status: 'selected',
            recipe: 'balanced',
            algorithm: 'static',
            plugins: [],
            recommended_models: ['leaf-a'],
            matched_signals: {},
            trace_decisions: ['decision-a'],
          },
          checks: {
            decision: true,
            model: true,
            recipe: true,
            algorithm: true,
            plugins: true,
            signals: true,
            alias: true,
            trace: true,
          },
          failures: [],
          latency_ms: 8,
        }),
      })
      return
    }

    if (path.endsWith('/run-plan')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          probe_id: summary.id,
          recipe_digest: 'sha256:recipe',
          model: summary.model,
          messages: runPlanMessages,
          tools: probeTools,
          request: {
            model: summary.model,
            messages: runPlanMessages,
            tools: probeTools,
            temperature: 0,
          },
          editable: true,
        }),
      })
      return
    }

    if (path !== '/api/recipe/probes') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ...summary,
          messages: runPlanMessages,
          tools: probeTools,
          repeat: 1,
          notes: 'A verified multi-turn correction fixture.',
        }),
      })
      return
    }

    const pageNumber = Number(url.searchParams.get('page') ?? '1')
    const item =
      pageNumber === 1
        ? summary
        : {
            ...summary,
            id: 'decision-a:variant-b',
            variant_id: 'variant-b',
            query_preview: 'Second page probe.',
          }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: [item],
        page: pageNumber,
        page_size: 50,
        total: 51,
        total_pages: 2,
        facets: {
          recipes: { balanced: 51 },
          decisions: { 'decision-a': 51 },
          tags: { messages: 1, verified: 51 },
          models: { 'team/custom-balanced': 51 },
          shapes: { messages: 1, text: 50, tools: 1 },
        },
        recipe_digest: 'sha256:recipe',
      }),
    })
  })
}

test.beforeEach(async ({ page }) => {
  await mockManagedRecipeWorkspace(page)
})

test('discovers installed catalog versions and backend role requirements without ZIP distribution', async ({
  page,
}) => {
  await page.goto('/config/entrypoints-recipes')

  await expect(page.getByRole('heading', { name: 'Built-in model catalogs' })).toBeVisible({
    timeout: 15_000,
  })
  await expect(page.getByText('2 version packages')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'latest', exact: true })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'v0.3', exact: true })).toBeVisible()
  await expect(page.getByText('Chorus V1').first()).toBeVisible()
  await expect(page.getByText('catalog verified').first()).toBeVisible()

  await page.getByText('1 backend role requirements').first().click()
  await expect(page.getByText('minimum 2').first()).toBeVisible()
  await expect(page.getByText('Qwen/Qwen3-8B').first()).toBeVisible()
  await expect(page.getByLabel('Recipe ZIP URL')).toHaveCount(0)
  await expect(page.getByRole('button', { name: /Import Recipe package/i })).toHaveCount(0)
})

test('shows managed metadata and pages, inspects, and validates probes server-side', async ({
  page,
}) => {
  const requests: string[] = []
  let chatRequests = 0
  await mockProbeAPI(page, requests)
  await page.route('**/api/router/v1/chat/completions', async (route) => {
    chatRequests += 1
    await route.abort()
  })

  await page.goto('/config/entrypoints-recipes')
  await expect(
    page.getByRole('heading', { name: 'Multi-Objective Mixture-of-Models' }),
  ).toBeVisible()
  await expect(page.getByText('51', { exact: true })).toBeVisible()
  await expect(page.getByText('metadata.yaml')).toBeVisible()
  await expect(page.getByTestId('managed-recipe-config-lock')).toContainText(
    'Managed Recipe controls the runtime configuration',
  )

  await page.getByRole('tab', { name: 'Models & Routing' }).click()
  await expect(page.getByRole('button', { name: 'Add unified model' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Add recipe' })).toHaveCount(0)

  await page.getByRole('tab', { name: 'Probes' }).click()
  await expect(page.getByText('Please correct the earlier answer.')).toBeVisible()
  expect(requests).toContain('GET /api/recipe/probes?page=1&page_size=50')

  const recipeFilter = page.getByRole('combobox', { name: 'Recipe', exact: true })
  await expect(recipeFilter).toContainText('balanced (51)')
  await recipeFilter.selectOption('balanced')
  await expect
    .poll(() => requests.includes('GET /api/recipe/probes?page=1&page_size=50&recipe=balanced'))
    .toBe(true)

  await page.getByRole('button', { name: `View ${summary.id} details` }).click()
  await expect(page.getByText('A verified multi-turn correction fixture.')).toBeVisible()
  await expect(page.getByText('lookup_source')).toBeVisible()

  await page.getByRole('button', { name: 'Validate' }).click()
  await expect(page.getByText('Route validated')).toBeVisible()
  await expect(page.getByText('Final model').locator('..')).toContainText('leaf-a')
  await expect(page.getByText('Request model').locator('..')).toContainText('team/custom-balanced')
  expect(requests).toContain('POST /api/recipe/probes/decision-a/variant-a/validate')
  expect(chatRequests).toBe(0)

  await page.getByRole('button', { name: 'Next' }).click()
  await expect(page.getByText('Second page probe.')).toBeVisible()
  expect(requests).toContain('GET /api/recipe/probes?page=2&page_size=50&recipe=balanced')
})

test('activates and restores an API-preinstalled custom package with explicit previews', async ({
  page,
}) => {
  const packageRequests: string[] = []
  const packageImportRequests: string[] = []
  const activationPreviewBodies: Array<Record<string, unknown>> = []
  const activationBodies: Array<Record<string, unknown>> = []
  const deactivationPreviewBodies: Array<Record<string, unknown>> = []
  const deactivationBodies: Array<Record<string, unknown>> = []
  const probeRequests: string[] = []
  let descriptorRequests = 0
  let configRequests = 0
  let activated = false
  let deactivated = false

  const activatedConfig = {
    ...config,
    providers: {
      defaults: { default_model: 'latency-leaf' },
      models: [{ name: 'latency-leaf' }],
    },
    routing: { modelCards: [{ name: 'latency-leaf' }], decisions: [] },
    entrypoints: [{ model_names: ['team/custom-latency'], recipe: 'latency-aware' }],
    recipes: [
      {
        name: 'latency-aware',
        description: 'Latency-optimized routing installed from a package.',
        routing: {
          decisions: [
            {
              name: 'latency-decision',
              priority: 100,
              rules: { operator: 'AND', conditions: [] },
              modelRefs: [{ model: 'latency-leaf', use_reasoning: false }],
            },
          ],
        },
      },
    ],
  }

  const currentPackage = {
    id: 'multi-objective',
    name: 'Multi-Objective Mixture-of-Models',
    version: '0.1.0',
    description: 'Current active Recipe.',
    recipe_digest: 'sha256:recipe',
    config_digest: `sha256:${'d'.repeat(64)}`,
    archive_sha256: `sha256:${'a'.repeat(64)}`,
    archive_verified: true,
    installed_at: '2026-08-11T08:00:00Z',
    active: true,
    counts: managedDescriptor.counts,
    warnings: [],
  }
  const importedPackage = {
    id: 'latency-aware',
    name: 'Latency-Aware Mixture-of-Models',
    version: '1.2.0',
    description: 'Routes interactive traffic to a latency-optimized pool.',
    recipe_digest: `sha256:${'c'.repeat(64)}`,
    config_digest: `sha256:${'e'.repeat(64)}`,
    archive_sha256: `sha256:${'b'.repeat(64)}`,
    archive_verified: true,
    installed_at: '2026-08-12T09:30:00Z',
    active: false,
    counts: { unified_models: 2, recipes: 1, decisions: 3, probes: 120 },
    warnings: [
      {
        code: 'environment_binding_required',
        path: '/providers/models/0/api_key',
        message:
          'Recipe requires environment variable LATENCY_PROVIDER_API_KEY to be explicitly authorized before activation.',
      },
    ],
  }

  await mockProbeAPI(page, probeRequests)
  await page.route('**/api/recipe/import', async (route) => {
    packageImportRequests.push(route.request().method())
    await route.fulfill({ status: 410 })
  })
  await page.route('**/api/router/config/all', async (route) => {
    configRequests += 1
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(activated ? activatedConfig : config),
    })
  })
  await page.route('**/api/recipe', async (route) => {
    descriptorRequests += 1
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(
        deactivated
          ? {
              ...managedDescriptor,
              activation_state: 'none',
              active_recipe_digest: undefined,
            }
          : activated
            ? {
                ...managedDescriptor,
                metadata: {
                  ...managedDescriptor.metadata,
                  id: importedPackage.id,
                  name: importedPackage.name,
                  version: importedPackage.version,
                  description: importedPackage.description,
                },
                digests: { recipe: importedPackage.recipe_digest },
                active_recipe_digest: importedPackage.recipe_digest,
                counts: importedPackage.counts,
              }
            : managedDescriptor,
      ),
    })
  })
  await page.route('**/api/recipe/packages**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    packageRequests.push(`${request.method()} ${url.pathname}${url.search}`)
    const items = [
      { ...currentPackage, active: !activated && !deactivated },
      { ...importedPackage, active: activated },
    ]
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items,
        page: 1,
        page_size: 25,
        total: items.length,
        total_pages: 1,
        ...(deactivated
          ? {}
          : {
              active_digest: activated
                ? importedPackage.recipe_digest
                : currentPackage.recipe_digest,
            }),
        activation_state: deactivated ? 'none' : 'active',
      }),
    })
  })
  await page.route('**/api/recipe/activate/preview', async (route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>
    activationPreviewBodies.push(body)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(
        activationPlan(
          String(body.recipe_digest || importedPackage.recipe_digest),
          ACTIVATION_PLAN_DIGEST,
        ),
      ),
    })
  })
  await page.route('**/api/recipe/deactivate/preview', async (route) => {
    deactivationPreviewBodies.push(route.request().postDataJSON() as Record<string, unknown>)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(activationPlan('sha256:source', DEACTIVATION_PLAN_DIGEST)),
    })
  })
  await page.route('**/api/recipe/activate', async (route) => {
    activationBodies.push(route.request().postDataJSON() as Record<string, unknown>)
    activated = true
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'active',
        recipe_digest: importedPackage.recipe_digest,
        previous_recipe_digest: currentPackage.recipe_digest,
      }),
    })
  })
  await page.route('**/api/recipe/deactivate', async (route) => {
    deactivationBodies.push(route.request().postDataJSON() as Record<string, unknown>)
    deactivated = true
    activated = false
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'inactive',
        previous_recipe_digest: importedPackage.recipe_digest,
      }),
    })
  })

  await page.goto('/config/entrypoints-recipes')
  await expect(page.getByRole('heading', { name: 'Custom package lifecycle' })).toBeVisible()
  expect(packageRequests).toContain('GET /api/recipe/packages?page=1&page_size=25')
  await expect(page.getByLabel('Recipe ZIP URL')).toHaveCount(0)
  await expect(page.getByRole('button', { name: /Import Recipe package/i })).toHaveCount(0)
  const importedPackageCard = page.getByRole('article', { name: importedPackage.name })
  await expect(importedPackageCard).toBeVisible()
  await expect(importedPackageCard.getByText('Integrity verified')).toBeVisible()
  await expect(page.getByText(importedPackage.archive_sha256)).toHaveCount(0)
  expect(activationBodies).toHaveLength(0)

  await page.getByLabel('Search installed packages').fill('latency aware')
  await expect
    .poll(() => packageRequests.some((request) => request.endsWith('&q=latency+aware')))
    .toBe(true)
  await page.getByRole('button', { name: 'Clear installed Recipe package search' }).click()

  await page
    .getByRole('button', {
      name: `Activate Recipe package ${importedPackage.name} ${importedPackage.version} (${importedPackage.id})`,
    })
    .click()
  const confirmation = page.getByRole('alertdialog')
  expect(activationPreviewBodies).toEqual([
    {
      recipe_digest: importedPackage.recipe_digest,
      acknowledge_warnings: true,
    },
  ])
  await expect(confirmation.getByText('LATENCY_PROVIDER_API_KEY')).toBeVisible()
  await expect(confirmation.getByText('/providers/models/0/api_key')).toBeVisible()
  await expect(confirmation).not.toContainText('private-release')
  expect(activationBodies).toHaveLength(0)
  await confirmation.getByRole('button', { name: 'Acknowledge warnings and activate' }).click()

  await expect(page.getByRole('heading', { name: importedPackage.name }).first()).toBeVisible()
  const activePackage = page.getByRole('article', { name: importedPackage.name })
  await expect(activePackage.getByText('Active', { exact: true })).toBeVisible()
  expect(activationBodies).toEqual([
    {
      recipe_digest: importedPackage.recipe_digest,
      acknowledge_warnings: true,
      expected_plan_digest: ACTIVATION_PLAN_DIGEST,
      confirm_stack_recreation: false,
    },
  ])
  expect(descriptorRequests).toBeGreaterThan(1)
  expect(configRequests).toBeGreaterThan(1)

  await page.getByRole('tab', { name: 'Models & Routing' }).click()
  await expect(page.getByText('team/custom-latency')).toBeVisible()
  await expect(page.getByText('latency-aware', { exact: true }).first()).toBeVisible()
  await page.getByRole('button', { name: 'Expand team/custom-latency' }).click()
  await expect(page.getByText('custom / unverified')).toBeVisible()
  await expect(page.getByText(/not an exact installed catalog model/)).toBeVisible()

  await page.getByRole('tab', { name: 'Probes' }).click()
  await expect(page.getByText('Please correct the earlier answer.')).toBeVisible()
  expect(probeRequests).toContain('GET /api/recipe/probes?page=1&page_size=50')

  await page.getByRole('tab', { name: 'Built-in Models' }).click()
  await page
    .getByRole('button', {
      name: `Restore source runtime from active Recipe package ${importedPackage.name} ${importedPackage.version} (${importedPackage.id})`,
    })
    .click()
  expect(deactivationPreviewBodies).toEqual([{}])
  const restoreConfirmation = page.getByRole('alertdialog', {
    name: 'Restore the source runtime configuration?',
  })
  await expect(restoreConfirmation).toContainText(
    'durable source configuration captured before the first Recipe package activation',
  )
  await restoreConfirmation.getByRole('button', { name: 'Restore source' }).click()

  await expect(page.getByText(/durable source runtime configuration/)).toBeVisible()
  await expect(page.getByTestId('managed-recipe-config-lock')).toHaveCount(0)
  await expect(activePackage.getByText('Active', { exact: true })).toHaveCount(0)
  expect(deactivationBodies).toEqual([
    {
      expected_plan_digest: DEACTIVATION_PLAN_DIGEST,
      confirm_stack_recreation: false,
    },
  ])

  await page.getByRole('tab', { name: 'Models & Routing' }).click()
  await expect(page.getByRole('button', { name: 'Add unified model' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Add recipe' })).toBeVisible()
  expect(packageImportRequests).toEqual([])
})

test('offers an explicit installed-package repair when activation state is inconsistent', async ({
  page,
}) => {
  const repairPreviewBodies: Array<Record<string, unknown>> = []
  const repairActivationBodies: Array<Record<string, unknown>> = []
  const repairPackage = {
    id: 'recovery',
    name: 'Recovery Recipe',
    version: '1.0.0',
    description: 'A known package for Router repair.',
    recipe_digest: `sha256:${'f'.repeat(64)}`,
    config_digest: `sha256:${'e'.repeat(64)}`,
    archive_sha256: `sha256:${'d'.repeat(64)}`,
    archive_verified: true,
    installed_at: '2026-08-12T10:00:00Z',
    active: false,
    counts: { unified_models: 1, recipes: 1, decisions: 1, probes: 2 },
    warnings: [],
  }
  await page.route('**/api/recipe', async (route) => {
    await route.fulfill({
      status: 409,
      contentType: 'application/json',
      body: JSON.stringify({
        ...managedDescriptor,
        activation_state: 'inconsistent',
        source_health: {
          ...managedDescriptor.source_health,
          status: 'inconsistent',
          issues: ['Active Recipe and runtime configuration are inconsistent.'],
        },
      }),
    })
  })
  await page.route('**/api/recipe/packages**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: [repairPackage],
        page: 1,
        page_size: 25,
        total: 1,
        total_pages: 1,
        activation_state: 'inconsistent',
      }),
    })
  })
  await page.route('**/api/recipe/activate/preview', async (route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>
    repairPreviewBodies.push(body)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(activationPlan(repairPackage.recipe_digest, ACTIVATION_PLAN_DIGEST)),
    })
  })
  await page.route('**/api/recipe/activate', async (route) => {
    repairActivationBodies.push(route.request().postDataJSON() as Record<string, unknown>)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'active', recipe_digest: repairPackage.recipe_digest }),
    })
  })

  await page.goto('/config/entrypoints-recipes')
  await expect(page.getByTestId('managed-recipe-config-lock')).toContainText(
    'Managed Recipe requires explicit repair',
  )
  await expect(page.getByText('Routing and probes fail closed in this state.')).toBeVisible()
  await expect(
    page.getByRole('article', { name: repairPackage.name }).getByText('Active', { exact: true }),
  ).toHaveCount(0)
  const repairButton = page.getByRole('button', {
    name: `Repair Router by activating Recipe package ${repairPackage.name} ${repairPackage.version} (${repairPackage.id})`,
  })
  await expect(repairButton).toBeEnabled()
  await repairButton.click()
  const repairDialog = page.getByRole('alertdialog', {
    name: `Repair with ${repairPackage.name} ${repairPackage.version}?`,
  })
  await expect(repairDialog).toBeVisible()
  expect(repairPreviewBodies).toEqual([{ recipe_digest: repairPackage.recipe_digest }])
  await repairDialog.getByRole('button', { name: `Repair with ${repairPackage.name}` }).click()
  await expect.poll(() => repairActivationBodies.length).toBe(1)
  expect(repairActivationBodies).toEqual([
    {
      recipe_digest: repairPackage.recipe_digest,
      expected_plan_digest: ACTIVATION_PLAN_DIGEST,
      confirm_stack_recreation: false,
    },
  ])
})

test('reports missing environment bindings without exposing values or changing runtime config', async ({
  page,
}) => {
  const bindingPackage = {
    id: 'private-provider',
    name: 'Private Provider Recipe',
    version: '1.0.0',
    description: 'Requires an explicitly authorized provider binding.',
    recipe_digest: `sha256:${'9'.repeat(64)}`,
    config_digest: `sha256:${'8'.repeat(64)}`,
    archive_sha256: `sha256:${'7'.repeat(64)}`,
    archive_verified: true,
    installed_at: '2026-08-12T10:30:00Z',
    active: false,
    counts: { unified_models: 1, recipes: 1, decisions: 1, probes: 4 },
    warnings: [
      {
        code: 'environment_binding_required',
        path: '/providers/models/0/api_key',
        message:
          'Recipe requires environment variable PRIVATE_PROVIDER_API_KEY to be explicitly authorized before activation.',
      },
    ],
  }
  let activationRequests = 0
  await page.route('**/api/recipe/packages**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: [bindingPackage],
        page: 1,
        page_size: 25,
        total: 1,
        total_pages: 1,
        active_digest: managedDescriptor.digests.recipe,
        activation_state: 'active',
      }),
    })
  })
  await page.route('**/api/recipe/activate', async (route) => {
    activationRequests += 1
    await route.fulfill({
      status: 409,
      contentType: 'application/json',
      body: JSON.stringify({
        error: 'environment_binding_required',
        message: 'Required Recipe environment bindings are not authorized or unavailable.',
      }),
    })
  })

  await page.goto('/config/entrypoints-recipes')
  await page
    .getByRole('button', {
      name: `Activate Recipe package ${bindingPackage.name} ${bindingPackage.version} (${bindingPackage.id})`,
    })
    .click()
  const confirmation = page.getByRole('alertdialog')
  await expect(confirmation.getByText('PRIVATE_PROVIDER_API_KEY')).toBeVisible()
  await expect(confirmation.getByText(bindingPackage.warnings[0].path)).toBeVisible()
  await confirmation.getByRole('button', { name: 'Acknowledge warnings and activate' }).click()

  await expect(page.getByText('Recipe environment binding unavailable')).toBeVisible()
  await expect(
    page.getByText(/Authorize and provide the required environment variables/),
  ).toBeVisible()
  await expect(page.getByText('Private Provider Recipe', { exact: true })).toBeVisible()
  await expect(page.getByText(/super-secret|private-release/)).toHaveCount(0)
  expect(activationRequests).toBe(1)
})

test('Edit opens a clean composer without sending and preserves the exact structured request', async ({
  page,
}) => {
  const requests: string[] = []
  const chatBodies: Array<Record<string, unknown>> = []
  await mockProbeAPI(page, requests)
  await page.route('**/api/router/v1/chat/completions', async (route) => {
    chatBodies.push(route.request().postDataJSON() as Record<string, unknown>)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        choices: [{ index: 0, message: { content: 'Edited probe answer.' } }],
      }),
    })
  })
  await page.addInitScript(() => {
    window.localStorage.setItem(
      'sr:chat:conversations',
      JSON.stringify([
        {
          id: 'legacy-conversation',
          createdAt: 1,
          updatedAt: 1,
          payload: [
            { id: 'legacy-message', role: 'user', content: 'Legacy conversation', timestamp: 1 },
          ],
        },
      ]),
    )
  })

  await page.goto('/config/entrypoints-recipes')
  await page.getByRole('tab', { name: 'Probes' }).click()
  await page.getByRole('button', { name: 'Edit' }).click()

  await expect(page).toHaveURL(/\/playground$/)
  const composer = page.getByPlaceholder('Ask me anything...')
  await expect(composer).toHaveValue('Please correct the earlier answer.')
  const transcript = page.getByTestId('chat-transcript')
  await expect(transcript.getByText('This is the first answer.')).toBeVisible()
  await expect(transcript.getByText('Legacy conversation')).toHaveCount(0)
  expect(chatBodies).toHaveLength(0)

  await composer.fill('Correct it and cite two sources.')
  await page.getByRole('button', { name: 'Send message' }).click()
  await expect(page.getByText('Edited probe answer.')).toBeVisible()
  expect(chatBodies).toEqual([
    {
      model: 'team/custom-balanced',
      messages: [
        probeMessages[0],
        probeMessages[1],
        { role: 'user', content: 'Correct it and cite two sources.' },
      ],
      tools: probeTools,
      temperature: 0,
      stream: true,
      max_completion_tokens: 2048,
    },
  ])
})

test('Run opens a clean chat and immediately sends the materialized probe request', async ({
  page,
}) => {
  const requests: string[] = []
  const chatBodies: Array<Record<string, unknown>> = []
  await mockProbeAPI(page, requests)
  await page.route('**/api/router/v1/chat/completions', async (route) => {
    chatBodies.push(route.request().postDataJSON() as Record<string, unknown>)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ choices: [{ index: 0, message: { content: 'Probe answer.' } }] }),
    })
  })

  await page.goto('/config/entrypoints-recipes')
  await page.getByRole('tab', { name: 'Probes' }).click()
  await page.getByRole('button', { name: 'Run' }).click()

  await expect(page).toHaveURL(/\/playground$/)
  await expect(page.getByText('Probe answer.')).toBeVisible()
  expect(requests).toContain('POST /api/recipe/probes/decision-a/variant-a/run-plan')
  expect(chatBodies).toEqual([
    {
      model: 'team/custom-balanced',
      messages: probeMessages,
      tools: probeTools,
      temperature: 0,
      stream: true,
      max_completion_tokens: 2048,
    },
  ])
})

test('Run renders a multimodal probe as readable text and a real image', async ({ page }) => {
  const requests: string[] = []
  const chatBodies: Array<Record<string, unknown>> = []
  const multimodalMessages = [
    { role: 'user', content: 'Keep the earlier visual context available.' },
    {
      role: 'assistant',
      content: [
        { type: 'output_text', text: 'Earlier visual context.' },
        { type: 'image_url', image_url: probeImageDataURL },
      ],
    },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Summarize the architecture shown in this diagram.' },
        { type: 'image_url', image_url: { url: probeImageDataURL } },
      ],
    },
  ]
  await mockProbeAPI(page, requests, multimodalMessages)
  await page.route('**/api/router/v1/chat/completions', async (route) => {
    chatBodies.push(route.request().postDataJSON() as Record<string, unknown>)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ choices: [{ index: 0, message: { content: 'Vision answer.' } }] }),
    })
  })

  await page.goto('/config/entrypoints-recipes')
  await page.getByRole('tab', { name: 'Probes' }).click()
  await page.getByRole('button', { name: 'Run' }).click()

  await expect(page).toHaveURL(/\/playground$/)
  await expect(
    page.getByText('Summarize the architecture shown in this diagram.', { exact: true }),
  ).toBeVisible()
  await expect(page.getByText('Earlier visual context.', { exact: true })).toBeVisible()
  const images = page.getByTestId('probe-message-image')
  await expect(images).toHaveCount(2)
  await expect
    .poll(() =>
      images.evaluateAll((elements) =>
        elements.every(
          (element) =>
            element instanceof HTMLImageElement &&
            element.currentSrc.startsWith('data:image/gif;base64,') &&
            element.naturalWidth > 0,
        ),
      ),
    )
    .toBe(true)
  await expect(page.getByText(/data:image\/gif;base64/)).toHaveCount(0)
  expect(chatBodies).toEqual([
    {
      model: 'team/custom-balanced',
      messages: multimodalMessages,
      tools: probeTools,
      temperature: 0,
      stream: true,
      max_completion_tokens: 2048,
    },
  ])
})
