import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const routerModels = [
  {
    name: 'category_classifier',
    type: 'intent_classification',
    loaded: true,
    state: 'ready',
    model_path: 'models/mmbert32k-intent-classifier-merged',
    registry: {
      local_path: 'models/mmbert32k-intent-classifier-merged',
      repo_id: 'vllm-sr/mmbert32k-intent-classifier-merged',
      purpose: 'domain-classification',
      description: 'Merged intent classifier for multilingual routing decisions.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      num_classes: 14,
      license: 'apache-2.0',
      model_card_url: 'https://huggingface.co/vllm-sr/mmbert32k-intent-classifier-merged',
      tags: ['text-classification', 'intent-classification'],
    },
    metadata: {
      model_type: 'mmbert_32k',
      threshold: '0.50',
    },
  },
  {
    name: 'fact_check_classifier',
    type: 'fact_check_classification',
    loaded: true,
    state: 'ready',
    model_path: 'models/mmbert32k-factcheck-classifier-merged',
    registry: {
      local_path: 'models/mmbert32k-factcheck-classifier-merged',
      repo_id: 'vllm-sr/mmbert32k-factcheck-classifier-merged',
      purpose: 'hallucination-sentinel',
      description: 'Fact-check classifier used during hallucination mitigation.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      num_classes: 2,
      license: 'apache-2.0',
      model_card_url: 'https://huggingface.co/vllm-sr/mmbert32k-factcheck-classifier-merged',
      tags: ['text-classification', 'fact-check'],
    },
    metadata: {
      model_type: 'mmbert_32k',
      threshold: '0.60',
      use_cpu: 'false',
    },
  },
  {
    name: 'feedback_detector',
    type: 'feedback_detection',
    loaded: true,
    state: 'ready',
    model_path: 'models/mmbert32k-feedback-detector-merged',
    registry: {
      local_path: 'models/mmbert32k-feedback-detector-merged',
      repo_id: 'vllm-sr/mmbert32k-feedback-detector-merged',
      purpose: 'feedback-detection',
      description: 'User feedback classifier for satisfaction and correction signals.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      num_classes: 4,
      license: 'apache-2.0',
      model_card_url: 'https://huggingface.co/vllm-sr/mmbert32k-feedback-detector-merged',
      tags: ['text-classification', 'feedback-detection'],
    },
    metadata: {
      model_type: 'mmbert_32k',
      threshold: '0.70',
      use_cpu: 'false',
    },
  },
  {
    name: 'jailbreak_classifier',
    type: 'security_detection',
    loaded: true,
    state: 'ready',
    model_path: 'models/mmbert32k-jailbreak-detector-merged',
    registry: {
      local_path: 'models/mmbert32k-jailbreak-detector-merged',
      repo_id: 'vllm-sr/mmbert32k-jailbreak-detector-merged',
      purpose: 'jailbreak-detection',
      description: 'Prompt injection and jailbreak detector aligned with the router registry.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      num_classes: 2,
      license: 'apache-2.0',
      model_card_url: 'https://huggingface.co/vllm-sr/mmbert32k-jailbreak-detector-merged',
      tags: ['text-classification', 'security'],
    },
    metadata: {
      model_type: 'mmbert_32k',
      enabled: 'true',
    },
  },
  {
    name: 'mmbert_embedding_model',
    recipe: 'default',
    type: 'embedding',
    loaded: true,
    state: 'ready',
    model_path: 'models/mmbert-embed-32k-2d-matryoshka',
    registry: {
      local_path: 'models/mmbert-embed-32k-2d-matryoshka',
      repo_id: 'vllm-sr/mmbert-embed-32k-2d-matryoshka',
      purpose: 'embedding',
      description: 'Multilingual 2D Matryoshka embedding model with long-context support.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      license: 'apache-2.0',
      model_card_url: 'https://huggingface.co/vllm-sr/mmbert-embed-32k-2d-matryoshka',
      tags: ['embedding', 'matryoshka', 'multilingual'],
    },
    metadata: {
      model_type: 'mmbert',
      provider: 'model_runtime',
      device: 'rocm:0',
      effective_input_tokens: '32768',
      max_sequence_length: '32768',
      default_dimension: '768',
      matryoshka_supported: 'true',
    },
  },
  {
    name: 'pii_classifier',
    type: 'pii_detection',
    loaded: true,
    state: 'ready',
    model_path: 'models/mmbert32k-pii-detector-merged',
    registry: {
      local_path: 'models/mmbert32k-pii-detector-merged',
      repo_id: 'vllm-sr/mmbert32k-pii-detector-merged',
      purpose: 'pii-detection',
      description: 'PII detector for multilingual redaction and routing.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      num_classes: 35,
      license: 'apache-2.0',
      model_card_url: 'https://huggingface.co/vllm-sr/mmbert32k-pii-detector-merged',
      tags: ['token-classification', 'pii'],
    },
    metadata: {
      model_type: 'mmbert_32k',
      threshold: '0.73',
    },
  },
]

const statusPayload = {
  overall: 'healthy',
  deployment_type: 'local',
  services: [
    { name: 'Router', status: 'running', healthy: true },
    { name: 'Dashboard', status: 'running', healthy: true },
  ],
  models: {
    models: routerModels,
    summary: {
      ready: true,
      phase: 'ready',
      message: 'Router models are ready.',
      loaded_models: 6,
      total_models: 6,
      updated_at: '2026-03-11T09:12:00Z',
    },
    system: {
      go_version: 'go1.24.0',
      architecture: 'amd64',
      os: 'linux',
      memory_usage: '256 MB',
      gpu_available: true,
    },
  },
}

const hourlyHistory = (name: string) => ({
  name,
  hours: Array.from({ length: 90 }, (_, index) => ({
    observedAt: new Date(Date.UTC(2026, 7, 24, 0, index * 60)).toISOString(),
    status: index === 41 ? ('starting' as const) : ('operational' as const),
  })),
})

const decisionModelCatalog = {
  system: { decision_model: { deployment: 'routing-judge' } },
  deployments: {
    'routing-judge': {
      provider: 'model_runtime',
      artifact: 'vllm-sr/Vela-2.0-4B',
      device: 'rocm:0',
    },
  },
}

const readyJudge = {
  name: 'routing-judge',
  managed: true,
  served_name: 'routing-judge',
  repo: 'vllm-sr/Vela-2.0-4B',
  ready: true,
  state: 'ready',
  device: 'rocm:0',
  desired_replicas: 1,
  ready_replicas: 1,
}

async function mockRouterInventoryShell(
  page: Page,
  status: unknown = statusPayload,
  config: unknown = { version: 'v0.5', providers: { models: [] }, recipes: [{ name: 'default' }] },
  inventory: unknown = { deployments: [] },
) {
  await mockAuthenticatedAppShell(page, {
    settings: {
      platform: 'amd',
    },
  })

  await page.route('**/api/router/api/v1/inventory/model-runtime', (route) =>
    route.fulfill({ json: inventory }),
  )

  await page.route('**/api/router/config/all', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
  })

  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(status),
    })
  })
}

test.describe('Router model inventory surfaces', () => {
  test('shows the configured decision model, routing questions and the Router-reported runtime', async ({
    page,
  }) => {
    const questions = [
      { name: 'task', question: { type: 'choice' } },
      { name: 'difficulty', question: { type: 'score' } },
      { name: 'precise_facts', question: { type: 'noul' } },
      { name: 'needs', question: { type: 'set' } },
      { name: 'correction', question: { type: 'noul' } },
    ]
    const config = {
      version: 'v0.5',
      global: { model_catalog: decisionModelCatalog },
      providers: { models: [] },
      entrypoints: [{ recipe: 'balanced', model_names: ['vllm-sr/balanced'] }],
      recipes: [
        {
          name: 'balanced',
          signals: {
            systemone: questions,
            pii: [{ name: 'personal_data' }],
            jailbreak: [{ name: 'prompt_attack' }],
            safety: [{ name: 'unsafe_request' }],
          },
          projections: { scores: [{ name: 'effort_score' }], mappings: [{ name: 'effort' }] },
        },
      ],
    }
    const models = ['pii_classifier', 'jailbreak_classifier', 'safety.unsafe_request'].map(
      (name) => ({
        name,
        recipe: 'balanced',
        type: 'classification',
        loaded: true,
        state: 'ready',
        model_path: 'a'.repeat(64),
        metadata: {
          deployment: 'routing-judge',
          resource_id: 'shared-vela',
          provider: 'model_runtime',
          device: 'rocm:0',
        },
      }),
    )
    await mockRouterInventoryShell(page, { ...statusPayload, models: { models } }, config)
    await page.goto('/dashboard')

    const intelligence = page.getByRole('region', { name: 'Router Intelligence' })
    const overview = intelligence.getByTestId('decision-model-overview')
    await expect(overview).toContainText('Vela-2.0-4B')
    // The model runtime does not list the deployment, so the Router's bindings speak for it.
    await expect(overview).toContainText('Ready')
    await expect(overview).toContainText('The Router reports it ready')
    await expect(
      intelligence.getByText('5 questions · 3 other signals · 2 projections'),
    ).toBeVisible()
    await intelligence.locator('summary').filter({ hasText: 'balanced' }).click()
    for (const question of questions) {
      await expect(intelligence.getByText(question.name, { exact: true })).toBeVisible()
    }
    await expect(intelligence.getByText('effort_score', { exact: true })).toBeVisible()
    await expect(intelligence.getByText('effort', { exact: true })).toBeVisible()
    await expect(intelligence.getByText('personal_data', { exact: true })).toBeVisible()
    await expect(intelligence.getByTestId('router-model-grid-preview')).toHaveCount(0)
    await expect(intelligence.getByText('Runtime inventory')).toHaveCount(0)

    await intelligence.getByRole('button', { name: 'Manage decision model' }).click()
    await expect(page).toHaveURL(/\/models\/decision(\?|$)/)
  })

  test('shows one decision model deployment state on Home and Status', async ({ page }) => {
    await mockRouterInventoryShell(
      page,
      statusPayload,
      { global: { model_catalog: decisionModelCatalog } },
      { deployments: [readyJudge] },
    )
    await page.goto('/dashboard')
    const overview = page.getByTestId('decision-model-overview')
    await expect(overview).toContainText('Vela-2.0-4B')
    await expect(overview).toContainText('Ready')
    await expect(overview).toContainText('rocm:0 · 1 of 1 replicas ready')

    await page.goto('/observe/status')
    const summary = page.getByTestId('status-decision-model')
    await expect(summary).toContainText('Vela-2.0-4B · routing-judge')
    await expect(summary).toContainText('Ready')
    await expect(summary).toContainText('rocm:0 · 1 of 1 replicas ready')
    await summary.getByRole('button', { name: 'Decision models' }).click()
    await expect(page).toHaveURL(/\/models\/decision(\?|$)/)
  })

  test('does not claim a configured decision model is ready without runtime evidence', async ({
    page,
  }) => {
    await mockRouterInventoryShell(page, statusPayload, {
      global: { model_catalog: decisionModelCatalog },
    })
    await page.goto('/dashboard')
    const overview = page.getByTestId('decision-model-overview')
    await expect(overview).toContainText('Vela-2.0-4B')
    await expect(overview).toContainText('Not reported')
    await expect(overview).toContainText(
      'The model runtime has not reported the routing-judge deployment yet',
    )
    await expect(overview).not.toContainText('Ready')

    await page.goto('/observe/status')
    await expect(page.getByTestId('status-decision-model')).toContainText('Not reported')
  })

  test('keeps service availability useful when router model metadata is absent', async ({
    page,
  }) => {
    await mockRouterInventoryShell(page, {
      ...statusPayload,
      models: {
        ...statusPayload.models,
        models: null,
        summary: {
          ...statusPayload.models.summary,
          loaded_models: 0,
          total_models: 0,
        },
      },
    })

    await page.goto('/observe/status')

    await expect(page.getByTestId('status-overview')).toContainText('Healthy')
    await expect(page.getByTestId('status-services-section')).toContainText('Router')
    await expect(page.getByTestId('status-services-section')).toContainText('Dashboard')
  })

  test('keeps the legacy classifier and embedding inventory off Home and Status', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1600, height: 1000 })
    await mockRouterInventoryShell(page)

    await page.goto('/dashboard')
    await expect(page.getByTestId('decision-model-overview')).toBeVisible()
    await expect(page.locator('[data-testid^="router-model-preview-"]')).toHaveCount(0)
    await expect(page.getByText('Runtime inventory')).toHaveCount(0)
    await expect(page.getByText(routerModels[4].registry.repo_id)).toHaveCount(0)

    await page.goto('/observe/status')
    await expect(page.getByTestId('status-router')).toBeVisible()
    await expect(page.getByTestId('status-decision-model')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Router models' })).toHaveCount(0)
    await expect(page.getByText('No router models reported')).toHaveCount(0)
    await expect(page.getByText(/classifiers? and embedding models/i)).toHaveCount(0)
    await expect(page.getByText(routerModels[0].registry.repo_id)).toHaveCount(0)
  })

  test('makes degraded service health explicit without hiding healthy services', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1600, height: 900 })

    await mockRouterInventoryShell(page, {
      ...statusPayload,
      overall: 'degraded',
      services: [
        ...statusPayload.services,
        { name: 'Telemetry', status: 'unavailable', healthy: false },
      ],
    })
    await page.goto('/observe/status')

    const overview = page.getByTestId('status-overview')
    await expect(overview).toContainText('Degraded')
    const services = page.getByTestId('status-services-section')
    await expect(services).toContainText('Router')
    await expect(services).toContainText('Operational')
    await expect(services).toContainText('Telemetry')
    await expect(services).toContainText('Unavailable')
  })

  test('renders a keyboard-accessible 90-hour service history', async ({ page }) => {
    await page.setViewportSize({ width: 1600, height: 900 })

    await mockRouterInventoryShell(page, {
      ...statusPayload,
      history: {
        windowHours: 90,
        through: '2026-08-27T17:00:00.000Z',
        services: [hourlyHistory('Router'), hourlyHistory('Dashboard')],
      },
    })
    await page.goto('/observe/status')

    const overview = page.getByTestId('status-overview')
    const servicesSection = page.getByTestId('status-services-section')
    await expect(overview).toContainText('Healthy')
    await expect(servicesSection.getByText('90-hour observed history')).toBeVisible()
    const routerRow = servicesSection.locator('article').filter({ hasText: 'Router' })
    const hours = routerRow.locator('[data-status-history-hour]')
    await expect(hours).toHaveCount(90)
    await hours.nth(89).focus()
    await expect(hours.nth(89)).toBeFocused()
    await page.keyboard.press('ArrowLeft')
    await expect(hours.nth(88)).toBeFocused()
    await expect(routerRow.getByRole('tooltip')).toContainText('UTC: Operational')
  })

  test('stacks the status overview without introducing horizontal overflow on mobile', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 })

    await mockRouterInventoryShell(page, {
      ...statusPayload,
      deployment_type: 'none',
      services: [],
      models: { models: [] },
    })
    await page.goto('/observe/status')

    await expect(page.getByRole('heading', { name: 'Status', exact: true, level: 1 })).toBeVisible()
    const overview = page.getByTestId('status-overview')
    await expect(overview).toBeVisible()
    await expect(overview).toContainText('No running services detected')
    await expect(overview).toContainText('Availability will appear when the Router starts.')
    await expect(page.getByLabel('Refresh system status')).toBeVisible()

    const pageMetrics = await page.getByTestId('status-page').evaluate((node) => ({
      clientWidth: node.clientWidth,
      scrollWidth: node.scrollWidth,
    }))
    expect(pageMetrics.scrollWidth).toBeLessThanOrEqual(pageMetrics.clientWidth + 1)

    const servicesSection = page.getByTestId('status-services-section')
    await servicesSection.scrollIntoViewIfNeeded()
    await expect(servicesSection).toContainText('No services reported')
    const servicesMetrics = await servicesSection.evaluate((node) => ({
      clientWidth: node.clientWidth,
      scrollWidth: node.scrollWidth,
    }))
    expect(servicesMetrics.scrollWidth).toBeLessThanOrEqual(servicesMetrics.clientWidth + 1)
  })

  test('lines the Router and Deployment cards up and keeps in-cluster addresses off the page', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1600, height: 1000 })
    await mockRouterInventoryShell(page, {
      ...statusPayload,
      version: 'v0.5.0-dev.e1ba40498f',
      endpoints: ['http://router:8080'],
      router_runtime: { phase: 'ready', ready: true, message: 'Router configuration is active.' },
      history: {
        windowHours: 90,
        through: '2026-08-27T17:00:00.000Z',
        services: [hourlyHistory('Router'), hourlyHistory('Dashboard')],
      },
    })
    await page.goto('/observe/status')

    const router = page.getByTestId('status-router')
    const deployment = page.getByTestId('status-deployment')
    await expect(deployment).toContainText('v0.5.0-dev.e1ba40498f')
    const [routerBox, deploymentBox] = await Promise.all([
      router.boundingBox(),
      deployment.boundingBox(),
    ])
    expect(routerBox).not.toBeNull()
    expect(deploymentBox).not.toBeNull()
    expect(Math.abs(routerBox!.y - deploymentBox!.y)).toBeLessThanOrEqual(1)
    expect(Math.abs(routerBox!.height - deploymentBox!.height)).toBeLessThanOrEqual(1)

    await expect(page.getByText('Public endpoints')).toHaveCount(0)
    await expect(page.getByText('http://router:8080')).toHaveCount(0)
    await expect(page.getByTestId('status-decision-model')).toBeVisible()
  })
})
