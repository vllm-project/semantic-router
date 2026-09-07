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
      repo_id: 'llm-semantic-router/mmbert32k-intent-classifier-merged',
      purpose: 'domain-classification',
      description: 'Merged intent classifier for multilingual routing decisions.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      num_classes: 14,
      license: 'apache-2.0',
      model_card_url:
        'https://huggingface.co/llm-semantic-router/mmbert32k-intent-classifier-merged',
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
      repo_id: 'llm-semantic-router/mmbert32k-factcheck-classifier-merged',
      purpose: 'hallucination-sentinel',
      description: 'Fact-check classifier used during hallucination mitigation.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      num_classes: 2,
      license: 'apache-2.0',
      model_card_url:
        'https://huggingface.co/llm-semantic-router/mmbert32k-factcheck-classifier-merged',
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
      repo_id: 'llm-semantic-router/mmbert32k-feedback-detector-merged',
      purpose: 'feedback-detection',
      description: 'User feedback classifier for satisfaction and correction signals.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      num_classes: 4,
      license: 'apache-2.0',
      model_card_url:
        'https://huggingface.co/llm-semantic-router/mmbert32k-feedback-detector-merged',
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
      repo_id: 'llm-semantic-router/mmbert32k-jailbreak-detector-merged',
      purpose: 'jailbreak-detection',
      description: 'Prompt injection and jailbreak detector aligned with the router registry.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      num_classes: 2,
      license: 'apache-2.0',
      model_card_url:
        'https://huggingface.co/llm-semantic-router/mmbert32k-jailbreak-detector-merged',
      tags: ['text-classification', 'security'],
    },
    metadata: {
      model_type: 'mmbert_32k',
      enabled: 'true',
    },
  },
  {
    name: 'mmbert_embedding_model',
    type: 'embedding',
    loaded: true,
    state: 'ready',
    model_path: 'models/mmbert-embed-32k-2d-matryoshka',
    registry: {
      local_path: 'models/mmbert-embed-32k-2d-matryoshka',
      repo_id: 'llm-semantic-router/mmbert-embed-32k-2d-matryoshka',
      purpose: 'embedding',
      description: 'Multilingual 2D Matryoshka embedding model with long-context support.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      license: 'apache-2.0',
      model_card_url: 'https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka',
      tags: ['embedding', 'matryoshka', 'multilingual'],
    },
    metadata: {
      model_type: 'mmbert',
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
      repo_id: 'llm-semantic-router/mmbert32k-pii-detector-merged',
      purpose: 'pii-detection',
      description: 'PII detector for multilingual redaction and routing.',
      parameter_size: '307M',
      embedding_dim: 768,
      max_context_length: 32768,
      num_classes: 35,
      license: 'apache-2.0',
      model_card_url: 'https://huggingface.co/llm-semantic-router/mmbert32k-pii-detector-merged',
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

async function mockRouterInventoryShell(page: Page, status: unknown = statusPayload) {
  await mockAuthenticatedAppShell(page, {
    settings: {
      platform: 'amd',
    },
  })

  await page.route('**/api/router/config/all', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        signals: {},
        decisions: [],
        providers: { models: [] },
        plugins: {},
      }),
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
  test('keeps service availability useful when router model metadata is absent', async ({ page }) => {
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

    await page.goto('/status')

    await expect(page.getByTestId('status-overview')).toContainText('Healthy')
    await expect(page.getByTestId('status-services-section')).toContainText('Router')
    await expect(page.getByTestId('status-services-section')).toContainText('Dashboard')
  })

  test('renders six preview cards and opens the canonical Models workspace', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1920, height: 1200 })

    await mockRouterInventoryShell(page)

    await page.goto('/dashboard')

    const previewGrid = page.getByTestId('router-model-grid-preview')
    await expect(previewGrid.locator('[data-testid^="router-model-preview-"]')).toHaveCount(6)

    const embeddingPreview = page.getByTestId('router-model-preview-mmbert_embedding_model')
    await expect(embeddingPreview).toContainText('models/mmbert-embed-32k-2d-matryoshka')
    await expect(embeddingPreview).toContainText('Embedding')
    await expect(embeddingPreview).not.toContainText('MmBertEmbeddingModel(')
    await expect(previewGrid.getByAltText('AMD platform')).toHaveCount(6)
    await expect(page.getByAltText('AMD', { exact: true })).toBeVisible()
    await expect(page.getByText('AMD GPU', { exact: true })).toHaveCount(0)

    await embeddingPreview.click()
    await expect(page).toHaveURL(/\/config\/models$/)
    await expect(page.getByRole('heading', { name: 'Models', exact: true }).first()).toBeVisible()
  })

  test('makes degraded service health explicit without hiding healthy services', async ({ page }) => {
    await page.setViewportSize({ width: 1600, height: 900 })

    await mockRouterInventoryShell(page, {
      ...statusPayload,
      overall: 'degraded',
      services: [
        ...statusPayload.services,
        { name: 'Telemetry', status: 'unavailable', healthy: false },
      ],
    })
    await page.goto('/status')

    const overview = page.getByTestId('status-overview')
    await expect(overview).toContainText('Degraded')
    const services = page.getByTestId('status-services-section')
    await expect(services).toContainText('Router')
    await expect(services).toContainText('Operational')
    await expect(services).toContainText('Telemetry')
    await expect(services).toContainText('Unavailable')
  })

  test('renders a keyboard-accessible 90-hour service history', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1600, height: 900 })

    await mockRouterInventoryShell(page, {
      ...statusPayload,
      history: {
        windowHours: 90,
        through: '2026-08-27T17:00:00.000Z',
        services: [hourlyHistory('Router'), hourlyHistory('Dashboard')],
      },
    })
    await page.goto('/status')

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
    await page.goto('/status')

    await expect(page.getByRole('heading', { name: 'System status' })).toBeVisible()
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

})
