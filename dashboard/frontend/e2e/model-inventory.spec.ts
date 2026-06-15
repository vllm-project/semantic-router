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
      model_card_url: 'https://huggingface.co/llm-semantic-router/mmbert32k-intent-classifier-merged',
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
      model_card_url: 'https://huggingface.co/llm-semantic-router/mmbert32k-factcheck-classifier-merged',
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
      model_card_url: 'https://huggingface.co/llm-semantic-router/mmbert32k-feedback-detector-merged',
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
      model_card_url: 'https://huggingface.co/llm-semantic-router/mmbert32k-jailbreak-detector-merged',
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

async function mockRouterInventoryShell(page: Page) {
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
      body: JSON.stringify(statusPayload),
    })
  })
}

test.describe('Router model inventory surfaces', () => {
  test('renders six preview cards and keeps embedding metadata clean in status view', async ({ page }) => {
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

    await embeddingPreview.click()
    await expect(page).toHaveURL(/\/status#model-mmbert-embedding-model$/)

    const fullCard = page.getByTestId('router-model-full-mmbert_embedding_model')
    await expect(fullCard).toContainText('Identity')
    await expect(fullCard).toContainText('Capabilities')
    await expect(fullCard).toContainText('Runtime & Config')
    await expect(fullCard).toContainText('models/mmbert-embed-32k-2d-matryoshka')
    await expect(fullCard).not.toContainText('MmBertEmbeddingModel(')
    await expect(fullCard.getByAltText('AMD platform')).toBeVisible()
  })

  test('keeps status model inventory and services reachable inside the page scroll container', async ({ page }) => {
    await page.setViewportSize({ width: 1600, height: 900 })

    await mockRouterInventoryShell(page)
    await page.goto('/status')

    const statusPage = page.getByTestId('status-page')
    const inventorySection = page.getByTestId('status-model-inventory-section')
    const servicesSection = page.getByTestId('status-services-section')
    const lastModelCard = page.getByTestId('router-model-full-pii_classifier')

    const metrics = await statusPage.evaluate((node) => ({
      overflowY: window.getComputedStyle(node).overflowY,
      scrollHeight: node.scrollHeight,
      clientHeight: node.clientHeight,
    }))

    expect(metrics.overflowY).toBe('auto')
    expect(metrics.scrollHeight).toBeGreaterThan(metrics.clientHeight)

    await expect(inventorySection).toBeVisible()
    await lastModelCard.scrollIntoViewIfNeeded()
    await expect(lastModelCard).toBeInViewport()

    await servicesSection.scrollIntoViewIfNeeded()
    await expect(servicesSection).toBeInViewport()
    await expect(servicesSection).toContainText('Router')
    await expect(servicesSection).toContainText('Dashboard')
  })
})
