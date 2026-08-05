import { expect, test, type Locator, type Page } from '@playwright/test';
import { mockAuthenticatedSession } from './support/auth';

const configResponse = {
  version: 'v0.3',
  listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
  providers: {
    defaults: {
      default_model: 'test-model',
      reasoning_families: {
        qwen3: {
          type: 'reasoning_effort',
          parameter: 'reasoning_effort',
        },
      },
    },
    models: [
      {
        name: 'test-model',
        provider_model_id: 'test-model',
        backend_refs: [
          {
            name: 'endpoint1',
            endpoint: '127.0.0.1:8000',
            protocol: 'http',
          },
        ],
      },
    ],
  },
  routing: {
    modelCards: [{ name: 'test-model', description: 'Primary routing model' }],
    signals: {
      domains: [{ name: 'business', description: 'Business' }],
      keywords: [{ name: 'pricing', value: 'pricing' }],
    },
    decisions: [
      {
        name: 'business-route',
        priority: 1,
        description: 'Route business requests',
        rules: {
          operator: 'AND',
          conditions: [{ type: 'domain', name: 'business' }],
        },
        modelRefs: [
          {
            model: 'test-model',
            use_reasoning: true,
            reasoning_effort: 'medium',
            lora_name: 'adapter-v1',
            weight: 1,
            reasoning_description: 'Use chain-of-thought for harder tasks',
          },
        ],
      },
    ],
  },
  global: {
    router: { strategy: 'priority' },
    services: {
      response_api: { enabled: true },
    },
    model_catalog: {
      modules: {
        prompt_guard: {
          enabled: true,
          model_ref: 'prompt_guard',
          threshold: 0.7,
          use_cpu: true,
        },
      },
      system: {
        prompt_guard: 'models/mmbert32k-jailbreak-detector-merged',
        domain_classifier: 'models/mmbert32k-intent-classifier-merged',
        pii_classifier: 'models/mmbert32k-pii-detector-merged',
      },
      embeddings: {
        semantic: {
          mmbert_model_path: 'models/mmbert-embed-32k-2d-matryoshka',
          bert_model_path: 'models/mom-embedding-bert',
          use_cpu: true,
          embedding_config: {
            model_type: 'mmbert',
            preload_embeddings: true,
            target_dimension: 768,
            target_layer: 22,
          },
        },
      },
    },
  },
};

const rawGlobalYaml = `router:
  strategy: priority
services:
  response_api:
    enabled: true
model_catalog:
  modules:
    prompt_guard:
      enabled: true
      model_ref: prompt_guard
      threshold: 0.7
      use_cpu: true
  system:
    prompt_guard: models/mmbert32k-jailbreak-detector-merged
    domain_classifier: models/mmbert32k-intent-classifier-merged
    pii_classifier: models/mmbert32k-pii-detector-merged
  embeddings:
    semantic:
      mmbert_model_path: models/mmbert-embed-32k-2d-matryoshka
      bert_model_path: models/mom-embedding-bert
      use_cpu: true
`;

const taxonomyClassifierResponse = {
  items: [
    {
      name: 'privacy_classifier',
      type: 'taxonomy',
      builtin: true,
      managed: true,
      editable: false,
      threshold: 0.3,
      security_threshold: 0.25,
      description: 'Built-in privacy routing taxonomy',
      source: {
        path: 'knowledge_bases/privacy/',
        taxonomy_file: 'taxonomy.json',
      },
      tiers: [
        { name: 'privacy_policy', description: 'Sensitive company content' },
        { name: 'frontier_reasoning', description: 'General frontier reasoning' },
      ],
      categories: [
        {
          name: 'proprietary_code',
          tier: 'privacy_policy',
          description: 'Internal code and repositories',
          exemplars: ['Review our private codebase'],
        },
        {
          name: 'general_research',
          tier: 'frontier_reasoning',
          description: 'Open research requests',
          exemplars: ['Summarize this public paper'],
        },
      ],
      tier_groups: {
        privacy_categories: ['proprietary_code'],
      },
      signal_references: [
        {
          name: 'privacy_policy',
          bind: {
            kind: 'tier',
            value: 'privacy_policy',
          },
        },
      ],
      bind_options: {
        tiers: ['privacy_policy', 'frontier_reasoning'],
        categories: ['proprietary_code', 'general_research'],
      },
    },
    {
      name: 'research_classifier',
      type: 'taxonomy',
      builtin: false,
      managed: true,
      editable: true,
      threshold: 0.41,
      security_threshold: 0.28,
      description: 'Custom research classifier',
      source: {
        path: 'classifiers/custom/research_classifier/',
        taxonomy_file: 'taxonomy.json',
      },
      tiers: [
        { name: 'internal', description: 'Private research assets' },
        { name: 'external', description: 'Public artifacts' },
      ],
      categories: [
        {
          name: 'lab_notes',
          tier: 'internal',
          description: 'Private notes',
          exemplars: ['Review our lab notes'],
        },
        {
          name: 'papers',
          tier: 'external',
          description: 'Published papers',
          exemplars: ['Summarize this paper'],
        },
      ],
      tier_groups: {},
      signal_references: [],
      bind_options: {
        tiers: ['internal', 'external'],
        categories: ['lab_notes', 'papers'],
      },
    },
  ],
};

async function mockConfigSurface(page: Page) {
  await mockAuthenticatedSession(page);

  await page.route('**/api/auth/bootstrap/can-register', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ canRegister: false }),
    });
  });

  await page.route('**/api/setup/state', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        setupMode: false,
        listenerPort: 8000,
        models: 1,
        decisions: 1,
        hasModels: true,
        hasDecisions: true,
        canActivate: true,
      }),
    });
  });

  await page.route('**/api/settings', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        readonlyMode: false,
        setupMode: false,
        platform: '',
        envoyUrl: '',
      }),
    });
  });

  await page.route('**/api/mcp/tools', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ tools: [] }) });
  });

  await page.route('**/api/mcp/servers', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) });
  });

  await page.route('**/api/router/config/all', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(configResponse) });
  });

  await page.route('**/api/router/config/global', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(configResponse.global),
    });
  });

  await page.route('**/api/router/config/global/raw', async route => {
    await route.fulfill({ status: 200, contentType: 'text/yaml', body: rawGlobalYaml });
  });

  await page.route('**/api/router/config/classifiers', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(taxonomyClassifierResponse),
    });
  });

  await page.route('**/api/status', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        overall: 'healthy',
        deployment_type: 'local',
        services: [{ name: 'router', status: 'healthy', healthy: true }],
        models: { models: [], summary: { loaded_models: 0, total_models: 0 } },
      }),
    });
  });
}

async function expectInside(container: Locator, child: Locator) {
  await child.scrollIntoViewIfNeeded();

  const [containerBox, childBox] = await Promise.all([
    container.boundingBox(),
    child.boundingBox(),
  ]);

  expect(containerBox).not.toBeNull();
  expect(childBox).not.toBeNull();

  const containerBounds = containerBox!;
  const childBounds = childBox!;

  expect(childBounds.x).toBeGreaterThanOrEqual(containerBounds.x - 1);
  expect(childBounds.y).toBeGreaterThanOrEqual(containerBounds.y - 1);
  expect(childBounds.x + childBounds.width).toBeLessThanOrEqual(containerBounds.x + containerBounds.width + 1);
  expect(childBounds.y + childBounds.height).toBeLessThanOrEqual(containerBounds.y + containerBounds.height + 1);
}

test.describe('Config surface layout regressions', () => {
  test('keeps decision model references editor controls inside the modal layout', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1100 });
    await mockConfigSurface(page);

    await page.goto('/config/decisions');
    const addDecisionButton = page.getByRole('button', { name: 'Add Decision' });
    await addDecisionButton.click();

    const modal = page.getByRole('dialog', { name: 'Add Decision' });
    const form = modal.locator('form').first();
    await expect(modal).toBeVisible();
    await expect(form).toBeVisible();

    const controls = [
      page.getByLabel('Model').first(),
      page.getByLabel('Reasoning effort').first(),
      page.getByLabel('Use reasoning').first(),
      page.getByLabel('LoRA adapter').first(),
      page.getByLabel('Weight').first(),
      page.getByLabel('Reasoning description').first(),
      page.getByRole('button', { name: 'Remove model reference' }),
    ];

    for (const control of controls) {
      await expect(control).toBeVisible();
      await expectInside(modal, control);
    }

    const [modalBox, viewport] = await Promise.all([
      modal.boundingBox(),
      page.viewportSize(),
    ]);
    expect(modalBox).not.toBeNull();
    expect(viewport).not.toBeNull();
    expect(modalBox!.x + modalBox!.width).toBeCloseTo(viewport!.width, 0);
    expect(modalBox!.y).toBeCloseTo(0, 0);
    expect(modalBox!.height).toBeCloseTo(viewport!.height, 0);

    const closeButton = modal.getByRole('button', { name: 'Close editor' });
    await expect(closeButton).toBeFocused();
    await modal.getByRole('button', { name: 'Add', exact: true }).focus();
    await page.keyboard.press('Tab');
    await expect(closeButton).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(modal).toBeHidden();
    await expect(addDecisionButton).toBeFocused();
  });

  test('expands the shared editor drawer to the mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await mockConfigSurface(page);

    await page.goto('/config/decisions');
    await page.getByRole('button', { name: 'Add Decision' }).click();

    const modal = page.getByRole('dialog', { name: 'Add Decision' });
    await expect(modal).toHaveCSS('animation-name', 'none');
    await expect.poll(async () => Math.round((await modal.boundingBox())?.x ?? -1)).toBe(0);
    const modalBox = await modal.boundingBox();
    expect(modalBox).not.toBeNull();
    expect(modalBox!.x).toBeCloseTo(0, 0);
    expect(modalBox!.y).toBeCloseTo(0, 0);
    expect(modalBox!.width).toBeCloseTo(390, 0);
    expect(modalBox!.height).toBeCloseTo(844, 0);
  });

  test('stacks model catalog section path and actions cleanly inside global cards', async ({ page }) => {
    await page.setViewportSize({ width: 1600, height: 1200 });
    await mockConfigSurface(page);

    await page.goto('/config/global-config');
    await expect(page.getByRole('heading', { name: 'Global Config Overview' })).toBeVisible();

    const systemBindingsCard = page.locator('article').filter({
      has: page.getByRole('heading', { name: 'System Model Bindings' }),
    }).first();
    const embeddingsCard = page.locator('article').filter({
      has: page.getByRole('heading', { name: 'Embedding Models' }),
    }).first();

    await expect(systemBindingsCard).toBeVisible();
    await expect(embeddingsCard).toBeVisible();
    await expect(systemBindingsCard.getByText('models/.../mmbert32k-jailbreak-detector-merged')).toBeVisible();
    await expect(embeddingsCard.getByText('models/mmbert-embed-32k-2d-matryoshka')).toBeVisible();

    const systemPath = systemBindingsCard.getByText('global.model_catalog.system');
    const systemButton = systemBindingsCard.getByRole('button', { name: 'Edit Section' });
    const embeddingPath = embeddingsCard.getByText('global.model_catalog.embeddings');
    const embeddingButton = embeddingsCard.getByRole('button', { name: 'Edit Section' });

    for (const [card, path, button] of [
      [systemBindingsCard, systemPath, systemButton],
      [embeddingsCard, embeddingPath, embeddingButton],
    ] as const) {
      await expect(path).toBeVisible();
      await expect(button).toBeVisible();
      await expectInside(card, path);
      await expectInside(card, button);

      const [pathBox, buttonBox] = await Promise.all([path.boundingBox(), button.boundingBox()]);
      expect(pathBox).not.toBeNull();
      expect(buttonBox).not.toBeNull();
      expect(buttonBox!.y).toBeGreaterThan(pathBox!.y + pathBox!.height - 1);
    }

    await expect(page.getByRole('heading', { name: 'Classifier Catalog' })).toHaveCount(0);
  });

  test('redirects the retired classifier route to the canonical knowledge base manager', async ({ page }) => {
    await page.setViewportSize({ width: 1600, height: 1200 });
    await mockConfigSurface(page);

    await page.goto('/config/classifiers');

    await expect(page).toHaveURL(/\/knowledge-bases\/bases$/);
    await expect(page.getByRole('heading', { name: 'Knowledge Bases' })).toBeVisible();
  });
});
