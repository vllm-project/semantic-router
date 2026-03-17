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
          mmbert_model_path: 'models/mom-embedding-ultra',
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
      mmbert_model_path: models/mom-embedding-ultra
      bert_model_path: models/mom-embedding-bert
      use_cpu: true
`;

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
    await page.getByRole('button', { name: 'Add Decision' }).click();

    const modal = page.locator('[class*="EditModal_modal"]').filter({
      has: page.getByRole('heading', { name: 'Add Decision' }),
    }).first();
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
    await expect(embeddingsCard.getByText('models/mom-embedding-ultra')).toBeVisible();

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
  });
});
