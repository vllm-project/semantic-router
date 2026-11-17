import { expect, test, type Locator, type Page } from '@playwright/test';
import { mockAuthenticatedAppShell } from './support/auth';

const configResponse = {
  version: 'v0.5',
  listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
  providers: {
    defaults: { model: 'test-model' },
    models: [
      {
        name: 'test-model',
        provider_model_id: 'test-model',
        reasoning: { family: 'qwen3.8' },
        backend_refs: [
          {
            name: 'endpoint1',
            endpoint: '127.0.0.1:8000',
            protocol: 'http',
          },
        ],
        description: 'Primary routing model',
      },
    ],
  },
  recipes: [
    {
      name: 'default',
      signals: {
        domains: [{ name: 'business', description: 'Business' }],
        keywords: [{ name: 'pricing', operator: 'OR', keywords: ['pricing'], case_sensitive: false }],
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
  ],
  global: {
    router: { strategy: 'priority' },
    services: {
      response_api: { enabled: true },
    },
    model_catalog: {
      system: {
        decision_model: { deployment: 'main-judge' },
      },
      bindings: {
        jailbreak: 'guard-judge',
      },
      embeddings: {
        default: 'qwen3-embedding',
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
  system:
    decision_model: {deployment: main-judge}
  bindings:
    jailbreak: guard-judge
  embeddings:
    default: {deployment: qwen3-embedding}
`;

async function mockConfigSurface(page: Page) {
  await mockAuthenticatedAppShell(page, {
    settings: { platform: '', readonlyMode: false },
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
  test('opens decision model settings directly and saves the decision model and a binding', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await mockConfigSurface(page);

    let decisionModel = 'main-judge';
    let patch: Record<string, unknown> | undefined;
    const deployments = {
      'main-judge': { provider: 'model_runtime', artifact: 'vllm-sr/Vela-2.0-4B' },
      'alternate-judge': { provider: 'model_runtime', artifact: 'vllm-sr/Vela-2.0-0.8B' },
    };
    await page.route('**/api/router/config/global', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          model_catalog: { system: { decision_model: { deployment: decisionModel } }, deployments },
        }),
      });
    });
    await page.route('**/api/router/config/global/update', async route => {
      patch = route.request().postDataJSON().model_catalog;
      decisionModel = 'alternate-judge';
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{"status":"success","warnings":[]}' });
    });

    await page.goto('/config/global-config#global-section-system_models');
    const card = page.locator('#global-section-system_models');
    await expect(card.getByRole('heading', { name: 'Decision Model & Bindings' })).toBeInViewport();
    await expect(card.getByText('main-judge', { exact: true })).toBeVisible();
    await card.getByRole('button', { name: 'Edit', exact: true }).click();

    const modal = page.getByRole('dialog', { name: 'Edit Decision Model & Bindings' });
    await expect(modal.getByLabel(/^Decision Model/)).toHaveValue('main-judge');
    await modal.getByLabel(/^Decision Model/).selectOption('alternate-judge');
    await modal.getByLabel('Domain', { exact: true }).selectOption('alternate-judge');
    await modal.getByRole('button', { name: 'Save', exact: true }).click();
    await expect(modal).toBeHidden();
    expect(patch).toEqual({
      system: { decision_model: { deployment: 'alternate-judge' } },
      bindings: { domain: 'alternate-judge' },
    });
    await expect(card.getByText('alternate-judge', { exact: true })).toBeVisible();
  });

  test('opens a new signal of any type in its drawer with schema-driven controls', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1100 });
    await mockConfigSurface(page);

    await page.goto('/routing/signals');
    await page.getByRole('button', { name: 'New signal' }).click();
    await page.getByRole('menuitem', { name: /Keywords/ }).click();

    const sheet = page.getByRole('dialog', { name: 'New keywords signal' });
    await expect(sheet).toBeVisible();
    await expect(sheet.getByRole('combobox', { name: 'Signal type' })).toBeVisible();
    await expectInside(sheet, sheet.getByRole('textbox', { name: 'Keywords' }));
    await sheet.getByRole('button', { name: /more settings at their defaults/ }).click();
    const caseSensitive = sheet.getByRole('switch', { name: 'Case Sensitive' });
    await expectInside(sheet, caseSensitive);
    await caseSensitive.click();
    await expect(caseSensitive).toBeChecked();
  });

  test('uses the same hierarchy for projection editors', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1100 });
    await mockConfigSurface(page);

    await page.goto('/routing/projections');
    await page.getByRole('button', { name: 'New projection' }).first().click();
    await page.getByRole('menuitem', { name: /Partition/ }).click();

    const sheet = page.getByRole('dialog', { name: 'New partition' });
    await expect(sheet.getByRole('heading', { name: 'Definition' })).toBeVisible();
    await expectInside(sheet, sheet.getByRole('combobox', { name: 'Semantics' }));
    await expectInside(sheet, sheet.getByRole('textbox', { name: 'Members' }));
  });

  test('uses the shared hierarchy for signal and decision drawers', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1100 });
    await mockConfigSurface(page);

    await page.goto('/routing/signals');
    await page.getByRole('row', { name: /pricing/ }).click();
    const signalSheet = page.getByRole('dialog', { name: 'pricing' });
    await expect(signalSheet.getByRole('heading', { name: 'Identity' })).toBeVisible();
    await expect(signalSheet.getByRole('heading', { name: 'Definition' })).toBeVisible();
    await expect(page).toHaveURL(/id=keyword%3Apricing/);
    await signalSheet.getByRole('button', { name: 'Close' }).last().click();
    await expect(signalSheet).toBeHidden();

    await page.goto('/routing/decisions');
    await page.getByRole('row', { name: /business-route/ }).click();
    const decisionSheet = page.getByRole('dialog', { name: 'business-route' });
    for (const tab of ['Overview', 'Rules', 'Models', 'Algorithm', 'Plugins', 'Advanced', 'Try it', 'YAML']) {
      await expect(decisionSheet.getByRole('tab', { name: new RegExp(`^${tab}`) })).toBeVisible();
    }
  });

  test('keeps decision model reference controls inside the drawer layout', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1100 });
    await mockConfigSurface(page);

    await page.goto('/routing/decisions?id=business-route&view=models');
    const sheet = page.getByRole('dialog', { name: 'business-route' });
    const controls = [
      sheet.getByRole('combobox', { name: 'Model', exact: true }),
      sheet.getByRole('radiogroup', { name: 'Use reasoning' }),
      sheet.getByRole('combobox', { name: 'Reasoning effort' }),
      sheet.getByRole('textbox', { name: 'LoRA adapter' }),
      sheet.getByRole('textbox', { name: 'Weight' }),
      sheet.getByRole('textbox', { name: 'Reasoning Description' }),
      sheet.getByRole('button', { name: 'Remove test-model' }),
    ];
    for (const control of controls) {
      await expect(control).toBeVisible();
      await expectInside(sheet, control);
    }

    await sheet.getByRole('textbox', { name: 'Weight' }).fill('2');
    await page.keyboard.press('Escape');
    const discardDialog = page.getByRole('alertdialog', { name: 'Discard unsaved changes?' });
    await expect(discardDialog).toBeVisible();
    await discardDialog.getByRole('button', { name: 'Discard changes' }).click();
    await expect(sheet).toBeHidden();
  });

  test('fits the decision drawer inside the mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await mockConfigSurface(page);

    await page.goto('/routing/decisions');
    await page.getByRole('button', { name: 'New decision' }).click();

    const sheet = page.getByRole('dialog', { name: 'New decision' });
    await expect(sheet).toBeVisible();
    const box = await sheet.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(390);
    expect(box!.y).toBeGreaterThanOrEqual(0);
    expect(box!.y + box!.height).toBeLessThanOrEqual(844);
  });

  test('stacks model catalog section path and actions cleanly inside global cards', async ({ page }) => {
    await page.setViewportSize({ width: 1600, height: 1200 });
    await mockConfigSurface(page);

    await page.goto('/config/global-config');
    await expect(page.getByRole('heading', { name: 'Global Config', exact: true })).toBeVisible();

    const systemBindingsCard = page.locator('article').filter({
      has: page.getByRole('heading', { name: 'Decision Model & Bindings' }),
    }).first();
    const embeddingsCard = page.locator('article').filter({
      has: page.getByRole('heading', { name: 'Embedding Models' }),
    }).first();

    await expect(systemBindingsCard).toBeVisible();
    await expect(embeddingsCard).toBeVisible();
    await expect(systemBindingsCard.getByText('guard-judge', { exact: true })).toBeVisible();
    await expect(embeddingsCard.getByText('qwen3-embedding')).toBeVisible();

    const systemPath = systemBindingsCard.getByText('global.model_catalog.system');
    const systemButton = systemBindingsCard.getByRole('button', { name: 'Edit' });
    const embeddingPath = embeddingsCard.getByText('global.model_catalog.embeddings');
    const embeddingButton = embeddingsCard.getByRole('button', { name: 'Edit' });

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
      const horizontallySeparated =
        buttonBox!.x + buttonBox!.width <= pathBox!.x ||
        pathBox!.x + pathBox!.width <= buttonBox!.x;
      const verticallySeparated =
        buttonBox!.y + buttonBox!.height <= pathBox!.y ||
        pathBox!.y + pathBox!.height <= buttonBox!.y;
      expect(horizontallySeparated || verticallySeparated).toBe(true);
    }

    await expect(page.getByRole('heading', { name: 'Classifier Catalog' })).toHaveCount(0);
  });

  test('falls back from the retired classifier route to Global Config', async ({ page }) => {
    await page.setViewportSize({ width: 1600, height: 1200 });
    await mockConfigSurface(page);

    await page.goto('/config/classifiers');

    await expect(page).toHaveURL(/\/config\/global-config$/);
    await expect(page.getByRole('heading', { name: 'Global Config', exact: true })).toBeVisible();
  });
});
