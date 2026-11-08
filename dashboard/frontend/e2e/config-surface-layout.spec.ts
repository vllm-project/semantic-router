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

  test('keeps signal controls readable and aligned in the shared editor', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1100 });
    await mockConfigSurface(page);

    await page.goto('/config/signals');
    await page.getByRole('button', { name: 'Add Signal' }).click();

    const modal = page.getByRole('dialog', { name: 'Add Signal' });
    const signalType = modal.getByLabel('Signal type');
    const signalOption = signalType.locator('option').first();
    await expect(signalType).toHaveCSS('color-scheme', 'dark');
    await expect(signalOption).toHaveCSS('color', 'rgb(245, 245, 247)');
    await expect(signalOption).toHaveCSS('background-color', 'rgb(17, 18, 22)');

    const checkbox = modal.getByLabel('Case Sensitive');
    const [checkboxBox, checkboxControlBox] = await Promise.all([
      checkbox.boundingBox(),
      checkbox.locator('..').boundingBox(),
    ]);
    expect(checkboxBox).not.toBeNull();
    expect(checkboxControlBox).not.toBeNull();
    expect(checkboxBox!.width).toBeCloseTo(checkboxBox!.height, 0);
    expect(checkboxBox!.width).toBeLessThanOrEqual(15);
    expect(checkboxControlBox!.height).toBeGreaterThanOrEqual(39.5);

    await checkbox.check();
    await expect(checkbox).toBeChecked();
    const checkmarkSize = await checkbox.evaluate((element) => {
      const checkmark = window.getComputedStyle(element, '::before');
      return {
        width: Number.parseFloat(checkmark.width),
        height: Number.parseFloat(checkmark.height),
      };
    });
    expect(checkmarkSize.width).toBeLessThanOrEqual(4);
    expect(checkmarkSize.height).toBeLessThanOrEqual(6);
  });

  test('uses the same hierarchy for projection editors', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1100 });
    await mockConfigSurface(page);

    await page.goto('/config/projections');
    await page.getByRole('button', { name: 'Add Partition' }).click();

    const modal = page.getByRole('dialog', { name: 'Add Projection Partition' });
    await expect(modal.getByRole('heading', { name: 'Identity' })).toBeVisible();
    await expect(modal.getByRole('heading', { name: 'Definition' })).toBeVisible();
    await expectInside(modal, modal.getByLabel('Semantics'));
    await expectInside(modal, modal.getByRole('button', { name: 'Add member' }));
  });

  test('uses the shared hierarchy for signal and decision view and edit flows', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1100 });
    await mockConfigSurface(page);

    await page.goto('/config/signals');
    await page.getByRole('button', { name: 'View keyword-pricing' }).click();
    const signalView = page.getByRole('dialog', { name: 'Signal: pricing' });
    await expect(signalView.getByRole('heading', { name: 'Identity' })).toBeVisible();
    await expect(signalView.getByRole('heading', { name: 'Definition' })).toBeVisible();
    await signalView.getByRole('button', { name: 'Close' }).first().click();

    await page.getByRole('button', { name: 'Edit keyword-pricing' }).click();
    const signalEdit = page.getByRole('dialog', { name: 'Edit Signal: pricing' });
    await expect(signalEdit.getByRole('heading', { name: 'Identity' })).toBeVisible();
    await expect(signalEdit.getByRole('heading', { name: 'Definition' })).toBeVisible();
    await signalEdit.getByRole('button', { name: 'Close editor' }).click();

    await page.goto('/config/decisions');
    await page.getByRole('button', { name: 'View business-route' }).click();
    const decisionView = page.getByRole('dialog', { name: 'Decision: business-route' });
    for (const section of ['Identity', 'Routing policy', 'Selection & runtime', 'Output behavior']) {
      await expect(decisionView.getByRole('heading', { name: section })).toBeVisible();
    }
  });

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
    const modelSelect = modal.getByRole('combobox', { name: 'Model', exact: true });
    await modelSelect.selectOption('test-model');

    const controls = [
      modelSelect,
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
    const rightGutter = viewport!.width - modalBox!.x - modalBox!.width;
    const bottomGutter = viewport!.height - modalBox!.y - modalBox!.height;
    expect(modalBox!.width).toBeLessThanOrEqual(1120);
    expect(modalBox!.x).toBeCloseTo(rightGutter, 0);
    expect(modalBox!.y).toBeCloseTo(bottomGutter, 0);

    const closeButton = modal.getByRole('button', { name: 'Close editor' });
    await expect(closeButton).toBeFocused();
    await modal.getByRole('button', { name: 'Add', exact: true }).focus();
    await page.keyboard.press('Tab');
    await expect(closeButton).toBeFocused();
    await page.keyboard.press('Escape');
    const discardDialog = page.getByRole('alertdialog', { name: 'Discard unsaved changes?' });
    await expect(discardDialog).toBeVisible();
    await discardDialog.getByRole('button', { name: 'Discard changes' }).click();
    await expect(modal).toBeHidden();
    await expect(addDecisionButton).toBeFocused();
  });

  test('fits the shared editor inside the mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await mockConfigSurface(page);

    await page.goto('/config/decisions');
    await page.getByRole('button', { name: 'Add Decision' }).click();

    const modal = page.getByRole('dialog', { name: 'Add Decision' });
    await expect(modal).toHaveCSS('animation-name', 'none');
    const modalBox = await modal.boundingBox();
    expect(modalBox).not.toBeNull();
    const rightGutter = 390 - modalBox!.x - modalBox!.width;
    expect(modalBox!.x).toBeGreaterThanOrEqual(0);
    expect(rightGutter).toBeGreaterThanOrEqual(0);
    expect(modalBox!.x).toBeLessThanOrEqual(8);
    expect(rightGutter).toBeLessThanOrEqual(8);
    expect(modalBox!.y).toBeGreaterThanOrEqual(0);
    expect(modalBox!.y + modalBox!.height).toBeLessThanOrEqual(844);
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
