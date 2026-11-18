import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const MODEL_COUNT = 305
const DEFAULT_MODEL = 'model-000-default'
const REFERENCED_MODEL = 'model-001-referenced'

type MockConfig = ReturnType<typeof buildConfig>

function modelName(index: number): string {
  if (index === 0) return DEFAULT_MODEL
  if (index === 1) return REFERENCED_MODEL
  if (index === 299) return 'model-299-needle'
  return `model-${String(index).padStart(3, '0')}`
}

function buildConfig() {
  const models = Array.from({ length: MODEL_COUNT }, (_, index) => {
    const name = modelName(index)
    const reasoningFamily =
      index % 3 === 0 ? 'family-alpha' : index % 3 === 1 ? 'family-beta' : undefined

    return {
      name,
      provider_model_id: `physical/Qwen-Scale-${String(index).padStart(3, '0')}`,
      api_format: 'openai',
      reasoning: reasoningFamily ? { family: reasoningFamily } : undefined,
      backend_refs:
        index % 2 === 0
          ? [
              {
                name: `local-rocm-${index % 4}`,
                endpoint: 'vllm:8000',
                protocol: 'http' as const,
                weight: 1,
              },
            ]
          : [],
      pricing: {
        currency: 'USD',
        prompt_per_1m: index / 100,
        completion_per_1m: index / 50,
      },
      description:
        index === 299
          ? 'Unique needle model for large inventory search coverage.'
          : `Scale-test routing model ${index}`,
      capabilities: index % 2 === 0 ? ['chat', 'reasoning'] : ['chat'],
      tags: [index % 2 === 0 ? 'rocm' : 'fallback'],
      param_size: `${8 + (index % 4) * 8}B`,
    }
  })

  return {
    version: 'v0.5',
    listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
    providers: {
      defaults: {
        model: DEFAULT_MODEL,
      },
      models,
    },
    recipes: [
      {
        name: 'default',
        signals: {},
        decisions: [
          {
            name: 'protected-production-route',
            description: 'Keeps one non-default model protected by a routing reference.',
            priority: 1,
            rules: { operator: 'AND' as const, conditions: [] },
            modelRefs: [
              {
                model: REFERENCED_MODEL,
                use_reasoning: false,
                weight: 1,
              },
            ],
          },
        ],
      },
    ],
    global: {},
  }
}

// Models without a backend have nothing to check; of the rest, a few are down or degraded.
function modelHealth(name: string, index: number) {
  if (index % 2 === 1 || index > MODEL_COUNT - 1) {
    return { model: name, status: 'unchecked', reason: 'No backend endpoint to check.' }
  }
  if (index % 20 === 4) {
    return { model: name, status: 'down', reason: 'The backend could not be reached.' }
  }
  const p95Ms = 300 + (index % 7) * 110
  return {
    model: name,
    status: index % 20 === 8 ? 'degraded' : 'healthy',
    reason: index % 20 === 8 ? 'Slow: answered in 2.8 s.' : undefined,
    latencyMs: p95Ms - 40,
    p95Ms,
    recentMs: [p95Ms - 60, p95Ms - 40, p95Ms],
  }
}

interface ModelInventoryMockOptions {
  readonlyUser?: boolean
}

async function mockLargeModelInventory(
  page: Page,
  options: ModelInventoryMockOptions = {},
): Promise<{ writes: MockConfig[] }> {
  const user = options.readonlyUser
    ? {
        id: 'admin-readonly',
        email: 'readonly@example.com',
        name: 'Read-only Admin',
        role: 'admin',
        permissions: ['config.read'],
      }
    : undefined

  await mockAuthenticatedAppShell(page, {
    ...(user ? { user } : {}),
    settings: { platform: 'amd', readonlyMode: false },
  })

  let config = buildConfig()
  const writes: MockConfig[] = []

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
      body: JSON.stringify(config.global),
    })
  })

  await page.route('**/api/router/config/update', async (route) => {
    const nextConfig = route.request().postDataJSON() as MockConfig
    writes.push(nextConfig)
    config = nextConfig
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ success: true }),
    })
  })

  await page.route('**/api/models/health*', async (route) => {
    const wanted = new URL(route.request().url()).searchParams.getAll('model')
    const models = config.providers.models
      .map((model, index) => modelHealth(model.name, index))
      .filter((entry) => wanted.length === 0 || wanted.includes(entry.model))
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ checkedAt: new Date().toISOString(), intervalSeconds: 30, models }),
    })
  })

  await page.route('**/api/router/api/v1/observability/replays/aggregate*', (route) =>
    route.fulfill({ status: 404, body: 'Router Replay is off' }),
  )

  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        overall: 'healthy',
        deployment_type: 'local',
        services: [{ name: 'router', status: 'running', healthy: true }],
        models: { models: [], summary: { loaded_models: 0, total_models: 0 } },
      }),
    })
  })

  return { writes }
}

function inventoryTable(page: Page) {
  return page.getByRole('table').filter({
    has: page.getByRole('columnheader', { name: 'Model', exact: true }),
  })
}

// A filter pill, not the column header of the same name.
function filterPill(page: Page, label: string) {
  return page
    .getByRole('button', { name: label, exact: true })
    .and(page.locator('[aria-haspopup="dialog"]'))
}

function inventoryPagination(page: Page) {
  return page.getByRole('group', { name: 'models pagination' })
}

async function openRowAction(page: Page, model: string, action: string) {
  await page.getByRole('button', { name: `Actions for ${model}`, exact: true }).click()
  // The previous row's menu can still be animating out; act in the open one.
  await page
    .locator('[role="menu"][data-state="open"]')
    .getByRole('menuitem', { name: action, exact: true })
    .click()
}

function bodyOverflow(page: Page) {
  return page.evaluate(() => getComputedStyle(document.body).overflow)
}

test.describe('Backend models inventory at 300+ scale', () => {
  test('keeps pricing clear of actions at 1320px', async ({ page }) => {
    await page.setViewportSize({ width: 1320, height: 549 })
    await mockLargeModelInventory(page)
    await page.goto('/models/backends')

    const table = inventoryTable(page)
    const pricing = table.getByRole('columnheader', { name: 'Price / 1M', exact: true })
    await expect(pricing).toBeVisible()
    const actionsButton = page.getByRole('button', { name: `Actions for ${DEFAULT_MODEL}` })
    const row = table.getByRole('row').filter({ has: actionsButton })
    const priceCell = row.getByRole('cell').filter({ hasText: /\$0\.00\s*\/\s*\$0\.00/ })
    await expect(priceCell).toBeVisible()

    // Text visibility alone misses an Actions cell covering Pricing.
    // Measure before clicking, since Playwright may scroll the table for us.
    for (const [price, actions] of [
      [pricing, table.getByRole('columnheader', { name: 'Actions', exact: true })],
      [priceCell, row.getByRole('cell').filter({ has: actionsButton })],
    ]) {
      const priceBounds = await price.boundingBox()
      const actionBounds = await actions.boundingBox()
      expect(priceBounds).not.toBeNull()
      expect(actionBounds).not.toBeNull()
      expect(actionBounds!.x).toBeGreaterThanOrEqual(priceBounds!.x + priceBounds!.width - 1)
    }

    await openRowAction(page, DEFAULT_MODEL, 'View details')
    await expect(page.getByRole('dialog', { name: DEFAULT_MODEL })).toBeVisible()
    await expect(page).toHaveURL(new RegExp(`m=${DEFAULT_MODEL}`))
  })

  test('adds and edits one model with one canonical write per action', async ({ page }) => {
    test.setTimeout(90_000)
    const { writes } = await mockLargeModelInventory(page)
    await page.goto('/models/backends')

    await expect(inventoryPagination(page)).toContainText(`1–25 of ${MODEL_COUNT} models`)
    await page.getByRole('button', { name: 'Connect model', exact: true }).click()

    const addDialog = page.getByRole('dialog', { name: 'Add models' })
    await addDialog.getByRole('button', { name: /vLLM Connect a private vLLM endpoint/ }).click()
    const connectionDialog = page.getByRole('dialog', { name: 'vLLM' })
    await connectionDialog.getByLabel('Base URL').fill('http://localhost:8000/v1')
    await connectionDialog.getByLabel(/API key/).fill('test-provider-key')
    await connectionDialog.getByPlaceholder('Or enter a model ID').fill('physical/Qwen-Scale-305')
    await connectionDialog.getByRole('button', { name: 'Add', exact: true }).click()

    const manualModelInput = connectionDialog.getByPlaceholder('Or enter a model ID')
    const advancedSettings = connectionDialog.getByText('Advanced settings', { exact: true })
    const manualModelBox = await manualModelInput.boundingBox()
    const advancedSettingsBox = await advancedSettings.boundingBox()
    expect(manualModelBox).not.toBeNull()
    expect(advancedSettingsBox).not.toBeNull()
    expect(
      (advancedSettingsBox?.y ?? 0) - ((manualModelBox?.y ?? 0) + (manualModelBox?.height ?? 0)),
    ).toBeGreaterThanOrEqual(14)

    await advancedSettings.click()
    await connectionDialog.getByLabel('Name prefix').fill('model-305-added')
    await connectionDialog.getByLabel('Reasoning family').selectOption('qwen3')
    await connectionDialog.getByLabel('Description').fill('Added from the 305-model inventory.')
    await connectionDialog.getByLabel('Parameter size').fill('32B')
    await connectionDialog.getByLabel('Context window').fill('131072')
    const capabilities = connectionDialog.getByLabel('Capabilities')
    await capabilities.fill('chat, reasoning')
    await capabilities.press('Enter')
    const tags = connectionDialog.getByLabel('Tags')
    await tags.fill('rocm, new')
    await tags.press('Enter')
    await connectionDialog.getByLabel('Input cost').fill('0.25')
    await connectionDialog.getByLabel('Output cost').fill('0.5')
    await connectionDialog.getByLabel('Max retries').fill('2')
    await connectionDialog.getByLabel('Load balancing').selectOption('LEAST_REQUEST')
    await connectionDialog.getByRole('button', { name: 'Add 1 model' }).click()

    await expect(connectionDialog).toBeHidden()
    await expect.poll(() => writes.length).toBe(1)
    const addedConfig = writes[0]
    expect(addedConfig).not.toHaveProperty('signals')
    expect(addedConfig).not.toHaveProperty('decisions')
    expect(addedConfig).not.toHaveProperty('model_config')
    expect(addedConfig.version).toBe('v0.5')
    expect(addedConfig.providers.defaults.model).toBe(DEFAULT_MODEL)
    expect(addedConfig.providers.defaults).not.toHaveProperty('default_model')
    expect(addedConfig.providers.models).toHaveLength(MODEL_COUNT + 1)
    expect(
      addedConfig.providers.models.find(
        (model) => model.name === 'model-305-added/physical/Qwen-Scale-305',
      ),
    ).toMatchObject({
      name: 'model-305-added/physical/Qwen-Scale-305',
      provider_model_id: 'physical/Qwen-Scale-305',
      api_format: 'chat_completions',
      reasoning: { family: 'qwen3' },
      backend_refs: [
        {
          name: 'vllm-primary',
          endpoint: 'http://localhost:8000/v1',
          provider: 'vllm',
          api_key: 'test-provider-key',
        },
      ],
      pricing: {
        currency: 'USD',
        prompt_per_1m: 0.25,
        completion_per_1m: 0.5,
        cached_input_per_1m: 0.25,
        cache_write_per_1m: 0.25,
      },
      reliability: { retry_count: 2, lb_policy: 'LEAST_REQUEST' },
      description: 'Added from the 305-model inventory.',
      capabilities: ['chat', 'reasoning'],
      tags: ['rocm', 'new'],
      param_size: '32B',
      context_window_size: 131072,
    })

    await expect(inventoryPagination(page)).toContainText(`of ${MODEL_COUNT + 1} models`)
    const search = page.getByPlaceholder('Search models, IDs or tags…')
    await search.fill('model-305-added/physical/Qwen-Scale-305')
    await expect(inventoryPagination(page)).toContainText('1–1 of 1 models')
    await openRowAction(page, 'model-305-added/physical/Qwen-Scale-305', 'Edit')

    const editDialog = page.getByRole('dialog', {
      name: 'Edit Model: model-305-added/physical/Qwen-Scale-305',
    })
    await expect(editDialog.getByLabel('Provider Model ID')).toHaveValue('physical/Qwen-Scale-305')
    await editDialog.getByLabel('Provider Model ID').fill('physical/Qwen-Scale-305-v2')
    await editDialog
      .getByLabel('Description', { exact: true })
      .fill('Updated without shrinking the large inventory.')
    await editDialog.getByRole('button', { name: 'Remove new', exact: true }).click()
    const tagInput = editDialog.getByRole('textbox', { name: 'Tags', exact: true })
    await tagInput.fill('updated')
    await tagInput.press('Enter')
    await editDialog.getByRole('button', { name: 'Save', exact: true }).click()

    await expect(editDialog).toBeHidden()
    await expect.poll(() => writes.length).toBe(2)
    const editedConfig = writes[1]
    expect(editedConfig).not.toHaveProperty('signals')
    expect(editedConfig).not.toHaveProperty('decisions')
    expect(editedConfig).not.toHaveProperty('model_config')
    expect(editedConfig.providers.defaults.model).toBe(DEFAULT_MODEL)
    expect(editedConfig.providers.models).toHaveLength(MODEL_COUNT + 1)
    expect(
      editedConfig.providers.models.find(
        (model) => model.name === 'model-305-added/physical/Qwen-Scale-305',
      ),
    ).toMatchObject({
      provider_model_id: 'physical/Qwen-Scale-305-v2',
      description: 'Updated without shrinking the large inventory.',
      tags: ['rocm', 'updated'],
    })
    expect(editedConfig.providers.models.some((model) => model.name === DEFAULT_MODEL)).toBe(true)
    expect(editedConfig.providers.models.some((model) => model.name === 'model-304')).toBe(true)
  })

  test('paginates 25 rows and composes kind and health filters with search', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    await mockLargeModelInventory(page)
    await page.goto('/models/backends')

    await expect(page.getByRole('heading', { name: 'Backend models', exact: true })).toBeVisible()
    await expect(inventoryPagination(page)).toContainText(`1–25 of ${MODEL_COUNT} models`)
    await expect(page.getByRole('button', { name: /^Actions for model-/ })).toHaveCount(25)
    await expect(inventoryTable(page).getByText(DEFAULT_MODEL, { exact: true })).toBeVisible()
    await expect(page.getByText('model-024', { exact: true })).toBeVisible()

    const modelHeader = page.getByRole('columnheader', { name: 'Model', exact: true })
    await expect
      .poll(async () => (await modelHeader.boundingBox())?.width ?? 0)
      .toBeGreaterThanOrEqual(260)
    await expect(page.getByRole('columnheader', { name: 'Price / 1M' })).toHaveText('Price / 1M')

    await inventoryPagination(page).getByRole('button', { name: 'Next page' }).click()
    await expect(inventoryPagination(page)).toContainText(`26–50 of ${MODEL_COUNT} models`)
    await expect(page.getByText('model-025', { exact: true })).toBeVisible()
    await expect(page.getByText('model-049', { exact: true })).toBeVisible()
    await expect(inventoryTable(page).getByText(DEFAULT_MODEL, { exact: true })).toHaveCount(0)

    await filterPill(page, 'Kind').click()
    await page.getByRole('menuitemcheckbox', { name: /^Chat/ }).click()
    await page.keyboard.press('Escape')
    await expect(inventoryPagination(page)).toContainText('1–25 of 50 models')
    await expect(inventoryTable(page).getByText('model-005', { exact: true })).toBeVisible()

    await page.getByRole('button', { name: 'Clear filters' }).click()
    await filterPill(page, 'Health').click()
    await page.getByRole('menuitemcheckbox', { name: /^Down/ }).click()
    await page.keyboard.press('Escape')
    await expect(inventoryPagination(page)).toContainText('1–16 of 16 models')
    const downRow = inventoryTable(page).getByRole('row').filter({ hasText: 'model-004' })
    await expect(downRow.getByText('Down', { exact: true })).toBeVisible()

    await page.getByRole('button', { name: 'Clear filters' }).click()
    await page.getByPlaceholder('Search models, IDs or tags…').fill('model-299-needle')
    await expect(inventoryPagination(page)).toContainText('1–1 of 1 models')
    await expect(page.getByText('model-299-needle', { exact: true })).toBeVisible()
    await expect(page.getByText('physical/Qwen-Scale-299', { exact: true })).toBeVisible()
  })

  test('keeps body lock and focus intact across the view-to-edit transition', async ({ page }) => {
    await mockLargeModelInventory(page)
    await page.goto('/models/backends')

    const actions = inventoryTable(page).getByRole('button', {
      name: `Actions for ${DEFAULT_MODEL}`,
    })
    await actions.focus()
    await openRowAction(page, DEFAULT_MODEL, 'View details')

    const viewSheet = page.getByRole('dialog', { name: DEFAULT_MODEL })
    await expect(viewSheet).toBeVisible()
    await expect.poll(() => bodyOverflow(page)).toBe('hidden')
    await viewSheet.getByRole('button', { name: 'Edit model', exact: true }).click()

    const editDialog = page.getByRole('dialog', { name: `Edit Model: ${DEFAULT_MODEL}` })
    await expect(editDialog).toBeVisible()
    await expect.poll(() => bodyOverflow(page)).toBe('hidden')
    await expect
      .poll(() => editDialog.evaluate((element) => element.contains(document.activeElement)))
      .toBe(true)

    // Closing the editor returns to the sheet it was opened from.
    await page.keyboard.press('Escape')
    await expect(editDialog).toBeHidden()
    await expect(viewSheet).toBeVisible()
    await expect.poll(() => bodyOverflow(page)).toBe('hidden')

    await page.keyboard.press('Escape')
    await expect(viewSheet).toBeHidden()
    await expect.poll(() => bodyOverflow(page)).not.toBe('hidden')
    await expect(page).not.toHaveURL(/m=/)
  })

  test('keeps selection across pages, blocks protected models, and saves a bulk delete once', async ({
    page,
  }) => {
    const { writes } = await mockLargeModelInventory(page)
    await page.goto('/models/backends')

    await expect(page.getByLabel(`Select model ${DEFAULT_MODEL}`)).toBeDisabled()
    await expect(page.getByLabel(`Select model ${REFERENCED_MODEL}`)).toBeDisabled()

    await openRowAction(page, DEFAULT_MODEL, 'Delete')
    await expect(
      page.getByText('Choose a different default model before deleting this model.'),
    ).toBeVisible()
    await expect(page.getByRole('alertdialog')).toHaveCount(0)
    expect(writes).toHaveLength(0)

    await openRowAction(page, REFERENCED_MODEL, 'Delete')
    await expect(
      page.getByText('Remove this model from 1 routing decision before deleting it.'),
    ).toBeVisible()
    await expect(page.getByRole('alertdialog')).toHaveCount(0)
    expect(writes).toHaveLength(0)

    await page.getByLabel('Select model model-002').check()
    await inventoryPagination(page).getByRole('button', { name: 'Next page' }).click()
    await page.getByLabel('Select model model-026').check()

    await expect(page.getByText('2 selected', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: /Delete selected/i }).click()

    const dialog = page.getByRole('alertdialog', { name: 'Delete 2 models?' })
    await expect(dialog).toBeVisible()
    await expect(dialog).toContainText('model-002')
    await expect(dialog).toContainText('model-026')
    await expect(dialog.getByRole('button', { name: 'Delete models' })).toBeDisabled()

    await dialog.getByLabel(/Type DELETE 2 to confirm/).fill('DELETE 2')
    await page.setViewportSize({ width: 320, height: 480 })
    const dialogBox = await dialog.boundingBox()
    expect(dialogBox).not.toBeNull()
    expect(dialogBox?.y ?? -1).toBeGreaterThanOrEqual(0)
    expect((dialogBox?.y ?? 0) + (dialogBox?.height ?? 0)).toBeLessThanOrEqual(480)
    expect(await bodyOverflow(page)).toBe('hidden')

    const confirmationInput = dialog.getByLabel(/Type DELETE 2 to confirm/)
    await confirmationInput.focus()
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    expect(await dialog.evaluate((element) => element.contains(document.activeElement))).toBe(true)

    await dialog.getByRole('button', { name: 'Delete models' }).click()

    await expect.poll(() => writes.length).toBe(1)
    expect(writes[0].providers.defaults.model).toBe(DEFAULT_MODEL)
    const savedModelNames = writes[0].providers.models.map((model) => model.name)
    expect(savedModelNames).toHaveLength(MODEL_COUNT - 2)
    expect(savedModelNames).not.toContain('model-002')
    expect(savedModelNames).not.toContain('model-026')
    expect(savedModelNames).toContain(DEFAULT_MODEL)
    expect(savedModelNames).toContain(REFERENCED_MODEL)
  })

  test('hides every write affordance when explicit permissions are read-only', async ({ page }) => {
    const { writes } = await mockLargeModelInventory(page, { readonlyUser: true })
    await page.goto('/models/backends')

    await expect(inventoryPagination(page)).toContainText(`1–25 of ${MODEL_COUNT} models`)
    await expect(page.getByRole('button', { name: 'Connect model' })).toHaveCount(0)
    await expect(page.getByRole('checkbox', { name: /^Select model / })).toHaveCount(0)
    await page.getByRole('button', { name: 'More model actions' }).click()
    await expect(page.getByRole('menuitem', { name: 'Reasoning families' })).toBeVisible()
    await expect(page.getByRole('menuitem', { name: 'Add a model manually' })).toHaveCount(0)
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: `Actions for ${DEFAULT_MODEL}`, exact: true }).click()
    await expect(page.getByRole('menuitem', { name: 'View details', exact: true })).toBeVisible()
    await expect(page.getByRole('menuitem', { name: 'Check health', exact: true })).toBeVisible()
    await expect(page.getByRole('menuitem', { name: 'Edit', exact: true })).toHaveCount(0)
    await expect(page.getByRole('menuitem', { name: 'Delete', exact: true })).toHaveCount(0)
    await page.keyboard.press('Escape')
    expect(writes).toHaveLength(0)
  })
})
