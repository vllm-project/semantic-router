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
    const reasoningFamily = index % 3 === 0
      ? 'family-alpha'
      : index % 3 === 1
        ? 'family-beta'
        : undefined

    return {
      name,
      provider_model_id: `physical/Qwen-Scale-${String(index).padStart(3, '0')}`,
      api_format: 'openai',
      reasoning_family: reasoningFamily,
      backend_refs: index % 2 === 0
        ? [{
            name: `local-rocm-${index % 4}`,
            endpoint: 'vllm:8000',
            protocol: 'http' as const,
            weight: 1,
          }]
        : [],
      pricing: {
        currency: 'USD',
        prompt_per_1m: index / 100,
        completion_per_1m: index / 50,
      },
    }
  })

  return {
    version: 'v0.3',
    listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
    providers: {
      defaults: {
        default_model: DEFAULT_MODEL,
        reasoning_families: {
          'family-alpha': { type: 'reasoning_effort', parameter: 'reasoning_effort' },
          'family-beta': { type: 'chat_template_kwargs', parameter: 'thinking' },
        },
      },
      models,
    },
    routing: {
      modelCards: models.map((model, index) => ({
        name: model.name,
        description: index === 299
          ? 'Unique needle model for large inventory search coverage.'
          : `Scale-test routing model ${index}`,
        capabilities: index % 2 === 0 ? ['chat', 'reasoning'] : ['chat'],
        tags: [index % 2 === 0 ? 'rocm' : 'fallback'],
        modality: 'text',
        param_size: `${8 + (index % 4) * 8}B`,
      })),
      signals: {},
      decisions: [{
        name: 'protected-production-route',
        description: 'Keeps one non-default model protected by a routing reference.',
        priority: 1,
        rules: { operator: 'AND' as const, conditions: [] },
        modelRefs: [{
          model: REFERENCED_MODEL,
          use_reasoning: false,
          weight: 1,
        }],
      }],
    },
    global: {},
    plugins: {},
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

  const config = buildConfig()
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
    writes.push(route.request().postDataJSON() as MockConfig)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ success: true }),
    })
  })

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

test.describe('Models inventory at 300+ scale', () => {
  test('paginates 25 rows and composes family filters with search', async ({ page }) => {
    await mockLargeModelInventory(page)
    await page.goto('/config/models')

    await expect(page.getByRole('heading', { name: 'Models', exact: true }).first()).toBeVisible()
    await expect(page.getByText(`1–25 of ${MODEL_COUNT} models`, { exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: /^Expand model-/ })).toHaveCount(25)
    await expect(page.getByText(DEFAULT_MODEL, { exact: true })).toBeVisible()
    await expect(page.getByText('model-024', { exact: true })).toBeVisible()

    await page.getByRole('button', { name: 'Next page' }).click()
    await expect(page.getByText(`26–50 of ${MODEL_COUNT} models`, { exact: true })).toBeVisible()
    await expect(page.getByText('model-025', { exact: true })).toBeVisible()
    await expect(page.getByText('model-049', { exact: true })).toBeVisible()
    await expect(page.getByText(DEFAULT_MODEL, { exact: true })).toHaveCount(0)

    await page.getByLabel('Reasoning family').selectOption('family-alpha')
    await expect(page.getByText('1–25 of 102 models', { exact: true })).toBeVisible()
    await expect(page.getByText(DEFAULT_MODEL, { exact: true })).toBeVisible()

    await page.getByRole('button', { name: 'Clear filters' }).click()
    await page.getByRole('searchbox', { name: 'Search name, ID, family, tag, or capability...' }).fill('model-299-needle')
    await expect(page.getByText('1–1 of 1 models', { exact: true })).toBeVisible()
    await expect(page.getByText('model-299-needle', { exact: true })).toBeVisible()
    await expect(page.getByText('physical/Qwen-Scale-299', { exact: true })).toBeVisible()
  })

  test('keeps selection across pages, blocks protected models, and saves a bulk delete once', async ({ page }) => {
    const { writes } = await mockLargeModelInventory(page)
    await page.goto('/config/models')

    await expect(page.getByLabel(`Select model ${DEFAULT_MODEL}`)).toBeDisabled()
    await expect(page.getByLabel(`Select model ${REFERENCED_MODEL}`)).toBeDisabled()

    await page.getByRole('button', { name: `Delete ${DEFAULT_MODEL}` }).click()
    await expect(page.getByText('Choose a different default model before deleting this model.')).toBeVisible()
    await expect(page.getByRole('alertdialog')).toHaveCount(0)
    expect(writes).toHaveLength(0)

    await page.getByRole('button', { name: `Delete ${REFERENCED_MODEL}` }).click()
    await expect(page.getByText('Remove this model from 1 routing decision before deleting it.')).toBeVisible()
    await expect(page.getByRole('alertdialog')).toHaveCount(0)
    expect(writes).toHaveLength(0)

    await page.getByLabel('Select model model-002').check()
    await page.getByRole('button', { name: 'Next page' }).click()
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
    expect(await page.evaluate(() => document.body.style.overflow)).toBe('hidden')

    const confirmationInput = dialog.getByLabel(/Type DELETE 2 to confirm/)
    await confirmationInput.focus()
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    expect(await dialog.evaluate((element) => element.contains(document.activeElement))).toBe(true)

    await dialog.getByRole('button', { name: 'Delete models' }).click()

    await expect.poll(() => writes.length).toBe(1)
    const savedModelNames = writes[0].providers.models.map((model) => model.name)
    expect(savedModelNames).toHaveLength(MODEL_COUNT - 2)
    expect(savedModelNames).not.toContain('model-002')
    expect(savedModelNames).not.toContain('model-026')
    expect(savedModelNames).toContain(DEFAULT_MODEL)
    expect(savedModelNames).toContain(REFERENCED_MODEL)
  })

  test('hides every write affordance when explicit permissions are read-only', async ({ page }) => {
    const { writes } = await mockLargeModelInventory(page, { readonlyUser: true })
    await page.goto('/config/models')

    await expect(page.getByText(`1–25 of ${MODEL_COUNT} models`, { exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Add Model' })).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Add Family' })).toHaveCount(0)
    await expect(page.getByRole('button', { name: /^Edit model-/ })).toHaveCount(0)
    await expect(page.getByRole('button', { name: /^Delete model-/ })).toHaveCount(0)
    await expect(page.getByRole('checkbox', { name: /^Select model / })).toHaveCount(0)
    await expect(page.getByRole('button', { name: `View ${DEFAULT_MODEL}` })).toBeVisible()
    expect(writes).toHaveLength(0)
  })
})
