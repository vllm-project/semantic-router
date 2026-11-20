import { expect, test, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'

import type { BuiltInModelCatalog } from '../src/types/modelCatalog'
import { mockAuthenticatedAppShell } from './support/auth'

const bundledCatalog: BuiltInModelCatalog = JSON.parse(
  readFileSync(
    new URL('../../../website/static/model-catalog/catalog.json', import.meta.url),
    'utf8',
  ),
)

const CATALOG_MODEL = 'anthropic/claude-opus-5.5'

const config = {
  version: 'v0.5',
  providers: {
    defaults: { model: 'local-qwen' },
    models: [
      {
        name: 'local-qwen',
        provider_model_id: 'Qwen/Qwen3-8B',
        backend_refs: [{ name: 'primary', endpoint: 'vllm:8000', provider: 'vllm' }],
        pricing: { currency: 'USD', prompt_per_1m: 0.2, completion_per_1m: 0.4 },
        reasoning: { family: 'qwen3' },
      },
      { name: 'native-kai', api_format: 'systemone', deployment: 'decision-kai' },
      {
        name: 'opus',
        catalog: CATALOG_MODEL,
        backend_refs: [
          { name: 'anthropic', endpoint: 'https://api.anthropic.com', provider: 'anthropic' },
        ],
      },
    ],
  },
  recipes: [
    {
      name: 'default',
      decisions: [
        {
          name: 'route',
          description: 'Answer everything.',
          priority: 1,
          rules: { operator: 'AND', conditions: [] },
          modelRefs: [{ model: 'local-qwen', use_reasoning: false }],
        },
      ],
    },
  ],
  global: {
    model_catalog: {
      deployments: {
        'decision-kai': {
          provider: 'model_runtime',
          artifact: 'vllm-sr/Decision-2.0-Kai-0.6B',
          device: 'auto',
        },
      },
    },
  },
}

async function mockBackendModels(page: Page) {
  await mockAuthenticatedAppShell(page)
  const writes: Array<typeof config> = []
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: config }))
  await page.route('**/api/router/config/global', (route) => route.fulfill({ json: config.global }))
  await page.route('**/api/models/catalog', (route) => route.fulfill({ json: bundledCatalog }))
  await page.route('**/api/router/config/update', async (route) => {
    writes.push(route.request().postDataJSON() as typeof config)
    await route.fulfill({ json: { success: true } })
  })
  return writes
}

async function gotoBackends(page: Page, path: string) {
  await page.goto(path)
  // A deep link opens a sheet over the page, which hides the heading from the accessibility tree.
  await expect(page.locator('h1').filter({ hasText: 'Backend models' })).toBeVisible({
    timeout: 30_000,
  })
}

test('a model sheet shows its shared deployment and published catalog evidence', async ({
  page,
}) => {
  await mockBackendModels(page)
  await gotoBackends(page, '/models/backends?m=native-kai&view=deployments')

  const deploymentSheet = page.getByRole('dialog', { name: 'native-kai' })
  await expect(deploymentSheet).toContainText('Deployment this model names')
  await expect(deploymentSheet).toContainText('decision-kai')
  await expect(deploymentSheet).toContainText('vllm-sr/Decision-2.0-Kai-0.6B')
  await page.keyboard.press('Escape')
  await expect(deploymentSheet).toBeHidden()

  await page.getByRole('row').filter({ hasText: 'opus' }).click()
  const sheet = page.getByRole('dialog', { name: 'opus' })
  await sheet.getByRole('tab', { name: /Evidence/ }).click()
  await expect(page).toHaveURL(/view=evidence/)
  await expect(sheet.getByText('Built-in catalog').first()).toBeVisible()
  const evaluations = bundledCatalog.evaluations.filter(
    (evaluation) => evaluation.model === CATALOG_MODEL,
  )
  await expect(sheet.getByRole('article')).toHaveCount(evaluations.length)
  await expect(page.getByText(/leaderboard|intelligence index|rank/i)).toHaveCount(0)

  await sheet.getByRole('tab', { name: 'Reasoning' }).click()
  await expect(sheet).toContainText('The catalog lists no reasoning controls for Claude Opus 5.5.')
  await expect(sheet.getByRole('button', { name: 'Set reasoning' })).toHaveCount(0)
  await page.keyboard.press('Escape')

  await page.getByRole('row').filter({ hasText: 'local-qwen' }).click()
  const qwen = page.getByRole('dialog', { name: 'local-qwen' })
  await qwen.getByRole('tab', { name: 'Reasoning' }).click()
  await expect(qwen).toContainText('Built-in reasoning family')
  await expect(qwen).toContainText('qwen3')
  await qwen.getByRole('tab', { name: 'Pricing' }).click()
  await expect(qwen).toContainText('$0.20')
  await expect(qwen).toContainText('$0.40')
})

test('makes another model the default after confirming', async ({ page }) => {
  const writes = await mockBackendModels(page)
  await gotoBackends(page, '/models/backends')

  await page.getByRole('button', { name: 'Actions for opus', exact: true }).click()
  await page.getByRole('menuitem', { name: 'Make default' }).click()
  const dialog = page.getByRole('alertdialog', { name: 'Make opus the default model?' })
  await dialog.getByRole('button', { name: 'Make default' }).click()
  await expect(dialog).toBeHidden()
  await expect.poll(() => writes.length).toBe(1)
  expect(writes[0].providers.defaults.model).toBe('opus')
  expect(writes[0].providers.models.map((model) => model.name)).toEqual([
    'local-qwen',
    'native-kai',
    'opus',
  ])
})

test('a catalog model proposed from the Model Hub opens the provider that serves it', async ({
  page,
}) => {
  await mockBackendModels(page)
  await gotoBackends(page, `/models/backends?add=${encodeURIComponent(CATALOG_MODEL)}`)

  const dialog = page.getByRole('dialog', { name: 'Anthropic' })
  await expect(dialog).toBeVisible()
  // The link is consumed once the dialog opens; a reload does not open it again.
  await expect(page).not.toHaveURL(/add=/)
  await expect(dialog.getByText('1 selected', { exact: true })).toBeVisible()
  await expect(dialog.getByRole('button', { name: 'Add 1 model' })).toBeEnabled()
  await dialog.getByRole('button', { name: 'Choose another provider' }).click()
  const providers = page.getByRole('dialog', { name: 'Add models' })
  await expect(providers).toContainText('Adding Claude Opus 5.5')
  await providers.getByRole('button', { name: 'Close' }).click()
  await expect(page).not.toHaveURL(/add=/)
})

test('reasoning families open in a sheet, and a decision count lists the decisions using a model', async ({
  page,
}) => {
  await mockBackendModels(page)
  await gotoBackends(page, '/models/backends')
  await expect(page.getByRole('tablist')).toHaveCount(0)

  await page.getByRole('button', { name: 'More model actions' }).click()
  await page.getByRole('menuitem', { name: 'Reasoning families' }).click()
  const families = page.getByRole('dialog', { name: 'Reasoning families' })
  await expect(families).toBeVisible()
  await expect(page).toHaveURL(/families=1/)
  await families.getByLabel('Search reasoning families').fill('qwen3')
  await families.getByText('qwen3', { exact: true }).click()
  await expect(page).toHaveURL(/family=qwen3/)
  const family = page.getByRole('dialog', { name: 'qwen3' })
  await expect(family).toContainText('Models using this family · 1')
  await family.getByRole('button', { name: 'local-qwen' }).click()
  const reasoning = page.getByRole('dialog', { name: 'local-qwen' })
  await expect(reasoning.getByRole('tab', { name: 'Reasoning', selected: true })).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(reasoning).toBeHidden()

  await page.getByRole('button', { name: '1 decision uses local-qwen' }).click()
  const usage = page.getByRole('dialog', { name: 'local-qwen' })
  await expect(page).toHaveURL(/view=usage/)
  await expect(usage.getByRole('tab', { name: /Used by/, selected: true })).toBeVisible()
  await usage.getByText('route', { exact: true }).click()
  await expect(page).toHaveURL(/\/routing\/decisions\?recipe=default&id=route/)
})
