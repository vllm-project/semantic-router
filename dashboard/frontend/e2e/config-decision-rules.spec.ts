import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

// The Router accepts a single condition as the whole rule tree.
const singleConditionRules = { type: 'jailbreak', name: 'prompt_injection' }

const configResponse = {
  version: 'v0.5',
  providers: {
    models: [
      {
        name: 'model-a',
        backend_refs: [{ name: 'local', endpoint: 'localhost:8000', protocol: 'http' }],
      },
    ],
  },
  recipes: [
    {
      name: 'default',
      signals: { jailbreak: [{ name: 'prompt_injection', predicate: { gte: 0.7 } }] },
      decisions: [
        {
          name: 'guard_route',
          description: 'Route detected prompt attacks.',
          priority: 120,
          rules: singleConditionRules,
          modelRefs: [{ model: 'model-a' }],
        },
      ],
    },
  ],
}

type SavedDecision = { name: string; description?: string; rules?: unknown; modelRefs?: unknown }
type SavedConfig = { recipes?: Array<{ decisions?: SavedDecision[] }> }

test('keeps a single-condition rule and untouched fields when a decision is saved', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: configResponse }))
  const saved: SavedConfig[] = []
  await page.route('**/api/router/config/update', async (route) => {
    saved.push(route.request().postDataJSON() as SavedConfig)
    await route.fulfill({ json: {} })
  })
  await page.goto('/routing/decisions')

  const row = page.getByRole('row', { name: /guard_route/ })
  await expect(row).toContainText('prompt_injection')
  await row.click()

  const sheet = page.getByRole('dialog', { name: 'guard_route' })
  await expect(sheet).toBeVisible()
  await sheet.getByRole('tab', { name: /Rules/ }).click()
  await expect(sheet.getByText('Single condition')).toBeVisible()
  await expect(sheet.getByLabel('Signal name')).toHaveValue('prompt_injection')
  await expect(sheet.getByText('Always matches')).toHaveCount(0)

  await sheet.getByRole('tab', { name: 'Overview' }).click()
  await sheet
    .getByRole('textbox', { name: 'Description' })
    .fill('Route detected prompt attacks to the guard model.')
  await sheet.getByRole('button', { name: 'Save and apply' }).click()

  await expect.poll(() => saved.length).toBe(1)
  const decision = saved[0].recipes?.[0]?.decisions?.find((entry) => entry.name === 'guard_route')
  expect(decision?.rules).toEqual(singleConditionRules)
  expect(decision?.description).toBe('Route detected prompt attacks to the guard model.')
  // The decision did not set use_reasoning, so saving does not add it.
  expect(decision?.modelRefs).toEqual([{ model: 'model-a' }])
})

test('builds a nested rule with All, Any and NOT groups', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: configResponse }))
  const saved: SavedConfig[] = []
  await page.route('**/api/router/config/update', async (route) => {
    saved.push(route.request().postDataJSON() as SavedConfig)
    await route.fulfill({ json: {} })
  })
  await page.goto('/routing/decisions?id=guard_route&view=rules')

  const sheet = page.getByRole('dialog', { name: 'guard_route' })
  await sheet.getByRole('button', { name: 'Condition', exact: true }).click()
  const names = sheet.getByLabel('Signal name')
  await expect(names).toHaveCount(2)
  await sheet.getByRole('combobox', { name: 'Signal type' }).nth(1).click()
  await page.getByRole('option', { name: /Jailbreak/ }).click()
  await names.nth(1).fill('prompt_injection')
  await sheet.getByRole('button', { name: 'Negate condition' }).nth(1).click()
  await sheet.getByRole('radio', { name: 'Any of' }).first().click()
  await sheet.getByRole('button', { name: 'Save and apply' }).click()

  await expect.poll(() => saved.length).toBe(1)
  const decision = saved[0].recipes?.[0]?.decisions?.find((entry) => entry.name === 'guard_route')
  expect(decision?.rules).toEqual({
    operator: 'OR',
    conditions: [
      { type: 'jailbreak', name: 'prompt_injection' },
      { operator: 'NOT', conditions: [{ type: 'jailbreak', name: 'prompt_injection' }] },
    ],
  })
})
