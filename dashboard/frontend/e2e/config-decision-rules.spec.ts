import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

// The Router accepts a single condition as the whole rule tree.
const singleConditionRules = { type: 'jailbreak', name: 'prompt_injection' }

const configResponse = {
  version: 'v0.3',
  providers: {
    models: [
      {
        name: 'model-a',
        backend_refs: [{ name: 'local', endpoint: 'localhost:8000', protocol: 'http' }],
      },
    ],
  },
  routing: {
    modelCards: [{ name: 'model-a', modality: 'text' }],
    signals: { jailbreak: [{ name: 'prompt_injection', threshold: 0.7 }] },
    decisions: [
      {
        name: 'guard_route',
        description: 'Route detected prompt attacks.',
        priority: 120,
        rules: singleConditionRules,
        modelRefs: [{ model: 'model-a', use_reasoning: false }],
      },
    ],
  },
}

type SavedConfig = { routing?: { decisions?: Array<{ name: string; rules?: unknown }> } }

test('keeps a single-condition rule when a decision is saved unchanged', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: configResponse }))
  const saved: SavedConfig[] = []
  await page.route('**/api/router/config/update', async (route) => {
    saved.push(route.request().postDataJSON() as SavedConfig)
    await route.fulfill({ json: {} })
  })
  await page.goto('/config/decisions')

  const editButton = page.getByRole('button', { name: 'Edit guard_route' })
  await expect(page.locator('tr', { has: editButton })).toContainText('1 condition')

  await page.getByRole('button', { name: 'View guard_route' }).click()
  const view = page.getByRole('dialog', { name: 'Decision: guard_route' })
  await expect(view.getByText('Single condition')).toBeVisible()
  await expect(view.getByText('jailbreak: prompt_injection')).toBeVisible()
  await expect(view.getByText('Unconditional match')).toHaveCount(0)
  await view.getByRole('button', { name: 'Close' }).first().click()

  await editButton.click()
  const editor = page.getByRole('dialog', { name: 'Edit Decision: guard_route' })
  await expect(editor.locator('label', { hasText: 'Root behavior' }).locator('select')).toHaveValue(
    'CONDITION',
  )
  await expect(editor.getByPlaceholder('technical')).toHaveValue('prompt_injection')
  await editor.getByRole('button', { name: 'Save', exact: true }).click()

  await expect.poll(() => saved.length).toBe(1)
  const decision = saved[0].routing?.decisions?.find((entry) => entry.name === 'guard_route')
  expect(decision?.rules).toEqual(singleConditionRules)
})
