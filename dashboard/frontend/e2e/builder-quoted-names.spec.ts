import { expect, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell, test } from './support/compiler'

// Uses the production Go compiler through its HTTP handler; only server data is a fixture.
// A context signal saves without a model deployment; a domain signal needs one.
const config = `version: v0.5
providers:
  models:
    - name: model-a
      backend_refs:
        - name: local
          endpoint: localhost:8000
          protocol: http
recipes:
  - name: default
    signals:
      context:
        - name: long prompts
          min_tokens: 32K
          description: Prompts that need a large context window.
    decisions:
      - name: coding_help_route
        description: Answer coding questions (debugging and reviews).
        priority: 100
        rules:
          operator: AND
          conditions:
            - type: context
              name: long prompts
        modelRefs:
          - model: model-a
`

const openBuilder = async (page: Page) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: config }))
  await page.goto('/builder')
}

const fieldInput = (page: Page, label: string) => page.getByLabel(new RegExp(`^${label}\\b`))

// The YAML view compiles the working copy when it opens.
const compiledOutput = async (page: Page, name: string) => {
  await page.getByRole('radio', { name: 'YAML', exact: true }).click()
  return page.locator('pre').filter({ hasText: `name: ${name}` })
}

test('saves a Builder signal whose name needs quotes', async ({ page }) => {
  await openBuilder(page)

  await page.getByRole('button', { name: /^Long prompts\b/i }).click()
  await fieldInput(page, 'Description').fill('Prompts with 32K tokens or more.')
  await expect(page.locator('pre').filter({ hasText: 'SIGNAL context' })).toContainText(
    'SIGNAL context "long prompts" {',
  )
  await page.getByRole('button', { name: 'Save', exact: true }).click()

  await expect(await compiledOutput(page, 'long prompts')).toContainText(
    'description: Prompts with 32K tokens or more.',
  )
})

test('saves a Builder route whose description contains a parenthesis', async ({ page }) => {
  await openBuilder(page)

  await page.getByRole('button', { name: /^coding_help_route\b/ }).click()
  await fieldInput(page, 'Priority').fill('150')
  await page.getByRole('button', { name: 'Save', exact: true }).click()

  await expect(await compiledOutput(page, 'coding_help_route')).toContainText('priority: 150')
})

test('deletes Builder entities whose headers need quotes', async ({ page }) => {
  await openBuilder(page)

  const route = page.getByRole('button', { name: /^coding_help_route\b/ })
  await route.click()
  await page.getByRole('button', { name: 'Delete', exact: true }).click()
  await expect(route).toHaveCount(0)

  const signal = page.getByRole('button', { name: /^Long prompts\b/i })
  await signal.click()
  await page.getByRole('button', { name: 'Delete', exact: true }).click()
  await expect(signal).toHaveCount(0)
})
