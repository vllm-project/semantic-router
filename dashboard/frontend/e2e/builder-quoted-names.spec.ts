import { expect, test, type Page } from '@playwright/test'
import path from 'node:path'

import { mockAuthenticatedAppShell } from './support/auth'

// Uses the real Go WASM compiler built by dashboard-build-wasm; only server data is a fixture.
const config = `version: v0.3
providers:
  models:
    - name: model-a
      backend_refs:
        - name: local
          endpoint: localhost:8000
          protocol: http
routing:
  modelCards:
    - name: model-a
      modality: text
  signals:
    domains:
      - name: computer science
        description: Computer science prompts.
        mmlu_categories: [computer science]
  decisions:
    - name: coding_help_route
      description: Answer coding questions (debugging and reviews).
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: domain
            name: computer science
      modelRefs:
        - model: model-a
`

const openBuilder = async (page: Page) => {
  await mockAuthenticatedAppShell(page)
  // Serve the pinned editor dependency locally so DSL mode works without a CDN.
  await page.route('https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs/**', async (route) => {
    const asset = new URL(route.request().url()).pathname.split('/min/vs/')[1]
    await route.fulfill({
      path: path.join(process.cwd(), 'node_modules/monaco-editor/min/vs', asset),
    })
  })
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: config }))
  await page.goto('/builder')
}

const fieldInput = (page: Page, label: string) =>
  page
    .locator('div[class*="fieldGroup"]', {
      has: page.locator('label', { hasText: new RegExp(`^\\s*${label}\\b`) }),
    })
    .locator('input')
    .first()

const compiledOutput = async (page: Page, name: string) => {
  await page.getByRole('button', { name: 'DSL', exact: true }).first().click()
  await page.getByRole('button', { name: 'Compile', exact: true }).click()
  return page.locator('pre').filter({ hasText: `name: ${name}` })
}

test('saves a Builder signal whose name needs quotes', async ({ page }) => {
  await openBuilder(page)

  await page.getByRole('button', { name: /^Computer science\b/i }).click()
  await fieldInput(page, 'Description').fill('Programming and systems prompts.')
  await expect(page.locator('pre').filter({ hasText: 'SIGNAL domain' })).toContainText(
    'SIGNAL domain "computer science" {',
  )
  await page.getByRole('button', { name: 'Save', exact: true }).click()

  await expect(await compiledOutput(page, 'computer science')).toContainText(
    'description: Programming and systems prompts.',
  )
})

test('saves a Builder route whose description contains a parenthesis', async ({ page }) => {
  await openBuilder(page)

  await page.getByRole('button', { name: /^Coding Help\b/ }).click()
  await fieldInput(page, 'Priority').fill('150')
  await page.getByRole('button', { name: 'Save', exact: true }).click()

  await expect(await compiledOutput(page, 'coding_help_route')).toContainText('priority: 150')
})

test('deletes Builder entities whose headers need quotes', async ({ page }) => {
  await openBuilder(page)

  const route = page.getByRole('button', { name: /^Coding Help\b/ })
  await route.click()
  await page.getByRole('button', { name: 'Delete', exact: true }).click()
  await expect(route).toHaveCount(0)

  const signal = page.getByRole('button', { name: /^Computer science\b/i })
  await signal.click()
  await page.getByRole('button', { name: 'Delete', exact: true }).click()
  await expect(signal).toHaveCount(0)
})
