import { expect, test } from '@playwright/test'
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
    keywords:
      - name: urgent
        operator: OR
        keywords: [urgent]
    classifiers:
      - name: safety-score
        type: local
        model_path: models/safety-score
        labels: [safe, unsafe]
  decisions:
    - name: unsafe_prompts_route
      description: Route unsafe prompts to a local model.
      priority: 125
      rules:
        operator: AND
        conditions:
          - type: classifier
            name: safety-score
            label: unsafe
            predicate:
              gte: 0.5
          - type: keyword
            name: urgent
            predicate:
              gt: 0.0000001
      modelRefs:
        - model: model-a
`
const condition =
  'classifier("safety-score", label: "unsafe", predicate: { gte: 0.5 }) AND keyword("urgent", predicate: { gt: 0.0000001 })'

test('keeps condition fields when a Builder route is edited and saved', async ({ page }) => {
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

  await page.getByRole('button', { name: /^Unsafe Prompts\b/ }).click()
  const result = page.locator('code').filter({ hasText: 'classifier(' })
  await expect(result).toHaveText(condition)

  // Re-selecting the same signal in the node editor keeps its fields.
  await page
    .locator('.react-flow__node')
    .filter({ hasText: 'safety-score' })
    .click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Edit Signal' }).click()
  const editor = page.getByRole('dialog', { name: 'Edit Signal' })
  await editor.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(editor).toHaveCount(0)
  await expect(result).toHaveText(condition)

  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await page.getByRole('button', { name: 'DSL', exact: true }).first().click()
  await page.getByRole('button', { name: 'Compile', exact: true }).click()
  const compiled = page.locator('pre').filter({ hasText: 'name: unsafe_prompts_route' })
  await expect(compiled).toContainText('label: unsafe')
  await expect(compiled).toContainText('gte: 0.5')
  await expect(compiled).toContainText('gt: 1e-07')
})
