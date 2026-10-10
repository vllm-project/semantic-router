import { expect } from '@playwright/test'
import path from 'node:path'

import { mockAuthenticatedAppShell, test } from './support/compiler'

// Uses the production Go compiler through its HTTP handler; only server data is a fixture.
const config = String.raw`version: v0.3
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
    structure:
      - name: numbered_steps
        description: Prompts that contain numbered list items such as "1. ..."
        feature:
          type: exists
          source:
            type: regex
            pattern: '(?m)^\s*\d+\.\s+'
`

test('keeps quotes and backslashes intact when a Builder signal is saved', async ({ page }) => {
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

  await page.getByText('Numbered Steps', { exact: true }).click()
  await page.getByRole('button', { name: 'Save', exact: true }).click()

  await page.getByRole('button', { name: 'DSL', exact: true }).first().click()
  await page.getByRole('button', { name: 'Compile', exact: true }).click()
  const compiled = page.locator('pre').filter({ hasText: 'name: numbered_steps' })
  await expect(compiled).toContainText(
    'description: Prompts that contain numbered list items such as "1. ..."',
  )
  await expect(compiled).toContainText(String.raw`pattern: (?m)^\s*\d+\.\s+`)
})
