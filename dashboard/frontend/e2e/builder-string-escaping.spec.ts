import { expect } from '@playwright/test'

import { mockAuthenticatedAppShell, test } from './support/compiler'

// Uses the production Go compiler through its HTTP handler; only server data is a fixture.
const config = String.raw`version: v0.5
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
      keywords:
        - name: numbered_steps
          description: Prompts that contain numbered list items such as "1. ..."
          operator: OR
          method: regex
          keywords:
            - '(?m)^\s*\d+\.\s+'
`

test('keeps quotes and backslashes intact when a Builder signal is saved', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: config }))
  await page.goto('/builder')

  await page.getByRole('button', { name: /^numbered_steps\b/ }).click()
  await page.getByRole('button', { name: 'Save', exact: true }).click()

  await page.getByRole('radio', { name: 'YAML', exact: true }).click()
  const compiled = page.locator('pre').filter({ hasText: 'name: numbered_steps' })
  await expect(compiled).toContainText(
    'description: Prompts that contain numbered list items such as "1. ..."',
  )
  await expect(compiled).toContainText(String.raw`- (?m)^\s*\d+\.\s+`)
})
