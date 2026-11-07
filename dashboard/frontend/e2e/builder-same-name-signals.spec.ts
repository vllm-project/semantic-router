import { expect, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell, test } from './support/compiler'

// Signal names are unique per type only: a keyword and a context signal share one name.
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
      - name: long_request
        operator: OR
        keywords: ["in detail"]
    context:
      - name: long_request
        min_tokens: 32K
  decisions:
    - name: keyword_route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: long_request
      modelRefs:
        - model: model-a
`

const openBuilder = async (page: Page) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: config }))
  await page.goto('/builder')
}

const sidebarSignal = (page: Page, signalType: string) =>
  page.getByRole('button', { name: /^Long Request\b/ }).filter({ hasText: signalType })

test('opens and deletes the selected one of two same-name signals', async ({ page }) => {
  await openBuilder(page)
  const keyword = sidebarSignal(page, 'keyword')
  const context = sidebarSignal(page, 'context')
  const typeBadge = page.locator('[class*="editorBadge"]')

  await context.click()
  await expect(typeBadge).toHaveText('context')
  await expect(context).toHaveAttribute('aria-current', 'page')
  await expect(keyword).not.toHaveAttribute('aria-current', 'page')

  await keyword.click()
  await expect(typeBadge).toHaveText('keyword')
  await expect(context).not.toHaveAttribute('aria-current', 'page')

  await context.click()
  await page.getByRole('button', { name: 'Delete', exact: true }).click()
  await expect(context).toHaveCount(0)
  await expect(keyword).toHaveCount(1)
})
