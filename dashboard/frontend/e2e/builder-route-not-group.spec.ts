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
      - name: legal_terms
        operator: OR
        keywords: [contract, liability]
      - name: opinion_request
        operator: OR
        keywords: [argue, persuade]
      - name: risk_markers
        operator: OR
        keywords: [lawsuit, penalty]
  decisions:
    - name: legal_review_route
      description: Review legal requests unless they only ask for an argument.
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: legal_terms
          - operator: NOT
            conditions:
              - operator: AND
                conditions:
                  - type: keyword
                    name: opinion_request
                  - operator: NOT
                    conditions:
                      - type: keyword
                        name: risk_markers
      modelRefs:
        - model: model-a
`
const condition =
  'keyword("legal_terms") AND NOT (keyword("opinion_request") AND NOT keyword("risk_markers"))'

test('keeps the group a NOT applies to when a Builder route is saved', async ({ page }) => {
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

  await page.getByRole('button', { name: /^Legal Review\b/ }).click()
  await expect(page.locator('code').filter({ hasText: 'legal_terms' })).toHaveText(condition)
  await expect(page.locator('pre').filter({ hasText: 'ROUTE legal_review_route' })).toContainText(
    `WHEN ${condition}`,
  )

  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await page.getByRole('button', { name: 'DSL', exact: true }).first().click()
  await page.getByRole('button', { name: 'Compile', exact: true }).click()
  const compiled = page.locator('pre').filter({ hasText: 'name: legal_review_route' })
  await expect(compiled).toContainText(
    '- operator: NOT conditions: - operator: AND conditions: - type: keyword name: opinion_request',
  )
})
