import { expect } from '@playwright/test'

import { mockAuthenticatedAppShell, test } from './support/compiler'

// Uses the production Go compiler through its HTTP handler; only server data is a fixture.
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
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: config }))
  await page.goto('/builder')

  await page.getByRole('button', { name: /^legal_review_route\b/ }).click()
  await expect(page.locator('code').filter({ hasText: 'legal_terms' })).toHaveText(condition)
  await expect(page.locator('pre').filter({ hasText: 'ROUTE legal_review_route' })).toContainText(
    `WHEN ${condition}`,
  )

  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await page.getByRole('radio', { name: 'YAML', exact: true }).click()
  const compiled = page.locator('pre').filter({ hasText: 'name: legal_review_route' })
  await expect(compiled).toContainText(
    '- operator: NOT conditions: - operator: AND conditions: - type: keyword name: opinion_request',
  )
})
