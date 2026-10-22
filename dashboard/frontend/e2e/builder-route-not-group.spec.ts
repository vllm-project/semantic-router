import { expect, type Page } from '@playwright/test'
import path from 'node:path'

import { mockAuthenticatedAppShell, test } from './support/compiler'

// Uses the production Go compiler through its HTTP handler; only server data is a fixture.
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

async function openBuilder(page: Page, yaml: string) {
  await mockAuthenticatedAppShell(page)
  // Serve the pinned editor dependency locally so DSL mode works without a CDN.
  await page.route('https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs/**', async (route) => {
    const asset = new URL(route.request().url()).pathname.split('/min/vs/')[1]
    await route.fulfill({
      path: path.join(process.cwd(), 'node_modules/monaco-editor/min/vs', asset),
    })
  })
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: yaml }))
  await page.goto('/builder')
}

test('keeps the group a NOT applies to when a Builder route is saved', async ({ page }) => {
  await openBuilder(page, config)

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

const classifierConfig = `version: v0.3
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
    classifiers:
      - name: content-risk
        type: local
        labels: [violence, hate]
    keywords:
      - name: legal_terms
        operator: OR
        keywords: [contract, liability]
  decisions:
    - name: risk_review_route
      description: Review risky legal requests.
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: classifier
            name: content-risk
            label: violence
            predicate:
              gte: 0.7
            on_error: match
          - type: keyword
            name: legal_terms
      modelRefs:
        - model: model-a
`

test('keeps leaf label, predicate and on_error when a Builder route is edited', async ({ page }) => {
  await openBuilder(page, classifierConfig)

  await page.getByRole('button', { name: /^Risk Review\b/ }).click()
  const leaf = 'classifier("content-risk", label: "violence", predicate: { gte: 0.7 }, on_error: "match")'
  await expect(page.locator('code').filter({ hasText: 'content-risk' })).toHaveText(
    `${leaf} AND keyword("legal_terms")`,
  )

  // Edit only the label through the shared editor's leaf dialog.
  await page.locator('.react-flow__node').filter({ hasText: 'content-risk' }).dblclick()
  const dialog = page.getByRole('dialog', { name: 'Edit Signal' })
  await expect(dialog.getByLabel('gte', { exact: true })).toHaveValue('0.7')
  await dialog.getByLabel('Label').fill('hate')
  await dialog.getByRole('button', { name: 'Save', exact: true }).click()
  const edited = leaf.replace('violence', 'hate')
  await expect(page.locator('pre').filter({ hasText: 'ROUTE risk_review_route' })).toContainText(
    `WHEN ${edited} AND keyword("legal_terms")`,
  )

  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await page.getByRole('button', { name: 'DSL', exact: true }).first().click()
  await page.getByRole('button', { name: 'Compile', exact: true }).click()
  const compiled = page.locator('pre').filter({ hasText: 'name: risk_review_route' })
  await expect(compiled).toContainText('type: classifier name: content-risk label: hate')
  await expect(compiled).toContainText('gte: 0.7')
  await expect(compiled).toContainText('on_error: match')
})

test('an incomplete raw WHEN expression blocks the Builder route save', async ({ page }) => {
  await openBuilder(page, config)

  await page.getByRole('button', { name: /^Legal Review\b/ }).click()
  await expect(page.locator('code').filter({ hasText: 'legal_terms' })).toHaveText(condition)
  const save = page.getByRole('button', { name: 'Save', exact: true })
  await expect(save).toBeEnabled()

  await page.getByLabel('Edit raw expression').check()
  const raw = page.getByLabel('Raw expression', { exact: true })
  await raw.fill('keyword("legal_terms") AND keyword("risk_markers"')
  await expect(save).toBeDisabled()
  await expect(save).toHaveAttribute('title', /Raw expression is incomplete/)

  await raw.fill('keyword("legal_terms") AND keyword("risk_markers")')
  await expect(save).toBeEnabled()
  await expect(page.locator('pre').filter({ hasText: 'ROUTE legal_review_route' })).toContainText(
    'WHEN keyword("legal_terms") AND keyword("risk_markers")',
  )
})
