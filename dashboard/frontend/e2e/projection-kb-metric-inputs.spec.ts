import { expect } from '@playwright/test'
import path from 'node:path'

import { mockAuthenticatedAppShell, test } from './support/compiler'

// The Builder test uses the production Go compiler through its HTTP handler.
const kbMetricInput = {
  type: 'kb_metric',
  kb: 'privacy_kb',
  metric: 'private_vs_public',
  weight: 1,
  value_source: 'score',
}

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
    projections: {
      scores: [{ name: 'privacy_bias', method: 'weighted_sum', inputs: [kbMetricInput] }],
    },
  },
}

const configYaml = `version: v0.3
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
  projections:
    scores:
      - name: privacy_bias
        method: weighted_sum
        inputs:
          - type: kb_metric
            kb: privacy_kb
            metric: private_vs_public
            weight: 1.0
            value_source: score
`

test('lists kb_metric score inputs by knowledge base and metric', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: configResponse }))
  await page.goto('/config/projections')

  const row = page.getByRole('row', { name: /privacy_bias/ })
  await expect(row).toContainText('kb_metric:privacy_kb:private_vs_public')

  await page.getByPlaceholder('Search partitions, scores, or mappings...').fill('private_vs_public')
  await expect(row).toBeVisible()
})

test('keeps kb_metric inputs when a Builder projection score is saved', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  // Serve the pinned editor dependency locally so DSL mode works without a CDN.
  await page.route('https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs/**', async (route) => {
    const asset = new URL(route.request().url()).pathname.split('/min/vs/')[1]
    await route.fulfill({
      path: path.join(process.cwd(), 'node_modules/monaco-editor/min/vs', asset),
    })
  })
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: configYaml }))
  await page.goto('/builder')

  await page.getByText('Privacy Bias', { exact: true }).click()
  await page.getByRole('button', { name: 'Save', exact: true }).click()

  await page.getByRole('button', { name: 'DSL', exact: true }).first().click()
  await page.getByRole('button', { name: 'Compile', exact: true }).click()
  const compiled = page.locator('pre').filter({ hasText: 'name: privacy_bias' })
  await expect(compiled).toContainText('kb: privacy_kb')
  await expect(compiled).toContainText('metric: private_vs_public')
})
