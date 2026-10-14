import { expect, type Page } from '@playwright/test'
import path from 'node:path'

import { mockAuthenticatedAppShell, test } from './support/compiler'

test.use({ screenshot: 'only-on-failure' })

// These tests use the production Go compiler through its HTTP handler.
// Deploy writes the imported values back, and the compiler preserves references instead of resolving its process
// environment, so references and $$ escapes must survive the import as written.
const referencesConfig = `version: v0.3
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
      - name: billing
        operator: OR
        keywords: ["invoice"]
  decisions:
    - name: billing_route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: billing
      modelRefs:
        - model: model-a
      plugins:
        - type: system_prompt
          configuration:
            enabled: true
            system_prompt: "Quote every price in $$USD."
        - type: header_mutation
          configuration:
            add:
              - name: Authorization
                value: "Bearer \${BILLING_GATEWAY_TOKEN}"
              - name: X-Tenant
                value: "\${TENANT_ID:-default-tenant}"
`

// The OpenAI RAG backend requires api_key, which the docs recommend setting via ${VAR}.
const requiredReferenceConfig = `${referencesConfig}        - type: rag
          configuration:
            enabled: true
            backend: openai
            backend_config:
              vector_store_id: vs_billing_docs
              api_key: \${OPENAI_API_KEY}
`

// Each typed value is valid only once its reference takes the default, and the
// RAG key is valid only as written.
const typedDefaultsConfig = `version: v0.3
providers:
  models:
    - name: model-a
      reliability:
        base_ejection_time: \${EJECTION_TIME:-30s}
      backend_refs:
        - name: local
          endpoint: \${BACKEND_HOST:-127.0.0.1}:8000
          protocol: http
routing:
  modelCards:
    - name: model-a
      modality: text
  signals:
    context:
      - name: long_context
        min_tokens: \${CTX_MIN:-4k}
  decisions:
    - name: long_route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: context
            name: long_context
      modelRefs:
        - model: model-a
      plugins:
        - type: rag
          configuration:
            enabled: true
            backend: openai
            backend_config:
              vector_store_id: vs_billing_docs
              api_key: \${OPENAI_API_KEY}
`

test.beforeEach(async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  // Serve the pinned editor dependency locally so the deploy diff works without a CDN.
  await page.route('https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs/**', async (route) => {
    const asset = new URL(route.request().url()).pathname.split('/min/vs/')[1]
    await route.fulfill({
      path: path.join(process.cwd(), 'node_modules/monaco-editor/min/vs', asset),
    })
  })
})

async function deployPreviewYaml(page: Page, config: string) {
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: config }))
  await page.route('**/api/router/config/deploy/preview', (route) =>
    route.fulfill({ json: { current: config, preview: config } }),
  )
  await page.goto('/builder')
  await expect(page.getByText('model-a', { exact: true }).first()).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)

  const preview = page.waitForRequest('**/api/router/config/deploy/preview')
  await page.getByRole('button', { name: 'Deploy', exact: true }).click()
  return ((await preview).postDataJSON() as { yaml: string }).yaml
}

test('keeps environment references and $$ escapes when deploying an imported config', async ({
  page,
}) => {
  const yaml = await deployPreviewYaml(page, referencesConfig)
  expect(yaml).toContain('Quote every price in $$USD.')
  expect(yaml).toContain('Bearer ${BILLING_GATEWAY_TOKEN}')
  expect(yaml).toContain('${TENANT_ID:-default-tenant}')
})

test('imports a config whose required field is an environment reference', async ({ page }) => {
  const yaml = await deployPreviewYaml(page, requiredReferenceConfig)
  expect(yaml).toContain('api_key: ${OPENAI_API_KEY}')
})

test('imports a config whose typed values take environment defaults', async ({ page }) => {
  const yaml = await deployPreviewYaml(page, typedDefaultsConfig)
  expect(yaml).toContain('min_tokens: ${CTX_MIN:-4k}')
  expect(yaml).toContain('api_key: ${OPENAI_API_KEY}')
})
