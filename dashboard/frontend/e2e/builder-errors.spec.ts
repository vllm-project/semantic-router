import { expect } from '@playwright/test'

import { mockAuthenticatedAppShell, test } from './support/compiler'

test.use({ screenshot: 'only-on-failure' })

// These tests use the production Go compiler through its HTTP handler.
// Only server data is a fixture; malformed config and DSL are parsed normally.
const invalidConfig = 'version: v0.5\nrecipes:\n  - name: default\n    unknown_field: true\n'
const validConfig = `version: v0.5
providers:
  models:
    - name: model-a
      backend_refs:
        - name: local
          endpoint: localhost:8000
          protocol: http
recipes:
  - name: default
    decisions:
      - name: default_route
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs:
          - model: model-a
`

test.beforeEach(async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/deploy/preview', async (route) => {
    await route.fulfill({ json: { current: validConfig, preview: validConfig } })
  })
})

test('shows automatic load failure and preserves the detailed error during manual import recovery', async ({
  page,
}) => {
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: invalidConfig }))
  await page.goto('/builder')
  await expect(page.getByRole('button', { name: 'Import', exact: true })).toBeEnabled()
  await test.info().attach('real-compiler-error', {
    body: await page.evaluate(
      async (source) =>
        (
          await fetch('/api/dsl/decompile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source }),
          })
        ).text(),
      invalidConfig,
    ),
    contentType: 'application/json',
  })
  await expect(page.getByRole('alert')).toContainText('unknown_field')
  await expect(page.getByRole('alert')).toContainText('Failed to load')
  await expect(page.getByText('1 error', { exact: true }).first()).toBeVisible()
  await expect(page.getByText('Valid', { exact: true })).toHaveCount(0)
  await page.screenshot({ path: test.info().outputPath('automatic-load-error.png') })

  await page.getByRole('button', { name: 'Import', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Import config' })
  await dialog.getByRole('textbox', { name: 'Router config YAML' }).fill(invalidConfig)
  await dialog.getByRole('button', { name: 'Import', exact: true }).click()
  await expect(dialog.getByRole('alert')).toContainText('unknown_field')
  await expect(dialog).toBeVisible()

  await dialog.getByRole('textbox', { name: 'Router config YAML' }).fill(validConfig)
  await dialog.getByRole('button', { name: 'Import', exact: true }).click()
  await expect(dialog).not.toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByText('Valid', { exact: true })).toBeVisible()
  await expect(page.getByText('model-a', { exact: true }).first()).toBeVisible()
})

test('shows a failed router request and recovers through Load from Router', async ({ page }) => {
  await page.route('**/api/router/config/yaml', (route) =>
    route.fulfill({ status: 503, body: 'unavailable' }),
  )
  await page.goto('/builder')
  await expect(page.getByRole('alert')).toContainText('HTTP 503')
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: validConfig }))
  await page.getByRole('button', { name: 'Import', exact: true }).click()
  await page.getByRole('button', { name: 'Load from Router' }).click()
  await expect(page.getByRole('dialog')).not.toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByText('model-a', { exact: true }).first()).toBeVisible()
})

test('shows Format parser errors in the DSL view and clears them after correction', async ({
  page,
}) => {
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: validConfig }))
  await page.goto('/builder')
  await expect(page.getByText('model-a', { exact: true }).first()).toBeVisible()
  await page.getByRole('radio', { name: 'DSL', exact: true }).click()
  const editor = page.getByRole('textbox', { name: 'Editor content', exact: true })
  await expect(editor).toBeVisible()
  // Monaco uses the emulated browser's platform, while ControlOrMeta uses the host OS.
  const selectAll = await page.evaluate(() =>
    navigator.userAgent.includes('Macintosh') ? 'Meta+A' : 'Control+A',
  )
  async function replaceSource(source: string) {
    await editor.focus()
    await page.keyboard.press(selectAll)
    await page.keyboard.type(source)
    await expect(page.locator('.monaco-editor .view-lines').first()).toHaveText(source)
  }
  await replaceSource('hello')
  await test.info().attach('real-format-error', {
    body: await page.evaluate(async () =>
      (
        await fetch('/api/dsl/format', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: 'hello' }),
        })
      ).text(),
    ),
    contentType: 'application/json',
  })
  await page.getByRole('button', { name: 'Format', exact: true }).click()
  const error = page.getByRole('alert').filter({ hasText: 'parse errors:' })
  await expect(error).toContainText('unexpected token "hello"')
  await expect(page.getByText('0 errors', { exact: true })).toHaveCount(0)
  await page.screenshot({ path: test.info().outputPath('format-error.png') })

  // Format round-trips through the compiler, which keeps a model only when it declares something.
  await replaceSource('MODEL "repaired" { description: "Fixed" }')
  await page.getByRole('button', { name: 'Format', exact: true }).click()
  await expect(page.getByRole('main').getByRole('alert')).toHaveCount(0)
  await expect(page.getByText('0 errors', { exact: true })).toBeVisible()
  // Formatting keeps the author in the DSL view.
  await expect(page.getByRole('radio', { name: 'DSL', exact: true })).toHaveAttribute(
    'aria-checked',
    'true',
  )
  await expect(page.locator('.monaco-editor').first()).toContainText('repaired')
})
