import { expect, test } from '@playwright/test'
import path from 'node:path'

import { mockAuthenticatedAppShell } from './support/auth'

test.use({ screenshot: 'only-on-failure' })

// These tests use the actual Go WASM compiler built by dashboard-build-wasm.
// Only server data is a fixture; malformed config and DSL are parsed normally.
const invalidConfig = 'version: v0.3\nrouting:\n  unknown_field: true\n'
const validConfig = `version: v0.3
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
`

test.beforeEach(async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  // Serve the pinned editor dependency locally so editing works without a CDN.
  await page.route('https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs/**', async (route) => {
    const asset = new URL(route.request().url()).pathname.split('/min/vs/')[1]
    await route.fulfill({
      path: path.join(process.cwd(), 'node_modules/monaco-editor/min/vs', asset),
    })
  })
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
  await test.info().attach('real-wasm-error', {
    body: await page.evaluate((yaml) => window.signalDecompile(yaml), invalidConfig),
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

test('shows Format parser errors with output closed and clears them after correction', async ({
  page,
}) => {
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: validConfig }))
  await page.goto('/builder')
  await expect(page.getByText('model-a', { exact: true }).first()).toBeVisible()
  await page.getByRole('button', { name: 'DSL', exact: true }).click()
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
  await page.getByTitle('Hide Output Panel').click()
  await test.info().attach('real-format-error', {
    body: await page.evaluate(() => window.signalFormat('hello')),
    contentType: 'application/json',
  })
  await page.getByRole('button', { name: 'Format', exact: true }).click()
  const error = page.getByRole('alert').filter({ hasText: 'parse errors:' })
  await expect(error).toContainText('unexpected token "hello"')
  await expect(page.getByText('0 errors', { exact: true })).toHaveCount(0)
  await page.screenshot({ path: test.info().outputPath('format-error.png') })

  await replaceSource('MODEL "repaired" {}')
  await page.getByRole('button', { name: 'Format', exact: true }).click()
  await expect(page.getByRole('main').getByRole('alert')).toHaveCount(0)
  await expect(page.getByText('0 errors', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Show Output Panel', exact: true })).toBeVisible()
  await expect(page.getByTitle('Hide Output Panel')).toHaveCount(0)
  await expect(page.locator('.monaco-editor').first()).toContainText('repaired')
})
