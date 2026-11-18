import { expect, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell, test } from './support/compiler'

test.use({ screenshot: 'only-on-failure' })

// Monaco ships in the app bundle: the Dashboard's Content-Security-Policy
// (`script-src 'self'`) blocks the CDN its loader would otherwise inject.
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
    decisions:
      - name: default_route
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs:
          - model: model-a
`
const deployed = config.replace('priority: 100', 'priority: 200')

async function openEditor(page: Page) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: config }))
  await page.route('**/api/router/config/deploy/preview', (route) =>
    route.fulfill({ json: { current: config, preview: deployed } }),
  )
  await page.goto('/builder')
  await expect(page.getByText('model-a', { exact: true }).first()).toBeVisible()
}

test('loads the DSL editor and the deploy diff from the app bundle', async ({ page, baseURL }) => {
  const requests: string[] = []
  const workers: string[] = []
  const errors: string[] = []
  page.on('request', (request) => requests.push(request.url()))
  page.on('worker', (worker) => workers.push(worker.url()))
  page.on('pageerror', (error) => errors.push(error.message))

  await openEditor(page)
  // The visual view does not pay for Monaco.
  expect(requests.filter((url) => url.includes('monacoSetup'))).toEqual([])

  await page.getByRole('radio', { name: 'DSL', exact: true }).click()
  const editor = page.locator('.monaco-editor').first()
  await expect(editor).toContainText('model-a')
  await expect(editor.locator('.view-line [class*="mtk"]').first()).toBeVisible()

  await page.getByRole('button', { name: 'Review & deploy' }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog.locator('.monaco-diff-editor')).toBeVisible()
  await expect(dialog.getByText('+1', { exact: true })).toBeVisible()
  expect(workers.some((url) => url.includes('editor.worker'))).toBe(true)
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toHaveCount(0)

  const origin = new URL(baseURL ?? page.url()).origin
  expect(requests.filter((url) => url.includes('cdn.jsdelivr.net'))).toEqual([])
  expect(requests.filter((url) => /^https?:/.test(url) && new URL(url).origin !== origin)).toEqual(
    [],
  )
  expect(errors).toEqual([])
})

test('falls back to plain text when the editor bundle cannot load', async ({ page }) => {
  await page.route(/\/monacoSetup[^/]*$/, (route) => route.abort())
  await openEditor(page)

  await page.getByRole('radio', { name: 'DSL', exact: true }).click()
  await expect(page.getByText('The code editor could not load')).toBeVisible()
  const source = page.getByRole('textbox', { name: 'Editor content', exact: true })
  await expect(source).toHaveValue(/model-a/)
  await source.press('End')
  await source.pressSequentially('\n# edited as plain text')
  await expect(source).toHaveValue(/# edited as plain text/)
  await expect(page.locator('.monaco-editor')).toHaveCount(0)

  await page.getByRole('button', { name: 'Review & deploy' }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByText(/changed region as plain text/)).toBeVisible()
  await expect(dialog.getByText(/^changed region · \d+ lines$/)).toBeVisible()
  await expect(page.getByTestId('route-load-error')).toHaveCount(0)
})

test('falls back to the plain diff when the editor worker cannot start', async ({ page }) => {
  await page.context().route(/editor\.worker/, (route) => route.abort())
  await openEditor(page)

  await page.getByRole('button', { name: 'Review & deploy' }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByText(/changed region as plain text/)).toBeVisible()
  await expect(dialog.locator('.monaco-diff-editor')).toHaveCount(0)
  await expect(page.getByTestId('route-load-error')).toHaveCount(0)
})
