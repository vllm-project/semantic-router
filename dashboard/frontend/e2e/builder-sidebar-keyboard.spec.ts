import { expect, test, type Locator, type Page } from '@playwright/test'

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
    - name: model-b
      backend_refs:
        - name: local
          endpoint: localhost:8001
          protocol: http
routing:
  modelCards:
    - name: model-a
      modality: text
    - name: model-b
      modality: text
`

async function tabUntilFocused(page: Page, target: Locator, maxPresses = 10) {
  for (let press = 0; press < maxPresses; press += 1) {
    if (await target.evaluate((element) => element === document.activeElement)) return
    await page.keyboard.press('Tab')
  }
}

test('selects Builder sidebar entities with the keyboard', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: config }))
  await page.goto('/builder')

  const home = page.getByRole('button', { name: 'Dashboard', exact: true })
  const modelA = page.getByRole('button', { name: /^model-a\b/ })
  const modelB = page.getByRole('button', { name: /^model-b\b/ })
  await expect(modelB).toBeVisible()
  await expect(home).toHaveAttribute('aria-current', 'page')

  await home.focus()
  await tabUntilFocused(page, modelB)
  await expect(modelB).toBeFocused()
  await page.keyboard.press('Enter')

  await expect(modelB).toHaveAttribute('aria-current', 'page')
  await expect(home).not.toHaveAttribute('aria-current')
  await expect(page.getByRole('button', { name: 'Delete', exact: true })).toBeVisible()

  await page.keyboard.press('Shift+Tab')
  await expect(modelA).toBeFocused()
  await page.keyboard.press('Space')
  await expect(modelA).toHaveAttribute('aria-current', 'page')
  await expect(modelB).not.toHaveAttribute('aria-current')

  await home.click()
  await expect(home).toHaveAttribute('aria-current', 'page')
  await expect(modelA).not.toHaveAttribute('aria-current')
})
