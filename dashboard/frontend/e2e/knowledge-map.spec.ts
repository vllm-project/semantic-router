import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const metadataResponse = {
  name: 'privacy_kb',
  description: 'Privacy exemplars',
  projection: 'pca_2d',
  model_type: 'mmbert',
  point_count: 356,
  label_count: 18,
  group_count: 4,
  label_names: ['proprietary_code'],
}

async function mockKnowledgeMap(page: Page) {
  await mockAuthenticatedAppShell(page)

  await page.route('**/api/auth/bootstrap/can-register', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ canRegister: false }),
    })
  })

  await page.route('**/api/router/config/kbs/privacy_kb/map/metadata', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(metadataResponse),
    })
  })

  await page.route('**/embedded/wizmap/**', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'text/html',
      body: '<!doctype html><html><body style="margin:0;background:#fff"><div id="wizmap-shell" style="width:100vw;height:100vh">wizmap shell</div></body></html>',
    })
  })
}

test.describe('knowledge map route', () => {
  test('renders a standalone full-screen map shell without dashboard chrome', async ({ page }) => {
    await mockKnowledgeMap(page)
    await page.goto('/knowledge-bases/privacy_kb/map')

    await expect(page.getByRole('link', { name: 'Back to Bases' })).toBeVisible()
    await expect(page.frameLocator('iframe[title="Knowledge map for privacy_kb"]').locator('#wizmap-shell')).toBeVisible()

    await expect(page.getByText('Dashboard')).toHaveCount(0)
    await expect(page.getByText('Projection')).toHaveCount(0)
    await expect(page.getByText('Points')).toHaveCount(0)
    await expect(page.getByText('Labels')).toHaveCount(0)
  })
})
