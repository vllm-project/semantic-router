import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const policy = {
  role_mappings: [
    {
      name: 'engineering-access',
      subjects: [{ kind: 'Group', name: 'platform-engineering' }],
      role: 'premium_tier',
      model_refs: ['openai/gpt5.4', 'qwen/qwen3.5-rocm'],
      priority: 10,
    },
  ],
  rate_tiers: [
    {
      name: 'engineering-tier',
      group: 'platform-engineering',
      user: '',
      rpm: 240,
      tpm: 120000,
    },
  ],
  updated_at: '2026-07-11T00:00:00Z',
}

const fragment = {
  role_bindings: [{ role: 'premium_tier' }],
  decisions: [],
  ratelimit: { enabled: true },
}

async function mockSecurityPolicy(page: Page) {
  await mockAuthenticatedAppShell(page)

  await page.route('**/api/security/policy', async (route) => {
    if (route.request().method() === 'PUT') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ policy, fragment, message: 'Security policy saved.' }),
      })
      return
    }

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(policy),
    })
  })

  await page.route('**/api/security/policy/preview', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(fragment),
    })
  })
}

test.describe('Security policy responsive layout', () => {
  test('keeps policy inputs usable and contains the rate table at 390px', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await mockSecurityPolicy(page)
    await page.goto('/security')

    await expect(page.getByRole('heading', { name: 'Security Policy' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Add Mapping' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Add Tier' })).toBeVisible()

    const routerRoleBox = await page.getByLabel('Router Role').boundingBox()
    const modelsBox = await page.getByLabel('Models (comma-separated)').boundingBox()
    const subjectTypeBox = await page.getByLabel('Subject type 1').boundingBox()
    const subjectNameBox = await page.getByLabel('Subject name 1').boundingBox()

    expect(routerRoleBox).not.toBeNull()
    expect(modelsBox).not.toBeNull()
    expect(subjectTypeBox).not.toBeNull()
    expect(subjectNameBox).not.toBeNull()
    expect(routerRoleBox!.width).toBeGreaterThan(250)
    expect(modelsBox!.width).toBeGreaterThan(250)
    expect(subjectTypeBox!.width).toBeGreaterThanOrEqual(120)
    expect(subjectNameBox!.width).toBeGreaterThan(130)

    const tableScroller = page.getByTestId('rate-tier-scroller')
    await expect(tableScroller).toHaveAttribute('tabindex', '0')
    const tableMetrics = await tableScroller.evaluate((element) => ({
      clientWidth: element.clientWidth,
      scrollWidth: element.scrollWidth,
      right: element.getBoundingClientRect().right,
    }))
    expect(tableMetrics.scrollWidth).toBeGreaterThan(tableMetrics.clientWidth)
    expect(tableMetrics.right).toBeLessThanOrEqual(390)

    await page.evaluate(() => window.scrollTo({ left: 200 }))
    await expect.poll(() => page.evaluate(() => window.scrollX)).toBe(0)

    await expect(
      page.getByRole('button', { name: 'Remove role mapping engineering-access' }),
    ).toBeVisible()
    await expect(
      page.getByRole('button', { name: 'Remove rate tier engineering-tier' }),
    ).toBeVisible()

    await page.getByLabel('Router Role').fill('research_tier')
    await expect(page.getByLabel('Router Role')).toHaveValue('research_tier')

    await page.getByRole('button', { name: 'Preview Config Fragment' }).click()
    await expect(page.getByRole('heading', { name: 'Router Config Fragment' })).toBeVisible()
    await expect(page.getByText('"premium_tier"')).toBeVisible()
  })
})
