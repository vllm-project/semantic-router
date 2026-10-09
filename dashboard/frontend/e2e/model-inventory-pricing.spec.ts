import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

for (const width of [768, 1280, 1320, 1440, 1600]) {
  test(`keeps model pricing unobscured at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 549 })
    await mockAuthenticatedAppShell(page)
    const models = ['priced-model', 'unpriced-model', 'backup-model'].map((name, index) => ({
      name,
      provider_model_id: name,
      api_format: 'openai',
      backend_refs: [{ name: 'primary', endpoint: 'localhost:8000', protocol: 'http' }],
      ...(index === 0 ? { pricing: { currency: 'USD', prompt_per_1m: 3.5 } } : {}),
    }))
    await page.route('**/api/router/config/all', (route) =>
      route.fulfill({
        json: {
          version: 'v0.3',
          providers: { defaults: { model: 'priced-model' }, models },
          routing: {
            modelCards: models.map((model) => ({ name: model.name })),
            decisions: [],
          },
        },
      }),
    )
    await page.route('**/api/router/config/global', (route) => route.fulfill({ json: {} }))
    await page.route('**/api/recipe', (route) => route.fulfill({ json: { managed: false } }))
    await page.goto('/config/models')

    const pricing = page.getByRole('columnheader', { name: 'Pricing', exact: true })
    await expect(pricing).toBeVisible()
    const table = page.getByRole('table').filter({ has: pricing })
    const actions = table.getByRole('columnheader', { name: 'Actions', exact: true })
    const pricedRow = table
      .getByRole('row')
      .filter({ has: page.getByRole('cell', { name: '$3.50 / 1M', exact: true }) })
    const price = pricedRow.getByRole('cell', { name: '$3.50 / 1M', exact: true })
    const rowActions = pricedRow
      .getByRole('cell')
      .filter({ has: page.getByRole('button', { name: 'View priced-model', exact: true }) })

    // Visibility alone passes when a sticky Actions cell paints over Pricing.
    // Check the header and body geometry before any auto-scrolling interaction.
    for (const [pricingCell, actionsCell] of [
      [pricing, actions],
      [price, rowActions],
    ]) {
      const pricingBounds = await pricingCell.boundingBox()
      const actionsBounds = await actionsCell.boundingBox()
      expect(pricingBounds).not.toBeNull()
      expect(actionsBounds).not.toBeNull()
      expect(actionsBounds!.x).toBeGreaterThanOrEqual(pricingBounds!.x + pricingBounds!.width - 1)
    }

    await pricing.scrollIntoViewIfNeeded()
    const labelVisible = await pricing.evaluate((header) => {
      const range = document.createRange()
      range.selectNodeContents(header)
      const text = range.getBoundingClientRect()
      const cell = header.getBoundingClientRect()
      return (
        text.left >= cell.left &&
        text.right <= cell.right &&
        [text.left + 1, text.right - 1].every((x) =>
          header.contains(document.elementFromPoint(x, text.top + text.height / 2)),
        )
      )
    })
    expect(labelVisible).toBe(true)

    // Actions remain reachable by scrolling, including on smaller screens.
    await pricedRow.getByRole('button', { name: 'View priced-model', exact: true }).click()
    await expect(page.getByRole('dialog')).toBeVisible()
  })
}
