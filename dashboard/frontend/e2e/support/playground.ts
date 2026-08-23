import { expect, type Page } from '@playwright/test'

export async function openComposerAddMenu(page: Page) {
  const trigger = page.getByRole('button', { name: 'Add to conversation' })
  await trigger.click()

  const menu = page.getByRole('menu', { name: 'Add to conversation' })
  await expect(menu).toBeVisible()
  return menu
}
