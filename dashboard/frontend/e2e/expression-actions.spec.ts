import { expect, test } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.goto('/e2e/support/expression-actions.html')
})

test('expression menu supports native activation, arrows, dismissal and focus return', async ({
  page,
}) => {
  const trigger = page.getByRole('button', { name: 'Expression actions', exact: true })
  await page.keyboard.press('Tab')
  await expect(trigger).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.getByRole('menuitem', { name: 'Edit Signal' })).toBeFocused()
  await page.keyboard.press('ArrowUp')
  await expect(page.getByRole('menuitem', { name: 'Delete', exact: true })).toBeFocused()
  await page.keyboard.press('Home')
  await expect(page.getByRole('menuitem', { name: 'Edit Signal' })).toBeFocused()
  await page.keyboard.press('End')
  await expect(page.getByRole('menuitem', { name: 'Delete', exact: true })).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('menu')).toHaveCount(0)
  await expect(trigger).toBeFocused()
  await page.keyboard.press('Enter')
  await page.keyboard.press('ArrowDown')
  await page.keyboard.press('Enter')
  await expect(page.getByRole('status', { name: 'Last action' })).toHaveText('wrap AND')
  await trigger.click()
  await page.keyboard.press('Tab')
  await expect(page.getByRole('menu')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'After actions' })).toBeFocused()
})

test('a keyboard user can open a graph node menu and change the expression', async ({ page }) => {
  // Reach the actual ReactFlow node through the document tab order.
  const node = page.locator('.react-flow__node').first()
  await expect(node).toBeVisible()
  for (let index = 0; index < 30; index += 1) {
    if (await node.evaluate((element) => element === document.activeElement)) break
    await page.keyboard.press('Tab')
  }
  await expect(node).toBeFocused()
  await page.keyboard.press('Shift+F10')
  await expect(page.getByRole('menuitem', { name: 'Edit Signal' })).toBeFocused()
  await page.keyboard.press('ArrowDown')
  await page.keyboard.press('ArrowDown')
  await page.keyboard.press('ArrowDown')
  await expect(page.getByRole('menuitem', { name: 'Wrap with NOT' })).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.getByRole('status', { name: 'Expression value' })).toHaveText(
    'NOT keyword("example")',
  )
})
