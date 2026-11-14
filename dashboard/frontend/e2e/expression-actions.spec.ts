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
  await expect(page.getByRole('status', { name: 'Expression value', exact: true })).toHaveText(
    'NOT keyword("example")',
  )
})

test('the edit dialog changes a leaf label, predicate and on_error', async ({ page }) => {
  const second = page.getByRole('region', { name: 'Second editor' })
  const value = page.getByRole('status', { name: 'Other editor value' })
  await expect(value).toHaveText('classifier("intent", label: "billing", predicate: { gte: 0.6 })')

  await second.locator('.react-flow__node').first().dblclick()
  const dialog = page.getByRole('dialog', { name: 'Edit Signal' })
  await expect(dialog.getByLabel('Label')).toHaveValue('billing')
  await dialog.getByLabel('Label').fill('support')
  await dialog.getByLabel('gte', { exact: true }).fill('0.75')
  await dialog.getByLabel('On error').selectOption('match')
  await dialog.getByRole('button', { name: 'Save', exact: true }).click()

  const expected = 'classifier("intent", label: "support", predicate: { gte: 0.75 }, on_error: "match")'
  await expect(value).toHaveText(expected)
  await expect(second.locator('code')).toHaveText(expected)
  await expect(second.locator('.react-flow__node').first()).toContainText('support')
  await expect(second.locator('.react-flow__node').first()).toContainText('gte 0.75')
})

test('two mounted editors keep independent undo histories', async ({ page }) => {
  const first = page.getByRole('status', { name: 'Expression value', exact: true })
  const secondValue = page.getByRole('status', { name: 'Other editor value' })
  const second = page.getByRole('region', { name: 'Second editor' })
  const original = 'classifier("intent", label: "billing", predicate: { gte: 0.6 })'

  const firstNode = page.locator('.react-flow__node').first()
  await firstNode.click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Wrap with NOT' }).click()
  await expect(first).toHaveText('NOT keyword("example")')

  await second.locator('.react-flow__node').first().dblclick()
  const dialog = page.getByRole('dialog', { name: 'Edit Signal' })
  await dialog.getByLabel('Label').fill('support')
  await dialog.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(secondValue).toHaveText(original.replace('billing', 'support'))

  // Ctrl+Z in the first editor undoes only the first editor's change.
  await page.locator('.react-flow__node').first().click()
  await page.keyboard.press('Control+z')
  await expect(first).toHaveText('keyword("example")')
  await expect(secondValue).toHaveText(original.replace('billing', 'support'))

  // The second editor's own Undo restores only its tree.
  await second.getByTitle('Undo (Ctrl+Z)').click()
  await expect(secondValue).toHaveText(original)
  await expect(first).toHaveText('keyword("example")')
})

test('an incomplete raw draft is reported and leaving raw mode restores the prior expression', async ({
  page,
}) => {
  const value = page.getByRole('status', { name: 'Expression value', exact: true })
  const status = page.getByRole('status', { name: 'Raw status' })
  await page.getByLabel('Edit raw expression').first().check()
  const raw = page.getByLabel('Raw expression', { exact: true })

  await raw.fill('keyword("other")')
  await expect(value).toHaveText('keyword("other")')
  await raw.fill('keyword("other") AND')
  await expect(status).toHaveText('incomplete')
  await raw.fill('')
  await expect(status).toHaveText('complete')
  await raw.fill('keyword("a") AND keyword("b"')
  await expect(status).toHaveText('incomplete')

  await page.getByLabel('Edit raw expression').first().uncheck()
  await expect(status).toHaveText('complete')
  await expect(value).toHaveText('keyword("example")')
})

test('Escape exits fullscreen and undo keeps working after the fullscreen remount', async ({ page }) => {
  const value = page.getByRole('status', { name: 'Expression value', exact: true })
  await page.locator('.react-flow__node').first().click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Wrap with NOT' }).click()
  await expect(value).toHaveText('NOT keyword("example")')

  await page.getByTitle('Fullscreen').first().click()
  const exit = page.getByTitle('Exit fullscreen (Esc)')
  await expect(exit.first()).toBeVisible()
  await page.keyboard.press('Control+z')
  await expect(value).toHaveText('keyword("example")')
  await page.keyboard.press('Escape')
  await expect(exit).toHaveCount(0)
})

