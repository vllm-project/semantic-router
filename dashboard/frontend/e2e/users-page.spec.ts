import { expect, test } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

test.describe('Users page', () => {
  test('creates a personal invitation with a clear dashboard role', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })

    await mockAuthenticatedAppShell(page, {
      user: {
        id: 'user-admin-2',
        email: 'ada@example.com',
        name: 'Ada Lovelace',
        role: 'admin',
        permissions: ['users.manage', 'users.view', 'config.read', 'config.write'],
      },
    })

    await page.route('**/api/admin/users**', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          users: [
            {
              id: 'user-1',
              email: 'reader@example.com',
              name: 'Read User',
              role: 'read',
              status: 'active',
              createdAt: 1734652800,
            },
          ],
        }),
      })
    })

    await page.goto('/users')

    let invitePayload: Record<string, unknown> | null = null
    await page.route('**/api/admin/invitations', async (route) => {
      invitePayload = route.request().postDataJSON() as Record<string, unknown>
      await route.fulfill({
        status: 201,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          invitation: {
            id: 'invite-1',
            email: invitePayload.email,
            name: invitePayload.name,
            role: invitePayload.role,
            expiresAt: 1893456000,
          },
          token: 'one-time-token',
        }),
      })
    })

    const inviteUserButton = page.getByRole('button', { name: 'Invite user' })
    const overflowBeforeDialog = await page.evaluate(() => document.body.style.overflow)
    await inviteUserButton.click()

    const dialog = page.getByRole('dialog', { name: 'Invite user' })
    const nameInput = dialog.getByLabel('Name')
    const emailInput = dialog.getByLabel('Email')
    await expect(nameInput).toBeFocused()
    expect(await page.evaluate(() => document.body.style.overflow)).toBe('hidden')
    await expect(dialog.getByRole('radio', { name: /Read/ })).toBeChecked()
    await dialog.getByText('Builder', { exact: true }).click()
    await nameInput.fill('Grace Hopper')
    await emailInput.fill('grace@example.com')

    const closeButton = dialog.getByRole('button', { name: 'Close invitation' })
    const submitButton = dialog.getByRole('button', { name: 'Create invitation' })
    await closeButton.focus()
    await page.keyboard.press('Shift+Tab')
    await expect(submitButton).toBeFocused()
    await page.keyboard.press('Tab')
    await expect(closeButton).toBeFocused()

    await submitButton.click()
    await expect.poll(() => invitePayload).toEqual({
      email: 'grace@example.com',
      name: 'Grace Hopper',
      role: 'write',
    })
    const readyDialog = page.getByRole('dialog', { name: 'Welcome Grace Hopper' })
    await expect(readyDialog).toBeVisible()
    await expect(readyDialog.getByText('One-time invitation URL')).toBeVisible()
    await readyDialog.getByRole('button', { name: 'Done' }).click()
    await expect(dialog).toBeHidden()
    await expect(inviteUserButton).toBeFocused()
    expect(await page.evaluate(() => document.body.style.overflow)).toBe(overflowBeforeDialog)
  })
})
