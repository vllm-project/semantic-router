import { expect, test } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const adminUser = {
  id: 'user-admin-2',
  email: 'ada@example.com',
  name: 'Ada Lovelace',
  role: 'admin',
  permissions: ['users.manage', 'users.view', 'config.read', 'config.write'],
}

test.describe('Users and access', () => {
  test('creates a personal invitation with a clear dashboard role and a copyable link', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockAuthenticatedAppShell(page, { user: adminUser })

    await page.route('**/api/admin/users**', async (route) => {
      await route.fulfill({
        json: {
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
        },
      })
    })
    await page.route('**/api/admin/permissions', (route) =>
      route.fulfill({
        json: {
          rolePermissions: {
            admin: ['users.manage'],
            write: ['config.write'],
            read: ['config.read'],
          },
        },
      }),
    )

    let invitePayload: Record<string, unknown> | null = null
    await page.route('**/api/admin/invitations', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ json: { invitations: [] } })
        return
      }
      invitePayload = route.request().postDataJSON() as Record<string, unknown>
      await route.fulfill({
        status: 201,
        json: {
          invitation: {
            id: 'invite-1',
            kind: invitePayload.kind,
            email: invitePayload.email,
            name: invitePayload.name,
            role: invitePayload.role,
            maxUses: 1,
            usedCount: 0,
            remainingUses: 1,
            status: 'pending',
            expiresAt: 1893456000,
            createdAt: 1893000000,
          },
          token: 'one-time-token',
        },
      })
    })

    await page.goto('/users')
    await expect(page).toHaveURL(/\/settings\/access$/)
    await expect(page.getByRole('heading', { name: 'Users & access' })).toBeVisible()
    await expect(page.getByRole('cell', { name: /Read User/ })).toBeVisible()

    const inviteUserButton = page.getByRole('button', { name: 'Invite user' })
    await inviteUserButton.click()

    const dialog = page.getByRole('dialog', { name: 'Invite user' })
    const nameInput = dialog.getByLabel('Name')
    const emailInput = dialog.getByLabel('Email')
    await expect(nameInput).toBeFocused()
    await expect(dialog.getByRole('radio', { name: /^Read/ })).toHaveAttribute(
      'aria-checked',
      'true',
    )
    await dialog.getByRole('radio', { name: /^Builder/ }).click()
    await dialog.getByRole('button', { name: 'Create invitation' }).click()
    await expect(dialog.getByText('Enter their name.')).toBeVisible()
    expect(invitePayload).toBeNull()

    await nameInput.fill('Grace Hopper')
    await emailInput.fill('grace@example.com')
    await dialog.getByRole('button', { name: 'Create invitation' }).click()
    await expect
      .poll(() => invitePayload)
      .toEqual({
        kind: 'personal',
        email: 'grace@example.com',
        name: 'Grace Hopper',
        role: 'write',
      })
    const readyDialog = page.getByRole('dialog', { name: 'Welcome Grace Hopper' })
    await expect(readyDialog.getByText('One-time invitation link')).toBeVisible()
    await expect(readyDialog.getByText('/invite/one-time-token')).toBeVisible()
    await readyDialog.getByRole('button', { name: 'Done' }).click()
    await expect(readyDialog).toBeHidden()
    await expect(inviteUserButton).toBeFocused()
  })

  test('changes a role from the member sheet and keeps the change in the URL', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockAuthenticatedAppShell(page, { user: adminUser })
    let reader = {
      id: 'user-1',
      email: 'reader@example.com',
      name: 'Read User',
      role: 'read',
      status: 'active',
      createdAt: 1734652800,
    }
    const patches: unknown[] = []
    await page.route('**/api/admin/users?**', (route) =>
      route.fulfill({ json: { users: [reader], total: 1 } }),
    )
    await page.route('**/api/admin/users/user-1', async (route) => {
      const body = route.request().postDataJSON() as { role: string; status: string }
      patches.push(body)
      reader = { ...reader, ...body }
      await route.fulfill({ json: reader })
    })
    await page.route('**/api/admin/permissions', (route) =>
      route.fulfill({
        json: {
          rolePermissions: {
            admin: ['users.manage'],
            write: ['config.write'],
            read: ['config.read'],
          },
        },
      }),
    )
    await page.route('**/api/admin/invitations', (route) =>
      route.fulfill({ json: { invitations: [] } }),
    )

    const passwords: unknown[] = []
    await page.route('**/api/admin/users/password', async (route) => {
      passwords.push(route.request().postDataJSON())
      await route.fulfill({ json: { ok: true } })
    })

    await page.goto('/settings/access')
    await page.getByRole('cell', { name: /Read User/ }).click()
    await expect(page).toHaveURL(/user=user-1/)
    const sheet = page.getByRole('dialog', { name: /Read User/ })
    await sheet.getByRole('radio', { name: 'Builder' }).click()
    await sheet.getByRole('button', { name: 'Save changes' }).click()
    await expect.poll(() => patches).toEqual([{ role: 'write', status: 'active' }])
    expect(passwords).toEqual([])

    await sheet.getByRole('switch', { name: 'Can sign in' }).click()
    await sheet.getByLabel('New password').fill('rotated-password')
    await sheet.getByRole('button', { name: 'Save changes' }).click()
    await expect
      .poll(() => patches)
      .toEqual([
        { role: 'write', status: 'active' },
        { role: 'write', status: 'inactive' },
      ])
    await expect.poll(() => passwords).toEqual([{ userId: 'user-1', password: 'rotated-password' }])
    await expect(page.getByText('User updated and password rotated')).toBeVisible()
  })
})
