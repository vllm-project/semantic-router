import { expect, test } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const rolePermissions = {
  admin: [
    'config.deploy',
    'config.read',
    'config.write',
    'evaluation.read',
    'evaluation.run',
    'evaluation.write',
    'logs.read',
    'mcp.manage',
    'mcp.read',
    'mlpipeline.manage',
    'openclaw.manage',
    'openclaw.read',
    'tools.use',
    'topology.read',
    'users.manage',
    'users.view',
  ],
  write: [
    'config.deploy',
    'config.read',
    'config.write',
    'evaluation.read',
    'evaluation.run',
    'evaluation.write',
    'logs.read',
    'mcp.manage',
    'mcp.read',
    'mlpipeline.manage',
    'openclaw.manage',
    'openclaw.read',
    'tools.use',
    'topology.read',
  ],
  read: [
    'config.read',
    'evaluation.read',
    'logs.read',
    'mcp.read',
    'openclaw.read',
    'tools.use',
    'topology.read',
  ],
}

test.describe('Users page', () => {
  test('shows selected role permissions in the create user dialog', async ({ page }) => {
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

    await page.route('**/api/admin/permissions', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rolePermissions,
          allPermissions: Array.from(new Set(Object.values(rolePermissions).flat())),
        }),
      })
    })

    await page.goto('/users')

    await page.getByRole('button', { name: 'Create user' }).click()

    const dialog = page.getByRole('dialog', { name: 'Create user' })
    await expect(dialog.getByText('Permissions', { exact: true })).toBeVisible()
    await expect(dialog.getByText('config.read')).toBeVisible()
    await expect(dialog.getByText('users.manage')).toHaveCount(0)

    await dialog.getByLabel('Role').selectOption('admin')
    await expect(dialog.getByText('users.manage')).toBeVisible()
    await expect(dialog.getByText('users.view')).toBeVisible()

    await dialog.getByLabel('Role').selectOption('write')
    await expect(dialog.getByText('config.deploy')).toBeVisible()
    await expect(dialog.getByText('users.manage')).toHaveCount(0)
  })
})
