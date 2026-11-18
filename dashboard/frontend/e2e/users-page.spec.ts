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

  test('sorts by column header and filters by role on the server, without a refresh button', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockAuthenticatedAppShell(page, { user: adminUser })
    const queries: URLSearchParams[] = []
    await page.route('**/api/admin/users?**', (route) => {
      queries.push(new URL(route.request().url()).searchParams)
      return route.fulfill({
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
          total: 1,
        },
      })
    })
    await page.route('**/api/admin/permissions', (route) =>
      route.fulfill({ json: { rolePermissions: { admin: [], write: [], read: [] } } }),
    )
    await page.route('**/api/admin/invitations', (route) =>
      route.fulfill({ json: { invitations: [] } }),
    )

    await page.goto('/settings/access')
    await expect(page.getByRole('cell', { name: /Read User/ })).toBeVisible()
    await expect(page.getByRole('button', { name: /refresh/i })).toHaveCount(0)
    await expect(page.getByRole('combobox', { name: 'Sort by' })).toHaveCount(0)
    const latest = () => Object.fromEntries(queries.at(-1)?.entries() ?? [])
    expect(latest()).toMatchObject({ sort: 'createdAt', order: 'desc' })

    const person = page.getByRole('columnheader', { name: 'Person' })
    await person.getByRole('button').click()
    await expect.poll(latest).toMatchObject({ sort: 'name', order: 'desc', page: '1' })
    await expect(person).toHaveAttribute('aria-sort', 'descending')
    await person.getByRole('button').click()
    await expect.poll(latest).toMatchObject({ sort: 'name', order: 'asc' })
    await expect(person).toHaveAttribute('aria-sort', 'ascending')
    await page.getByRole('columnheader', { name: 'Last sign-in' }).getByRole('button').click()
    await expect.poll(latest).toMatchObject({ sort: 'lastLoginAt', order: 'desc' })

    await page
      .getByRole('button', { name: 'Role', exact: true })
      .and(page.locator('[aria-haspopup="dialog"]'))
      .click()
    await page.getByRole('menuitemcheckbox', { name: /Admin/ }).click()
    await expect.poll(() => latest().role).toBe('admin')
    await page.getByRole('menuitemcheckbox', { name: /Builder/ }).click()
    await expect.poll(() => latest().role).toBe('admin,write')
  })

  test('filters the audit log with chips and exports what matches', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockAuthenticatedAppShell(page, { user: adminUser })
    await page.route('**/api/admin/users?**', (route) =>
      route.fulfill({
        json: {
          users: [{ id: 'user-1', email: 'grace@example.com', name: 'Grace Hopper', role: 'read' }],
          total: 1,
        },
      }),
    )
    await page.route('**/api/admin/permissions', (route) =>
      route.fulfill({ json: { rolePermissions: { admin: [], write: [], read: [] } } }),
    )
    await page.route('**/api/admin/invitations', (route) =>
      route.fulfill({ json: { invitations: [] } }),
    )
    const queries: URLSearchParams[] = []
    await page.route('**/api/admin/audit-logs?**', (route) => {
      const query = new URL(route.request().url()).searchParams
      queries.push(query)
      return route.fulfill({
        json: {
          logs: [
            {
              id: 7,
              userId: 'user-1',
              action: 'user.update',
              resource: 'users/user-2',
              method: 'PATCH',
              path: '/api/admin/users/user-2',
              ip: '192.0.2.1',
              userAgent: 'console',
              statusCode: 204,
              createdAt: 1_893_456_000,
            },
          ],
          total: 1,
          page: 1,
          limit: Number(query.get('limit') ?? 20),
          ...(query.get('facets') === '1'
            ? {
                facets: {
                  users: [{ value: 'user-1', count: 12 }],
                  actions: [{ value: 'user.update', count: 4 }],
                },
              }
            : {}),
        },
      })
    })

    await page.goto('/settings/access?tab=audit')
    await expect(page.getByText('Grace Hopper').first()).toBeVisible()
    await expect(page.getByRole('button', { name: /refresh/i })).toHaveCount(0)
    expect(queries[0]?.get('facets')).toBe('1')
    const latest = () => Object.fromEntries(queries.at(-1)?.entries() ?? [])

    await page.getByRole('button', { name: 'Person' }).click()
    await page.getByRole('menuitemcheckbox', { name: /Grace Hopper/ }).click()
    await expect.poll(() => latest().user).toBe('user-1')
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: 'Result' }).click()
    await page.getByRole('menuitemcheckbox', { name: /Succeeded/ }).click()
    await page.getByRole('menuitemcheckbox', { name: /Server error/ }).click()
    await expect.poll(() => latest().status).toBe('success,server_error')
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: 'Time' }).click()
    await page.getByRole('menuitemradio', { name: 'Last 7 days' }).click()
    await expect.poll(() => Number(latest().from)).toBeGreaterThan(0)
    await page.keyboard.press('Escape')

    const download = page.waitForEvent('download')
    await page.getByRole('button', { name: 'Export CSV' }).click()
    expect((await download).suggestedFilename()).toMatch(/^audit-log-\d{4}-\d{2}-\d{2}\.csv$/)
    expect(latest()).toMatchObject({ user: 'user-1', status: 'success,server_error', limit: '200' })
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
