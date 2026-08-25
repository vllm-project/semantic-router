import { describe, expect, it } from 'vitest'

import { resolveAccessControlPage } from './AccessControlPageSupport'

describe('Access Control page permissions', () => {
  it('loads Dashboard identities for viewers without exposing member actions', () => {
    const result = resolveAccessControlPage(
      {
        permissions: ['users.view'],
        managementPermissions: ['user.read'],
      },
      '/access/users',
    )

    expect(result.canReadDashboardMembers).toBe(true)
    expect(result.canManageDashboardMembers).toBe(false)
    expect(result.canManage).toBe(false)
  })

  it('keeps invitation and login management behind users.manage', () => {
    const result = resolveAccessControlPage(
      {
        permissions: ['users.manage'],
        managementPermissions: ['user.read'],
      },
      '/access/users',
    )

    expect(result.canReadDashboardMembers).toBe(true)
    expect(result.canManageDashboardMembers).toBe(true)
  })
})
