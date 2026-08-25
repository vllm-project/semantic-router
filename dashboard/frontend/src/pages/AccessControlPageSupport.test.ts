import { readFileSync } from 'node:fs'
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

  it('separates key reveal from lifecycle and policy management', () => {
    const revealer = resolveAccessControlPage(
      { managementPermissions: ['key.read', 'key.reveal'] },
      '/access/api-keys',
    )
    const manager = resolveAccessControlPage(
      { managementPermissions: ['key.read', 'key.manage'] },
      '/access/api-keys',
    )

    expect(revealer.canRevealKeys).toBe(true)
    expect(revealer.canManage).toBe(false)
    expect(manager.canRevealKeys).toBe(false)
    expect(manager.canManage).toBe(true)
  })

  it('threads reveal, lifecycle, and policy controls through distinct UI capabilities', () => {
    const detail = readFileSync(new URL('./APIKeyDetail.tsx', import.meta.url), 'utf8')
    const overlays = readFileSync(
      new URL('./AccessControlDetailOverlays.tsx', import.meta.url),
      'utf8',
    )

    expect(detail).toContain('{canReveal ? (')
    expect(detail).toContain('{key && effectiveCanManage ? (')
    expect(overlays).toContain('canReveal={canRevealKeys}')
    expect(overlays).toContain('canEditPolicy={canManage}')
  })
})
