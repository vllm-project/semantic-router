import { describe, expect, it } from 'vitest'

import {
  canAccessMLSetup,
  canAccessReplayFlowDetails,
  canManageUsers,
  canViewUsers,
  canWriteConfig,
} from './accessControl'

describe('config write access', () => {
  it('allows explicit config writers and write-capable roles', () => {
    expect(canWriteConfig({ role: 'read', permissions: ['config.write'] })).toBe(true)
    expect(canWriteConfig({ role: 'admin' })).toBe(true)
    expect(canWriteConfig({ role: 'write' })).toBe(true)
  })

  it('keeps read-only users out of mutating dashboard controls', () => {
    expect(canWriteConfig({ role: 'read', permissions: ['config.read'] })).toBe(false)
    expect(canWriteConfig({ role: 'write', permissions: ['config.read'] })).toBe(false)
    expect(canWriteConfig({ role: 'admin', permissions: [] })).toBe(false)
    expect(canWriteConfig(null)).toBe(false)
  })

  it('treats an explicit permissions list as authoritative on every protected surface', () => {
    const explicitReader = { role: 'admin', permissions: ['config.read'] }
    const emptyAdmin = { role: 'admin', permissions: [] }

    expect(canAccessReplayFlowDetails(explicitReader)).toBe(false)
    expect(canAccessMLSetup(explicitReader)).toBe(false)
    expect(canAccessReplayFlowDetails(emptyAdmin)).toBe(false)
    expect(canAccessMLSetup(emptyAdmin)).toBe(false)
  })

  it('uses legacy role fallback only when permissions are absent', () => {
    expect(canAccessReplayFlowDetails({ role: 'write' })).toBe(true)
    expect(canAccessMLSetup({ role: 'admin' })).toBe(true)
    expect(canAccessReplayFlowDetails({ role: 'read', permissions: ['config.write'] })).toBe(true)
    expect(canAccessMLSetup({ role: 'read', permissions: ['mlpipeline.manage'] })).toBe(true)
  })

  it('uses effective user permissions for user-management surfaces', () => {
    expect(canViewUsers({ role: 'read', permissions: ['users.view'] })).toBe(true)
    expect(canViewUsers({ role: 'read', permissions: ['users.manage'] })).toBe(true)
    expect(canManageUsers({ role: 'read', permissions: ['users.manage'] })).toBe(true)
    expect(canManageUsers({ role: 'read', permissions: ['users.view'] })).toBe(false)
    expect(canViewUsers({ role: 'admin', permissions: [] })).toBe(false)
    expect(canManageUsers({ role: 'admin', permissions: [] })).toBe(false)
    expect(canViewUsers({ role: 'admin' })).toBe(true)
    expect(canManageUsers({ role: 'admin' })).toBe(true)
  })
})
