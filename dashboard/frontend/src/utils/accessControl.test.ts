import { describe, expect, it } from 'vitest'

import {
  canAccessMLSetup,
  canAccessDashboardPath,
  canAccessReplayFlowDetails,
  canManageInferenceAccess,
  canDeployConfig,
  canManageMCP,
  canManageOpenClaw,
  canManageUsers,
  canRunEvaluation,
  canViewUsers,
  canWriteConfig,
  canWriteEvaluation,
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
    expect(canAccessReplayFlowDetails({ role: 'read' })).toBe(false)
    expect(canAccessMLSetup({ role: 'admin' })).toBe(true)
    expect(canAccessReplayFlowDetails({ role: 'read', permissions: ['replay.read'] })).toBe(false)
    expect(
      canAccessReplayFlowDetails({
        role: 'write',
        permissions: ['config.write', 'replay.read'],
      }),
    ).toBe(true)
    expect(canAccessMLSetup({ role: 'read', permissions: ['mlpipeline.manage'] })).toBe(true)
  })

  it('maps dashboard routes to their backend read permissions', () => {
    expect(canAccessDashboardPath({ permissions: ['topology.read'] }, '/status')).toBe(true)
    expect(canAccessDashboardPath({ permissions: ['logs.read'] }, '/status')).toBe(false)
    expect(canAccessDashboardPath({ permissions: ['logs.read'] }, '/logs')).toBe(true)
    expect(canAccessDashboardPath({ role: 'read' }, '/logs')).toBe(false)
    expect(canAccessDashboardPath({ role: 'write' }, '/logs')).toBe(true)
    expect(canAccessDashboardPath({ permissions: ['logs.read'] }, '/plugins/response-cache')).toBe(
      true,
    )
    expect(canAccessDashboardPath({ permissions: ['config.read'] }, '/status')).toBe(false)
    expect(canAccessDashboardPath({ permissions: ['replay.read'] }, '/insights/record-1')).toBe(
      true,
    )
    expect(canAccessDashboardPath({ permissions: ['evaluation.read'] }, '/evaluation')).toBe(true)
    expect(canAccessDashboardPath({ permissions: ['mcp.read'] }, '/config/mcp')).toBe(true)
    expect(canAccessDashboardPath({ permissions: ['config.read'] }, '/config/mcp')).toBe(false)
    expect(canAccessDashboardPath({ role: 'read' }, '/topology')).toBe(true)
    expect(canAccessDashboardPath({ role: 'read' }, '/status')).toBe(true)
    expect(canAccessDashboardPath({ permissions: ['access.read'] }, '/access/api-keys')).toBe(true)
    expect(canAccessDashboardPath({ permissions: ['config.read'] }, '/access/api-keys')).toBe(false)
  })

  it('separates read, write, run, and manage actions', () => {
    expect(canDeployConfig({ permissions: ['config.deploy'] })).toBe(true)
    expect(canDeployConfig({ permissions: ['config.write'] })).toBe(false)
    expect(canWriteEvaluation({ permissions: ['evaluation.write'] })).toBe(true)
    expect(canWriteEvaluation({ permissions: ['evaluation.read'] })).toBe(false)
    expect(canRunEvaluation({ permissions: ['evaluation.run'] })).toBe(true)
    expect(canRunEvaluation({ permissions: ['evaluation.write'] })).toBe(false)
    expect(canManageMCP({ permissions: ['mcp.manage'] })).toBe(true)
    expect(canManageMCP({ permissions: ['mcp.read'] })).toBe(false)
    expect(canManageOpenClaw({ permissions: ['openclaw.manage'] })).toBe(true)
    expect(canManageInferenceAccess({ permissions: ['access.manage'] })).toBe(true)
    expect(canManageInferenceAccess({ permissions: ['access.read'] })).toBe(false)
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
