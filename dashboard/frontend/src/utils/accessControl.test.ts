import { describe, expect, it } from 'vitest'

import {
  canAccessMLSetup,
  canAccessDashboardPath,
  canAccessReplayFlowDetails,
  canManageInferenceAccess,
  canDeployConfig,
  canManageAgent,
  canManageAgentTools,
  canManageOpenClaw,
  canManageUsers,
  canRunEvaluation,
  canManageRouting,
  canReadKeyScopedRouting,
  canReadRouting,
  canReadRoutingCatalog,
  canRevealInferenceKey,
  canSelfManageInferenceAccess,
  canViewUsers,
  canWriteConfig,
  canWriteEvaluation,
  isModelConsumer,
  resolveAccessLandingPath,
} from './accessControl'

describe('config write access', () => {
  it('allows only explicit config writers', () => {
    expect(canWriteConfig({ role: 'read', permissions: ['config.write'] })).toBe(true)
    expect(canWriteConfig({ role: 'admin' })).toBe(false)
    expect(canWriteConfig({ role: 'write' })).toBe(false)
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

  it('never infers capabilities from a dashboard role', () => {
    expect(canAccessReplayFlowDetails({ role: 'write' })).toBe(false)
    expect(canAccessReplayFlowDetails({ role: 'read' })).toBe(false)
    expect(canAccessMLSetup({ role: 'admin' })).toBe(false)
    expect(canAccessReplayFlowDetails({ role: 'read', permissions: ['replay.read'] })).toBe(false)
    expect(canAccessReplayFlowDetails({ managementPermissions: ['log_payload.read'] })).toBe(true)
    expect(canAccessMLSetup({ role: 'read', permissions: ['mlpipeline.manage'] })).toBe(true)
  })

  it('maps dashboard routes to their backend read permissions', () => {
    expect(canAccessDashboardPath({ permissions: ['topology.read'] }, '/status')).toBe(true)
    expect(canAccessDashboardPath({ permissions: ['logs.read'] }, '/status')).toBe(false)
    expect(canAccessDashboardPath({ managementPermissions: ['log.read'] }, '/logs')).toBe(true)
    expect(canAccessDashboardPath({ role: 'read' }, '/logs')).toBe(false)
    expect(canAccessDashboardPath({ role: 'write' }, '/logs')).toBe(false)
    expect(
      canAccessDashboardPath({ permissions: ['logs.read'] }, '/plugins/context-compression'),
    ).toBe(true)
    expect(canAccessDashboardPath({ permissions: ['config.read'] }, '/status')).toBe(false)
    expect(
      canAccessDashboardPath(
        { managementPermissions: ['log.read', 'usage.read'] },
        '/insights/record-1',
      ),
    ).toBe(true)
    expect(canAccessDashboardPath({ permissions: ['evaluation.read'] }, '/evaluation')).toBe(true)
    expect(canAccessDashboardPath({ managementPermissions: ['agent.read'] }, '/config/agent')).toBe(
      true,
    )
    expect(canAccessDashboardPath({ managementPermissions: ['tool.read'] }, '/config/agent')).toBe(
      true,
    )
    expect(canAccessDashboardPath({ permissions: ['config.read'] }, '/config/agent')).toBe(false)
    expect(canAccessDashboardPath({ managementPermissions: ['routing.read'] }, '/topology')).toBe(
      true,
    )
    expect(canAccessDashboardPath({ permissions: ['topology.read'] }, '/topology')).toBe(false)
    expect(canAccessDashboardPath({ role: 'read' }, '/topology')).toBe(false)
    expect(canAccessDashboardPath({ role: 'read' }, '/status')).toBe(false)
    expect(
      canAccessDashboardPath({ managementPermissions: ['key.read'] }, '/access/api-keys'),
    ).toBe(true)
    expect(
      canAccessDashboardPath(
        {
          managementPermissions: ['key.read', 'delegation.use'],
          managementUserId: 'router-user-1',
        },
        '/access/api-keys',
      ),
    ).toBe(true)
    expect(canAccessDashboardPath({ permissions: ['config.read'] }, '/access/api-keys')).toBe(false)
    expect(
      canAccessDashboardPath(
        { permissions: ['config.read'], managementPermissions: ['routing.read'] },
        '/config/entrypoints-recipes',
      ),
    ).toBe(true)
    expect(
      canAccessDashboardPath(
        { permissions: ['config.read'], managementPermissions: [] },
        '/config/entrypoints-recipes',
      ),
    ).toBe(false)
    expect(canAccessDashboardPath({ managementPermissions: ['routing.read'] }, '/builder')).toBe(
      true,
    )
    expect(canAccessDashboardPath({ permissions: ['config.read'] }, '/builder')).toBe(false)
    expect(canAccessDashboardPath({ managementPermissions: ['usage.read'] }, '/access/usage')).toBe(
      true,
    )
    expect(
      canAccessDashboardPath({ managementPermissions: ['usage.read'] }, '/access/not-a-view'),
    ).toBe(false)
    expect(canAccessDashboardPath({ managementPermissions: ['usage.read'] }, '/access/users')).toBe(
      false,
    )
  })

  it('keeps model consumers inside routing and their own access surfaces', () => {
    const consumer = {
      role: 'read',
      permissions: ['config.read', 'topology.read'],
      managementPermissions: [
        'agent.read',
        'agent.use',
        'delegation.use',
        'key.read',
        'access_policy.read',
        'routing_context.read',
        'tool.invoke',
        'tool.read',
        'usage.read',
      ],
    }
    expect(isModelConsumer(consumer)).toBe(true)
    expect(canReadRouting(consumer)).toBe(false)
    expect(canReadKeyScopedRouting(consumer)).toBe(true)
    expect(canReadRoutingCatalog(consumer)).toBe(true)
    expect(canAccessDashboardPath(consumer, '/config/entrypoints-recipes')).toBe(true)
    expect(canAccessDashboardPath(consumer, '/topology')).toBe(true)
    expect(canAccessDashboardPath(consumer, '/playground')).toBe(true)
    expect(canAccessDashboardPath(consumer, '/config/models')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/config/signals')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/config/projections')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/config/decisions')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/builder')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/fleet-sim')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/config/global-config')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/config/agent')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/status')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/insights')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/evaluation')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/access/usage')).toBe(true)
    expect(canAccessDashboardPath(consumer, '/access/api-keys')).toBe(true)
    expect(canAccessDashboardPath(consumer, '/access/users')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/access/access-groups')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/access/budgets')).toBe(false)
    expect(canAccessDashboardPath(consumer, '/access/audit-logs')).toBe(false)
  })

  it('classifies consumers from effective capabilities instead of dashboard role names', () => {
    expect(
      isModelConsumer({
        role: 'custom-role-name',
        permissions: ['config.read'],
        managementPermissions: ['delegation.use', 'key.read', 'usage.read', 'routing.read'],
      }),
    ).toBe(true)
    expect(
      isModelConsumer({
        role: 'read',
        permissions: ['config.read'],
        managementPermissions: ['key.read', 'usage.read', 'routing.read'],
      }),
    ).toBe(false)
    expect(
      isModelConsumer({
        role: 'read',
        permissions: ['config.read', 'status.read'],
        managementPermissions: ['delegation.use', 'routing.read'],
      }),
    ).toBe(false)
    expect(
      isModelConsumer({
        role: 'read',
        permissions: ['config.read'],
        managementPermissions: ['delegation.use', 'key.manage', 'routing.read'],
      }),
    ).toBe(true)
    expect(
      isModelConsumer({
        role: 'read',
        permissions: ['config.read'],
        managementPermissions: ['delegation.use', 'membership.manage', 'routing.read'],
      }),
    ).toBe(true)
  })

  it('keeps scoped key and membership managers out of global workspace surfaces', () => {
    for (const scopedPermission of ['key.manage', 'membership.manage']) {
      const scopedManager = {
        role: 'read',
        permissions: ['config.read'],
        managementPermissions: ['delegation.use', 'key.read', scopedPermission],
      }

      expect(isModelConsumer(scopedManager)).toBe(true)
      expect(canAccessDashboardPath(scopedManager, '/config/agent')).toBe(false)
      expect(canAccessDashboardPath(scopedManager, '/status')).toBe(false)
    }
  })

  it('lands Access on the first surface granted to the current identity', () => {
    expect(resolveAccessLandingPath({ managementPermissions: ['usage.read'] })).toBe(
      '/access/usage',
    )
    expect(resolveAccessLandingPath({ managementPermissions: ['key.read', 'key.reveal'] })).toBe(
      '/access/api-keys',
    )
    expect(resolveAccessLandingPath({ managementPermissions: ['team.read'] })).toBe(
      '/access/teams',
    )
    expect(resolveAccessLandingPath({ managementPermissions: [] })).toBeNull()
  })

  it('keeps read-only Evaluation visible only with evaluation.read', () => {
    const reader = {
      role: 'read',
      permissions: ['config.read', 'topology.read', 'replay.read', 'evaluation.read'],
      managementPermissions: ['key.read', 'usage.read', 'log.read', 'delegation.use'],
    }
    expect(isModelConsumer(reader)).toBe(true)
    expect(canAccessDashboardPath(reader, '/insights')).toBe(true)
    expect(canAccessDashboardPath(reader, '/evaluation')).toBe(true)
    expect(
      canAccessDashboardPath(
        {
          role: 'read',
          permissions: ['config.read'],
          managementPermissions: ['delegation.use'],
        },
        '/evaluation',
      ),
    ).toBe(false)
  })

  it('separates read, write, run, and manage actions', () => {
    expect(canDeployConfig({ permissions: ['config.deploy'] })).toBe(true)
    expect(canDeployConfig({ permissions: ['config.write'] })).toBe(false)
    expect(canWriteEvaluation({ permissions: ['evaluation.write'] })).toBe(true)
    expect(canWriteEvaluation({ permissions: ['evaluation.read'] })).toBe(false)
    expect(canRunEvaluation({ permissions: ['evaluation.run'] })).toBe(true)
    expect(canRunEvaluation({ permissions: ['evaluation.write'] })).toBe(false)
    expect(canManageAgent({ managementPermissions: ['agent.read', 'agent.manage'] })).toBe(true)
    expect(canManageAgent({ managementPermissions: ['agent.read'] })).toBe(false)
    expect(canManageAgentTools({ managementPermissions: ['tool.read', 'tool.manage'] })).toBe(true)
    expect(canManageAgentTools({ managementPermissions: ['tool.read'] })).toBe(false)
    expect(canManageOpenClaw({ permissions: ['openclaw.manage'] })).toBe(true)
    expect(canAccessDashboardPath({ permissions: ['openclaw.read'] }, '/openclaw')).toBe(true)
    expect(canManageInferenceAccess({ managementPermissions: ['key.manage'] })).toBe(true)
    expect(canManageInferenceAccess({ managementPermissions: ['key.read'] })).toBe(false)
    expect(canSelfManageInferenceAccess({ managementPermissions: ['key.manage'] })).toBe(true)
    expect(
      canSelfManageInferenceAccess({ managementPermissions: ['key.read', 'delegation.use'] }),
    ).toBe(false)
    expect(canRevealInferenceKey({ managementPermissions: ['key.reveal'] })).toBe(true)
    expect(canRevealInferenceKey({ managementPermissions: ['key.manage'] })).toBe(false)
    expect(canReadRouting({ role: 'read', managementPermissions: ['routing.read'] })).toBe(true)
    expect(canManageRouting({ role: 'admin', managementPermissions: ['routing.read'] })).toBe(false)
    expect(
      canManageRouting({ role: 'read', managementPermissions: ['routing.read', 'routing.manage'] }),
    ).toBe(true)
    expect(canReadRouting({ role: 'admin', permissions: ['config.write'] })).toBe(false)
    expect(
      canReadKeyScopedRouting({
        managementPermissions: [
          'delegation.use',
          'key.read',
          'access_policy.read',
          'routing_context.read',
        ],
      }),
    ).toBe(true)
    expect(
      canReadKeyScopedRouting({
        managementPermissions: ['delegation.use', 'key.read', 'access_policy.read'],
      }),
    ).toBe(false)
  })

  it('authorizes the complete routing workspace only from Router Management capabilities', () => {
    const routingAdmin = {
      role: 'admin',
      permissions: [],
      managementPermissions: ['routing.read', 'routing.manage'],
    }
    const sameCapabilitiesWithoutAdminRole = {
      role: 'custom-dashboard-role',
      permissions: [],
      managementPermissions: ['routing.read', 'routing.manage'],
    }

    for (const user of [routingAdmin, sameCapabilitiesWithoutAdminRole]) {
      expect(canAccessDashboardPath(user, '/topology')).toBe(true)
      expect(canAccessDashboardPath(user, '/config/models')).toBe(true)
      expect(canAccessDashboardPath(user, '/config/entrypoints-recipes')).toBe(true)
      expect(canManageRouting(user)).toBe(true)
    }

    expect(
      canManageRouting({ role: 'admin', permissions: ['config.write'], managementPermissions: [] }),
    ).toBe(false)
  })

  it('fails routing closed when identity projection is unavailable', () => {
    const disconnectedAdmin = {
      role: 'admin',
      permissions: ['config.read', 'config.write'],
      managementPermissions: [],
      managementIdentityStatus: 'error' as const,
      managementIdentityError: 'Principal link is unavailable.',
    }
    expect(canAccessDashboardPath(disconnectedAdmin, '/config/entrypoints-recipes')).toBe(false)
    expect(canReadRouting(disconnectedAdmin)).toBe(false)
    expect(canManageRouting(disconnectedAdmin)).toBe(false)
  })

  it('uses effective user permissions for user-management surfaces', () => {
    expect(canViewUsers({ role: 'read', permissions: ['users.view'] })).toBe(true)
    expect(canViewUsers({ role: 'read', permissions: ['users.manage'] })).toBe(true)
    expect(canManageUsers({ role: 'read', permissions: ['users.manage'] })).toBe(true)
    expect(canManageUsers({ role: 'read', permissions: ['users.view'] })).toBe(false)
    expect(canViewUsers({ role: 'admin', permissions: [] })).toBe(false)
    expect(canManageUsers({ role: 'admin', permissions: [] })).toBe(false)
    expect(canViewUsers({ role: 'admin' })).toBe(false)
    expect(canManageUsers({ role: 'admin' })).toBe(false)
  })
})
