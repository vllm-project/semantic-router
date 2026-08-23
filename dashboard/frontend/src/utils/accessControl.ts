export interface PermissionUser {
  role?: string
  permissions?: string[]
  managementPermissions?: string[]
  managementUserId?: string
  managementIdentityStatus?: 'ready' | 'error'
  managementIdentityError?: string
}

const CONFIG_READ_PERMISSION = 'config.read'
const CONFIG_DEPLOY_PERMISSION = 'config.deploy'
const CONFIG_WRITE_PERMISSION = 'config.write'
const EVALUATION_READ_PERMISSION = 'evaluation.read'
const EVALUATION_RUN_PERMISSION = 'evaluation.run'
const EVALUATION_WRITE_PERMISSION = 'evaluation.write'
const LOGS_READ_PERMISSION = 'logs.read'
const ML_PIPELINE_MANAGE_PERMISSION = 'mlpipeline.manage'
const OPENCLAW_READ_PERMISSION = 'openclaw.read'
const OPENCLAW_MANAGE_PERMISSION = 'openclaw.manage'
const TOPOLOGY_READ_PERMISSION = 'topology.read'
const USERS_VIEW_PERMISSION = 'users.view'
const USERS_MANAGE_PERMISSION = 'users.manage'
const STATUS_READ_PERMISSION = 'status.read'

const managementHas = (user: PermissionUser | null | undefined, permission: string) =>
  Array.isArray(user?.managementPermissions) && user.managementPermissions.includes(permission)

const managementHasAny = (user: PermissionUser | null | undefined, ...permissions: string[]) =>
  permissions.some((permission) => managementHas(user, permission))

export function canReadRouting(user?: PermissionUser | null): boolean {
  return managementHas(user, 'routing.read')
}

export function canManageRouting(user?: PermissionUser | null): boolean {
  return canReadRouting(user) && managementHas(user, 'routing.manage')
}

function isRoutingWorkspacePath(pathname: string): boolean {
  return [
    '/builder',
    '/config/models',
    '/config/signals',
    '/config/projections',
    '/config/decisions',
    '/config/entrypoints-recipes',
    '/config/entrypoints',
    '/config/recipes',
  ].some((path) => pathname === path || pathname.startsWith(`${path}/`))
}

function hasPermission(user: PermissionUser | null | undefined, permission: string): boolean {
  return Array.isArray(user?.permissions) && user.permissions.includes(permission)
}

function canAccessWithPermission(
  user: PermissionUser | null | undefined,
  permission: string,
): boolean {
  return hasPermission(user, permission)
}

export function canAccessReplayFlowDetails(user?: PermissionUser | null): boolean {
  return managementHas(user, 'log_payload.read')
}

export function canWriteConfig(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, CONFIG_WRITE_PERMISSION)
}

export function canDeployConfig(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, CONFIG_DEPLOY_PERMISSION)
}

export function canAccessMLSetup(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, ML_PIPELINE_MANAGE_PERMISSION)
}

export function canWriteEvaluation(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, EVALUATION_WRITE_PERMISSION)
}

export function canRunEvaluation(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, EVALUATION_RUN_PERMISSION)
}

export function canReadAgent(user?: PermissionUser | null): boolean {
  return managementHas(user, 'agent.read')
}

export function canUseAgent(user?: PermissionUser | null): boolean {
  return canReadAgent(user) && managementHas(user, 'agent.use')
}

export function canManageAgent(user?: PermissionUser | null): boolean {
  return canReadAgent(user) && managementHas(user, 'agent.manage')
}

export function canReadAgentTools(user?: PermissionUser | null): boolean {
  return managementHas(user, 'tool.read')
}

export function canInvokeAgentTools(user?: PermissionUser | null): boolean {
  return canReadAgentTools(user) && managementHas(user, 'tool.invoke')
}

export function canManageAgentTools(user?: PermissionUser | null): boolean {
  return canReadAgentTools(user) && managementHas(user, 'tool.manage')
}

export function canUseBuilderAgent(user?: PermissionUser | null): boolean {
  return canUseAgent(user) && canManageRouting(user)
}

export function canPublishRouting(user?: PermissionUser | null): boolean {
  return canReadRouting(user) && managementHas(user, 'routing.publish')
}

export function canManageOpenClaw(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, OPENCLAW_MANAGE_PERMISSION)
}

export function canAccessDashboardPath(
  user: PermissionUser | null | undefined,
  pathname: string,
): boolean {
  const normalizedPath = pathname.trim().toLowerCase()

  if (normalizedPath.startsWith('/playground')) {
    return canUseAgent(user)
  }

  if (normalizedPath.startsWith('/config/agent')) {
    return canReadAgent(user) || canReadAgentTools(user)
  }

  if (isRoutingWorkspacePath(normalizedPath)) {
    return canReadRouting(user)
  }

  if (isModelConsumer(user)) {
    if (
      normalizedPath.startsWith('/fleet-sim') ||
      normalizedPath.startsWith('/ml-setup') ||
      normalizedPath.startsWith('/status') ||
      normalizedPath.startsWith('/openclaw') ||
      normalizedPath.startsWith('/plugins') ||
      normalizedPath.startsWith('/monitoring') ||
      normalizedPath.startsWith('/tracing')
    ) {
      return false
    }

    if (
      normalizedPath.startsWith('/access/users') ||
      normalizedPath.startsWith('/access/access-groups') ||
      normalizedPath.startsWith('/access/budgets') ||
      normalizedPath.startsWith('/access/audit-logs')
    ) {
      return false
    }
    if (normalizedPath.startsWith('/config/')) {
      return [
        '/config/models',
        '/config/signals',
        '/config/projections',
        '/config/decisions',
        '/config/entrypoints-recipes',
      ].some((path) => normalizedPath === path || normalizedPath.startsWith(`${path}/`))
    }
  }

  if (normalizedPath.startsWith('/access')) {
    if (normalizedPath === '/access') {
      return canReadInferenceAccess(user) || canSelfManageInferenceAccess(user)
    }
    const required: Array<[string, string]> = [
      ['/access/api-keys', 'key.read'],
      ['/access/users', 'user.read'],
      ['/access/teams', 'team.read'],
      ['/access/access-groups', 'access_policy.read'],
      ['/access/budgets', 'rate_policy.read'],
      ['/access/usage', 'usage.read'],
      ['/access/audit-logs', 'audit.read'],
    ]
    const match = required.find(
      ([path]) => normalizedPath === path || normalizedPath.startsWith(`${path}/`),
    )
    if (!match) return false
    return managementHas(user, match[1])
  }
  if (normalizedPath.startsWith('/ml-setup')) return canAccessMLSetup(user)
  if (normalizedPath.startsWith('/topology')) {
    return canReadRouting(user)
  }
  if (normalizedPath.startsWith('/status')) {
    return (
      canAccessWithPermission(user, STATUS_READ_PERMISSION) ||
      hasPermission(user, TOPOLOGY_READ_PERMISSION)
    )
  }
  if (
    normalizedPath.startsWith('/plugins') ||
    normalizedPath.startsWith('/monitoring') ||
    normalizedPath.startsWith('/tracing')
  ) {
    return canAccessWithPermission(user, LOGS_READ_PERMISSION)
  }
  if (normalizedPath.startsWith('/logs')) {
    return managementHas(user, 'log.read')
  }
  if (normalizedPath.startsWith('/insights')) {
    return managementHas(user, 'log.read') && managementHas(user, 'usage.read')
  }
  if (normalizedPath.startsWith('/evaluation')) {
    return canAccessWithPermission(user, EVALUATION_READ_PERMISSION)
  }
  if (normalizedPath.startsWith('/openclaw')) {
    return canAccessWithPermission(user, OPENCLAW_READ_PERMISSION)
  }
  if (normalizedPath.startsWith('/config') || normalizedPath.startsWith('/fleet-sim')) {
    return canAccessWithPermission(user, CONFIG_READ_PERMISSION)
  }

  return true
}

export function canViewUsers(user?: PermissionUser | null): boolean {
  return hasPermission(user, USERS_VIEW_PERMISSION) || hasPermission(user, USERS_MANAGE_PERMISSION)
}

export function canManageUsers(user?: PermissionUser | null): boolean {
  return hasPermission(user, USERS_MANAGE_PERMISSION)
}

export function canManageInferenceAccess(user?: PermissionUser | null): boolean {
  return managementHasAny(
    user,
    'key.manage',
    'user.manage',
    'team.manage',
    'access_policy.manage',
    'rate_policy.manage',
  )
}

export function canReadInferenceAccess(user?: PermissionUser | null): boolean {
  return managementHasAny(
    user,
    'key.read',
    'user.read',
    'team.read',
    'access_policy.read',
    'rate_policy.read',
    'usage.read',
    'log.read',
    'audit.read',
  )
}

export function canSelfManageInferenceAccess(user?: PermissionUser | null): boolean {
  return managementHas(user, 'key.manage')
}

export function isModelConsumer(user?: PermissionUser | null): boolean {
  if (!user || !managementHas(user, 'delegation.use')) return false
  return (
    !hasPermission(user, CONFIG_WRITE_PERMISSION) &&
    !hasPermission(user, LOGS_READ_PERMISSION) &&
    !hasPermission(user, STATUS_READ_PERMISSION) &&
    !managementHasAny(
      user,
      'key.manage',
      'user.manage',
      'team.manage',
      'membership.manage',
      'access_policy.manage',
      'rate_policy.manage',
      'routing.manage',
    )
  )
}

export function canViewOwnUsage(user?: PermissionUser | null): boolean {
  return managementHas(user, 'usage.read')
}
