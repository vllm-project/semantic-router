export interface PermissionUser {
  role?: string
  permissions?: string[]
}

const WRITE_CAPABLE_ROLES = new Set(['admin', 'write'])
const READ_CAPABLE_ROLES = new Set(['admin', 'write', 'read'])
const CONFIG_READ_PERMISSION = 'config.read'
const CONFIG_DEPLOY_PERMISSION = 'config.deploy'
const CONFIG_WRITE_PERMISSION = 'config.write'
const EVALUATION_READ_PERMISSION = 'evaluation.read'
const EVALUATION_RUN_PERMISSION = 'evaluation.run'
const EVALUATION_WRITE_PERMISSION = 'evaluation.write'
const LOGS_READ_PERMISSION = 'logs.read'
const ML_PIPELINE_MANAGE_PERMISSION = 'mlpipeline.manage'
const MCP_READ_PERMISSION = 'mcp.read'
const MCP_MANAGE_PERMISSION = 'mcp.manage'
const OPENCLAW_READ_PERMISSION = 'openclaw.read'
const OPENCLAW_MANAGE_PERMISSION = 'openclaw.manage'
const REPLAY_READ_PERMISSION = 'replay.read'
const TOPOLOGY_READ_PERMISSION = 'topology.read'
const USERS_VIEW_PERMISSION = 'users.view'
const USERS_MANAGE_PERMISSION = 'users.manage'
const ACCESS_READ_PERMISSION = 'access.read'
const ACCESS_MANAGE_PERMISSION = 'access.manage'
const ACCESS_SELF_PERMISSION = 'access.self'
const USAGE_SELF_PERMISSION = 'usage.self'
const STATUS_READ_PERMISSION = 'status.read'

function hasPermission(user: PermissionUser | null | undefined, permission: string): boolean {
  return Array.isArray(user?.permissions) && user.permissions.includes(permission)
}

function canAccessWithPermission(
  user: PermissionUser | null | undefined,
  permission: string,
  fallbackRoles: ReadonlySet<string> = WRITE_CAPABLE_ROLES,
): boolean {
  if (Array.isArray(user?.permissions)) {
    return hasPermission(user, permission)
  }

  if (!user) return false
  const normalizedRole = typeof user.role === 'string' ? user.role.trim().toLowerCase() : ''
  return fallbackRoles.has(normalizedRole)
}

export function canAccessReplayFlowDetails(user?: PermissionUser | null): boolean {
  // replay.read grants structural record access. The dashboard backend only leaves
  // request/response bodies and tool payloads unredacted for config writers.
  return canAccessWithPermission(user, CONFIG_WRITE_PERMISSION)
}

export function canWriteConfig(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, CONFIG_WRITE_PERMISSION)
}

export function canDeployConfig(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, CONFIG_DEPLOY_PERMISSION)
}

export function canVerifyModels(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, STATUS_READ_PERMISSION)
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

export function canManageMCP(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, MCP_MANAGE_PERMISSION)
}

export function canManageOpenClaw(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, OPENCLAW_MANAGE_PERMISSION)
}

export function canAccessDashboardPath(
  user: PermissionUser | null | undefined,
  pathname: string,
): boolean {
  const normalizedPath = pathname.trim().toLowerCase()

  if (isModelConsumer(user)) {
    if (
      normalizedPath.startsWith('/knowledge-bases') ||
      normalizedPath.startsWith('/taxonomy') ||
      normalizedPath.startsWith('/fleet-sim') ||
      normalizedPath.startsWith('/ml-setup') ||
      normalizedPath.startsWith('/status') ||
      normalizedPath.startsWith('/clawos') ||
      normalizedPath.startsWith('/openclaw') ||
      normalizedPath.startsWith('/plugins') ||
      normalizedPath.startsWith('/response-cache') ||
      normalizedPath.startsWith('/context-compression') ||
      normalizedPath.startsWith('/monitoring') ||
      normalizedPath.startsWith('/tracing')
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
    if (canAccessWithPermission(user, ACCESS_READ_PERMISSION, READ_CAPABLE_ROLES)) return true
    if (!hasPermission(user, ACCESS_SELF_PERMISSION)) return false
    if (normalizedPath === '/access') return true
    return ['/access/api-keys', '/access/teams', '/access/usage'].some(
      (path) => normalizedPath === path || normalizedPath.startsWith(`${path}/`),
    )
  }
  if (normalizedPath.startsWith('/users')) return canViewUsers(user)
  if (normalizedPath.startsWith('/ml-setup')) return canAccessMLSetup(user)
  if (normalizedPath.startsWith('/topology')) {
    return canAccessWithPermission(user, TOPOLOGY_READ_PERMISSION, READ_CAPABLE_ROLES)
  }
  if (normalizedPath.startsWith('/status')) {
    return (
      canAccessWithPermission(user, STATUS_READ_PERMISSION, READ_CAPABLE_ROLES) ||
      hasPermission(user, TOPOLOGY_READ_PERMISSION)
    )
  }
  if (
    normalizedPath.startsWith('/plugins') ||
    normalizedPath.startsWith('/response-cache') ||
    normalizedPath.startsWith('/context-compression') ||
    normalizedPath.startsWith('/monitoring') ||
    normalizedPath.startsWith('/tracing')
  ) {
    return canAccessWithPermission(user, LOGS_READ_PERMISSION)
  }
  if (normalizedPath.startsWith('/logs')) {
    return (
      canAccessWithPermission(user, LOGS_READ_PERMISSION) ||
      hasPermission(user, USAGE_SELF_PERMISSION)
    )
  }
  if (normalizedPath.startsWith('/insights')) {
    return canAccessWithPermission(user, REPLAY_READ_PERMISSION, READ_CAPABLE_ROLES)
  }
  if (normalizedPath.startsWith('/evaluation')) {
    return canAccessWithPermission(user, EVALUATION_READ_PERMISSION, READ_CAPABLE_ROLES)
  }
  if (normalizedPath.startsWith('/clawos') || normalizedPath.startsWith('/openclaw')) {
    return canAccessWithPermission(user, OPENCLAW_READ_PERMISSION, READ_CAPABLE_ROLES)
  }
  if (normalizedPath.startsWith('/config/mcp')) {
    return canAccessWithPermission(user, MCP_READ_PERMISSION, READ_CAPABLE_ROLES)
  }
  if (
    normalizedPath.startsWith('/builder') ||
    normalizedPath.startsWith('/config') ||
    normalizedPath.startsWith('/knowledge-bases') ||
    normalizedPath.startsWith('/taxonomy') ||
    normalizedPath.startsWith('/fleet-sim')
  ) {
    return canAccessWithPermission(user, CONFIG_READ_PERMISSION, READ_CAPABLE_ROLES)
  }

  return true
}

export function canViewUsers(user?: PermissionUser | null): boolean {
  if (Array.isArray(user?.permissions)) {
    return (
      hasPermission(user, USERS_VIEW_PERMISSION) || hasPermission(user, USERS_MANAGE_PERMISSION)
    )
  }

  return user?.role?.trim().toLowerCase() === 'admin'
}

export function canManageUsers(user?: PermissionUser | null): boolean {
  if (Array.isArray(user?.permissions)) {
    return hasPermission(user, USERS_MANAGE_PERMISSION)
  }

  return user?.role?.trim().toLowerCase() === 'admin'
}

export function canManageInferenceAccess(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, ACCESS_MANAGE_PERMISSION, new Set(['admin']))
}

export function canReadInferenceAccess(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, ACCESS_READ_PERMISSION, READ_CAPABLE_ROLES)
}

export function canSelfManageInferenceAccess(user?: PermissionUser | null): boolean {
  return canManageInferenceAccess(user) || hasPermission(user, ACCESS_SELF_PERMISSION)
}

export function isModelConsumer(user?: PermissionUser | null): boolean {
  if (!user) return false
  if (Array.isArray(user.permissions)) {
    return (
      hasPermission(user, ACCESS_SELF_PERMISSION) &&
      hasPermission(user, USAGE_SELF_PERMISSION) &&
      !hasPermission(user, CONFIG_WRITE_PERMISSION) &&
      !hasPermission(user, ACCESS_READ_PERMISSION) &&
      !hasPermission(user, LOGS_READ_PERMISSION) &&
      !hasPermission(user, STATUS_READ_PERMISSION)
    )
  }
  return user.role?.trim().toLowerCase() === 'read'
}

export function canViewOwnUsage(user?: PermissionUser | null): boolean {
  return hasPermission(user, USAGE_SELF_PERMISSION)
}
