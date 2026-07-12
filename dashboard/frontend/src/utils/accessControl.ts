export interface PermissionUser {
  role?: string
  permissions?: string[]
}

const WRITE_CAPABLE_ROLES = new Set(['admin', 'write'])
const READ_CAPABLE_ROLES = new Set(['admin', 'write', 'read'])
const ADMIN_ROLES = new Set(['admin'])
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
const SECURITY_MANAGE_PERMISSION = 'security.manage'
const TOPOLOGY_READ_PERMISSION = 'topology.read'
const USERS_VIEW_PERMISSION = 'users.view'
const USERS_MANAGE_PERMISSION = 'users.manage'

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

export function canManageSecurity(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, SECURITY_MANAGE_PERMISSION, ADMIN_ROLES)
}

export function canAccessDashboardPath(
  user: PermissionUser | null | undefined,
  pathname: string,
): boolean {
  const normalizedPath = pathname.trim().toLowerCase()

  if (normalizedPath.startsWith('/users')) return canViewUsers(user)
  if (normalizedPath.startsWith('/ml-setup')) return canAccessMLSetup(user)
  if (normalizedPath.startsWith('/topology')) {
    return canAccessWithPermission(user, TOPOLOGY_READ_PERMISSION, READ_CAPABLE_ROLES)
  }
  if (
    normalizedPath.startsWith('/status') ||
    normalizedPath.startsWith('/logs') ||
    normalizedPath.startsWith('/monitoring') ||
    normalizedPath.startsWith('/tracing')
  ) {
    return canAccessWithPermission(user, LOGS_READ_PERMISSION, READ_CAPABLE_ROLES)
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
    normalizedPath.startsWith('/security') ||
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
