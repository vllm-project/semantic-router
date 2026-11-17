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
const FEEDBACK_SUBMIT_PERMISSION = 'feedback.submit'
const LOGS_READ_PERMISSION = 'logs.read'
const MCP_READ_PERMISSION = 'mcp.read'
const MCP_MANAGE_PERMISSION = 'mcp.manage'
const REPLAY_READ_PERMISSION = 'replay.read'
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

export function canWriteEvaluation(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, EVALUATION_WRITE_PERMISSION)
}

export function canRunEvaluation(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, EVALUATION_RUN_PERMISSION)
}

export function canSubmitFeedback(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, FEEDBACK_SUBMIT_PERMISSION, READ_CAPABLE_ROLES)
}

export function canManageMCP(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, MCP_MANAGE_PERMISSION)
}

export function canAccessDashboardPath(
  user: PermissionUser | null | undefined,
  pathname: string,
): boolean {
  const normalizedPath = pathname.trim().toLowerCase()
  const startsWith = (...prefixes: string[]) =>
    prefixes.some((prefix) => normalizedPath.startsWith(prefix))

  if (startsWith('/settings/access', '/users')) return canViewUsers(user)
  if (startsWith('/routing/topology', '/topology', '/observe/status', '/status')) {
    return canAccessWithPermission(user, TOPOLOGY_READ_PERMISSION, READ_CAPABLE_ROLES)
  }
  if (
    startsWith(
      '/observe/logs',
      '/observe/metrics',
      '/observe/traces',
      '/logs',
      '/monitoring',
      '/tracing',
    )
  ) {
    return canAccessWithPermission(user, LOGS_READ_PERMISSION)
  }
  if (startsWith('/observe/requests', '/observe/runs', '/insights')) {
    return canAccessWithPermission(user, REPLAY_READ_PERMISSION, READ_CAPABLE_ROLES)
  }
  if (startsWith('/observe/evaluations', '/evaluation')) {
    return canAccessWithPermission(user, EVALUATION_READ_PERMISSION, READ_CAPABLE_ROLES)
  }
  if (startsWith('/settings/integrations', '/config/mcp')) {
    return canAccessWithPermission(user, MCP_READ_PERMISSION, READ_CAPABLE_ROLES)
  }
  if (
    normalizedPath === '/api/router/docs' ||
    startsWith(
      '/routing/',
      '/models/',
      '/settings/router',
      '/settings/schema',
      '/builder',
      '/models',
      '/decision-model',
      '/config',
    )
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
