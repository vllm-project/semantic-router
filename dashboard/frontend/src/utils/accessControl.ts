export interface PermissionUser {
  role?: string
  permissions?: string[]
}

const WRITE_CAPABLE_ROLES = new Set(['admin', 'write'])
const CONFIG_WRITE_PERMISSION = 'config.write'
const ML_PIPELINE_MANAGE_PERMISSION = 'mlpipeline.manage'

function hasPermission(user: PermissionUser | null | undefined, permission: string): boolean {
  return Array.isArray(user?.permissions) && user.permissions.includes(permission)
}

function hasWriteCapableRole(user: PermissionUser | null | undefined): boolean {
  if (!user) {
    return false
  }

  const normalizedRole = typeof user.role === 'string' ? user.role.trim().toLowerCase() : ''
  return WRITE_CAPABLE_ROLES.has(normalizedRole)
}

function canAccessWithPermission(
  user: PermissionUser | null | undefined,
  permission: string,
): boolean {
  if (Array.isArray(user?.permissions)) {
    return hasPermission(user, permission)
  }

  return hasWriteCapableRole(user)
}

export function canAccessReplayFlowDetails(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, CONFIG_WRITE_PERMISSION)
}

export function canWriteConfig(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, CONFIG_WRITE_PERMISSION)
}

export function canAccessMLSetup(user?: PermissionUser | null): boolean {
  return canAccessWithPermission(user, ML_PIPELINE_MANAGE_PERMISSION)
}
