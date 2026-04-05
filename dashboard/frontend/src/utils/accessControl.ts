export interface PermissionUser {
  role?: string
  permissions?: string[]
}

const WRITE_CAPABLE_ROLES = new Set(['admin', 'write'])
const CONFIG_WRITE_PERMISSION = 'config.write'
const ML_PIPELINE_MANAGE_PERMISSION = 'mlpipeline.manage'

function hasPermission(user: PermissionUser | null | undefined, permission: string): boolean {
  if (!user) {
    return false
  }

  if (Array.isArray(user.permissions) && user.permissions.length > 0) {
    return user.permissions.includes(permission)
  }

  return false
}

function hasWriteCapableRole(user: PermissionUser | null | undefined): boolean {
  if (!user) {
    return false
  }

  const normalizedRole = typeof user.role === 'string' ? user.role.trim().toLowerCase() : ''
  return WRITE_CAPABLE_ROLES.has(normalizedRole)
}

export function canAccessReplayFlowDetails(user?: PermissionUser | null): boolean {
  if (hasPermission(user, CONFIG_WRITE_PERMISSION)) {
    return true
  }

  return hasWriteCapableRole(user)
}

export function canAccessMLSetup(user?: PermissionUser | null): boolean {
  if (hasPermission(user, ML_PIPELINE_MANAGE_PERMISSION)) {
    return true
  }

  return hasWriteCapableRole(user)
}
