export interface PermissionUser {
  role?: string
  permissions?: string[]
}

const WRITE_CAPABLE_ROLES = new Set(['admin', 'write'])
const ML_PIPELINE_MANAGE_PERMISSION = 'mlpipeline.manage'

export function canAccessMLSetup(user?: PermissionUser | null): boolean {
  if (!user) {
    return false
  }

  if (Array.isArray(user.permissions) && user.permissions.length > 0) {
    return user.permissions.includes(ML_PIPELINE_MANAGE_PERMISSION)
  }

  const normalizedRole = typeof user.role === 'string' ? user.role.trim().toLowerCase() : ''
  return WRITE_CAPABLE_ROLES.has(normalizedRole)
}
