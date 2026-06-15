export type UsersPageRolePermissions = Record<string, string[]>

export type UsersPageRolePermissionsPayload = {
  rolePermissions?: UsersPageRolePermissions
}

export const EMPTY_ROLE_PERMISSIONS = Object.freeze({}) as UsersPageRolePermissions
