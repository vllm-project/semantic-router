export interface AccountPermissionGroup {
  key: string
  label: string
  permissions: string[]
}

export function getAccountInitials(name?: string, email?: string): string {
  const source = (name || email || 'User').trim()
  if (!source) return 'U'

  const words = source.split(/\s+/).filter(Boolean)
  if (words.length >= 2) return `${words[0][0]}${words[1][0]}`.toUpperCase()
  return source.slice(0, 2).toUpperCase()
}

export function formatAccountRole(role?: string): string {
  const normalized = role?.trim()
  if (!normalized) return 'Unknown role'
  return normalized
    .split(/[._-]+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(' ')
}

function permissionNamespace(permission: string): string {
  const separatorIndex = permission.search(/[.:]/)
  return separatorIndex > 0 ? permission.slice(0, separatorIndex) : 'general'
}

export function groupAccountPermissions(permissions: readonly string[]): AccountPermissionGroup[] {
  const uniquePermissions = Array.from(
    new Set(permissions.map((permission) => permission.trim()).filter(Boolean)),
  )
  const groups = new Map<string, string[]>()
  uniquePermissions.forEach((permission) => {
    const namespace = permissionNamespace(permission)
    groups.set(namespace, [...(groups.get(namespace) || []), permission])
  })

  return Array.from(groups.entries()).map(([key, groupedPermissions]) => ({
    key,
    label: key === 'general' ? 'General' : formatAccountRole(key),
    permissions: groupedPermissions,
  }))
}
