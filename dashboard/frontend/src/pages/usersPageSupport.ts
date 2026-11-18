export type UsersPageRolePermissions = Record<string, string[]>

export type UsersPageRolePermissionsPayload = {
  rolePermissions?: UsersPageRolePermissions
}

export const EMPTY_ROLE_PERMISSIONS = Object.freeze({}) as UsersPageRolePermissions

// Mirrors MaxPasswordBytes in dashboard/backend/auth/password.go.
export const MAX_PASSWORD_BYTES = 72

// bcrypt counts bytes, not characters: 25 CJK characters are 75 bytes. A
// maxLength attribute is not a substitute, it counts UTF-16 units and would
// silently truncate a pasted password.
export const passwordByteLength = (password: string) => new TextEncoder().encode(password).length

// Blank means "keep the current password" when editing; a new account has none.
export const validateUserPassword = (password: string, mode: 'create' | 'edit') => {
  if (!password) {
    return mode === 'create' ? 'A password is required to create a user.' : null
  }

  const byteLength = passwordByteLength(password)
  if (byteLength > MAX_PASSWORD_BYTES) {
    return `Password must be at most ${MAX_PASSWORD_BYTES} bytes (this one is ${byteLength}). Passwords are measured in bytes, so accented and non-Latin characters count as more than one.`
  }

  return null
}

// The role write can commit and the password write still fail, so the message
// has to say which of the two happened.
export const describeUserUpdateFailure = (error: unknown, roleAndStatusSaved: boolean) => {
  const reason = error instanceof Error ? error.message : String(error)
  if (!roleAndStatusSaved) {
    return reason
  }

  return `Role and status were saved. The password was not changed: ${reason}`
}

export const isUsersRequestAbortError = (error: unknown) =>
  error instanceof DOMException
    ? error.name === 'AbortError'
    : error instanceof Error && error.name === 'AbortError'

export const createLatestUsersRequest = () => {
  let sequence = 0
  let activeController: AbortController | null = null

  return {
    start() {
      activeController?.abort()
      const controller = new AbortController()
      const requestSequence = ++sequence
      activeController = controller

      return {
        signal: controller.signal,
        isCurrent: () => requestSequence === sequence,
        finish: () => {
          if (requestSequence === sequence) {
            activeController = null
          }
        },
      }
    },
    abort() {
      sequence += 1
      activeController?.abort()
      activeController = null
    },
  }
}

export type AdminUser = {
  id: string
  email: string
  name: string
  role: string
  status: string
  createdAt?: number
  updatedAt?: number
  lastLoginAt?: number
}

export type UsersPagePayload = {
  users?: AdminUser[]
  total?: number
  page?: number
  active?: number
  privileged?: number
}

export const DASHBOARD_ROLES = ['admin', 'write', 'read'] as const

export type DashboardRoleTone = 'brand' | 'violet' | 'neutral'

// The backend's role ids with the names the dashboard shows for them.
export const ROLE_META: Record<
  string,
  { label: string; description: string; tone: DashboardRoleTone }
> = {
  admin: {
    label: 'Admin',
    description: 'Everything, including people, roles and router settings.',
    tone: 'brand',
  },
  write: { label: 'Builder', description: 'Create, tune and deploy routing work.', tone: 'violet' },
  read: {
    label: 'Read',
    description: 'Explore configuration, requests and results.',
    tone: 'neutral',
  },
}

export const roleLabel = (role: string) => ROLE_META[role]?.label ?? role

/** "3 min ago", "2 days ago"; a Unix timestamp in seconds. */
export function relativeTime(seconds: number | undefined, now = Date.now()): string {
  if (!seconds) return 'Never'
  const delta = Math.max(0, now - seconds * 1000)
  const minute = 60_000
  const hour = 60 * minute
  const day = 24 * hour
  if (delta < minute) return 'Just now'
  if (delta < hour) return `${Math.floor(delta / minute)} min ago`
  if (delta < day) return `${Math.floor(delta / hour)} h ago`
  if (delta < 30 * day) {
    const days = Math.floor(delta / day)
    return `${days} ${days === 1 ? 'day' : 'days'} ago`
  }
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(new Date(seconds * 1000))
}

export function absoluteTime(seconds: number | undefined): string {
  if (!seconds) return '—'
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(seconds * 1000))
}

export const responseText = async (response: Response) =>
  (await response.text()) || `Request failed (${response.status})`

/** The page an invitation link opens, for a token the backend returned. */
export function invitationUrl(token: string): string {
  return new URL(`/invite/${encodeURIComponent(token)}`, window.location.origin).toString()
}
