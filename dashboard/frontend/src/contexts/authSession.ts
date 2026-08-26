import type { ManagementNamespaceSummary } from '../utils/routerManagementTypes'
import type { ManagementIdentityStatus } from './managementIdentityRecovery'

export interface AuthUser {
  id: string
  email: string
  name: string
  role?: string
  permissions?: string[]
  managementPermissions?: string[]
  managementScopes?: ManagementNamespaceSummary[]
  /** Router Management identity. Never substitute the local Dashboard user id. */
  managementPrincipalId?: string
  managementNamespaceId?: string
  managementUserId?: string
  managementTeams?: Array<{
    teamId: string
    name: string
    role: 'admin' | 'member'
    status: string
  }>
  managementSelfServicePolicy?: {
    maxKeysPerUser: number
    maxDelegatedSessions: number
    delegatedSessionTtlSeconds: number
    allowTeamKeyDelegation: boolean
    automaticFirstKey: boolean
    revision: number
  }
  /** Whether Router Management identity projection completed for this session. */
  managementIdentityStatus?: ManagementIdentityStatus
  /** User-facing failure detail. Router permissions remain empty while this is set. */
  managementIdentityError?: string
}

export interface AuthSessionRefreshResult {
  user: AuthUser | null
  unauthorized: boolean
}

export function hasAuthenticatedSession(user: AuthUser | null): boolean {
  return Boolean(user)
}

export async function fetchCurrentAuthUser(
  fetcher: typeof fetch = fetch,
): Promise<AuthSessionRefreshResult> {
  const response = await fetcher('/api/auth/me', { credentials: 'same-origin' })

  if (response.status === 401) {
    return { user: null, unauthorized: true }
  }

  if (!response.ok) {
    return { user: null, unauthorized: false }
  }

  const payload = (await response.json()) as { user?: AuthUser | null }
  return { user: payload?.user ?? null, unauthorized: false }
}
