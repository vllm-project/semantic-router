import type {
  AccessAPIKey,
  AccessBudget,
  AccessGroup,
  AccessOverview,
  AccessPage,
  AccessTeam,
  AccessUser,
  InlineRateLimitPolicyDraft,
  UsageSummary,
} from '../utils/inferenceAccessApi'
import type { ProductIconName } from '../components/ProductIcon'
import {
  canAccessDashboardPath,
  canManageUsers,
  canRevealInferenceKey,
  canSelfManageInferenceAccess,
  canViewUsers,
  type PermissionUser,
} from '../utils/accessControl'

export type AccessView =
  | 'api-keys'
  | 'users'
  | 'teams'
  | 'access-groups'
  | 'budgets'
  | 'usage'
  | 'request-logs'
  | 'audit-logs'

export type AccessEditor =
  | { kind: 'user'; value: Partial<AccessUser> }
  | { kind: 'team'; value: Partial<AccessTeam> }
  | {
      kind: 'key'
      value: Partial<AccessAPIKey>
      ownerType: 'user' | 'team'
      rateLimitMode: 'inherit' | 'budget' | 'custom'
      inlineRateLimit?: InlineRateLimitPolicyDraft
    }
  | { kind: 'group'; value: Partial<AccessGroup> }
  | { kind: 'budget'; value: Partial<AccessBudget> }

export const ACCESS_NAV_ITEMS: Array<{
  id: AccessView
  icon: ProductIconName
  label: string
  section: 'Control' | 'Identity' | 'Policy' | 'Observe'
  description: string
}> = [
  {
    id: 'usage',
    icon: 'activity',
    label: 'Usage',
    section: 'Observe',
    description: 'Traffic, performance, and access posture',
  },
  {
    id: 'api-keys',
    icon: 'key',
    label: 'API Keys',
    section: 'Control',
    description: 'Credentials and model scope',
  },
  {
    id: 'users',
    icon: 'user',
    label: 'Users',
    section: 'Identity',
    description: 'People, access, and invitations',
  },
  {
    id: 'teams',
    icon: 'team',
    label: 'Teams',
    section: 'Identity',
    description: 'Shared membership and limits',
  },
  {
    id: 'access-groups',
    icon: 'shield',
    label: 'Access Groups',
    section: 'Policy',
    description: 'Reusable model grants',
  },
  {
    id: 'budgets',
    icon: 'budget',
    label: 'Budgets',
    section: 'Policy',
    description: 'Request, token, spend, and concurrency limits',
  },
  {
    id: 'request-logs',
    icon: 'logs',
    label: 'Request Logs',
    section: 'Observe',
    description: 'Request-level accounting',
  },
  {
    id: 'audit-logs',
    icon: 'audit',
    label: 'Audit',
    section: 'Observe',
    description: 'Administrative changes',
  },
]

export const EMPTY_ACCESS_OVERVIEW: AccessOverview = {
  users: null,
  teams: null,
  activeKeys: null,
  expiringKeys: null,
  accessGroups: null,
  enabledBudgets: null,
  requestsToday: 0,
  successfulToday: 0,
  tokensToday: 0,
  p95LatencyMs: 0,
}

export const EMPTY_ACCESS_USAGE: UsageSummary = {
  granularity: 'hour',
  requests: 0,
  successful: 0,
  failed: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  activeKeys: 0,
  averageLatencyMs: 0,
  p95LatencyMs: 0,
  averageTtftMs: 0,
  p95TtftMs: 0,
  costs: [],
  series: [],
  byModel: [],
  byUser: [],
  byTeam: [],
  byKey: [],
}

export const emptyAccessPage = <T>(): AccessPage<T> => ({
  items: [],
  total: 0,
  limit: 10,
  hasMore: false,
})

export const accessPageQuery = (
  state: { page: number; pageSize: number; query: string },
  cursor?: string,
) => ({
  q: state.query.trim() || undefined,
  limit: state.pageSize,
  cursor,
})

const hasManagementPermission = (user: PermissionUser | null, permission: string) =>
  user?.managementPermissions?.includes(permission) ?? false

export function resolveAccessControlPage(user: PermissionUser | null, pathname: string) {
  const routeView = (
    pathname === '/logs' ? 'request-logs' : pathname.split('/').filter(Boolean)[1] || 'usage'
  ) as AccessView
  const activeView = ACCESS_NAV_ITEMS.some((item) => item.id === routeView) ? routeView : 'usage'
  const canAdministerDirectory = Boolean(
    user?.managementPermissions?.some((permission) =>
      ['user.manage', 'team.manage', 'access_policy.manage', 'rate_policy.manage'].includes(
        permission,
      ),
    ),
  )
  const selfService =
    Boolean(user?.managementUserId) && canSelfManageInferenceAccess(user) && !canAdministerDirectory
  const canReadRouting = hasManagementPermission(user, 'routing.read')
  const canManage =
    !selfService &&
    (activeView === 'api-keys'
      ? hasManagementPermission(user, 'key.manage')
      : activeView === 'users'
        ? hasManagementPermission(user, 'user.manage')
        : activeView === 'teams'
          ? hasManagementPermission(user, 'team.manage')
          : activeView === 'access-groups'
            ? hasManagementPermission(user, 'access_policy.manage') && canReadRouting
            : activeView === 'budgets'
              ? hasManagementPermission(user, 'rate_policy.manage')
              : false)

  return {
    activeView,
    activeMeta: ACCESS_NAV_ITEMS.find((item) => item.id === activeView) ?? ACCESS_NAV_ITEMS[0],
    visibleNavItems: ACCESS_NAV_ITEMS.filter((item) =>
      canAccessDashboardPath(user, item.id === 'request-logs' ? '/logs' : `/access/${item.id}`),
    ),
    selfService,
    canManage,
    canRevealKeys: canRevealInferenceKey(user),
    canReadDashboardMembers: canViewUsers(user),
    canManageDashboardMembers: canManageUsers(user),
    canReadUsers: hasManagementPermission(user, 'user.read'),
    canReadTeams: hasManagementPermission(user, 'team.read'),
    canReadGroups: hasManagementPermission(user, 'access_policy.read'),
    canReadBudgets: hasManagementPermission(user, 'rate_policy.read'),
  }
}
