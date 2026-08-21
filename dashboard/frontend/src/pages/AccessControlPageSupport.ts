import type {
  AccessAPIKey,
  AccessBudget,
  AccessGroup,
  AccessOverview,
  AccessPage,
  AccessTeam,
  AccessUser,
  UsageSummary,
} from '../utils/inferenceAccessApi'

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
  | { kind: 'key'; value: Partial<AccessAPIKey>; ownerType: 'user' | 'team' }
  | { kind: 'group'; value: Partial<AccessGroup> }
  | { kind: 'budget'; value: Partial<AccessBudget> }

export const ACCESS_NAV_ITEMS: Array<{
  id: AccessView
  label: string
  section: 'Control' | 'Identity' | 'Policy' | 'Observe'
  description: string
}> = [
  {
    id: 'api-keys',
    label: 'API Keys',
    section: 'Control',
    description: 'Credentials and model scope',
  },
  {
    id: 'users',
    label: 'Users',
    section: 'Identity',
    description: 'People, access, and invitations',
  },
  { id: 'teams', label: 'Teams', section: 'Identity', description: 'Shared membership and limits' },
  {
    id: 'access-groups',
    label: 'Access Groups',
    section: 'Policy',
    description: 'Reusable model grants',
  },
  { id: 'budgets', label: 'Budgets', section: 'Policy', description: 'RPM, TPM, and daily quota' },
  {
    id: 'usage',
    label: 'Usage',
    section: 'Observe',
    description: 'Traffic, performance, and access posture',
  },
  {
    id: 'request-logs',
    label: 'Request Logs',
    section: 'Observe',
    description: 'Request-level accounting',
  },
  { id: 'audit-logs', label: 'Audit', section: 'Observe', description: 'Administrative changes' },
]

export const EMPTY_ACCESS_OVERVIEW: AccessOverview = {
  users: 0,
  teams: 0,
  activeKeys: 0,
  expiringKeys: 0,
  accessGroups: 0,
  enabledBudgets: 0,
  requestsToday: 0,
  successfulToday: 0,
  tokensToday: 0,
  p95LatencyMs: 0,
}

export const EMPTY_ACCESS_USAGE: UsageSummary = {
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
  offset: 0,
  hasMore: false,
})

export const accessPageQuery = (state: { page: number; pageSize: number; query: string }) => ({
  q: state.query.trim() || undefined,
  limit: state.pageSize,
  offset: (state.page - 1) * state.pageSize,
})

export const accessRangeStart = (preset: '24h' | '7d' | '30d') => {
  const hours = preset === '24h' ? 24 : preset === '7d' ? 24 * 7 : 24 * 30
  return new Date(Date.now() - hours * 60 * 60 * 1000).toISOString()
}
