import type { DashboardMemberInvitation } from '../utils/dashboardMemberInvitations'
import type {
  AccessAPIKey,
  AccessAuditEvent,
  AccessBudget,
  AccessGroup,
  AccessOverview,
  AccessPage,
  AccessTeam,
  AccessUsageEvent,
  AccessUser,
  UsageSummary,
} from '../utils/inferenceAccessApi'
import type { AccessView } from './AccessControlPageSupport'
import type { UsageScope } from './accessControlUsageRange'
import type { AccessControlSelectorSources } from './accessControlSelectorSources'

export interface DashboardMember {
  id: string
  email: string
  name: string
  role: string
  status: string
  permissions?: string[]
  createdAt?: number
  lastLoginAt?: number
}

export type IdentityTab = 'users' | 'invitations'

export interface AccessControlPageState {
  page: number
  pageSize: number
  query: string
}

export interface AccessControlViewProps {
  view: AccessView
  overview: AccessOverview
  usage: UsageSummary
  selectors: AccessControlSelectorSources
  users: AccessUser[]
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  groups: AccessGroup[]
  budgets: AccessBudget[]
  entityTotals: Record<'users' | 'teams' | 'api-keys' | 'access-groups' | 'budgets', number>
  dashboardMembers: DashboardMember[]
  invitations: DashboardMemberInvitation[]
  identityTab: IdentityTab
  onIdentityTabChange: (value: IdentityTab) => void
  requestPage: AccessPage<AccessUsageEvent>
  auditPage: AccessPage<AccessAuditEvent>
  pageState: AccessControlPageState
  onPageStateChange: (value: AccessControlPageState) => void
  usageScope: UsageScope
  onUsageScopeChange: (value: AccessControlViewProps['usageScope']) => void
  loading: boolean
  canManage: boolean
  canManageDashboardMembers: boolean
  ownerName: (item: Pick<AccessAPIKey, 'ownerType' | 'ownerId'>) => string
  onOpenKey: (id: string) => void
  onOpenLog: (id: string) => void
  onOpenEntity: (id: string) => void
  onOpenDashboardMember: (id: string) => void
  onInvitationsChanged: () => void
}
