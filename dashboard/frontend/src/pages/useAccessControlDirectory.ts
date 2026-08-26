import { useCallback, useEffect, useRef, useState } from 'react'
import {
  inferenceAccessApi,
  type AccessAPIKey,
  type AccessAuditEvent,
  type AccessBudget,
  type AccessGroup,
  type AccessPage,
  type AccessTeam,
  type AccessUsageEvent,
  type AccessUser,
} from '../utils/inferenceAccessApi'
import {
  dashboardMemberInvitationApi,
  type DashboardMemberInvitation,
} from '../utils/dashboardMemberInvitations'
import type {
  AccessControlPageState,
  AccessControlViewProps,
  DashboardMember,
} from './AccessControlViewTypes'
import {
  createAccessEntityDeletionTombstones,
  omitDeletedAccessEntities,
} from './accessEntityDeletionState'
import {
  EMPTY_ACCESS_OVERVIEW,
  EMPTY_ACCESS_USAGE,
  emptyAccessPage,
  type AccessView,
} from './AccessControlPageSupport'
import type { UsageScope } from './accessControlUsageRange'
import { loadAllDashboardMembers } from './dashboardMemberDirectory'
import { useAccessControlCurrentView } from './useAccessControlViewData'

export type EntityTotals = AccessControlViewProps['entityTotals']

interface AccessControlDirectoryOptions {
  activeView: AccessView
  requestedPageQuery: string
  currentUserEmail?: string
  currentUserName?: string
  selfUserId: string
  selfService: boolean
  canManageDashboardMembers: boolean
  canReadDashboardMembers: boolean
  canReadUsers: boolean
  canReadTeams: boolean
  canReadGroups: boolean
  canReadBudgets: boolean
}

export const useAccessControlDirectory = ({
  activeView,
  requestedPageQuery,
  currentUserEmail,
  currentUserName,
  selfUserId,
  selfService,
  canManageDashboardMembers,
  canReadDashboardMembers,
  canReadUsers,
  canReadTeams,
  canReadGroups,
  canReadBudgets,
}: AccessControlDirectoryOptions) => {
  const [overview, setOverview] = useState(EMPTY_ACCESS_OVERVIEW)
  const [usage, setUsage] = useState(EMPTY_ACCESS_USAGE)
  const [users, setUsers] = useState<AccessUser[]>([])
  const [teams, setTeams] = useState<AccessTeam[]>([])
  const [keys, setKeys] = useState<AccessAPIKey[]>([])
  const [groups, setGroups] = useState<AccessGroup[]>([])
  const [budgets, setBudgets] = useState<AccessBudget[]>([])
  const [entityTotals, setEntityTotals] = useState<EntityTotals>({
    users: 0,
    teams: 0,
    'api-keys': 0,
    'access-groups': 0,
    budgets: 0,
  })
  const [dashboardMembers, setDashboardMembers] = useState<DashboardMember[]>([])
  const [invitations, setInvitations] = useState<DashboardMemberInvitation[]>([])
  const [requestPage, setRequestPage] = useState<AccessPage<AccessUsageEvent>>(emptyAccessPage)
  const [auditPage, setAuditPage] = useState<AccessPage<AccessAuditEvent>>(emptyAccessPage)
  const [loading, setLoading] = useState(true)
  const [viewLoading, setViewLoading] = useState(false)
  const [error, setError] = useState('')
  const [pageState, setPageState] = useState<AccessControlPageState>({
    page: 1,
    pageSize: 10,
    query: activeView === 'request-logs' ? requestedPageQuery : '',
  })
  const [pageCursors, setPageCursors] = useState<Record<string, Record<number, string>>>({})
  const [usageScope, setUsageScope] = useState<UsageScope>({
    type: 'global',
    id: '',
    model: '',
    range: 'today',
    granularity: 'auto',
    customFrom: '',
    customTo: '',
  })
  const [liveState, setLiveState] = useState<'checking' | 'live' | 'error'>('checking')
  const deletionTombstonesRef = useRef(createAccessEntityDeletionTombstones())
  const dashboardIdentityLoadGenerationRef = useRef(0)
  const userDirectoryLoadedRef = useRef(false)

  useEffect(() => {
    if (activeView === 'users') userDirectoryLoadedRef.current = false
    setPageState((current) => ({
      ...current,
      page: 1,
      query: activeView === 'request-logs' ? requestedPageQuery : '',
    }))
    setPageCursors({})
  }, [activeView, requestedPageQuery])

  useEffect(() => {
    setPageCursors({})
  }, [pageState.pageSize, pageState.query])

  const rememberNextCursor = useCallback(
    (view: AccessView, page: AccessPage<unknown>) => {
      if (!page.nextCursor) return
      setPageCursors((current) => {
        if (current[view]?.[pageState.page + 1] === page.nextCursor) return current
        return {
          ...current,
          [view]: { ...(current[view] || {}), [pageState.page + 1]: page.nextCursor! },
        }
      })
    },
    [pageState.page],
  )

  const displayTotal = useCallback(
    (page: AccessPage<unknown>) =>
      (pageState.page - 1) * pageState.pageSize + page.items.length + (page.hasMore ? 1 : 0),
    [pageState.page, pageState.pageSize],
  )

  const loadDashboardIdentities = useCallback(async () => {
    if (!canReadDashboardMembers) return
    const generation = ++dashboardIdentityLoadGenerationRef.current
    const [membersResult, invitationsResult] = await Promise.allSettled([
      loadAllDashboardMembers(),
      canManageDashboardMembers
        ? dashboardMemberInvitationApi.list()
        : Promise.resolve({ items: [] as DashboardMemberInvitation[] }),
    ])
    if (generation !== dashboardIdentityLoadGenerationRef.current) return
    if (membersResult.status === 'fulfilled') {
      setDashboardMembers(
        omitDeletedAccessEntities(
          deletionTombstonesRef.current,
          'dashboard-member',
          membersResult.value,
        ),
      )
    }
    if (invitationsResult.status === 'fulfilled') setInvitations(invitationsResult.value.items)
    if (membersResult.status === 'rejected' && invitationsResult.status === 'rejected') {
      throw membersResult.reason
    }
  }, [canManageDashboardMembers, canReadDashboardMembers])

  const loadCatalog = useCallback(
    async (showSpinner = true) => {
      if (showSpinner) setLoading(true)
      setError('')
      setLiveState('checking')
      try {
        const [nextOverview, nextKeys, ownTeams] = await Promise.all([
          inferenceAccessApi.overview(),
          selfService
            ? inferenceAccessApi.selfKeys({ limit: 25 })
            : Promise.resolve(emptyAccessPage<AccessAPIKey>()),
          selfService
            ? inferenceAccessApi.selfTeams()
            : Promise.resolve({ items: [], members: [], accessGroups: [], budgets: [] }),
        ])
        setOverview(nextOverview)
        setEntityTotals((current) => ({
          users: nextOverview.users === null ? current.users : Number(nextOverview.users),
          teams: nextOverview.teams === null ? current.teams : Number(nextOverview.teams),
          'api-keys':
            nextOverview.activeKeys === null
              ? current['api-keys']
              : Number(nextOverview.activeKeys),
          'access-groups':
            nextOverview.accessGroups === null
              ? current['access-groups']
              : Number(nextOverview.accessGroups),
          budgets:
            nextOverview.enabledBudgets === null
              ? current.budgets
              : Number(nextOverview.enabledBudgets),
        }))
        if (selfService) {
          setKeys(omitDeletedAccessEntities(deletionTombstonesRef.current, 'key', nextKeys.items))
          const visibleMembers = [...ownTeams.members]
          if (selfUserId && !visibleMembers.some((member) => member.id === selfUserId)) {
            visibleMembers.push({
              id: selfUserId,
              email: currentUserEmail || '',
              name: currentUserName || currentUserEmail || 'You',
              status: 'active',
            })
          }
          setUsers(
            visibleMembers.map((member) => ({
              ...member,
              accessGroupIds: [],
              memberships: ownTeams.items.flatMap((team) =>
                team.members.filter((membership) => membership.userId === member.id),
              ),
            })),
          )
          setTeams(ownTeams.items)
          setGroups(ownTeams.accessGroups)
          setBudgets(ownTeams.budgets)
          setEntityTotals((current) => ({ ...current, teams: ownTeams.items.length }))
        }
        setLiveState('live')
        await loadDashboardIdentities().catch(() => undefined)
      } catch (nextError) {
        setLiveState('error')
        setError(nextError instanceof Error ? nextError.message : 'Could not load access data')
      } finally {
        if (showSpinner) setLoading(false)
      }
    },
    [currentUserEmail, currentUserName, loadDashboardIdentities, selfService, selfUserId],
  )

  useEffect(() => {
    void loadCatalog()
  }, [loadCatalog])

  const loadCurrentView = useAccessControlCurrentView({
    activeView,
    selfService,
    canReadUsers,
    canReadTeams,
    canReadGroups,
    canReadBudgets,
    usageScope,
    pageState,
    pageCursors,
    deletionTombstonesRef,
    userDirectoryLoadedRef,
    rememberNextCursor,
    displayTotal,
    setUsage,
    setUsers,
    setTeams,
    setKeys,
    setGroups,
    setBudgets,
    setEntityTotals,
    setRequestPage,
    setAuditPage,
    setViewLoading,
    setError,
  })

  return {
    overview,
    usage,
    users,
    setUsers,
    teams,
    setTeams,
    keys,
    setKeys,
    groups,
    setGroups,
    budgets,
    setBudgets,
    entityTotals,
    setEntityTotals,
    dashboardMembers,
    setDashboardMembers,
    invitations,
    requestPage,
    auditPage,
    loading,
    viewLoading,
    error,
    setError,
    pageState,
    setPageState,
    usageScope,
    setUsageScope,
    liveState,
    deletionTombstonesRef,
    userDirectoryLoadedRef,
    loadDashboardIdentities,
    loadCatalog,
    loadCurrentView,
  }
}
