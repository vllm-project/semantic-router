import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
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
  type CreatedAccessAPIKey,
  type UsageFilter,
} from '../utils/inferenceAccessApi'
import { claimInvitationOnboarding } from '../utils/invitationOnboarding'
import {
  dashboardMemberInvitationApi,
  type DashboardMemberInvitation,
} from '../utils/dashboardMemberInvitations'
import AccessControlViews, { type DashboardMember, type IdentityTab } from './AccessControlViews'
import AccessControlPageOverlays from './AccessControlPageOverlays'
import AccessControlWorkspace from './AccessControlWorkspace'
import { accessControlSelectorSources } from './accessControlSelectorSources'
import {
  EMPTY_ACCESS_OVERVIEW,
  EMPTY_ACCESS_USAGE,
  accessPageQuery,
  emptyAccessPage,
  resolveAccessControlPage,
  type AccessEditor,
  type AccessView,
} from './AccessControlPageSupport'
import { usageRangeBounds, type UsageScope } from './accessControlUsageRange'
import styles from './AccessControlPage.module.css'

type PageState = { page: number; pageSize: number; query: string }
type EntityTotals = Record<'users' | 'teams' | 'api-keys' | 'access-groups' | 'budgets', number>
type EntityEditorReturn = { kind: Exclude<AccessEditor['kind'], 'key'>; id: string }
const AccessControlPage: React.FC = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const { user: currentUser } = useAuth()
  const {
    activeView,
    activeMeta,
    visibleNavItems,
    selfService,
    canManage,
    canRevealKeys,
    canManageDashboardMembers,
    canReadUsers,
    canReadTeams,
    canReadGroups,
    canReadBudgets,
    canReadDashboardMembers,
  } = resolveAccessControlPage(currentUser, location.pathname)
  const selfUserId = currentUser?.managementUserId || ''
  const detailParams = new URLSearchParams(location.search)
  const invitationOnboardingRequested = detailParams.get('onboarding') === 'invitation'
  const detailKeyId = activeView === 'api-keys' ? detailParams.get('key') || '' : ''
  const detailLogId = activeView === 'request-logs' ? detailParams.get('log') || '' : ''
  const requestedPageQuery = detailParams.get('q')?.trim() || ''
  const requestedCreateKind = detailParams.get('create') || ''
  const entityKind =
    activeView === 'users'
      ? 'user'
      : activeView === 'teams'
        ? 'team'
        : activeView === 'access-groups'
          ? 'group'
          : activeView === 'budgets'
            ? 'budget'
            : null
  const detailEntityId = entityKind ? detailParams.get('item') || '' : ''
  const detailMemberId = activeView === 'users' ? detailParams.get('member') || '' : ''

  const [overview, setOverview] = useState(EMPTY_ACCESS_OVERVIEW)
  const [usage, setUsage] = useState(EMPTY_ACCESS_USAGE)
  const [users, setUsers] = useState<AccessUser[]>([])
  const [teams, setTeams] = useState<AccessTeam[]>([])
  const [keys, setKeys] = useState<AccessAPIKey[]>([])
  const [groups, setGroups] = useState<AccessGroup[]>([])
  const [budgets, setBudgets] = useState<AccessBudget[]>([])
  const [ownerLabels, setOwnerLabels] = useState<Record<string, string>>({})
  const [resourceLabels, setResourceLabels] = useState<Record<string, string>>({})
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
  const [toast, setToast] = useState('')
  const [editor, setEditor] = useState<AccessEditor | null>(null)
  const [entityEditorReturn, setEntityEditorReturn] = useState<EntityEditorReturn | null>(null)
  const [editorError, setEditorError] = useState('')
  const [saving, setSaving] = useState(false)
  const [createdKey, setCreatedKey] = useState<CreatedAccessAPIKey | null>(null)
  const [inviteOpen, setInviteOpen] = useState(false)
  const [identityTab, setIdentityTab] = useState<IdentityTab>('users')
  const [pageState, setPageState] = useState<PageState>({
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
  const createRequestHandledRef = useRef(false)

  useEffect(() => {
    if (!invitationOnboardingRequested) return
    const onboardingKey = claimInvitationOnboarding(selfUserId)?.onboardingKey
    const nextParams = new URLSearchParams(location.search)
    nextParams.delete('onboarding')
    const suffix = nextParams.toString()
    navigate(`${location.pathname}${suffix ? `?${suffix}` : ''}`, { replace: true, state: null })
    if (onboardingKey?.id && onboardingKey.secret) setCreatedKey(onboardingKey)
  }, [invitationOnboardingRequested, location.pathname, location.search, navigate, selfUserId])

  useEffect(() => {
    if (!toast) return
    const timeout = window.setTimeout(() => setToast(''), 3200)
    return () => window.clearTimeout(timeout)
  }, [toast])

  useEffect(() => {
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
    const [members, invitationsResponse] = await Promise.all([
      (async () => {
        const response = await fetch('/api/admin/users?page=1&limit=200')
        if (!response.ok) throw new Error(await response.text())
        const payload = (await response.json()) as { users: DashboardMember[] }
        return payload.users || []
      })(),
      canManageDashboardMembers
        ? dashboardMemberInvitationApi.list()
        : Promise.resolve({ items: [] as DashboardMemberInvitation[] }),
    ])
    setDashboardMembers(members)
    setInvitations(invitationsResponse.items)
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
          setKeys(nextKeys.items)
          const visibleMembers = [...ownTeams.members]
          if (selfUserId && !visibleMembers.some((member) => member.id === selfUserId)) {
            visibleMembers.push({
              id: selfUserId,
              email: currentUser?.email || '',
              name: currentUser?.name || currentUser?.email || 'You',
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
    [currentUser?.email, currentUser?.name, loadDashboardIdentities, selfService, selfUserId],
  )

  useEffect(() => {
    void loadCatalog()
  }, [loadCatalog])

  const usageFilter = useMemo<UsageFilter>(() => {
    const bounds = usageRangeBounds(usageScope)
    const filter: UsageFilter = {
      model: usageScope.model || undefined,
      from: bounds.from,
      to: bounds.to,
      granularity: usageScope.granularity,
      timezoneOffset: new Date().getTimezoneOffset(),
    }
    if (usageScope.type === 'user') filter.userId = usageScope.id || undefined
    if (usageScope.type === 'team') filter.teamId = usageScope.id || undefined
    if (usageScope.type === 'key') filter.keyId = usageScope.id || undefined
    return filter
  }, [usageScope])

  const loadCurrentView = useCallback(async () => {
    setViewLoading(true)
    setError('')
    try {
      if (activeView === 'usage') {
        setUsage(
          await (selfService ? inferenceAccessApi.selfUsage : inferenceAccessApi.usage)(
            usageFilter,
          ),
        )
      }
      if (activeView === 'users' && canReadUsers && !selfService) {
        const page = await inferenceAccessApi.users(
          accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
        )
        setUsers(page.items)
        rememberNextCursor(activeView, page)
        setEntityTotals((current) => ({ ...current, users: displayTotal(page) }))
      }
      if (activeView === 'teams' && canReadTeams && !selfService) {
        const page = await inferenceAccessApi.teams(
          accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
        )
        setTeams(page.items)
        rememberNextCursor(activeView, page)
        setEntityTotals((current) => ({ ...current, teams: displayTotal(page) }))
      }
      if (activeView === 'api-keys') {
        const page = selfService
          ? await inferenceAccessApi.selfKeys(
              accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
            )
          : await inferenceAccessApi.keys(
              accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
            )
        setKeys(page.items)
        rememberNextCursor(activeView, page)
        setEntityTotals((current) => ({ ...current, 'api-keys': displayTotal(page) }))
      }
      if (activeView === 'access-groups' && canReadGroups && !selfService) {
        const page = await inferenceAccessApi.groups(
          accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
        )
        setGroups(page.items)
        rememberNextCursor(activeView, page)
        setEntityTotals((current) => ({ ...current, 'access-groups': displayTotal(page) }))
      }
      if (activeView === 'budgets' && canReadBudgets && !selfService) {
        const page = await inferenceAccessApi.budgets(
          accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
        )
        setBudgets(page.items)
        rememberNextCursor(activeView, page)
        setEntityTotals((current) => ({ ...current, budgets: displayTotal(page) }))
      }
      if (activeView === 'request-logs') {
        const page = await (selfService
          ? inferenceAccessApi.selfRequestLogs({
              ...usageFilter,
              ...accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
            })
          : inferenceAccessApi.requestLogs({
              ...usageFilter,
              ...accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
            }))
        setRequestPage({ ...page, total: displayTotal(page) })
        rememberNextCursor(activeView, page)
      }
      if (activeView === 'audit-logs') {
        const page = await inferenceAccessApi.auditLogs(
          accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
        )
        setAuditPage({ ...page, total: displayTotal(page) })
        rememberNextCursor(activeView, page)
      }
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not load this view')
    } finally {
      setViewLoading(false)
    }
  }, [
    activeView,
    canReadBudgets,
    canReadGroups,
    canReadTeams,
    canReadUsers,
    displayTotal,
    pageCursors,
    pageState,
    rememberNextCursor,
    selfService,
    usageFilter,
  ])

  useEffect(() => {
    void loadCurrentView()
  }, [loadCurrentView])

  useEffect(() => {
    if (activeView !== 'api-keys' || selfService) return
    let cancelled = false
    const owners = Array.from(
      new Map(
        keys.map((key) => [
          `${key.ownerType}:${key.ownerId}`,
          { type: key.ownerType, id: key.ownerId },
        ]),
      ).entries(),
    ).filter(([cacheKey]) => !ownerLabels[cacheKey])
    if (!owners.length) return

    void (async () => {
      const next: Record<string, string> = {}
      for (let offset = 0; offset < owners.length; offset += 6) {
        const batch = owners.slice(offset, offset + 6)
        const results = await Promise.allSettled(
          batch.map(([, owner]) =>
            owner.type === 'user'
              ? inferenceAccessApi.userSummary(owner.id)
              : inferenceAccessApi.teamSummary(owner.id),
          ),
        )
        if (cancelled) return
        results.forEach((result, index) => {
          if (result.status === 'fulfilled') next[batch[index][0]] = result.value.name
        })
      }
      if (!cancelled && Object.keys(next).length) {
        setOwnerLabels((current) => ({ ...current, ...next }))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [activeView, keys, ownerLabels, selfService])

  useEffect(() => {
    if (activeView !== 'access-groups') return
    let cancelled = false
    const resources = Array.from(
      new Map<string, AccessGroup['resources'][number]>(
        groups.flatMap((group) =>
          group.resources.map(
            (resource) => [`${resource.resourceType}:${resource.resourceId}`, resource] as const,
          ),
        ),
      ).entries(),
    ).filter(([cacheKey]) => !resourceLabels[cacheKey])
    if (!resources.length) return

    void (async () => {
      const next: Record<string, string> = {}
      for (let offset = 0; offset < resources.length; offset += 6) {
        const batch = resources.slice(offset, offset + 6)
        const results = await Promise.allSettled(
          batch.map(([, resource]) =>
            resource.resourceType === 'model'
              ? accessControlSelectorSources.models.detail(resource.resourceId)
              : accessControlSelectorSources.entrypoints.detail(resource.resourceId),
          ),
        )
        if (cancelled) return
        results.forEach((result, index) => {
          if (result.status === 'fulfilled') next[batch[index][0]] = result.value.name
        })
      }
      if (!cancelled && Object.keys(next).length) {
        setResourceLabels((current) => ({ ...current, ...next }))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [activeView, groups, resourceLabels])

  const ownerName = useCallback(
    (item: Pick<AccessAPIKey, 'ownerType' | 'ownerId'>) =>
      ownerLabels[`${item.ownerType}:${item.ownerId}`] ||
      (item.ownerType === 'user'
        ? users.find((user) => user.id === item.ownerId)?.name || item.ownerId
        : teams.find((team) => team.id === item.ownerId)?.name || item.ownerId || 'Unassigned'),
    [ownerLabels, teams, users],
  )

  const resourceName = useCallback(
    (resourceType: 'model' | 'entrypoint', resourceId: string) =>
      resourceLabels[`${resourceType}:${resourceId}`] || resourceId,
    [resourceLabels],
  )

  const openCreate = useCallback(
    (kind?: Exclude<AccessEditor['kind'], 'user'>) => {
      setEditorError('')
      setEntityEditorReturn(null)
      const target =
        kind ||
        (activeView === 'api-keys'
          ? 'key'
          : activeView === 'teams'
            ? 'team'
            : activeView === 'access-groups'
              ? 'group'
              : 'budget')
      if (target === 'team')
        setEditor({
          kind: 'team',
          value: {
            status: 'active',
            members: [],
            accessGroupIds: [],
            budgetId: '',
          },
        })
      if (target === 'key')
        setEditor(() => {
          const personalKeyCount = keys.filter(
            (key) => key.ownerType === 'user' && key.ownerId === selfUserId,
          ).length
          const maxPersonalKeys = currentUser?.managementSelfServicePolicy?.maxKeysPerUser ?? 0
          const firstManagedTeam = currentUser?.managementSelfServicePolicy?.allowTeamKeyDelegation
            ? teams.find((team) =>
                team.members.some(
                  (membership) => membership.userId === selfUserId && membership.role === 'admin',
                ),
              )
            : undefined
          const ownerType =
            selfService && personalKeyCount >= maxPersonalKeys && firstManagedTeam ? 'team' : 'user'
          const ownerId = ownerType === 'team' ? firstManagedTeam?.id || '' : selfUserId
          return {
            kind: 'key',
            ownerType,
            rateLimitMode: 'inherit',
            value: {
              ownerType,
              ownerId,
              contextTeamId: ownerType === 'team' ? ownerId : undefined,
              status: 'active',
              accessGroupIds: [],
            },
          }
        })
      if (target === 'group') setEditor({ kind: 'group', value: { resources: [] } })
      if (target === 'budget')
        setEditor({
          kind: 'budget',
          value: {
            rules: [
              {
                metric: 'requests',
                algorithm: 'sliding_log',
                limit: '60',
                window: 'PT1M',
                accounting: 'request',
                enforcement: 'enforce',
              },
              {
                metric: 'total_tokens',
                algorithm: 'sliding_log',
                limit: '100000',
                window: 'PT1M',
                accounting: 'response_actual',
                enforcement: 'enforce',
              },
            ],
            enabled: true,
          },
        })
    },
    [
      activeView,
      currentUser?.managementSelfServicePolicy?.maxKeysPerUser,
      currentUser?.managementSelfServicePolicy?.allowTeamKeyDelegation,
      keys,
      selfService,
      selfUserId,
      teams,
    ],
  )

  const saveEditor = async () => {
    if (!editor) return
    const returnTarget = entityEditorReturn
    setSaving(true)
    setEditorError('')
    try {
      if (editor.kind === 'user') {
        if (!editor.value.id) throw new Error('User id is required')
        await inferenceAccessApi.saveUser(editor.value as Partial<AccessUser> & { id: string })
      }
      if (editor.kind === 'team') {
        if (!selfService && (!editor.value.accessGroupIds?.length || !editor.value.budgetId)) {
          throw new Error('Choose model access and a quota for this team.')
        }
        if (selfService && editor.value.id) {
          await inferenceAccessApi.saveSelfTeam(
            editor.value as Partial<AccessTeam> & { id: string },
          )
        } else {
          await inferenceAccessApi.saveTeam(editor.value)
        }
      }
      if (editor.kind === 'group') {
        if (!editor.value.resources?.length) {
          throw new Error('Choose at least one Mixture-of-Model or single model.')
        }
        await inferenceAccessApi.saveGroup(editor.value)
      }
      if (editor.kind === 'budget') await inferenceAccessApi.saveBudget(editor.value)
      if (editor.kind === 'key') {
        const value = { ...editor.value, ownerType: editor.ownerType }
        if (value.id) {
          await inferenceAccessApi.saveKey(value as Partial<AccessAPIKey> & { id: string })
        } else {
          const inlineRateLimit =
            editor.rateLimitMode === 'custom' ? editor.inlineRateLimit : undefined
          if (
            inlineRateLimit &&
            (!inlineRateLimit.name.trim() || inlineRateLimit.rules.length === 0)
          ) {
            throw new Error('Name the custom quota and add at least one limit.')
          }
          setCreatedKey(
            selfService
              ? await inferenceAccessApi.createSelfKey(
                  value.name || 'My API key',
                  value.ownerType || 'user',
                  value.ownerId || selfUserId,
                  value.contextTeamId,
                )
              : await inferenceAccessApi.createKey(value, inlineRateLimit),
          )
        }
      }
      setEditor(null)
      setEntityEditorReturn(null)
      setToast('Saved')
      await Promise.all([loadCatalog(false), loadCurrentView()])
      if (returnTarget) {
        navigate(`${location.pathname}?item=${encodeURIComponent(returnTarget.id)}`, {
          replace: true,
        })
      }
    } catch (nextError) {
      setEditorError(nextError instanceof Error ? nextError.message : 'Could not save changes')
    } finally {
      setSaving(false)
    }
  }

  const remove = async (kind: 'user' | 'team' | 'group' | 'budget', id: string) => {
    try {
      if (kind === 'user') await inferenceAccessApi.deleteUser(id)
      if (kind === 'team') await inferenceAccessApi.deleteTeam(id)
      if (kind === 'group') await inferenceAccessApi.deleteGroup(id)
      if (kind === 'budget') await inferenceAccessApi.deleteBudget(id)
      setToast('Deleted')
      await Promise.all([loadCatalog(false), loadCurrentView()])
      return true
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Delete failed')
      return false
    }
  }

  const invite = () => setInviteOpen(true)

  const openDetail = (kind: 'key' | 'log' | 'item' | 'member', id: string) => {
    navigate(`${location.pathname}?${kind}=${encodeURIComponent(id)}`)
  }

  const closeDetail = () => navigate(location.pathname, { replace: true })

  const managedTeams = currentUser?.managementSelfServicePolicy?.allowTeamKeyDelegation
    ? teams.filter((team) =>
        team.members.some(
          (membership) => membership.userId === selfUserId && membership.role === 'admin',
        ),
      )
    : []
  const personalKeyCount = keys.filter(
    (key) => key.ownerType === 'user' && key.ownerId === selfUserId,
  ).length
  const maxPersonalKeys = currentUser?.managementSelfServicePolicy?.maxKeysPerUser ?? 0
  const hasCreateAction = ['api-keys', 'teams', 'access-groups', 'budgets'].includes(activeView)
  const canCreateCurrent =
    canManage ||
    (selfService &&
      activeView === 'api-keys' &&
      (personalKeyCount < maxPersonalKeys || managedTeams.length > 0))
  const createLabel =
    activeView === 'api-keys'
      ? 'Create key'
      : activeView === 'access-groups'
        ? 'New group'
        : activeView === 'budgets'
          ? 'New budget'
          : 'New team'

  useEffect(() => {
    if (activeView !== 'api-keys' || requestedCreateKind !== 'key') {
      createRequestHandledRef.current = false
      return
    }

    if (loading || viewLoading || createRequestHandledRef.current) {
      return
    }

    createRequestHandledRef.current = true
    const nextParams = new URLSearchParams(location.search)
    nextParams.delete('create')
    nextParams.delete('from')

    if (canCreateCurrent) {
      openCreate('key')
    } else if (selfService && keys[0]) {
      nextParams.set('key', keys[0].id)
      setToast('Your API key is ready to manage')
    }

    const nextSearch = nextParams.toString()
    navigate(`${location.pathname}${nextSearch ? `?${nextSearch}` : ''}`, { replace: true })
  }, [
    activeView,
    canCreateCurrent,
    keys,
    loading,
    location.pathname,
    location.search,
    navigate,
    openCreate,
    requestedCreateKind,
    selfService,
    viewLoading,
  ])

  return (
    <div className={styles.page}>
      <AccessControlWorkspace
        activeView={activeView}
        activeMeta={activeMeta}
        visibleNavItems={visibleNavItems}
        overview={overview}
        liveState={liveState}
        canInvite={activeView === 'users' && canManageDashboardMembers}
        canCreate={canCreateCurrent && hasCreateAction}
        createLabel={createLabel}
        error={error}
        loading={loading}
        toast={toast}
        onCheck={() => void loadCatalog(false)}
        onNavigate={(view) => navigate(view === 'request-logs' ? '/logs' : `/access/${view}`)}
        onInvite={invite}
        onCreate={() => openCreate()}
        onDismissError={() => setError('')}
      >
        <AccessControlViews
          view={activeView}
          overview={overview}
          usage={usage}
          selectors={accessControlSelectorSources}
          users={users}
          teams={teams}
          keys={keys}
          groups={groups}
          budgets={budgets}
          entityTotals={entityTotals}
          dashboardMembers={dashboardMembers}
          invitations={invitations}
          identityTab={identityTab}
          onIdentityTabChange={setIdentityTab}
          requestPage={requestPage}
          auditPage={auditPage}
          pageState={pageState}
          onPageStateChange={setPageState}
          usageScope={usageScope}
          onUsageScopeChange={setUsageScope}
          loading={viewLoading}
          canManage={canManage}
          canManageDashboardMembers={canManageDashboardMembers}
          ownerName={ownerName}
          resourceName={resourceName}
          onOpenKey={(id) => openDetail('key', id)}
          onOpenLog={(id) => openDetail('log', id)}
          onOpenEntity={(id) => openDetail('item', id)}
          onOpenDashboardMember={(id) => openDetail('member', id)}
          onInvitationsChanged={() => void loadDashboardIdentities()}
        />
      </AccessControlWorkspace>
      <AccessControlPageOverlays
        editor={editor}
        createdKey={createdKey}
        inviteOpen={inviteOpen}
        detail={{
          keyId: detailKeyId,
          logId: detailLogId,
          entityId: detailEntityId,
          memberId: detailMemberId,
          entityKind,
        }}
        catalog={{ users, teams, keys, groups, budgets }}
        permissions={{
          canManage,
          canRevealKeys,
          canManageDashboardMembers,
          selfService,
          selfUserId,
        }}
        editorError={editorError}
        saving={saving}
        onEditorChange={setEditor}
        onEditorClose={() => {
          const returnTarget = entityEditorReturn
          setEditor(null)
          setEntityEditorReturn(null)
          setEditorError('')
          if (returnTarget) {
            navigate(`${location.pathname}?item=${encodeURIComponent(returnTarget.id)}`, {
              replace: true,
            })
          }
        }}
        onEditorSave={() => void saveEditor()}
        onCreatedKeyClose={() => setCreatedKey(null)}
        onCreatedKeyDetails={(keyId) => {
          setCreatedKey(null)
          openDetail('key', keyId)
        }}
        onInviteClose={() => setInviteOpen(false)}
        onInviteCreated={() => {
          setToast('Invitation ready')
          void loadDashboardIdentities()
        }}
        onDetailClose={closeDetail}
        onCatalogChanged={() => void Promise.all([loadCatalog(false), loadCurrentView()])}
        onDashboardMembersChanged={() => void loadDashboardIdentities()}
        onEditKey={(key) => {
          setEntityEditorReturn(null)
          setEditor({
            kind: 'key',
            value: key,
            ownerType: key.ownerType,
            rateLimitMode: key.budgetId ? 'budget' : 'inherit',
          })
          closeDetail()
        }}
        onEditEntity={(kind, item) => {
          setEntityEditorReturn({ kind, id: item.id })
          if (kind === 'user') setEditor({ kind, value: item as AccessUser })
          if (kind === 'team') {
            void inferenceAccessApi
              .teamForEdit(item.id)
              .then((team) => setEditor({ kind, value: team }))
              .catch((nextError) =>
                setToast(
                  nextError instanceof Error ? nextError.message : 'Could not load the team.',
                ),
              )
          }
          if (kind === 'group') setEditor({ kind, value: item as AccessGroup })
          if (kind === 'budget') setEditor({ kind, value: item as AccessBudget })
          closeDetail()
        }}
        onDeleteEntity={(kind, id) => {
          void remove(kind, id).then((deleted) => {
            if (deleted) closeDetail()
          })
        }}
        onEditModelAccess={(accessUser) => {
          setEditor({ kind: 'user', value: accessUser })
          closeDetail()
        }}
        resourceName={resourceName}
      />
    </div>
  )
}

export default AccessControlPage
