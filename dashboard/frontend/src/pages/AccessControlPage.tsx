import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import {
  canAccessDashboardPath,
  canManageInferenceAccess,
  canManageUsers,
  canReadInferenceAccess,
  canSelfManageInferenceAccess,
} from '../utils/accessControl'
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
import {
  dashboardMemberInvitationApi,
  type DashboardMemberInvitation,
} from '../utils/dashboardMemberInvitations'
import AccessControlDialog from './AccessControlDialog'
import {
  AccessEntityDetail,
  APIKeyDetail,
  DashboardMemberDetail,
  RequestLogDetail,
} from './AccessControlDetails'
import AccessControlViews, { type DashboardMember, type IdentityTab } from './AccessControlViews'
import DashboardMemberInviteDialog from './DashboardMemberInviteDialog'
import {
  ACCESS_NAV_ITEMS,
  EMPTY_ACCESS_OVERVIEW,
  EMPTY_ACCESS_USAGE,
  accessPageQuery,
  accessRangeStart,
  emptyAccessPage,
  type AccessEditor,
  type AccessView,
} from './AccessControlPageSupport'
import styles from './AccessControlPage.module.css'

type PageState = { page: number; pageSize: number; query: string }
type EntityTotals = Record<'users' | 'teams' | 'api-keys' | 'access-groups' | 'budgets', number>

const AccessControlPage: React.FC = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const { user: currentUser } = useAuth()
  const canManage = canManageInferenceAccess(currentUser)
  const canReadAll = canReadInferenceAccess(currentUser)
  const canSelfManage = canSelfManageInferenceAccess(currentUser)
  const selfService = canSelfManage && !canReadAll && !canManage
  const canManageDashboardMembers = canManageUsers(currentUser)
  const routeView = (
    location.pathname === '/logs'
      ? 'request-logs'
      : location.pathname.split('/').filter(Boolean)[1] || 'usage'
  ) as AccessView
  const activeView = ACCESS_NAV_ITEMS.some((item) => item.id === routeView) ? routeView : 'usage'
  const activeMeta =
    ACCESS_NAV_ITEMS.find((item) => item.id === activeView) ||
    ACCESS_NAV_ITEMS.find((item) => item.id === 'usage')!
  const visibleNavItems = ACCESS_NAV_ITEMS.filter((item) =>
    canAccessDashboardPath(
      currentUser,
      item.id === 'request-logs' ? '/logs' : `/access/${item.id}`,
    ),
  )
  const detailParams = new URLSearchParams(location.search)
  const detailKeyId = activeView === 'api-keys' ? detailParams.get('key') || '' : ''
  const detailLogId = activeView === 'request-logs' ? detailParams.get('log') || '' : ''
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
  const [editorError, setEditorError] = useState('')
  const [saving, setSaving] = useState(false)
  const [createdKey, setCreatedKey] = useState<CreatedAccessAPIKey | null>(null)
  const [inviteOpen, setInviteOpen] = useState(false)
  const [identityTab, setIdentityTab] = useState<IdentityTab>('users')
  const [pageState, setPageState] = useState<PageState>({ page: 1, pageSize: 10, query: '' })
  const [usageScope, setUsageScope] = useState<{
    type: 'global' | 'user' | 'team' | 'key'
    id: string
    model: string
    range: '24h' | '7d' | '30d'
  }>({ type: 'global', id: '', model: '', range: '24h' })
  const [liveState, setLiveState] = useState<'checking' | 'live' | 'error'>('checking')
  const createRequestHandledRef = useRef(false)

  useEffect(() => {
    if (!toast) return
    const timeout = window.setTimeout(() => setToast(''), 3200)
    return () => window.clearTimeout(timeout)
  }, [toast])

  useEffect(() => {
    setPageState((current) => ({ ...current, page: 1, query: '' }))
  }, [activeView])

  const loadDashboardIdentities = useCallback(async () => {
    if (!canManageDashboardMembers) return
    const [members, invitationsResponse] = await Promise.all([
      (async () => {
        const collected: DashboardMember[] = []
        for (let page = 1; page <= 50; page += 1) {
          const response = await fetch(`/api/admin/users?page=${page}&limit=200`)
          if (!response.ok) throw new Error(await response.text())
          const payload = (await response.json()) as { users: DashboardMember[]; total: number }
          collected.push(...(payload.users || []))
          if (collected.length >= payload.total || !payload.users?.length) break
        }
        return collected
      })(),
      dashboardMemberInvitationApi.list(),
    ])
    setDashboardMembers(members)
    setInvitations(invitationsResponse.items)
  }, [canManageDashboardMembers])

  const loadCatalog = useCallback(
    async (showSpinner = true) => {
      if (showSpinner) setLoading(true)
      setError('')
      setLiveState('checking')
      try {
        const [nextOverview, nextKeys, ownTeams] = await Promise.all([
          selfService ? inferenceAccessApi.selfOverview() : inferenceAccessApi.overview(),
          selfService
            ? inferenceAccessApi.selfKeys()
            : inferenceAccessApi.keys(canReadAll ? { limit: 100 } : { limit: 1 }),
          selfService ? inferenceAccessApi.selfTeams() : Promise.resolve({ items: [] }),
        ])
        setOverview(nextOverview)
        setKeys(nextKeys.items)
        setEntityTotals((current) => ({ ...current, 'api-keys': nextKeys.total }))
        if (canReadAll) {
          const [nextUsers, nextTeams, nextGroups, nextBudgets] = await Promise.all([
            inferenceAccessApi.users({ limit: 100 }),
            inferenceAccessApi.teams({ limit: 100 }),
            inferenceAccessApi.groups({ limit: 100 }),
            inferenceAccessApi.budgets({ limit: 100 }),
          ])
          setUsers(nextUsers.items)
          setTeams(nextTeams.items)
          setGroups(nextGroups.items)
          setBudgets(nextBudgets.items)
          setEntityTotals({
            users: nextUsers.total,
            teams: nextTeams.total,
            'api-keys': nextKeys.total,
            'access-groups': nextGroups.total,
            budgets: nextBudgets.total,
          })
        } else {
          setUsers([])
          setTeams(ownTeams.items)
          setGroups([])
          setBudgets([])
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
    [canReadAll, loadDashboardIdentities, selfService],
  )

  useEffect(() => {
    void loadCatalog()
  }, [loadCatalog])

  const usageFilter = useMemo<UsageFilter>(() => {
    const filter: UsageFilter = {
      model: usageScope.model || undefined,
      from: accessRangeStart(usageScope.range),
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
      if (activeView === 'users' && canReadAll) {
        if (pageState.page !== 1) return
        const collected: AccessUser[] = []
        let total = 0
        for (let offset = 0; offset < 10_000; offset += 100) {
          const page = await inferenceAccessApi.users({
            q: pageState.query.trim() || undefined,
            limit: 100,
            offset,
          })
          total = page.total
          collected.push(...page.items)
          if (collected.length >= total || !page.items.length) break
        }
        setUsers(collected)
        setEntityTotals((current) => ({ ...current, users: total }))
      }
      if (activeView === 'teams' && canReadAll) {
        const page = await inferenceAccessApi.teams(accessPageQuery(pageState))
        setTeams(page.items)
        setEntityTotals((current) => ({ ...current, teams: page.total }))
      }
      if (activeView === 'api-keys') {
        const page = selfService
          ? await inferenceAccessApi.selfKeys()
          : await inferenceAccessApi.keys(accessPageQuery(pageState))
        setKeys(page.items)
        setEntityTotals((current) => ({ ...current, 'api-keys': page.total }))
      }
      if (activeView === 'access-groups' && canReadAll) {
        const page = await inferenceAccessApi.groups(accessPageQuery(pageState))
        setGroups(page.items)
        setEntityTotals((current) => ({ ...current, 'access-groups': page.total }))
      }
      if (activeView === 'budgets' && canReadAll) {
        const page = await inferenceAccessApi.budgets(accessPageQuery(pageState))
        setBudgets(page.items)
        setEntityTotals((current) => ({ ...current, budgets: page.total }))
      }
      if (activeView === 'request-logs') {
        setRequestPage(
          await (selfService
            ? inferenceAccessApi.selfRequestLogs({ ...usageFilter, ...accessPageQuery(pageState) })
            : inferenceAccessApi.requestLogs({ ...usageFilter, ...accessPageQuery(pageState) })),
        )
      }
      if (activeView === 'audit-logs') {
        setAuditPage(await inferenceAccessApi.auditLogs(accessPageQuery(pageState)))
      }
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not load this view')
    } finally {
      setViewLoading(false)
    }
  }, [activeView, canReadAll, pageState, selfService, usageFilter])

  useEffect(() => {
    void loadCurrentView()
  }, [loadCurrentView])

  const ownerName = useCallback(
    (item: Pick<AccessAPIKey, 'userId' | 'teamId'>) =>
      item.userId
        ? users.find((user) => user.id === item.userId)?.name || item.userId
        : teams.find((team) => team.id === item.teamId)?.name || item.teamId || 'Unassigned',
    [teams, users],
  )

  const openCreate = useCallback(
    (kind?: AccessEditor['kind']) => {
      setEditorError('')
      const target =
        kind ||
        (activeView === 'api-keys'
          ? 'key'
          : activeView === 'users'
            ? 'user'
            : activeView === 'teams'
              ? 'team'
              : activeView === 'access-groups'
                ? 'group'
                : 'budget')
      if (target === 'user') setEditor({ kind: 'user', value: { status: 'active' } })
      if (target === 'team') setEditor({ kind: 'team', value: { status: 'active', userIds: [] } })
      if (target === 'key')
        setEditor({
          kind: 'key',
          ownerType: 'user',
          value: {
            status: 'active',
            accessGroupIds: [],
            budget: { rpm: 0, tpm: 0, dailyTokens: 0 },
          },
        })
      if (target === 'group')
        setEditor({ kind: 'group', value: { modelPatterns: ['vllm-sr/mom-*'], bindings: [] } })
      if (target === 'budget')
        setEditor({
          kind: 'budget',
          value: {
            scopeType: 'global',
            scopeId: '',
            rpm: 60,
            tpm: 100000,
            dailyTokens: 1000000,
            enabled: true,
          },
        })
    },
    [activeView],
  )

  const saveEditor = async () => {
    if (!editor) return
    setSaving(true)
    setEditorError('')
    try {
      if (editor.kind === 'user') await inferenceAccessApi.saveUser(editor.value)
      if (editor.kind === 'team') await inferenceAccessApi.saveTeam(editor.value)
      if (editor.kind === 'group') await inferenceAccessApi.saveGroup(editor.value)
      if (editor.kind === 'budget') await inferenceAccessApi.saveBudget(editor.value)
      if (editor.kind === 'key') {
        const value = { ...editor.value }
        if (editor.ownerType === 'user') value.teamId = undefined
        else value.userId = undefined
        if (value.id) {
          await inferenceAccessApi.saveKey(value as Partial<AccessAPIKey> & { id: string })
        } else {
          setCreatedKey(
            selfService
              ? await inferenceAccessApi.createSelfKey(value.name || 'My API key')
              : await inferenceAccessApi.createKey(value),
          )
        }
      }
      setEditor(null)
      setToast('Saved')
      await loadCatalog(false)
    } catch (nextError) {
      setEditorError(nextError instanceof Error ? nextError.message : 'Could not save changes')
    } finally {
      setSaving(false)
    }
  }

  const remove = async (kind: 'user' | 'team' | 'group' | 'budget', id: string, label: string) => {
    if (!window.confirm(`Delete ${label}? This cannot be undone.`)) return false
    try {
      if (kind === 'user') await inferenceAccessApi.deleteUser(id)
      if (kind === 'team') await inferenceAccessApi.deleteTeam(id)
      if (kind === 'group') await inferenceAccessApi.deleteGroup(id)
      if (kind === 'budget') await inferenceAccessApi.deleteBudget(id)
      setToast('Deleted')
      await loadCatalog(false)
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

  const hasCreateAction = ['api-keys', 'users', 'teams', 'access-groups', 'budgets'].includes(
    activeView,
  )
  const canCreateCurrent =
    canManage || (selfService && activeView === 'api-keys' && entityTotals['api-keys'] === 0)
  const createLabel =
    activeView === 'users'
      ? 'New user'
      : activeView === 'api-keys'
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
      <header className={styles.hero}>
        <div className={styles.heroCopy}>
          <div className={styles.heroTopline}>
            <span className={styles.eyebrow}>Access Control</span>
            <span className={styles.heroBrand}>
              <img src="/vllm.png" alt="" />
              vllm-sr
            </span>
          </div>
          <h1>Every model. The right audience.</h1>
          <p>Give users and teams exactly the models and capacity they need.</p>
        </div>
        <div className={styles.heroPulse}>
          <button
            type="button"
            className={`${styles.liveButton} ${styles[`live${liveState}`]}`}
            onClick={() => void loadCatalog(false)}
            aria-label="Check access-control service"
          >
            <span />{' '}
            {liveState === 'checking' ? 'Checking' : liveState === 'live' ? 'Live' : 'Retry'}
          </button>
          <div>
            <strong>{overview.requestsToday.toLocaleString('en-US')}</strong>
            <span>requests today</span>
          </div>
          <div>
            <strong>{overview.tokensToday.toLocaleString('en-US')}</strong>
            <span>tokens today</span>
          </div>
        </div>
      </header>

      <nav className={styles.sectionNav} aria-label="Access control">
        {visibleNavItems.map((item) => (
          <button
            type="button"
            key={item.id}
            className={activeView === item.id ? styles.sectionNavActive : ''}
            onClick={() => navigate(item.id === 'request-logs' ? '/logs' : `/access/${item.id}`)}
            aria-current={activeView === item.id ? 'page' : undefined}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <main className={styles.surface}>
        <div className={styles.surfaceHeader}>
          <div>
            <span>{activeMeta.section}</span>
            <h2>{activeMeta.label}</h2>
            <p>{activeMeta.description}</p>
          </div>
          <div className={styles.headerActions}>
            {activeView === 'users' && canManageDashboardMembers ? (
              <button type="button" className={styles.secondaryButton} onClick={() => invite()}>
                Invite user
              </button>
            ) : null}
            {canCreateCurrent && hasCreateAction ? (
              <button type="button" className={styles.primaryButton} onClick={() => openCreate()}>
                <span aria-hidden="true">＋</span> {createLabel}
              </button>
            ) : null}
          </div>
        </div>

        {error ? (
          <div className={styles.inlineError} role="alert">
            {error}
            <button type="button" onClick={() => setError('')}>
              Dismiss
            </button>
          </div>
        ) : null}
        {loading ? (
          <div className={styles.skeletonGrid}>
            <i />
            <i />
            <i />
            <i />
          </div>
        ) : null}
        {!loading ? (
          <AccessControlViews
            view={activeView}
            overview={overview}
            usage={usage}
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
            onOpenKey={(id) => openDetail('key', id)}
            onOpenLog={(id) => openDetail('log', id)}
            onOpenEntity={(id) => openDetail('item', id)}
            onOpenDashboardMember={(id) => openDetail('member', id)}
            onInvitationsChanged={() => void loadDashboardIdentities()}
          />
        ) : null}
      </main>

      {toast ? (
        <div className={styles.toast} role="status">
          <span>✓</span>
          {toast}
        </div>
      ) : null}
      {editor ? (
        <AccessControlDialog
          editor={editor}
          users={users}
          teams={teams}
          keys={keys}
          groups={groups}
          budgets={budgets}
          selfService={selfService}
          error={editorError}
          saving={saving}
          onChange={setEditor}
          onClose={() => {
            setEditor(null)
            setEditorError('')
          }}
          onSave={() => void saveEditor()}
        />
      ) : null}
      {createdKey ? (
        <AccessControlDialog secret={createdKey} onClose={() => setCreatedKey(null)} />
      ) : null}
      <DashboardMemberInviteDialog
        isOpen={inviteOpen}
        roleOptions={['admin', 'write', 'read']}
        teams={teams}
        onClose={() => {
          setInviteOpen(false)
        }}
        onCreated={() => {
          setToast('Invitation ready')
          void loadDashboardIdentities()
        }}
      />
      {detailKeyId ? (
        <APIKeyDetail
          keyId={detailKeyId}
          users={users}
          teams={teams}
          groups={groups}
          budgets={budgets}
          canManage={canManage || selfService}
          canEditPolicy={canManage}
          selfService={selfService}
          onEdit={(key) => {
            setEditor({
              kind: 'key',
              value: key,
              ownerType: key.userId ? 'user' : 'team',
            })
            closeDetail()
          }}
          onClose={closeDetail}
          onChanged={() => void loadCatalog(false)}
        />
      ) : null}
      {detailLogId ? (
        <RequestLogDetail
          logId={detailLogId}
          users={users}
          teams={teams}
          keys={keys}
          selfService={selfService}
          onClose={closeDetail}
        />
      ) : null}
      {detailEntityId && entityKind ? (
        <AccessEntityDetail
          kind={entityKind}
          id={detailEntityId}
          users={users}
          teams={teams}
          keys={keys}
          canManage={canManage}
          onEdit={(kind, item) => {
            if (kind === 'user') setEditor({ kind, value: item as AccessUser })
            if (kind === 'team') setEditor({ kind, value: item as AccessTeam })
            if (kind === 'group') setEditor({ kind, value: item as AccessGroup })
            if (kind === 'budget') setEditor({ kind, value: item as AccessBudget })
            closeDetail()
          }}
          onDelete={(kind, id, label) => {
            void remove(kind, id, label).then((deleted) => {
              if (deleted) closeDetail()
            })
          }}
          onClose={closeDetail}
        />
      ) : null}
      {detailMemberId ? (
        <DashboardMemberDetail
          memberId={detailMemberId}
          canManage={canManageDashboardMembers}
          onChanged={() => void loadDashboardIdentities()}
          onClose={closeDetail}
        />
      ) : null}
    </div>
  )
}

export default AccessControlPage
