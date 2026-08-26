import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import {
  inferenceAccessApi,
  type AccessAPIKey,
  type AccessBudget,
  type AccessGroup,
  type AccessTeam,
  type AccessUser,
  type CreatedAccessAPIKey,
} from '../utils/inferenceAccessApi'
import { claimInvitationOnboarding } from '../utils/invitationOnboarding'
import { ManagementApiError } from '../utils/managementApiContract'
import { canReadInternalUsageDimensions } from '../utils/accessControl'
import AccessControlViews, { type IdentityTab } from './AccessControlViews'
import AccessControlPageOverlays from './AccessControlPageOverlays'
import AccessControlWorkspace from './AccessControlWorkspace'
import {
  rememberDeletedAccessEntity,
  type DeletableAccessEntityKind,
} from './accessEntityDeletionState'
import { accessControlSelectorSources } from './accessControlSelectorSources'
import { resolveAccessControlPage, type AccessEditor } from './AccessControlPageSupport'
import { deleteUnifiedUser, type UnifiedUserDeletionProgress } from './unifiedUserDeletion'
import { useAccessControlDirectory, type EntityTotals } from './useAccessControlDirectory'
import { useAccessControlLabels } from './useAccessControlViewData'
import styles from './AccessControlPage.module.css'

type EntityEditorReturn = { kind: Exclude<AccessEditor['kind'], 'key'>; id: string }
type RouterEntityKind = Exclude<DeletableAccessEntityKind, 'dashboard-member' | 'key'>

const dashboardResponseError = async (response: Response, fallback: string) =>
  (await response.text()).trim() || fallback

const deleteDashboardLogin = async (memberId: string) => {
  const response = await fetch(`/api/admin/users/${encodeURIComponent(memberId)}`, {
    method: 'DELETE',
  })
  // A retry after the first step completed is intentionally idempotent.
  if (!response.ok && response.status !== 404) {
    throw new Error(await dashboardResponseError(response, 'Could not remove Dashboard login'))
  }
}

const deleteRouterEntityIfPresent = async (action: () => Promise<unknown>) => {
  try {
    await action()
  } catch (error) {
    // DELETE retries are successful when a previous attempt already removed the resource.
    if (error instanceof ManagementApiError && error.status === 404) return
    throw error
  }
}

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

  const [toast, setToast] = useState('')
  const [editor, setEditor] = useState<AccessEditor | null>(null)
  const [entityEditorReturn, setEntityEditorReturn] = useState<EntityEditorReturn | null>(null)
  const [editorError, setEditorError] = useState('')
  const [saving, setSaving] = useState(false)
  const [createdKey, setCreatedKey] = useState<CreatedAccessAPIKey | null>(null)
  const [inviteOpen, setInviteOpen] = useState(false)
  const [identityTab, setIdentityTab] = useState<IdentityTab>('users')
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

  const {
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
  } = useAccessControlDirectory({
    activeView,
    requestedPageQuery,
    currentUserEmail: currentUser?.email,
    currentUserName: currentUser?.name,
    selfUserId,
    selfService,
    canManageDashboardMembers,
    canReadDashboardMembers,
    canReadUsers,
    canReadTeams,
    canReadGroups,
    canReadBudgets,
    canReadInternalUsageDimensions: canReadInternalUsageDimensions(currentUser),
  })
  const { ownerName, resourceName } = useAccessControlLabels({
    activeView,
    selfService,
    keys,
    groups,
    users,
    teams,
  })

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
        userDirectoryLoadedRef.current = false
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

  const removeLocalEntity = (kind: RouterEntityKind, id: string) => {
    rememberDeletedAccessEntity(deletionTombstonesRef.current, kind, id)
    if (kind === 'user') setUsers((current) => current.filter((item) => item.id !== id))
    if (kind === 'team') setTeams((current) => current.filter((item) => item.id !== id))
    if (kind === 'group') setGroups((current) => current.filter((item) => item.id !== id))
    if (kind === 'budget') setBudgets((current) => current.filter((item) => item.id !== id))
    const totalKey: keyof EntityTotals =
      kind === 'user'
        ? 'users'
        : kind === 'team'
          ? 'teams'
          : kind === 'group'
            ? 'access-groups'
            : 'budgets'
    setEntityTotals((current) => ({
      ...current,
      [totalKey]: Math.max(0, current[totalKey] - 1),
    }))
  }

  const refreshAfterDelete = () => {
    void Promise.all([loadCatalog(false), loadCurrentView({ refreshUserDirectory: true })])
  }

  const remove = async (kind: RouterEntityKind, id: string) => {
    await deleteRouterEntityIfPresent(() => {
      if (kind === 'user') return inferenceAccessApi.deleteUser(id)
      if (kind === 'team') return inferenceAccessApi.deleteTeam(id)
      if (kind === 'group') return inferenceAccessApi.deleteGroup(id)
      return inferenceAccessApi.deleteBudget(id)
    })
    removeLocalEntity(kind, id)
    setToast('Deleted')
    refreshAfterDelete()
  }

  const removeLogin = async (memberId: string) => {
    await deleteDashboardLogin(memberId)
    rememberDeletedAccessEntity(deletionTombstonesRef.current, 'dashboard-member', memberId)
    setDashboardMembers((current) => current.filter((member) => member.id !== memberId))
    setToast('Login removed')
    void loadDashboardIdentities().catch(() => undefined)
  }

  const removeKeyLocally = (keyId: string) => {
    rememberDeletedAccessEntity(deletionTombstonesRef.current, 'key', keyId)
    setKeys((current) => current.filter((key) => key.id !== keyId))
    closeDetail()
    setToast('API key deleted')
    void Promise.all([loadCatalog(false), loadCurrentView()])
  }

  const removeUnifiedUser = async (
    memberId: string,
    modelUserId: string,
    progress: UnifiedUserDeletionProgress,
  ) => {
    const completed = await deleteUnifiedUser(progress, {
      removeDashboardLogin: async () => {
        await deleteDashboardLogin(memberId)
        rememberDeletedAccessEntity(deletionTombstonesRef.current, 'dashboard-member', memberId)
        setDashboardMembers((current) => current.filter((member) => member.id !== memberId))
      },
      deleteModelIdentity: () =>
        deleteRouterEntityIfPresent(() => inferenceAccessApi.deleteUser(modelUserId)),
    })
    removeLocalEntity('user', modelUserId)
    setToast('User deleted')
    refreshAfterDelete()
    void loadDashboardIdentities().catch(() => undefined)
    return completed
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
        onDashboardMembersChanged={() => void loadDashboardIdentities().catch(() => undefined)}
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
        onDeleteEntity={remove}
        onDeleteKey={removeKeyLocally}
        onRemoveDashboardLogin={removeLogin}
        onDeleteUnifiedUser={removeUnifiedUser}
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
