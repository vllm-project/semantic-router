import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from 'react'
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
  type UsageFilter,
  type UsageSummary,
} from '../utils/inferenceAccessApi'
import {
  omitDeletedAccessEntities,
  type AccessEntityDeletionTombstones,
} from './accessEntityDeletionState'
import { loadAllAccessUsers } from './accessIdentityDirectory'
import { accessControlSelectorSources } from './accessControlSelectorSources'
import { accessPageQuery, type AccessView } from './AccessControlPageSupport'
import type { AccessControlPageState, AccessControlViewProps } from './AccessControlViewTypes'
import { usageRangeBounds, type UsageScope } from './accessControlUsageRange'

type EntityTotals = AccessControlViewProps['entityTotals']

interface CurrentViewOptions {
  activeView: AccessView
  selfService: boolean
  canReadUsers: boolean
  canReadTeams: boolean
  canReadGroups: boolean
  canReadBudgets: boolean
  canReadInternalUsageDimensions: boolean
  usageScope: UsageScope
  pageState: AccessControlPageState
  pageCursors: Record<string, Record<number, string>>
  deletionTombstonesRef: MutableRefObject<AccessEntityDeletionTombstones>
  userDirectoryLoadedRef: MutableRefObject<boolean>
  rememberNextCursor: (view: AccessView, page: AccessPage<unknown>) => void
  displayTotal: (page: AccessPage<unknown>) => number
  setUsage: Dispatch<SetStateAction<UsageSummary>>
  setUsers: Dispatch<SetStateAction<AccessUser[]>>
  setTeams: Dispatch<SetStateAction<AccessTeam[]>>
  setKeys: Dispatch<SetStateAction<AccessAPIKey[]>>
  setGroups: Dispatch<SetStateAction<AccessGroup[]>>
  setBudgets: Dispatch<SetStateAction<AccessBudget[]>>
  setEntityTotals: Dispatch<SetStateAction<EntityTotals>>
  setRequestPage: Dispatch<SetStateAction<AccessPage<AccessUsageEvent>>>
  setAuditPage: Dispatch<SetStateAction<AccessPage<AccessAuditEvent>>>
  setViewLoading: Dispatch<SetStateAction<boolean>>
  setError: Dispatch<SetStateAction<string>>
}

export const useAccessControlCurrentView = ({
  activeView,
  selfService,
  canReadUsers,
  canReadTeams,
  canReadGroups,
  canReadBudgets,
  canReadInternalUsageDimensions,
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
}: CurrentViewOptions) => {
  const viewLoadGenerationRef = useRef(0)
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

  const loadCurrentView = useCallback(
    async (options: { refreshUserDirectory?: boolean } = {}) => {
      if (
        activeView === 'users' &&
        userDirectoryLoadedRef.current &&
        !options.refreshUserDirectory
      ) {
        return
      }
      const generation = ++viewLoadGenerationRef.current
      const isLatest = () => generation === viewLoadGenerationRef.current
      setViewLoading(true)
      setError('')
      try {
        if (activeView === 'usage') {
          const nextUsage = await (
            selfService ? inferenceAccessApi.selfUsage : inferenceAccessApi.usage
          )(usageFilter, { internalDimensions: canReadInternalUsageDimensions })
          if (!isLatest()) return
          setUsage(nextUsage)
        }
        if (
          activeView === 'users' &&
          canReadUsers &&
          !selfService &&
          (!userDirectoryLoadedRef.current || options.refreshUserDirectory)
        ) {
          const nextUsers = omitDeletedAccessEntities(
            deletionTombstonesRef.current,
            'user',
            await loadAllAccessUsers(inferenceAccessApi.users),
          )
          if (!isLatest()) return
          setUsers(nextUsers)
          setEntityTotals((current) => ({ ...current, users: nextUsers.length }))
          userDirectoryLoadedRef.current = true
        }
        if (activeView === 'teams' && canReadTeams && !selfService) {
          const page = await inferenceAccessApi.teams(
            accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
          )
          if (!isLatest()) return
          const visibleItems = omitDeletedAccessEntities(
            deletionTombstonesRef.current,
            'team',
            page.items,
          )
          setTeams(visibleItems)
          rememberNextCursor(activeView, page)
          setEntityTotals((current) => ({
            ...current,
            teams: displayTotal({ ...page, items: visibleItems }),
          }))
        }
        if (activeView === 'api-keys') {
          const page = selfService
            ? await inferenceAccessApi.selfKeys(
                accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
              )
            : await inferenceAccessApi.keys(
                accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
              )
          if (!isLatest()) return
          const visibleItems = omitDeletedAccessEntities(
            deletionTombstonesRef.current,
            'key',
            page.items,
          )
          setKeys(visibleItems)
          rememberNextCursor(activeView, page)
          setEntityTotals((current) => ({
            ...current,
            'api-keys': displayTotal({ ...page, items: visibleItems }),
          }))
        }
        if (activeView === 'access-groups' && canReadGroups && !selfService) {
          const page = await inferenceAccessApi.groups(
            accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
          )
          if (!isLatest()) return
          const visibleItems = omitDeletedAccessEntities(
            deletionTombstonesRef.current,
            'group',
            page.items,
          )
          setGroups(visibleItems)
          rememberNextCursor(activeView, page)
          setEntityTotals((current) => ({
            ...current,
            'access-groups': displayTotal({ ...page, items: visibleItems }),
          }))
        }
        if (activeView === 'budgets' && canReadBudgets && !selfService) {
          const page = await inferenceAccessApi.budgets(
            accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
          )
          if (!isLatest()) return
          const visibleItems = omitDeletedAccessEntities(
            deletionTombstonesRef.current,
            'budget',
            page.items,
          )
          setBudgets(visibleItems)
          rememberNextCursor(activeView, page)
          setEntityTotals((current) => ({
            ...current,
            budgets: displayTotal({ ...page, items: visibleItems }),
          }))
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
          if (!isLatest()) return
          setRequestPage({ ...page, total: displayTotal(page) })
          rememberNextCursor(activeView, page)
        }
        if (activeView === 'audit-logs') {
          const page = await inferenceAccessApi.auditLogs(
            accessPageQuery(pageState, pageCursors[activeView]?.[pageState.page]),
          )
          if (!isLatest()) return
          setAuditPage({ ...page, total: displayTotal(page) })
          rememberNextCursor(activeView, page)
        }
      } catch (nextError) {
        if (isLatest()) {
          setError(nextError instanceof Error ? nextError.message : 'Could not load this view')
        }
      } finally {
        if (isLatest()) setViewLoading(false)
      }
    },
    [
      activeView,
      canReadBudgets,
      canReadGroups,
      canReadInternalUsageDimensions,
      canReadTeams,
      canReadUsers,
      deletionTombstonesRef,
      displayTotal,
      pageCursors,
      pageState,
      rememberNextCursor,
      selfService,
      setAuditPage,
      setBudgets,
      setEntityTotals,
      setError,
      setGroups,
      setKeys,
      setRequestPage,
      setTeams,
      setUsage,
      setUsers,
      setViewLoading,
      usageFilter,
      userDirectoryLoadedRef,
    ],
  )

  useEffect(() => {
    void loadCurrentView()
  }, [loadCurrentView])

  return loadCurrentView
}

interface AccessControlLabelOptions {
  activeView: AccessView
  selfService: boolean
  keys: AccessAPIKey[]
  groups: AccessGroup[]
  users: AccessUser[]
  teams: AccessTeam[]
}

export const useAccessControlLabels = ({
  activeView,
  selfService,
  keys,
  groups,
  users,
  teams,
}: AccessControlLabelOptions) => {
  const [ownerLabels, setOwnerLabels] = useState<Record<string, string>>({})
  const [resourceLabels, setResourceLabels] = useState<Record<string, string>>({})

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
      resourceLabels[`${resourceType}:${resourceId}`] ||
      (resourceType === 'model' ? 'Model name unavailable' : 'Mixture-of-Model name unavailable'),
    [resourceLabels],
  )

  return { ownerName, resourceName }
}
