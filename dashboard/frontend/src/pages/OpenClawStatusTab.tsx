import React, { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'

import ConfirmDialog from '../components/ConfirmDialog'
import type { RoomBridgeEnvelope } from '../components/openclawRoomBridge'
import {
  listenForSurfaceEvents,
  postRoomContextToFrame,
  postRoomEventToFrame,
  subscribeRoomEvents,
} from '../components/openclawRoomBridge'
import type { WSOutboundMessage } from '../components/clawRoomChatSupport'
import {
  filterAndSortOpenClawWorkers,
  getOpenClawPageCount,
  getOpenClawWorkerHealth,
  OPENCLAW_CONTAINERS_PAGE_SIZE,
  paginateOpenClawItems,
  type WorkerCatalogSort,
  type WorkerHealthFilter,
} from '../utils/openClawCatalogSupport'
import {
  createLatestOpenClawRequest,
  fetchOpenClawJSON,
  getOpenClawErrorMessage,
  type LatestOpenClawRequest,
} from '../utils/openClawRequestSupport'
import { OpenClawCatalogControls } from './OpenClawCatalogControls'
import styles from './OpenClawPage.module.css'
import { truncateText, type OpenClawStatus } from './OpenClawPageSupport'
import { OpenClawRequestNotice } from './OpenClawRequestNotice'

interface StatusTabProps {
  containers: OpenClawStatus[]
  readOnly: boolean
  statusError?: string | null
  statusLoading: boolean
  onRefresh: () => void
}

type LifecycleAction = 'start' | 'stop'

interface RetryLifecycleAction {
  action: LifecycleAction
  name: string
}

export const StatusTab: React.FC<StatusTabProps> = ({
  containers,
  statusLoading,
  statusError,
  onRefresh,
  readOnly,
}) => {
  const [search, setSearch] = useState('')
  const deferredSearch = useDeferredValue(search)
  const [healthFilter, setHealthFilter] = useState<WorkerHealthFilter>('all')
  const [teamFilter, setTeamFilter] = useState('all')
  const [sort, setSort] = useState<WorkerCatalogSort>('name-asc')
  const [page, setPage] = useState(1)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [actionError, setActionError] = useState('')
  const [retryAction, setRetryAction] = useState<RetryLifecycleAction | null>(null)
  const [deleteError, setDeleteError] = useState('')
  const [selectedContainer, setSelectedContainer] = useState<string | null>(null)
  const [pendingDeleteContainer, setPendingDeleteContainer] = useState<string | null>(null)
  const [gatewayToken, setGatewayToken] = useState('')
  const [tokenLoading, setTokenLoading] = useState(false)
  const [tokenError, setTokenError] = useState('')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [linkedRoomId, setLinkedRoomId] = useState('')
  const [linkedRoomLoading, setLinkedRoomLoading] = useState(false)
  const [linkedRoomError, setLinkedRoomError] = useState('')
  const [roomBridgeEvents, setRoomBridgeEvents] = useState<WSOutboundMessage[]>([])
  const [surfaceEvents, setSurfaceEvents] = useState<RoomBridgeEnvelope[]>([])
  const iframeContainerRef = useRef<HTMLDivElement | null>(null)
  const iframeRef = useRef<HTMLIFrameElement | null>(null)
  const tokenRequestRef = useRef<LatestOpenClawRequest | null>(null)
  const roomRequestRef = useRef<LatestOpenClawRequest | null>(null)
  const mutationRequestRef = useRef<LatestOpenClawRequest | null>(null)
  if (!tokenRequestRef.current) tokenRequestRef.current = createLatestOpenClawRequest()
  if (!roomRequestRef.current) roomRequestRef.current = createLatestOpenClawRequest()
  if (!mutationRequestRef.current) mutationRequestRef.current = createLatestOpenClawRequest()

  const selected = containers.find((container) => container.containerName === selectedContainer)
  const filteredContainers = useMemo(
    () => filterAndSortOpenClawWorkers(containers, deferredSearch, healthFilter, teamFilter, sort),
    [containers, deferredSearch, healthFilter, sort, teamFilter],
  )
  const pageCount = getOpenClawPageCount(filteredContainers.length, OPENCLAW_CONTAINERS_PAGE_SIZE)
  const safePage = Math.min(page, pageCount)
  const visibleContainers = paginateOpenClawItems(
    filteredContainers,
    safePage,
    OPENCLAW_CONTAINERS_PAGE_SIZE,
  )
  const teamOptions = useMemo(() => {
    const entries = new Map<string, string>()
    for (const container of containers) {
      if (container.teamId?.trim()) {
        entries.set(container.teamId, container.teamName?.trim() || container.teamId)
      }
    }
    return [...entries.entries()].sort((left, right) => left[1].localeCompare(right[1]))
  }, [containers])

  useEffect(() => setPage(1), [deferredSearch, healthFilter, sort, teamFilter])
  useEffect(() => {
    if (page > pageCount) setPage(pageCount)
  }, [page, pageCount])
  useEffect(() => {
    const onFullscreenChange = () => setIsFullscreen(Boolean(document.fullscreenElement))
    document.addEventListener('fullscreenchange', onFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', onFullscreenChange)
  }, [])
  useEffect(
    () => () => {
      tokenRequestRef.current?.cancel()
      roomRequestRef.current?.cancel()
      mutationRequestRef.current?.cancel()
    },
    [],
  )
  useEffect(() => {
    if (!readOnly) return
    mutationRequestRef.current?.cancel()
    setSelectedContainer(null)
    setPendingDeleteContainer(null)
    setActionLoading(null)
    setActionError('')
    setRetryAction(null)
    setDeleteError('')
  }, [readOnly])

  const loadGatewayToken = useCallback(async () => {
    if (!selectedContainer || !selected?.healthy) return
    await tokenRequestRef.current?.run(
      (signal) =>
        fetchOpenClawJSON<{ token?: string }>(
          `/api/openclaw/token?name=${encodeURIComponent(selectedContainer)}`,
          {},
          signal,
        ),
      {
        onStart: () => {
          setGatewayToken('')
          setTokenLoading(true)
          setTokenError('')
        },
        onSuccess: (data) => {
          if (!data.token) throw new Error('The OpenClaw gateway returned no access token.')
          setGatewayToken(data.token)
        },
        onError: (error) => {
          setTokenError(getOpenClawErrorMessage(error, 'Failed to connect to the gateway.'))
        },
        onFinish: () => setTokenLoading(false),
      },
    )
  }, [selected?.healthy, selectedContainer])

  const loadLinkedRoom = useCallback(async () => {
    if (!selected?.teamId) {
      roomRequestRef.current?.cancel()
      setLinkedRoomId('')
      setLinkedRoomError('')
      return
    }
    await roomRequestRef.current?.run(
      (signal) =>
        fetchOpenClawJSON<Array<{ id?: string }>>(
          `/api/openclaw/rooms?teamId=${encodeURIComponent(selected.teamId || '')}`,
          {},
          signal,
        ),
      {
        onStart: () => {
          setLinkedRoomLoading(true)
          setLinkedRoomError('')
        },
        onSuccess: (rooms) => {
          setLinkedRoomId(Array.isArray(rooms) && rooms[0]?.id ? rooms[0].id : '')
        },
        onError: (error) => {
          setLinkedRoomId('')
          setLinkedRoomError(
            getOpenClawErrorMessage(error, 'Failed to load room collaboration context.'),
          )
        },
        onFinish: () => setLinkedRoomLoading(false),
      },
    )
  }, [selected?.teamId])

  useEffect(() => {
    if (selected?.healthy && selectedContainer) void loadGatewayToken()
    else {
      tokenRequestRef.current?.cancel()
      setGatewayToken('')
      setTokenError('')
    }
  }, [loadGatewayToken, selected?.healthy, selectedContainer])

  useEffect(() => {
    void loadLinkedRoom()
  }, [loadLinkedRoom])

  useEffect(() => {
    if (!linkedRoomId || !selectedContainer || !selected?.healthy) {
      setRoomBridgeEvents([])
      setSurfaceEvents([])
      return
    }
    const subscription = subscribeRoomEvents(linkedRoomId, (event) => {
      setRoomBridgeEvents((previous) => [...previous.slice(-19), event])
      if (iframeRef.current) postRoomEventToFrame(iframeRef.current, linkedRoomId, event)
    })
    const unsubscribeSurface = listenForSurfaceEvents(
      linkedRoomId,
      (_roomId, payload) => {
        subscription.sendSurfaceEvent(payload, {
          senderType: 'worker',
          senderId: selectedContainer,
          senderName: selectedContainer,
        })
        setSurfaceEvents((previous) => [
          ...previous.slice(-9),
          { source: 'clawos-room-bridge', type: 'surface_event', roomId: linkedRoomId, payload },
        ])
      },
      () => iframeRef.current?.contentWindow ?? null,
    )
    return () => {
      unsubscribeSurface()
      subscription.close()
    }
  }, [linkedRoomId, selected?.healthy, selectedContainer])

  const latestRoomBridgeEvent = useMemo(() => {
    const last = roomBridgeEvents[roomBridgeEvents.length - 1]
    if (!last) return ''
    return last.message?.content || last.chunk || last.type
  }, [roomBridgeEvents])

  const runLifecycleAction = async (action: LifecycleAction, name: string) => {
    if (readOnly) return
    await mutationRequestRef.current?.run(
      (signal) =>
        fetchOpenClawJSON<void>(
          `/api/openclaw/${action}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ containerName: name }),
          },
          signal,
        ),
      {
        onStart: () => {
          setActionLoading(name)
          setActionError('')
          setRetryAction(null)
        },
        onSuccess: onRefresh,
        onError: (error) => {
          setActionError(getOpenClawErrorMessage(error, `Failed to ${action} ${name}.`))
          setRetryAction({ action, name })
        },
        onFinish: () => setActionLoading(null),
      },
    )
  }

  const confirmDelete = async () => {
    const name = pendingDeleteContainer
    if (readOnly || !name) return
    await mutationRequestRef.current?.run(
      (signal) =>
        fetchOpenClawJSON<void>(
          `/api/openclaw/containers/${encodeURIComponent(name)}`,
          { method: 'DELETE' },
          signal,
        ),
      {
        onStart: () => {
          setActionLoading(name)
          setDeleteError('')
        },
        onSuccess: () => {
          if (selectedContainer === name) setSelectedContainer(null)
          setPendingDeleteContainer(null)
          onRefresh()
        },
        onError: (error) => {
          setDeleteError(getOpenClawErrorMessage(error, `Failed to remove ${name}.`))
        },
        onFinish: () => setActionLoading(null),
      },
    )
  }

  if (statusLoading && containers.length === 0) {
    return (
      <div className={styles.loading} role="status">
        <div className={styles.spinner} />
        <p>Checking OpenClaw containers…</p>
      </div>
    )
  }

  if (!readOnly && selectedContainer && selected?.healthy) {
    if (tokenLoading) {
      return (
        <div className={styles.loading} role="status">
          Connecting to gateway…
        </div>
      )
    }
    if (tokenError || !gatewayToken) {
      return (
        <div className={styles.embeddedFailure}>
          <OpenClawRequestNotice
            title="Gateway connection failed"
            message={tokenError || 'The gateway returned no access token.'}
            onRetry={() => void loadGatewayToken()}
          />
          <button
            type="button"
            className={styles.btnSecondary}
            onClick={() => setSelectedContainer(null)}
          >
            Back to containers
          </button>
        </div>
      )
    }

    const proxyBase = `/embedded/openclaw/${encodeURIComponent(selectedContainer)}/`
    const gatewayUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}${proxyBase}`
    const roomContext = linkedRoomId ? `&roomId=${encodeURIComponent(linkedRoomId)}` : ''
    const iframeSrc = `${proxyBase}#gatewayUrl=${encodeURIComponent(gatewayUrl)}&token=${encodeURIComponent(gatewayToken)}${roomContext}`
    const openInNewTab = () => window.open(iframeSrc, '_blank', 'noopener,noreferrer')
    const toggleFullscreen = async () => {
      try {
        if (!document.fullscreenElement && iframeContainerRef.current?.requestFullscreen) {
          await iframeContainerRef.current.requestFullscreen()
          return
        }
        if (document.fullscreenElement) {
          await document.exitFullscreen()
          return
        }
      } catch {
        openInNewTab()
        return
      }
      openInNewTab()
    }

    return (
      <div>
        <div className={styles.embeddedHeader}>
          <div className={styles.embeddedHeaderLeft}>
            <button
              type="button"
              className={styles.btnSecondary}
              onClick={() => setSelectedContainer(null)}
            >
              ← Back to containers
            </button>
            <span className={styles.embeddedContainerMeta}>
              {selected.containerName} · port {selected.port}
            </span>
          </div>
          <button
            type="button"
            className={styles.btnSecondary}
            onClick={() => void toggleFullscreen()}
          >
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </button>
        </div>
        {linkedRoomError ? (
          <OpenClawRequestNotice
            title="Room collaboration is unavailable"
            message={linkedRoomError}
            onRetry={() => void loadLinkedRoom()}
          />
        ) : null}
        {linkedRoomLoading ? (
          <div className={styles.inlineLoading} role="status">
            Loading room context…
          </div>
        ) : null}
        <div ref={iframeContainerRef} className={styles.iframeContainer}>
          {linkedRoomId || latestRoomBridgeEvent || surfaceEvents.length > 0 ? (
            <div className={styles.roomBridgePanel} data-testid="claw-room-bridge-activity">
              <div className={styles.roomBridgePanelHeader}>Room collaboration</div>
              {linkedRoomId ? (
                <div className={styles.roomBridgeMeta}>Room: {linkedRoomId}</div>
              ) : null}
              {latestRoomBridgeEvent ? (
                <div className={styles.roomBridgeEvent}>
                  Latest: {truncateText(latestRoomBridgeEvent, 120)}
                </div>
              ) : null}
              {surfaceEvents.length > 0 ? (
                <div className={styles.roomBridgeEvent}>
                  Surface:{' '}
                  {truncateText(
                    JSON.stringify(surfaceEvents[surfaceEvents.length - 1].payload || {}),
                    120,
                  )}
                </div>
              ) : null}
            </div>
          ) : null}
          <iframe
            ref={iframeRef}
            key={`${gatewayToken}:${linkedRoomId}`}
            className={styles.iframe}
            src={iframeSrc}
            title={`OpenClaw Control UI — ${selectedContainer}`}
            sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
            onLoad={() => {
              if (iframeRef.current && linkedRoomId) {
                postRoomContextToFrame(iframeRef.current, linkedRoomId)
              }
            }}
          />
        </div>
      </div>
    )
  }

  return (
    <div>
      {statusError ? (
        <OpenClawRequestNotice
          title="Container status is unavailable"
          message={statusError}
          onRetry={onRefresh}
        />
      ) : null}
      {actionError ? (
        <OpenClawRequestNotice
          title="Container action failed"
          message={actionError}
          onRetry={
            retryAction
              ? () => void runLifecycleAction(retryAction.action, retryAction.name)
              : undefined
          }
          onDismiss={() => {
            setActionError('')
            setRetryAction(null)
          }}
        />
      ) : null}

      <OpenClawCatalogControls
        searchLabel="Search containers"
        searchValue={search}
        filterLabel="Health"
        filterValue={healthFilter}
        filterOptions={[
          { value: 'all', label: 'All health states' },
          { value: 'healthy', label: 'Healthy' },
          { value: 'starting', label: 'Starting' },
          { value: 'stopped', label: 'Stopped' },
        ]}
        sortValue={sort}
        sortOptions={[
          { value: 'name-asc', label: 'Name A–Z' },
          { value: 'team-asc', label: 'Team A–Z' },
          { value: 'status', label: 'Health status' },
          { value: 'created-desc', label: 'Recently created' },
        ]}
        itemCount={filteredContainers.length}
        totalCount={containers.length}
        itemLabel="containers"
        page={safePage}
        pageSize={OPENCLAW_CONTAINERS_PAGE_SIZE}
        onSearchChange={setSearch}
        onFilterChange={(value) => setHealthFilter(value as WorkerHealthFilter)}
        onSortChange={(value) => setSort(value as WorkerCatalogSort)}
        onPageChange={setPage}
      />
      <label className={styles.enterpriseTeamFilter}>
        <span>Team</span>
        <select value={teamFilter} onChange={(event) => setTeamFilter(event.target.value)}>
          <option value="all">All teams</option>
          <option value="unassigned">Unassigned</option>
          {teamOptions.map(([id, name]) => (
            <option key={id} value={id}>
              {name}
            </option>
          ))}
        </select>
      </label>

      {visibleContainers.length === 0 ? (
        <>
          <div className={styles.emptyState}>
            <div className={styles.emptyStateText}>
              {containers.length === 0
                ? readOnly
                  ? 'No OpenClaw containers are provisioned.'
                  : 'No OpenClaw containers are provisioned. Use Claw Worker to create one.'
                : 'No containers match the current search, team, and health filters.'}
            </div>
          </div>
          <div className={styles.statusActionsCentered}>
            <button type="button" className={styles.btnSecondary} onClick={onRefresh}>
              Refresh Status
            </button>
          </div>
        </>
      ) : (
        <>
          <div className={styles.containerTableWrap}>
            <table className={styles.containerTable}>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Team</th>
                  <th>Port</th>
                  <th>Health</th>
                  <th>Error</th>
                  {!readOnly ? <th>Actions</th> : null}
                </tr>
              </thead>
              <tbody>
                {visibleContainers.map((container) => {
                  const health = getOpenClawWorkerHealth(container)
                  return (
                    <tr key={container.containerName}>
                      <td className={styles.containerTableName}>{container.containerName}</td>
                      <td>{container.teamName?.trim() || 'Unassigned'}</td>
                      <td className={styles.containerTablePort}>{container.port}</td>
                      <td>
                        <span
                          className={`${styles.healthBadge} ${styles[`healthBadge_${health}`]}`}
                        >
                          {health === 'healthy'
                            ? 'Healthy'
                            : health === 'starting'
                              ? 'Starting'
                              : 'Stopped'}
                        </span>
                      </td>
                      <td className={styles.containerErrorCell}>{container.error || '—'}</td>
                      {!readOnly ? (
                        <td>
                          <div className={styles.containerActions}>
                            {container.healthy ? (
                              <button
                                type="button"
                                className={`${styles.btnSmall} ${styles.btnSmallPrimary}`}
                                onClick={() => setSelectedContainer(container.containerName)}
                                disabled={Boolean(actionLoading)}
                              >
                                Dashboard
                              </button>
                            ) : null}
                            <button
                              type="button"
                              className={styles.btnSmall}
                              onClick={() =>
                                void runLifecycleAction(
                                  container.running ? 'stop' : 'start',
                                  container.containerName,
                                )
                              }
                              disabled={Boolean(actionLoading)}
                            >
                              {actionLoading === container.containerName
                                ? 'Working…'
                                : container.running
                                  ? 'Stop'
                                  : 'Start'}
                            </button>
                            <button
                              type="button"
                              className={`${styles.btnSmall} ${styles.btnSmallDanger}`}
                              onClick={() => {
                                setDeleteError('')
                                setPendingDeleteContainer(container.containerName)
                              }}
                              disabled={Boolean(actionLoading)}
                            >
                              Remove
                            </button>
                          </div>
                        </td>
                      ) : null}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <div className={styles.statusActions}>
            <button type="button" className={styles.btnSecondary} onClick={onRefresh}>
              Refresh Status
            </button>
          </div>
        </>
      )}

      <ConfirmDialog
        isOpen={Boolean(pendingDeleteContainer)}
        eyebrow="Remove OpenClaw container"
        title={`Remove ${pendingDeleteContainer || 'container'}?`}
        description="This stops and permanently removes the Docker container."
        details={
          pendingDeleteContainer ? (
            <div>
              <div>Container: {pendingDeleteContainer}</div>
              {deleteError ? (
                <div className={styles.confirmInlineError} role="alert">
                  {deleteError} Retry the removal or cancel.
                </div>
              ) : null}
            </div>
          ) : undefined
        }
        confirmLabel="Remove container"
        pending={Boolean(actionLoading)}
        tone="danger"
        onCancel={() => {
          setDeleteError('')
          setPendingDeleteContainer(null)
        }}
        onConfirm={confirmDelete}
      />
    </div>
  )
}
