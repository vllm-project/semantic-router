/**
 * MCP configuration panel for server and tool management.
 */

import React, { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'

import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { toolRegistry } from '../tools'
import type { RegisteredTool } from '../tools'
import { useMCPServers } from '../tools/mcp'
import type { MCPServerConfig, MCPServerState } from '../tools/mcp'
import type { LatestMCPRequestRunner } from '../tools/mcp/requestSupport'
import { canManageMCP } from '../utils/accessControl'
import ConfirmDialog from './ConfirmDialog'
import { MCPAvailableToolsSection } from './MCPAvailableToolsSection'
import styles from './MCPConfigPanel.module.css'
import { MCPRequestNotice } from './MCPRequestNotice'
import { MCPServerDialog } from './MCPServerDialog'
import { MCPServersSection } from './MCPServersSection'
import { MCPToolDetailModal } from './MCPToolDetailModal'
import type {
  BuiltInTool,
  ServerFilter,
  ServerSort,
  ToolSort,
  ToolSourceFilter,
  UnifiedTool,
} from './mcpConfigPanelTypes'
import {
  buildUnifiedTools,
  filterAndSortServers,
  filterAndSortUnifiedTools,
} from './mcpConfigPanelUtils'
import {
  createLatestMCPRequestRunner,
  fetchMCPToolsDatabase,
  getMCPRequestErrorMessage,
} from './mcpRequestSupport'

interface MCPConfigPanelProps {
  onClose?: () => void
  embedded?: boolean
}

type ConnectionMode = 'connect' | 'disconnect'
type RetryAction =
  | { kind: 'connection'; id: string; mode: ConnectionMode; name: string }
  | { kind: 'delete'; server: MCPServerConfig }

interface ActionError {
  message: string
  retry: RetryAction
}

export const MCPConfigPanel: React.FC<MCPConfigPanelProps> = ({ onClose, embedded = false }) => {
  const { isReadonly } = useReadonly()
  const { user } = useAuth()
  const canManageServers = canManageMCP(user)
  const mutationDisabled = isReadonly || !canManageServers
  const {
    servers,
    tools,
    loading,
    error,
    toolsError,
    addServer,
    updateServer,
    deleteServer,
    connect,
    disconnect,
    testConnection,
    refreshServers,
    refreshTools,
  } = useMCPServers()

  const [showAddDialog, setShowAddDialog] = useState(false)
  const [editingServer, setEditingServer] = useState<MCPServerConfig | null>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [actionError, setActionError] = useState<ActionError | null>(null)
  const [expandedServers, setExpandedServers] = useState<Set<string>>(new Set())
  const [builtInExpanded, setBuiltInExpanded] = useState(false)
  const [toolsDbExpanded, setToolsDbExpanded] = useState(false)
  const [toolsDbTools, setToolsDbTools] = useState<BuiltInTool[]>([])
  const [toolsDbLoading, setToolsDbLoading] = useState(false)
  const [toolsDbError, setToolsDbError] = useState<string | null>(null)
  const [registryTools] = useState<RegisteredTool[]>(() => toolRegistry.getAll())
  const [toolSearch, setToolSearch] = useState('')
  const [toolSourceFilter, setToolSourceFilter] = useState<ToolSourceFilter>('all')
  const [toolSort, setToolSort] = useState<ToolSort>('name-asc')
  const [selectedTool, setSelectedTool] = useState<UnifiedTool | null>(null)
  const [pendingDeleteServer, setPendingDeleteServer] = useState<MCPServerConfig | null>(null)
  const [toolsSectionExpanded, setToolsSectionExpanded] = useState(true)
  const [serversSectionExpanded, setServersSectionExpanded] = useState(true)
  const [serverFilter, setServerFilter] = useState<ServerFilter>('all')
  const [serverSearch, setServerSearch] = useState('')
  const [serverSort, setServerSort] = useState<ServerSort>('name-asc')
  const deferredToolSearch = useDeferredValue(toolSearch)
  const deferredServerSearch = useDeferredValue(serverSearch)
  const toolsDbRequestRef = useRef<LatestMCPRequestRunner | null>(null)
  const mutationRequestRef = useRef<LatestMCPRequestRunner | null>(null)
  if (!toolsDbRequestRef.current) toolsDbRequestRef.current = createLatestMCPRequestRunner()
  if (!mutationRequestRef.current) mutationRequestRef.current = createLatestMCPRequestRunner()

  const loadToolsDb = useCallback(async () => {
    await toolsDbRequestRef.current?.run(fetchMCPToolsDatabase, {
      onStart: () => {
        setToolsDbLoading(true)
        setToolsDbError(null)
      },
      onSuccess: setToolsDbTools,
      onError: (requestError) => {
        setToolsDbError(
          getMCPRequestErrorMessage(requestError, 'Failed to load the tools database.'),
        )
      },
      onFinish: () => setToolsDbLoading(false),
    })
  }, [])

  useEffect(() => {
    void loadToolsDb()
    return () => toolsDbRequestRef.current?.cancel()
  }, [loadToolsDb])

  useEffect(() => () => mutationRequestRef.current?.cancel(), [])

  useEffect(() => {
    if (!mutationDisabled) return
    mutationRequestRef.current?.cancel()
    setActionLoading(null)
    setActionError(null)
    setPendingDeleteServer(null)
    setShowAddDialog(false)
    setEditingServer(null)
  }, [mutationDisabled])

  const allAvailableTools = useMemo(
    () => buildUnifiedTools(servers, registryTools, toolsDbTools),
    [servers, registryTools, toolsDbTools],
  )
  const filteredTools = useMemo(
    () =>
      filterAndSortUnifiedTools(allAvailableTools, deferredToolSearch, toolSourceFilter, toolSort),
    [allAvailableTools, deferredToolSearch, toolSort, toolSourceFilter],
  )
  const filteredServers = useMemo(
    () => filterAndSortServers(servers, serverFilter, deferredServerSearch, serverSort),
    [deferredServerSearch, serverFilter, serverSort, servers],
  )

  const toggleServerExpand = useCallback((serverId: string) => {
    setExpandedServers((current) => {
      const next = new Set(current)
      if (next.has(serverId)) next.delete(serverId)
      else next.add(serverId)
      return next
    })
  }, [])

  const runConnectionAction = useCallback(
    async (id: string, name: string, mode: ConnectionMode) => {
      if (mutationDisabled) return
      await mutationRequestRef.current?.run(
        (signal) => (mode === 'connect' ? connect(id, signal) : disconnect(id, signal)),
        {
          onStart: () => {
            setActionLoading(id)
            setActionError(null)
          },
          onSuccess: () => {
            if (mode === 'connect') {
              setExpandedServers((current) => new Set(current).add(id))
            }
          },
          onError: (requestError) => {
            const verb = mode === 'connect' ? 'connect to' : 'disconnect from'
            setActionError({
              message: getMCPRequestErrorMessage(requestError, `Failed to ${verb} ${name}.`),
              retry: { kind: 'connection', id, mode, name },
            })
          },
          onFinish: () => setActionLoading(null),
        },
      )
    },
    [connect, disconnect, mutationDisabled],
  )

  const handleToggleConnection = useCallback(
    (server: MCPServerState) => {
      const mode: ConnectionMode = server.status === 'connected' ? 'disconnect' : 'connect'
      void runConnectionAction(server.config.id, server.config.name, mode)
    },
    [runConnectionAction],
  )

  const handleDeleteRequest = useCallback(
    (id: string) => {
      if (mutationDisabled) return
      const server = servers.find((candidate) => candidate.config.id === id)?.config
      if (server) {
        setActionError(null)
        setPendingDeleteServer(server)
      }
    },
    [mutationDisabled, servers],
  )

  const handleConfirmDelete = useCallback(async () => {
    if (mutationDisabled || !pendingDeleteServer) return
    const server = pendingDeleteServer
    await mutationRequestRef.current?.run((signal) => deleteServer(server.id, signal), {
      onStart: () => {
        setActionLoading(server.id)
        setActionError(null)
      },
      onSuccess: () => setPendingDeleteServer(null),
      onError: (requestError) => {
        setPendingDeleteServer(null)
        setActionError({
          message: getMCPRequestErrorMessage(requestError, `Failed to delete ${server.name}.`),
          retry: { kind: 'delete', server },
        })
      },
      onFinish: () => setActionLoading(null),
    })
  }, [deleteServer, mutationDisabled, pendingDeleteServer])

  const handleRetryAction = useCallback(() => {
    const retry = actionError?.retry
    if (!retry) return
    setActionError(null)
    if (retry.kind === 'delete') {
      setPendingDeleteServer(retry.server)
      return
    }
    void runConnectionAction(retry.id, retry.name, retry.mode)
  }, [actionError, runConnectionAction])

  const handleRefresh = useCallback(() => {
    void Promise.all([refreshServers(), refreshTools(), loadToolsDb()])
  }, [loadToolsDb, refreshServers, refreshTools])

  const mcpToolsCount = tools.length
  const builtInCount = registryTools.length + toolsDbTools.length
  const connectedCount = servers.filter((server) => server.status === 'connected').length
  const totalToolsCount = mcpToolsCount + builtInCount

  if (loading) {
    return (
      <div className={`${styles.panel} ${embedded ? styles.embeddedPanel : ''}`} aria-busy="true">
        {!embedded ? (
          <div className={styles.header}>
            <h2>MCP Servers &amp; Tools</h2>
            {onClose ? (
              <button
                type="button"
                className={styles.closeBtn}
                onClick={onClose}
                aria-label="Close MCP panel"
              >
                ×
              </button>
            ) : null}
          </div>
        ) : null}
        <div className={styles.loading} role="status">
          Loading MCP servers and tools…
        </div>
      </div>
    )
  }

  return (
    <div className={`${styles.panel} ${embedded ? styles.embeddedPanel : ''}`}>
      {!embedded ? (
        <div className={styles.header}>
          <h2>MCP Servers &amp; Tools</h2>
          <div className={styles.headerActions}>
            <button
              type="button"
              className={styles.refreshBtn}
              onClick={handleRefresh}
              title="Refresh MCP data"
              aria-label="Refresh MCP data"
            >
              ↻
            </button>
            {onClose ? (
              <button
                type="button"
                className={styles.closeBtn}
                onClick={onClose}
                aria-label="Close MCP panel"
              >
                ×
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      {embedded ? (
        <div className={styles.embeddedToolbar}>
          <div className={styles.embeddedSummary}>
            {totalToolsCount} tools ({mcpToolsCount} from MCP, {builtInCount} built-in) ·{' '}
            {connectedCount} connected servers
          </div>
          <button
            type="button"
            className={styles.refreshBtn}
            onClick={handleRefresh}
            title="Refresh MCP data"
            aria-label="Refresh MCP data"
          >
            ↻
          </button>
        </div>
      ) : null}

      {error ? (
        <MCPRequestNotice
          title="MCP servers are unavailable"
          message={error}
          retryLabel="Retry servers"
          onRetry={() => void refreshServers()}
        />
      ) : null}
      {toolsError ? (
        <MCPRequestNotice
          title="MCP tools are unavailable"
          message={toolsError}
          retryLabel="Retry tools"
          onRetry={() => void refreshTools()}
        />
      ) : null}
      {toolsDbError ? (
        <MCPRequestNotice
          title="Semantic Router tools are unavailable"
          message={toolsDbError}
          retryLabel="Retry tools database"
          onRetry={() => void loadToolsDb()}
        />
      ) : null}
      {actionError ? (
        <MCPRequestNotice
          title="MCP server action failed"
          message={actionError.message}
          retryLabel={actionError.retry.kind === 'delete' ? 'Review and retry' : 'Retry action'}
          onRetry={handleRetryAction}
          onDismiss={() => setActionError(null)}
        />
      ) : null}
      {!canManageServers ? (
        <div className={styles.permissionNotice} role="note">
          You can inspect MCP servers and tools. Connect, edit, and delete actions require the{' '}
          <code>mcp.manage</code> permission.
        </div>
      ) : null}

      <div className={styles.serverList}>
        <MCPAvailableToolsSection
          allAvailableTools={allAvailableTools}
          filteredTools={filteredTools}
          toolSearch={toolSearch}
          toolSort={toolSort}
          toolSourceFilter={toolSourceFilter}
          toolsSectionExpanded={toolsSectionExpanded}
          onSearchChange={setToolSearch}
          onSelectTool={setSelectedTool}
          onSortChange={setToolSort}
          onSourceFilterChange={setToolSourceFilter}
          onToggleExpanded={() => setToolsSectionExpanded((current) => !current)}
        />

        <MCPServersSection
          actionLoading={actionLoading}
          builtInExpanded={builtInExpanded}
          expandedServers={expandedServers}
          filteredServers={filteredServers}
          isReadonly={mutationDisabled}
          registryTools={registryTools}
          serverFilter={serverFilter}
          serverSearch={serverSearch}
          serverSort={serverSort}
          servers={servers}
          serversSectionExpanded={serversSectionExpanded}
          toolsDbExpanded={toolsDbExpanded}
          toolsDbLoading={toolsDbLoading}
          toolsDbTools={toolsDbTools}
          onDeleteServer={handleDeleteRequest}
          onEditServer={setEditingServer}
          onSelectTool={setSelectedTool}
          onServerFilterChange={setServerFilter}
          onServerSearchChange={setServerSearch}
          onServerSortChange={setServerSort}
          onToggleBuiltInExpanded={() => setBuiltInExpanded((current) => !current)}
          onToggleConnection={handleToggleConnection}
          onToggleServerExpand={toggleServerExpand}
          onToggleServersSection={() => setServersSectionExpanded((current) => !current)}
          onToggleToolsDbExpanded={() => setToolsDbExpanded((current) => !current)}
        />
      </div>

      <div className={styles.footer}>
        <button
          type="button"
          className={styles.addBtn}
          onClick={() => setShowAddDialog(true)}
          disabled={mutationDisabled || Boolean(actionLoading)}
        >
          + Add MCP Server
        </button>
        {!embedded ? (
          <div className={styles.summary}>
            {totalToolsCount} tools ({mcpToolsCount} from MCP, {builtInCount} built-in) ·{' '}
            {connectedCount} connected servers
          </div>
        ) : null}
      </div>

      {!mutationDisabled && (showAddDialog || editingServer) ? (
        <MCPServerDialog
          server={editingServer}
          onClose={() => {
            setShowAddDialog(false)
            setEditingServer(null)
          }}
          onSave={async (config) => {
            if (editingServer) await updateServer(editingServer.id, config)
            else await addServer(config)
            setShowAddDialog(false)
            setEditingServer(null)
          }}
          onTest={testConnection}
        />
      ) : null}

      {selectedTool ? (
        <MCPToolDetailModal tool={selectedTool} onClose={() => setSelectedTool(null)} />
      ) : null}

      <ConfirmDialog
        isOpen={Boolean(pendingDeleteServer)}
        eyebrow="MCP server"
        title={pendingDeleteServer ? `Delete ${pendingDeleteServer.name}?` : 'Delete MCP server?'}
        description="This removes the server configuration and disconnects its tools from the control plane. This action cannot be undone."
        confirmLabel="Delete server"
        pending={Boolean(pendingDeleteServer && actionLoading === pendingDeleteServer.id)}
        onCancel={() => setPendingDeleteServer(null)}
        onConfirm={handleConfirmDelete}
      />
    </div>
  )
}

export default MCPConfigPanel
