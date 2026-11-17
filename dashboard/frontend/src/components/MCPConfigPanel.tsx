/**
 * MCP configuration panel for server and tool management.
 */

import React, { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ArrowUpRight, Lock, Plus } from 'lucide-react'

import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { toolRegistry } from '../tools'
import type { RegisteredTool } from '../tools'
import { useMCPServers } from '../tools/mcp'
import type { MCPServerConfig, MCPServerState } from '../tools/mcp'
import type { LatestMCPRequestRunner } from '../tools/mcp/requestSupport'
import { Alert, Button, Skeleton, toast } from '../ui'
import { canManageMCP, canUseTools } from '../utils/accessControl'
import ConfirmDialog from './ConfirmDialog'
import { MCPAvailableToolsSection } from './MCPAvailableToolsSection'
import { MCPRequestNotice } from './MCPRequestNotice'
import { MCPServerDialog } from './MCPServerDialog'
import { MCPServerSheet } from './MCPServerSheet'
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
  /** Frames the panel, given the header actions and the content. */
  frame?: (actions: React.ReactNode, content: React.ReactNode) => React.ReactNode
}

type ConnectionMode = 'connect' | 'disconnect'
type RetryAction =
  | { kind: 'connection'; id: string; mode: ConnectionMode; name: string }
  | { kind: 'delete'; server: MCPServerConfig }

interface ActionError {
  message: string
  retry: RetryAction
}

export const MCPConfigPanel: React.FC<MCPConfigPanelProps> = ({
  frame = (actions, content) => (
    <>
      {actions}
      {content}
    </>
  ),
}) => {
  const { serverReadonly, isLoading: readonlyLoading } = useReadonly()
  const { user, isLoading: authLoading } = useAuth()
  const canManageServers = canManageMCP(user)
  const mutationDisabled = authLoading || readonlyLoading || serverReadonly || !canManageServers
  const canRunTools = !authLoading && canUseTools(user)
  const {
    servers,
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
  const [params, setParams] = useSearchParams()

  const [showAddDialog, setShowAddDialog] = useState(false)
  const [editingServer, setEditingServer] = useState<MCPServerConfig | null>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [actionError, setActionError] = useState<ActionError | null>(null)
  const [toolsDbTools, setToolsDbTools] = useState<BuiltInTool[]>([])
  const [toolsDbLoading, setToolsDbLoading] = useState(false)
  const [toolsDbError, setToolsDbError] = useState<string | null>(null)
  const [registryTools] = useState<RegisteredTool[]>(() => toolRegistry.getAll())
  const [toolSearch, setToolSearch] = useState('')
  const [toolSourceFilter, setToolSourceFilter] = useState<ToolSourceFilter>('all')
  const [toolSort, setToolSort] = useState<ToolSort>('name-asc')
  const [selectedTool, setSelectedTool] = useState<UnifiedTool | null>(null)
  const [pendingDeleteServer, setPendingDeleteServer] = useState<MCPServerConfig | null>(null)
  const [serverFilter, setServerFilter] = useState<ServerFilter>('all')
  const [serverSearch, setServerSearch] = useState('')
  const [serverSort, setServerSort] = useState<ServerSort>('name-asc')
  const deferredToolSearch = useDeferredValue(toolSearch)
  const deferredServerSearch = useDeferredValue(serverSearch)
  const toolsDbRequestRef = useRef<LatestMCPRequestRunner | null>(null)
  const mutationRequestRef = useRef<LatestMCPRequestRunner | null>(null)
  if (!toolsDbRequestRef.current) toolsDbRequestRef.current = createLatestMCPRequestRunner()
  if (!mutationRequestRef.current) mutationRequestRef.current = createLatestMCPRequestRunner()

  const activeServerId = params.get('server')
  const activeServer = servers.find((server) => server.config.id === activeServerId) ?? null
  const sheetTab =
    params.get('tab') === 'test' && canRunTools
      ? 'test'
      : params.get('tab') === 'connection'
        ? 'connection'
        : 'tools'
  const openServer = (id: string, tab = 'tools') => {
    const next = new URLSearchParams(params)
    next.set('server', id)
    if (tab === 'tools') next.delete('tab')
    else next.set('tab', tab)
    setParams(next)
  }
  const closeServer = () => {
    const next = new URLSearchParams(params)
    next.delete('server')
    next.delete('tab')
    setParams(next)
  }

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
          onSuccess: () =>
            toast.success(mode === 'connect' ? `${name} connected` : `${name} disconnected`),
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

  const handleToggleEnabled = useCallback(
    async (server: MCPServerState) => {
      if (mutationDisabled) return
      const enabled = !server.config.enabled
      await mutationRequestRef.current?.run(
        (signal) => updateServer(server.config.id, { ...server.config, enabled }, signal),
        {
          onStart: () => {
            setActionLoading(server.config.id)
            setActionError(null)
          },
          onSuccess: () =>
            toast(enabled ? `${server.config.name} turned on` : `${server.config.name} turned off`),
          onError: (requestError) =>
            toast.error(`${server.config.name} was not updated`, {
              description: getMCPRequestErrorMessage(requestError, 'Failed to update the server.'),
            }),
          onFinish: () => setActionLoading(null),
        },
      )
    },
    [mutationDisabled, updateServer],
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
      onSuccess: () => {
        setPendingDeleteServer(null)
        setParams((current) => {
          if (current.get('server') !== server.id) return current
          const next = new URLSearchParams(current)
          next.delete('server')
          next.delete('tab')
          return next
        })
        toast(`${server.name} deleted`)
      },
      onError: (requestError) => {
        setPendingDeleteServer(null)
        setActionError({
          message: getMCPRequestErrorMessage(requestError, `Failed to delete ${server.name}.`),
          retry: { kind: 'delete', server },
        })
      },
      onFinish: () => setActionLoading(null),
    })
  }, [deleteServer, mutationDisabled, pendingDeleteServer, setParams])

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

  const actions = (
    <>
      <Button variant="outline" asChild>
        <a
          href="https://modelcontextprotocol.io/docs/getting-started/intro"
          target="_blank"
          rel="noreferrer"
        >
          About MCP
          <ArrowUpRight />
        </a>
      </Button>
      <Button
        onClick={() => setShowAddDialog(true)}
        disabled={mutationDisabled || Boolean(actionLoading)}
      >
        <Plus />
        Add server
      </Button>
    </>
  )

  const content = loading ? (
    <div className="flex flex-col gap-3" aria-busy="true">
      <Skeleton className="h-5 w-40" />
      <Skeleton className="h-40 w-full rounded-xl" />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, index) => (
          <Skeleton key={index} className="h-36 rounded-xl" />
        ))}
      </div>
    </div>
  ) : (
    <div className="flex flex-col gap-8">
      {error ||
      toolsError ||
      toolsDbError ||
      actionError ||
      (!readonlyLoading && serverReadonly) ||
      (!authLoading && !canManageServers) ? (
        <div className="flex flex-col gap-3">
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
          {!readonlyLoading && serverReadonly ? (
            <Alert tone="neutral" title="View only" icon={<Lock />}>
              The server-wide read-only policy disables MCP server changes. Runtime config mount
              availability does not affect MCP management.
            </Alert>
          ) : !authLoading && !canManageServers ? (
            <Alert tone="neutral" title="View only" icon={<Lock />}>
              You can inspect MCP servers and tools. Connect, edit, and delete actions require the{' '}
              <code className="font-mono text-[12px]">mcp.manage</code> permission.
            </Alert>
          ) : null}
        </div>
      ) : null}

      <MCPServersSection
        servers={servers}
        filteredServers={filteredServers}
        serverFilter={serverFilter}
        serverSearch={serverSearch}
        serverSort={serverSort}
        isReadonly={mutationDisabled}
        actionLoading={actionLoading}
        activeServerId={activeServerId ?? undefined}
        onServerFilterChange={setServerFilter}
        onServerSearchChange={setServerSearch}
        onServerSortChange={setServerSort}
        onOpenServer={(id) => openServer(id)}
        onToggleConnection={handleToggleConnection}
        onToggleEnabled={(server) => void handleToggleEnabled(server)}
        onEditServer={setEditingServer}
        onDeleteServer={handleDeleteRequest}
        onAddServer={() => setShowAddDialog(true)}
        onRefresh={handleRefresh}
      />

      <MCPAvailableToolsSection
        allAvailableTools={allAvailableTools}
        filteredTools={filteredTools}
        toolSearch={toolSearch}
        toolSort={toolSort}
        toolSourceFilter={toolSourceFilter}
        onSearchChange={setToolSearch}
        onSelectTool={setSelectedTool}
        onSortChange={setToolSort}
        onSourceFilterChange={setToolSourceFilter}
      />
      {toolsDbLoading && toolsDbTools.length === 0 ? (
        <p className="-mt-6 text-xs text-subtle-foreground" role="status">
          Loading router tools…
        </p>
      ) : null}
    </div>
  )

  return (
    <>
      {frame(actions, content)}

      <MCPServerSheet
        server={activeServer}
        open={Boolean(activeServer)}
        tab={sheetTab}
        canManage={!mutationDisabled}
        canRunTools={canRunTools}
        actionLoading={actionLoading}
        onTabChange={(tab) => activeServer && openServer(activeServer.config.id, tab)}
        onOpenChange={(open) => {
          if (!open) closeServer()
        }}
        onToggleConnection={handleToggleConnection}
        onToggleEnabled={(server) => void handleToggleEnabled(server)}
        onEdit={(server) => setEditingServer(server.config)}
        onDelete={handleDeleteRequest}
        onSelectTool={setSelectedTool}
      />

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
            toast.success(editingServer ? `${config.name} updated` : `${config.name} added`, {
              description: editingServer
                ? 'Changes apply the next time it connects.'
                : 'Connect it to list its tools.',
            })
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
        tone="danger"
        pending={Boolean(pendingDeleteServer && actionLoading === pendingDeleteServer.id)}
        onCancel={() => setPendingDeleteServer(null)}
        onConfirm={handleConfirmDelete}
      />
    </>
  )
}

export default MCPConfigPanel
