/**
 * MCP Configuration Panel Component
 * Configuration panel for MCP servers and tools management
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react'
import { useMCPServers } from '../tools/mcp'
import type { MCPServerConfig } from '../tools/mcp'
import { toolRegistry } from '../tools'
import type { RegisteredTool } from '../tools'
import { useReadonly } from '../contexts/ReadonlyContext'
import styles from './MCPConfigPanel.module.css'
import { MCPAvailableToolsSection } from './MCPAvailableToolsSection'
import { MCPServerDialog } from './MCPServerDialog'
import { MCPServersSection } from './MCPServersSection'
import { MCPToolDetailModal } from './MCPToolDetailModal'
import type { BuiltInTool, ServerFilter, UnifiedTool } from './mcpConfigPanelTypes'
import {
  buildUnifiedTools,
  filterServers,
  filterUnifiedTools,
} from './mcpConfigPanelUtils'

interface MCPConfigPanelProps {
  onClose?: () => void
  embedded?: boolean
}

export const MCPConfigPanel: React.FC<MCPConfigPanelProps> = ({
  onClose,
  embedded = false,
}) => {
  const { isReadonly } = useReadonly()
  const {
    servers,
    tools,
    loading,
    error,
    addServer,
    updateServer,
    deleteServer,
    connect,
    disconnect,
    testConnection,
    refreshServers,
  } = useMCPServers()

  const [showAddDialog, setShowAddDialog] = useState(false)
  const [editingServer, setEditingServer] = useState<MCPServerConfig | null>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [expandedServers, setExpandedServers] = useState<Set<string>>(new Set())
  const [builtInExpanded, setBuiltInExpanded] = useState(false)
  const [toolsDbTools, setToolsDbTools] = useState<BuiltInTool[]>([])
  const [toolsDbLoading, setToolsDbLoading] = useState(false)
  const [registryTools] = useState<RegisteredTool[]>(() => toolRegistry.getAll())
  const [toolSearch, setToolSearch] = useState('')
  const [selectedTool, setSelectedTool] = useState<UnifiedTool | null>(null)
  const [toolsSectionExpanded, setToolsSectionExpanded] = useState(true)
  const [serverFilter, setServerFilter] = useState<ServerFilter>('all')

  useEffect(() => {
    const fetchToolsDb = async () => {
      setToolsDbLoading(true)
      try {
        const response = await fetch('/api/tools-db')
        if (response.ok) {
          const data = await response.json()
          setToolsDbTools(data || [])
        }
      } catch (err) {
        console.error('Failed to load tools_db:', err)
      } finally {
        setToolsDbLoading(false)
      }
    }

    fetchToolsDb()
  }, [])

  const allAvailableTools = useMemo(
    () => buildUnifiedTools(servers, registryTools, toolsDbTools),
    [servers, registryTools, toolsDbTools],
  )
  const filteredTools = useMemo(
    () => filterUnifiedTools(allAvailableTools, toolSearch),
    [allAvailableTools, toolSearch],
  )
  const filteredServers = useMemo(
    () => filterServers(servers, serverFilter),
    [servers, serverFilter],
  )

  const toggleServerExpand = useCallback((serverId: string) => {
    setExpandedServers(prev => {
      const next = new Set(prev)
      if (next.has(serverId)) {
        next.delete(serverId)
      } else {
        next.add(serverId)
      }
      return next
    })
  }, [])

  const handleToggleConnection = useCallback(async (server: typeof servers[number]) => {
    if (isReadonly) {
      return
    }

    setActionLoading(server.config.id)
    try {
      if (server.status === 'connected') {
        await disconnect(server.config.id)
      } else {
        await connect(server.config.id)
        setExpandedServers(prev => new Set(prev).add(server.config.id))
      }
    } catch (err) {
      console.error('Connection toggle failed:', err)
    } finally {
      setActionLoading(null)
    }
  }, [connect, disconnect, isReadonly])

  const handleDelete = useCallback(async (id: string) => {
    if (isReadonly) {
      return
    }
    if (!window.confirm('Are you sure you want to delete this MCP server?')) {
      return
    }

    setActionLoading(id)
    try {
      await deleteServer(id)
    } catch (err) {
      console.error('Delete failed:', err)
    } finally {
      setActionLoading(null)
    }
  }, [deleteServer, isReadonly])

  const mcpToolsCount = tools.length
  const builtInCount = registryTools.length + toolsDbTools.length
  const connectedCount = servers.filter(server => server.status === 'connected').length
  const totalToolsCount = mcpToolsCount + builtInCount

  if (loading) {
    return (
      <div className={`${styles.panel} ${embedded ? styles.embeddedPanel : ''}`}>
        {!embedded && (
          <div className={styles.header}>
            <h2>🔌 MCP Servers & Tools</h2>
            {onClose && <button className={styles.closeBtn} onClick={onClose}>×</button>}
          </div>
        )}
        <div className={styles.loading}>Loading...</div>
      </div>
    )
  }

  return (
    <div className={`${styles.panel} ${embedded ? styles.embeddedPanel : ''}`}>
      {!embedded && (
        <div className={styles.header}>
          <h2>🔌 MCP Servers & Tools</h2>
          <div className={styles.headerActions}>
            <button className={styles.refreshBtn} onClick={() => refreshServers()} title="Refresh">
              ↻
            </button>
            {onClose && <button className={styles.closeBtn} onClick={onClose}>×</button>}
          </div>
        </div>
      )}

      {embedded && (
        <div className={styles.embeddedToolbar}>
          <div className={styles.embeddedSummary}>
            {totalToolsCount} tools ({mcpToolsCount} from MCP, {builtInCount} built-in) • {connectedCount} connected servers
          </div>
          <button className={styles.refreshBtn} onClick={() => refreshServers()} title="Refresh">
            ↻
          </button>
        </div>
      )}

      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.serverList}>
        <MCPAvailableToolsSection
          allAvailableTools={allAvailableTools}
          filteredTools={filteredTools}
          toolSearch={toolSearch}
          toolsSectionExpanded={toolsSectionExpanded}
          onSearchChange={setToolSearch}
          onSelectTool={setSelectedTool}
          onToggleExpanded={() => setToolsSectionExpanded(prev => !prev)}
        />

        <MCPServersSection
          actionLoading={actionLoading}
          builtInExpanded={builtInExpanded}
          expandedServers={expandedServers}
          filteredServers={filteredServers}
          isReadonly={isReadonly}
          registryTools={registryTools}
          serverFilter={serverFilter}
          servers={servers}
          toolsDbLoading={toolsDbLoading}
          toolsDbTools={toolsDbTools}
          onDeleteServer={handleDelete}
          onEditServer={setEditingServer}
          onServerFilterChange={setServerFilter}
          onToggleBuiltInExpanded={() => setBuiltInExpanded(prev => !prev)}
          onToggleConnection={handleToggleConnection}
          onToggleServerExpand={toggleServerExpand}
        />
      </div>

      <div className={styles.footer}>
        <button
          className={styles.addBtn}
          onClick={() => {
            if (isReadonly) {
              return
            }
            setShowAddDialog(true)
          }}
          disabled={isReadonly}
        >
          + Add MCP Server
        </button>
        {!embedded && (
          <div className={styles.summary}>
            {totalToolsCount} tools ({mcpToolsCount} from MCP, {builtInCount} built-in) • {connectedCount} connected servers
          </div>
        )}
      </div>

      {(showAddDialog || editingServer) && (
        <MCPServerDialog
          server={editingServer}
          onClose={() => {
            setShowAddDialog(false)
            setEditingServer(null)
          }}
          onSave={async config => {
            if (editingServer) {
              await updateServer(editingServer.id, config)
            } else {
              await addServer(config)
            }
            setShowAddDialog(false)
            setEditingServer(null)
          }}
          onTest={testConnection}
        />
      )}

      {selectedTool && (
        <MCPToolDetailModal tool={selectedTool} onClose={() => setSelectedTool(null)} />
      )}
    </div>
  )
}

export default MCPConfigPanel
