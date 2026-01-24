/**
 * MCP Configuration Panel Component
 * Configuration panel for MCP servers and tools management
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react'
import { useMCPServers } from '../tools/mcp'
import type { MCPServerConfig, MCPServerState, MCPTransportType, MCPToolDefinition } from '../tools/mcp'
import { toolRegistry } from '../tools'
import type { RegisteredTool } from '../tools'
import styles from './MCPConfigPanel.module.css'

// Built-in tool type definitions
interface BuiltInToolParameter {
  type: string
  description?: string
  enum?: string[]
  default?: unknown
}

interface BuiltInTool {
  tool: {
    type: string
    function: {
      name: string
      description: string
      parameters: {
        type: string
        properties: Record<string, BuiltInToolParameter>
        required?: string[]
      }
    }
  }
  description: string
  category: string
  tags: string[]
}

// Unified tool type - for Available Tools section
interface UnifiedTool {
  id: string
  name: string
  description: string
  source: string           // Source name
  sourceType: 'mcp' | 'frontend' | 'backend'
  parameters: {
    name: string
    type: string
    description?: string
    required: boolean
  }[]
  rawTool: MCPToolDefinition | RegisteredTool | BuiltInTool
}

// Max visible height for tools grid (scrollable)
const TOOLS_GRID_MAX_HEIGHT = 320

interface MCPConfigPanelProps {
  onClose?: () => void
}

export const MCPConfigPanel: React.FC<MCPConfigPanelProps> = ({ onClose }) => {
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
  // Collapse state: connected servers expanded by default, disconnected collapsed
  const [expandedServers, setExpandedServers] = useState<Set<string>>(new Set())
  // Built-in tools expand state
  const [builtInExpanded, setBuiltInExpanded] = useState(false)
  // Backend tools_db.json tool data
  const [toolsDbTools, setToolsDbTools] = useState<BuiltInTool[]>([])
  const [toolsDbLoading, setToolsDbLoading] = useState(false)
  // Frontend registered built-in tools
  const [registryTools, setRegistryTools] = useState<RegisteredTool[]>([])

  // ========== Available Tools section states ==========
  // Tool search
  const [toolSearch, setToolSearch] = useState('')
  // Tool detail modal
  const [selectedTool, setSelectedTool] = useState<UnifiedTool | null>(null)
  // Available Tools section collapse state
  const [toolsSectionExpanded, setToolsSectionExpanded] = useState(true)
  // MCP Servers filter state: 'all' | 'connected' | 'disconnected'
  const [serverFilter, setServerFilter] = useState<'all' | 'connected' | 'disconnected'>('all')

  // Initialize collapse state - default to collapsed (empty set)
  // Tools details are collapsed by default for better UX
  useEffect(() => {
    // Keep expandedServers empty by default - tools details collapsed
    // Users can manually expand if needed
  }, [servers.length]) // Only update when server count changes

  // Get frontend registered built-in tools
  useEffect(() => {
    const tools = toolRegistry.getAll()
    setRegistryTools(tools)
  }, [])

  // Get backend tools_db.json tools (optional)
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

  // Aggregate all available tools
  const allAvailableTools = useMemo<UnifiedTool[]>(() => {
    const result: UnifiedTool[] = []

    // 1. MCP server tools (connected only)
    servers.filter(s => s.status === 'connected').forEach(server => {
      server.tools?.forEach(tool => {
        const schema = tool.inputSchema
        const properties = (schema?.properties || {}) as Record<string, { type?: string; description?: string }>
        const required = schema?.required || []
        
        result.push({
          id: `mcp-${server.config.id}-${tool.name}`,
          name: tool.name,
          description: tool.description || '',
          source: server.config.name,
          sourceType: 'mcp',
          parameters: Object.entries(properties).map(([name, prop]) => ({
            name,
            type: prop.type || 'any',
            description: prop.description,
            required: required.includes(name)
          })),
          rawTool: tool
        })
      })
    })

    // 2. Frontend built-in tools
    registryTools.forEach(tool => {
      const params = tool.definition.function.parameters
      const properties = (params?.properties || {}) as Record<string, { type?: string; description?: string }>
      const required = params?.required || []
      
      result.push({
        id: `frontend-${tool.metadata.id}`,
        name: tool.metadata.displayName,
        description: tool.definition.function.description,
        source: 'Built-in',
        sourceType: 'frontend',
        parameters: Object.entries(properties).map(([name, prop]) => ({
          name,
          type: prop.type || 'any',
          description: prop.description,
          required: required.includes(name)
        })),
        rawTool: tool
      })
    })

    // 3. Backend tools_db tools
    toolsDbTools.forEach(tool => {
      const params = tool.tool.function.parameters
      const properties = params?.properties || {}
      const required = params?.required || []
      
      result.push({
        id: `backend-${tool.tool.function.name}`,
        name: tool.tool.function.name,
        description: tool.tool.function.description,
        source: 'Semantic Router',
        sourceType: 'backend',
        parameters: Object.entries(properties).map(([name, prop]) => ({
          name,
          type: prop.type || 'any',
          description: prop.description,
          required: required.includes(name)
        })),
        rawTool: tool
      })
    })

    return result
  }, [servers, registryTools, toolsDbTools])

  // Search filter
  const filteredTools = useMemo(() => {
    if (!toolSearch.trim()) return allAvailableTools
    const search = toolSearch.toLowerCase()
    return allAvailableTools.filter(tool => 
      tool.name.toLowerCase().includes(search) ||
      tool.description.toLowerCase().includes(search) ||
      tool.source.toLowerCase().includes(search)
    )
  }, [allAvailableTools, toolSearch])

  // MCP Servers filter
  const filteredServers = useMemo(() => {
    if (serverFilter === 'all') return servers
    if (serverFilter === 'connected') {
      return servers.filter(s => s.status === 'connected')
    }
    // disconnected - includes 'disconnected', 'error', 'connecting'
    return servers.filter(s => s.status !== 'connected')
  }, [servers, serverFilter])
  // Toggle server expand/collapse
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

  // Handle connect/disconnect
  const handleToggleConnection = useCallback(async (server: MCPServerState) => {
    setActionLoading(server.config.id)
    try {
      if (server.status === 'connected') {
        await disconnect(server.config.id)
      } else {
        await connect(server.config.id)
        // Auto expand after connection
        setExpandedServers(prev => new Set(prev).add(server.config.id))
      }
    } catch (err) {
      console.error('Connection toggle failed:', err)
    } finally {
      setActionLoading(null)
    }
  }, [connect, disconnect])

  // Handle delete
  const handleDelete = useCallback(async (id: string) => {
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
  }, [deleteServer])

  // Get status icon
  const getStatusIcon = (status: MCPServerState['status']) => {
    switch (status) {
      case 'connected':
        return <span className={styles.statusDot} data-status="connected">●</span>
      case 'connecting':
        return <span className={styles.statusDot} data-status="connecting">◐</span>
      case 'error':
        return <span className={styles.statusDot} data-status="error">●</span>
      default:
        return <span className={styles.statusDot} data-status="disconnected">○</span>
    }
  }

  // Get transport type label
  const getTransportLabel = (transport: MCPTransportType) => {
    switch (transport) {
      case 'stdio':
        return 'Stdio'
      case 'streamable-http':
        return 'HTTP'
      default:
        return transport
    }
  }

  // Parse tool parameters
  const renderToolParameters = (tool: MCPToolDefinition) => {
    const schema = tool.inputSchema
    if (!schema || schema.type !== 'object') {
      return <span className={styles.noParams}>No parameters</span>
    }
    
    const properties = schema.properties || {}
    const required = schema.required || []
    const params = Object.entries(properties)
    
    if (params.length === 0) {
      return <span className={styles.noParams}>No parameters</span>
    }

    return (
      <div className={styles.paramsList}>
        {params.map(([name, prop]) => {
          const propData = prop as { type?: string; description?: string }
          return (
            <div key={name} className={styles.paramItem}>
              <span className={styles.paramName}>{name}</span>
              <span className={styles.paramType}>({propData.type || 'any'})</span>
              {required.includes(name) && <span className={styles.paramRequired}>*</span>}
              {propData.description && (
                <span className={styles.paramDesc}>{propData.description}</span>
              )}
            </div>
          )
        })}
      </div>
    )
  }

  // Render built-in tool parameters
  const renderBuiltInToolParameters = (tool: BuiltInTool) => {
    const params = tool.tool.function.parameters
    if (!params || !params.properties) {
      return <span className={styles.noParams}>No parameters</span>
    }
    
    const properties = params.properties
    const required = params.required || []
    const entries = Object.entries(properties)
    
    if (entries.length === 0) {
      return <span className={styles.noParams}>No parameters</span>
    }

    return (
      <div className={styles.paramsList}>
        {entries.map(([name, prop]) => (
          <div key={name} className={styles.paramItem}>
            <span className={styles.paramName}>{name}</span>
            <span className={styles.paramType}>({prop.type || 'any'})</span>
            {required.includes(name) && <span className={styles.paramRequired}>*</span>}
            {prop.description && (
              <span className={styles.paramDesc}>{prop.description}</span>
            )}
          </div>
        ))}
      </div>
    )
  }

  // Render frontend registered tool parameters
  const renderRegistryToolParameters = (tool: RegisteredTool) => {
    const params = tool.definition.function.parameters
    if (!params || params.type !== 'object') {
      return <span className={styles.noParams}>No parameters</span>
    }
    
    const properties = params.properties || {}
    const required = params.required || []
    const entries = Object.entries(properties)
    
    if (entries.length === 0) {
      return <span className={styles.noParams}>No parameters</span>
    }

    return (
      <div className={styles.paramsList}>
        {entries.map(([name, prop]) => {
          const propData = prop as { type?: string; description?: string }
          return (
            <div key={name} className={styles.paramItem}>
              <span className={styles.paramName}>{name}</span>
              <span className={styles.paramType}>({propData.type || 'any'})</span>
              {required.includes(name) && <span className={styles.paramRequired}>*</span>}
              {propData.description && (
                <span className={styles.paramDesc}>{propData.description}</span>
              )}
            </div>
          )
        })}
      </div>
    )
  }

  // Calculate statistics
  const connectedCount = servers.filter(s => s.status === 'connected').length
  const mcpToolsCount = tools.length
  const builtInCount = registryTools.length + toolsDbTools.length
  const totalToolsCount = mcpToolsCount + builtInCount

  if (loading) {
    return (
      <div className={styles.panel}>
        <div className={styles.header}>
          <h2>🔌 MCP Servers & Tools</h2>
          {onClose && <button className={styles.closeBtn} onClick={onClose}>×</button>}
        </div>
        <div className={styles.loading}>Loading...</div>
      </div>
    )
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h2>🔌 MCP Servers & Tools</h2>
        <div className={styles.headerActions}>
          <button 
            className={styles.refreshBtn} 
            onClick={() => refreshServers()}
            title="Refresh"
          >
            ↻
          </button>
          {onClose && <button className={styles.closeBtn} onClick={onClose}>×</button>}
        </div>
      </div>

      {error && (
        <div className={styles.error}>
          {error}
        </div>
      )}

      <div className={styles.serverList}>
        {/* ========== Available Tools Section ========== */}
        <div className={styles.availableToolsSection}>
          <div 
            className={styles.sectionHeader}
            onClick={() => setToolsSectionExpanded(!toolsSectionExpanded)}
          >
            <div className={styles.sectionTitle}>
              <span className={styles.expandIcon}>
                {toolsSectionExpanded ? '▼' : '▶'}
              </span>
              <span>🧰 Available Tools</span>
              <span className={styles.toolCountBadge}>
                {allAvailableTools.length} tools
              </span>
            </div>
            {toolsSectionExpanded && allAvailableTools.length > 0 && (
              <div className={styles.toolSearchWrapper} onClick={e => e.stopPropagation()}>
                <input
                  type="text"
                  className={styles.toolSearchInput}
                  placeholder="🔍 Search tools..."
                  value={toolSearch}
                  onChange={e => setToolSearch(e.target.value)}
                />
                {toolSearch && (
                  <button 
                    className={styles.clearSearchBtn}
                    onClick={() => setToolSearch('')}
                  >
                    ×
                  </button>
                )}
              </div>
            )}
          </div>

          {toolsSectionExpanded && (
            <>
              {/* Empty state when no tools available */}
              {allAvailableTools.length === 0 && (
                <div className={styles.noToolsAvailable}>
                  <span className={styles.emptyIcon}>🔧</span>
                  <p>No tools available</p>
                  <span className={styles.emptyHint}>
                    Connect an MCP server or add built-in tools to see them here
                  </span>
                </div>
              )}

              {/* Tools grid when tools exist */}
              {allAvailableTools.length > 0 && (
                <>
                  <div 
                    className={styles.toolsGridWrapper}
                    style={{ maxHeight: TOOLS_GRID_MAX_HEIGHT }}
                  >
                    <div className={styles.toolsGrid}>
                      {filteredTools.map(tool => (
                        <div 
                          key={tool.id} 
                          className={`${styles.toolGridCard} ${styles[`source_${tool.sourceType}`]}`}
                          onClick={() => setSelectedTool(tool)}
                        >
                          <div className={styles.toolGridHeader}>
                            <span className={styles.toolGridIcon}>🔧</span>
                            <span className={styles.toolGridName}>{tool.name}</span>
                          </div>
                          <div className={styles.toolGridDesc}>
                            {tool.description.length > 60 
                              ? tool.description.slice(0, 60) + '...' 
                              : tool.description || 'No description'}
                          </div>
                          <div className={styles.toolGridFooter}>
                            <span className={`${styles.sourceTypeBadge} ${styles[tool.sourceType]}`}>
                              {tool.sourceType === 'mcp' ? '🔌 MCP' : 
                               tool.sourceType === 'frontend' ? '⚡ Frontend' : '🌐 Backend'}
                            </span>
                            <span className={styles.sourceNameBadge}>{tool.source}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Tool count footer */}
                  {filteredTools.length > 0 && (
                    <div className={styles.toolsGridFooter}>
                      <span className={styles.toolCountInfo}>
                        {toolSearch ? `${filteredTools.length} of ${allAvailableTools.length} tools` : `${allAvailableTools.length} tools available`}
                      </span>
                      {filteredTools.length > 6 && (
                        <span className={styles.scrollHint}>↕ Scroll to see more</span>
                      )}
                    </div>
                  )}

                  {filteredTools.length === 0 && toolSearch && (
                    <div className={styles.noToolsFound}>
                      No matching tools found
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </div>

        {/* ========== MCP Servers Section ========== */}
        <div className={styles.mcpServersSection}>
          <div className={styles.sectionHeader}>
            <div className={styles.sectionTitle}>
              <span>📡 MCP Servers</span>
              <span className={styles.serverCountBadge}>
                {connectedCount} / {servers.length} connected
              </span>
            </div>
            {/* Server filter buttons */}
            {servers.length > 0 && (
              <div className={styles.serverFilterWrapper}>
                <button
                  className={`${styles.filterBtn} ${serverFilter === 'all' ? styles.filterActive : ''}`}
                  onClick={() => setServerFilter('all')}
                >
                  All ({servers.length})
                </button>
                <button
                  className={`${styles.filterBtn} ${serverFilter === 'connected' ? styles.filterActive : ''}`}
                  onClick={() => setServerFilter('connected')}
                >
                  Connected ({connectedCount})
                </button>
                <button
                  className={`${styles.filterBtn} ${serverFilter === 'disconnected' ? styles.filterActive : ''}`}
                  onClick={() => setServerFilter('disconnected')}
                >
                  Disconnected ({servers.length - connectedCount})
                </button>
              </div>
            )}
          </div>

          {/* MCP Servers Section */}
          {servers.length === 0 && registryTools.length === 0 && toolsDbTools.length === 0 ? (
            <div className={styles.empty}>
              No MCP servers configured and no built-in tools available.
              <br />
              Click "Add MCP Server" to get started.
            </div>
          ) : (
            <>
              {/* MCP Servers */}
              {filteredServers.length === 0 && servers.length > 0 && (
                <div className={styles.noServersFiltered}>
                  No {serverFilter} servers found
                </div>
              )}
              {filteredServers.map((server) => {
                const isExpanded = expandedServers.has(server.config.id)
                const hasTools = server.status === 'connected' && server.tools && server.tools.length > 0

                return (
                  <div key={server.config.id} className={styles.serverCard}>
                    {/* Server Header - Clickable to expand/collapse */}
                    <div 
                      className={styles.serverHeader}
                      onClick={() => hasTools && toggleServerExpand(server.config.id)}
                      style={{ cursor: hasTools ? 'pointer' : 'default' }}
                    >
                      <div className={styles.serverInfo}>
                        {hasTools && (
                          <span className={styles.expandIcon}>
                            {isExpanded ? '▼' : '▶'}
                          </span>
                        )}
                        {getStatusIcon(server.status)}
                        <span className={styles.serverName}>{server.config.name}</span>
                        <span className={styles.transportBadge}>
                          {getTransportLabel(server.config.transport)}
                        </span>
                        <span className={`${styles.statusBadge} ${styles[server.status]}`}>
                          {server.status}
                        </span>
                        {hasTools && (
                          <span className={styles.toolCount}>
                            {server.tools!.length} tools
                          </span>
                        )}
                      </div>
                      <div className={styles.serverActions} onClick={e => e.stopPropagation()}>
                        <button
                          className={styles.actionBtn}
                          onClick={() => handleToggleConnection(server)}
                          disabled={actionLoading === server.config.id}
                          title={server.status === 'connected' ? 'Disconnect' : 'Connect'}
                        >
                          {actionLoading === server.config.id ? '...' : 
                            server.status === 'connected' ? '⏹' : '▶'}
                        </button>
                        <button
                          className={styles.actionBtn}
                          onClick={() => setEditingServer(server.config)}
                          title="Edit"
                        >
                          ⚙
                        </button>
                        <button
                          className={styles.actionBtn}
                          onClick={() => handleDelete(server.config.id)}
                          disabled={actionLoading === server.config.id}
                          title="Delete"
                        >
                          🗑
                        </button>
                      </div>
                    </div>

                    {/* Server Description */}
                    {server.config.description && (
                      <div className={styles.serverDescription}>
                        {server.config.description}
                      </div>
                    )}

                    {/* Server Error */}
                    {server.error && (
                      <div className={styles.serverError}>
                        {server.error}
                      </div>
                    )}

                    {/* Disconnected hint */}
                    {server.status !== 'connected' && !server.error && (
                      <div className={styles.connectionHint}>
                        Click ▶ to connect and load tools
                      </div>
                    )}

                    {/* Tools List - Collapsible */}
                    {hasTools && isExpanded && (
                      <div className={styles.toolsContainer}>
                        {server.tools!.map((tool) => (
                          <div key={tool.name} className={styles.toolCard}>
                            <div className={styles.toolHeader}>
                              <span className={styles.toolIcon}>🔧</span>
                              <span className={styles.toolName}>{tool.name}</span>
                            </div>
                            {tool.description && (
                              <div className={styles.toolDescription}>
                                {tool.description}
                              </div>
                            )}
                            <div className={styles.toolParams}>
                              <span className={styles.paramsLabel}>Parameters:</span>
                              {renderToolParameters(tool)}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}

              {/* Built-in Tools Section - Frontend registered tools */}
              {registryTools.length > 0 && (
                <div className={`${styles.serverCard} ${styles.builtInSection}`}>
                  <div 
                    className={styles.serverHeader}
                    onClick={() => setBuiltInExpanded(!builtInExpanded)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className={styles.serverInfo}>
                      <span className={styles.expandIcon}>
                        {builtInExpanded ? '▼' : '▶'}
                      </span>
                      <span className={styles.statusDot} data-status="connected">●</span>
                      <span className={styles.serverName}>Built-in Tools</span>
                      <span className={styles.transportBadge}>Frontend</span>
                      <span className={`${styles.statusBadge} ${styles.connected}`}>
                        active
                      </span>
                      <span className={styles.toolCount}>
                        {registryTools.length} tools
                      </span>
                    </div>
                  </div>
                  <div className={styles.serverDescription}>
                    Executable tools registered in the frontend (web search, open web, etc.)
                  </div>

                  {builtInExpanded && (
                    <div className={styles.toolsContainer}>
                      {registryTools.map((tool) => (
                        <div key={tool.metadata.id} className={styles.toolCard}>
                          <div className={styles.toolHeader}>
                            <span className={styles.toolIcon}>🔧</span>
                            <span className={styles.toolName}>{tool.metadata.displayName}</span>
                            <span className={styles.categoryBadge}>{tool.metadata.category}</span>
                          </div>
                          <div className={styles.toolDescription}>
                            {tool.definition.function.description}
                          </div>
                          <div className={styles.toolParams}>
                            <span className={styles.paramsLabel}>Parameters:</span>
                            {renderRegistryToolParameters(tool)}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Tools DB Section - Backend tools_db.json tools */}
              {toolsDbTools.length > 0 && (
                <div className={`${styles.serverCard} ${styles.toolsDbSection}`}>
                  <div 
                    className={styles.serverHeader}
                    onClick={() => setBuiltInExpanded(!builtInExpanded)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className={styles.serverInfo}>
                      <span className={styles.expandIcon}>
                        {builtInExpanded ? '▼' : '▶'}
                      </span>
                      <span className={styles.statusDot} data-status="connected">●</span>
                      <span className={styles.serverName}>Semantic Router Tools</span>
                      <span className={styles.transportBadge}>Backend</span>
                      <span className={`${styles.statusBadge} ${styles.connected}`}>
                        active
                      </span>
                      <span className={styles.toolCount}>
                        {toolsDbTools.length} tools
                      </span>
                    </div>
                  </div>
                  <div className={styles.serverDescription}>
                    Tool definitions for semantic routing (from tools_db.json)
                  </div>

                  {builtInExpanded && (
                    <div className={styles.toolsContainer}>
                      {toolsDbLoading ? (
                        <div className={styles.toolsLoading}>Loading tools...</div>
                      ) : (
                        toolsDbTools.map((tool) => (
                          <div key={tool.tool.function.name} className={styles.toolCard}>
                            <div className={styles.toolHeader}>
                              <span className={styles.toolIcon}>🔧</span>
                              <span className={styles.toolName}>{tool.tool.function.name}</span>
                              <span className={styles.categoryBadge}>{tool.category}</span>
                            </div>
                            <div className={styles.toolDescription}>
                              {tool.tool.function.description}
                            </div>
                            {tool.tags && tool.tags.length > 0 && (
                              <div className={styles.toolTags}>
                                {tool.tags.map(tag => (
                                  <span key={tag} className={styles.tagBadge}>{tag}</span>
                                ))}
                              </div>
                            )}
                            <div className={styles.toolParams}>
                              <span className={styles.paramsLabel}>Parameters:</span>
                              {renderBuiltInToolParameters(tool)}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <div className={styles.footer}>
        <button 
          className={styles.addBtn}
          onClick={() => setShowAddDialog(true)}
        >
          + Add MCP Server
        </button>
        <div className={styles.summary}>
          {totalToolsCount} tools ({mcpToolsCount} from MCP, {builtInCount} built-in) • {connectedCount} connected servers
        </div>
      </div>

      {/* Add/Edit Dialog */}
      {(showAddDialog || editingServer) && (
        <MCPServerDialog
          server={editingServer}
          onSave={async (config) => {
            if (editingServer) {
              await updateServer(editingServer.id, config)
            } else {
              await addServer(config)
            }
            setShowAddDialog(false)
            setEditingServer(null)
          }}
          onTest={testConnection}
          onClose={() => {
            setShowAddDialog(false)
            setEditingServer(null)
          }}
        />
      )}

      {/* Tool Detail Modal */}
      {selectedTool && (
        <ToolDetailModal
          tool={selectedTool}
          onClose={() => setSelectedTool(null)}
        />
      )}
    </div>
  )
}

// ========== Tool Detail Modal Component ==========
interface ToolDetailModalProps {
  tool: UnifiedTool
  onClose: () => void
}

const ToolDetailModal: React.FC<ToolDetailModalProps> = ({ tool, onClose }) => {
  return (
    <div className={styles.dialogOverlay} onClick={onClose}>
      <div className={styles.toolDetailDialog} onClick={e => e.stopPropagation()}>
        <div className={styles.dialogHeader}>
          <div className={styles.toolDetailTitle}>
            <span className={styles.toolDetailIcon}>🔧</span>
            <h3>{tool.name}</h3>
            <span className={`${styles.sourceTypeBadge} ${styles[tool.sourceType]}`}>
              {tool.sourceType === 'mcp' ? '🔌 MCP' : 
               tool.sourceType === 'frontend' ? '⚡ Frontend' : '🌐 Backend'}
            </span>
          </div>
          <button className={styles.closeBtn} onClick={onClose}>×</button>
        </div>

        <div className={styles.toolDetailContent}>
          {/* Source info */}
          <div className={styles.toolDetailSource}>
            <span className={styles.detailLabel}>Source:</span>
            <span className={styles.detailValue}>{tool.source}</span>
          </div>

          {/* Description */}
          <div className={styles.toolDetailDescription}>
            <span className={styles.detailLabel}>Description:</span>
            <p>{tool.description || 'No description'}</p>
          </div>

          {/* Parameters list */}
          <div className={styles.toolDetailParams}>
            <span className={styles.detailLabel}>
              Parameters ({tool.parameters.length}):
            </span>
            {tool.parameters.length === 0 ? (
              <p className={styles.noParamsHint}>This tool requires no parameters</p>
            ) : (
              <div className={styles.paramDetailList}>
                {tool.parameters.map(param => (
                  <div key={param.name} className={styles.paramDetailItem}>
                    <div className={styles.paramDetailHeader}>
                      <span className={styles.paramDetailName}>{param.name}</span>
                      <span className={styles.paramDetailType}>({param.type})</span>
                      {param.required && (
                        <span className={styles.paramDetailRequired}>Required</span>
                      )}
                    </div>
                    {param.description && (
                      <div className={styles.paramDetailDesc}>
                        {param.description}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className={styles.dialogFooter}>
          <button className={styles.cancelBtn} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

// ========== Server Dialog Component ==========
/**
 * Parse headers string to object
 * Format: "Header-Name: value" (one per line)
 */
function parseHeaders(headersStr: string): Record<string, string> | undefined {
  if (!headersStr.trim()) return undefined
  
  const headers: Record<string, string> = {}
  const lines = headersStr.split('\n')
  
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) continue
    
    const colonIndex = trimmed.indexOf(':')
    if (colonIndex === -1) continue
    
    const key = trimmed.slice(0, colonIndex).trim()
    const value = trimmed.slice(colonIndex + 1).trim()
    
    if (key && value) {
      headers[key] = value
    }
  }
  
  return Object.keys(headers).length > 0 ? headers : undefined
}

interface MCPServerDialogProps {
  server: MCPServerConfig | null
  onSave: (config: Omit<MCPServerConfig, 'id'>) => Promise<void>
  onTest: (config: MCPServerConfig) => Promise<{ success: boolean; error?: string }>
  onClose: () => void
}

const MCPServerDialog: React.FC<MCPServerDialogProps> = ({
  server,
  onSave,
  onTest,
  onClose,
}) => {
  const [name, setName] = useState(server?.name || '')
  const [description, setDescription] = useState(server?.description || '')
  const [transport, setTransport] = useState<MCPTransportType>(server?.transport || 'stdio')
  const [enabled, setEnabled] = useState(server?.enabled ?? true)
  
  // Stdio config
  const [command, setCommand] = useState(server?.connection?.command || '')
  const [args, setArgs] = useState(server?.connection?.args?.join('\n') || '')
  
  // HTTP config
  const [url, setUrl] = useState(server?.connection?.url || '')
  // Headers for authentication (key:value per line)
  const [headers, setHeaders] = useState(() => {
    const h = server?.connection?.headers
    if (!h || Object.keys(h).length === 0) return ''
    return Object.entries(h).map(([k, v]) => `${k}: ${v}`).join('\n')
  })
  
  // Options
  const [timeout, setTimeout] = useState(server?.options?.timeout?.toString() || '30000')
  const [autoReconnect, setAutoReconnect] = useState(server?.options?.autoReconnect ?? true)
  
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ success: boolean; error?: string } | null>(null)

  const handleSave = async () => {
    setSaving(true)
    try {
      const config: Omit<MCPServerConfig, 'id'> = {
        name,
        description: description || undefined,
        transport,
        enabled,
        connection: transport === 'stdio'
          ? {
              command,
              args: args.split('\n').filter(a => a.trim()),
            }
          : {
              url,
              headers: parseHeaders(headers),
            },
        options: {
          timeout: parseInt(timeout) || 30000,
          autoReconnect,
        },
      }
      await onSave(config)
    } catch (err) {
      console.error('Save failed:', err)
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const config: MCPServerConfig = {
        id: server?.id || 'test',
        name,
        description,
        transport,
        enabled,
        connection: transport === 'stdio'
          ? { command, args: args.split('\n').filter(a => a.trim()) }
          : { url, headers: parseHeaders(headers) },
        options: { timeout: parseInt(timeout) || 30000 },
      }
      const result = await onTest(config)
      setTestResult(result)
    } catch (err) {
      setTestResult({ success: false, error: err instanceof Error ? err.message : 'Test failed' })
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className={styles.dialogOverlay} onClick={onClose}>
      <div className={styles.dialog} onClick={e => e.stopPropagation()}>
        <div className={styles.dialogHeader}>
          <h3>{server ? 'Edit MCP Server' : 'Add MCP Server'}</h3>
          <button className={styles.closeBtn} onClick={onClose}>×</button>
        </div>

        <div className={styles.dialogContent}>
          <div className={styles.formGroup}>
            <label>Name *</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="My MCP Server"
            />
          </div>

          <div className={styles.formGroup}>
            <label>Description</label>
            <input
              type="text"
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Optional description"
            />
          </div>

          <div className={styles.formGroup}>
            <label>Transport Protocol *</label>
            <div className={styles.radioGroup}>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  name="transport"
                  value="stdio"
                  checked={transport === 'stdio'}
                  onChange={() => setTransport('stdio')}
                />
                <span>Stdio</span>
                <small>Local command line (filesystem, git, etc.)</small>
              </label>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  name="transport"
                  value="streamable-http"
                  checked={transport === 'streamable-http'}
                  onChange={() => setTransport('streamable-http')}
                />
                <span>Streamable HTTP</span>
                <small>Remote service with streaming support</small>
              </label>
            </div>
          </div>

          {transport === 'stdio' ? (
            <>
              <div className={styles.formGroup}>
                <label>Command *</label>
                <input
                  type="text"
                  value={command}
                  onChange={e => setCommand(e.target.value)}
                  placeholder="npx"
                />
              </div>
              <div className={styles.formGroup}>
                <label>Arguments (one per line)</label>
                <textarea
                  value={args}
                  onChange={e => setArgs(e.target.value)}
                  placeholder={"-y\n@modelcontextprotocol/server-filesystem\n/Users/workspace"}
                  rows={4}
                />
              </div>
            </>
          ) : (
            <>
              <div className={styles.formGroup}>
                <label>URL *</label>
                <input
                  type="text"
                  value={url}
                  onChange={e => setUrl(e.target.value)}
                  placeholder="https://api.example.com/mcp"
                />
              </div>
              <div className={styles.formGroup}>
                <label>Headers (for authentication, one per line)</label>
                <textarea
                  value={headers}
                  onChange={e => setHeaders(e.target.value)}
                  placeholder={"Authorization: Bearer your-token\nX-API-Key: your-api-key"}
                  rows={3}
                />
                <small className={styles.fieldHint}>Format: Header-Name: value (one per line)</small>
              </div>
            </>
          )}

          <div className={styles.formGroup}>
            <label>Timeout (ms)</label>
            <input
              type="number"
              value={timeout}
              onChange={e => setTimeout(e.target.value)}
              placeholder="30000"
            />
          </div>

          <div className={styles.formGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={autoReconnect}
                onChange={e => setAutoReconnect(e.target.checked)}
              />
              <span>Auto Reconnect</span>
            </label>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={enabled}
                onChange={e => setEnabled(e.target.checked)}
              />
              <span>Enabled</span>
            </label>
          </div>

          {testResult && (
            <div className={testResult.success ? styles.testSuccess : styles.testError}>
              {testResult.success ? '✓ Connection successful!' : `✗ ${testResult.error}`}
            </div>
          )}
        </div>

        <div className={styles.dialogFooter}>
          <button className={styles.cancelBtn} onClick={onClose}>
            Cancel
          </button>
          <button 
            className={styles.testBtn} 
            onClick={handleTest}
            disabled={testing || !name || (transport === 'stdio' ? !command : !url)}
          >
            {testing ? 'Testing...' : 'Test Connection'}
          </button>
          <button 
            className={styles.saveBtn} 
            onClick={handleSave}
            disabled={saving || !name || (transport === 'stdio' ? !command : !url)}
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default MCPConfigPanel
