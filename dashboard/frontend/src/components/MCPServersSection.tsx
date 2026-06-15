import React from 'react'
import type { MCPServerConfig, MCPServerState } from '../tools/mcp'
import type { RegisteredTool } from '../tools'
import styles from './MCPConfigPanel.module.css'
import type { BuiltInTool, ServerFilter } from './mcpConfigPanelTypes'
import {
  extractBuiltInToolParameters,
  extractMCPToolParameters,
  extractRegisteredToolParameters,
  getTransportLabel,
} from './mcpConfigPanelUtils'
import { MCPToolParameterList } from './MCPToolParameterList'

interface MCPServersSectionProps {
  actionLoading: string | null
  builtInExpanded: boolean
  expandedServers: Set<string>
  filteredServers: MCPServerState[]
  isReadonly: boolean
  registryTools: RegisteredTool[]
  serverFilter: ServerFilter
  servers: MCPServerState[]
  toolsDbLoading: boolean
  toolsDbTools: BuiltInTool[]
  onDeleteServer: (id: string) => void
  onEditServer: (server: MCPServerConfig) => void
  onServerFilterChange: (filter: ServerFilter) => void
  onToggleBuiltInExpanded: () => void
  onToggleConnection: (server: MCPServerState) => void
  onToggleServerExpand: (id: string) => void
}

export const MCPServersSection: React.FC<MCPServersSectionProps> = ({
  actionLoading,
  builtInExpanded,
  expandedServers,
  filteredServers,
  isReadonly,
  registryTools,
  serverFilter,
  servers,
  toolsDbLoading,
  toolsDbTools,
  onDeleteServer,
  onEditServer,
  onServerFilterChange,
  onToggleBuiltInExpanded,
  onToggleConnection,
  onToggleServerExpand,
}) => {
  const connectedCount = servers.filter(server => server.status === 'connected').length

  return (
    <div className={styles.mcpServersSection}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionTitle}>
          <span>📡 MCP Servers</span>
          <span className={styles.serverCountBadge}>
            {connectedCount} / {servers.length} connected
          </span>
        </div>
        {servers.length > 0 && (
          <div className={styles.serverFilterWrapper}>
            <button
              className={`${styles.filterBtn} ${serverFilter === 'all' ? styles.filterActive : ''}`}
              onClick={() => onServerFilterChange('all')}
            >
              All ({servers.length})
            </button>
            <button
              className={`${styles.filterBtn} ${serverFilter === 'connected' ? styles.filterActive : ''}`}
              onClick={() => onServerFilterChange('connected')}
            >
              Connected ({connectedCount})
            </button>
            <button
              className={`${styles.filterBtn} ${serverFilter === 'disconnected' ? styles.filterActive : ''}`}
              onClick={() => onServerFilterChange('disconnected')}
            >
              Disconnected ({servers.length - connectedCount})
            </button>
          </div>
        )}
      </div>

      {servers.length === 0 && registryTools.length === 0 && toolsDbTools.length === 0 ? (
        <div className={styles.empty}>
          No MCP servers configured and no built-in tools available.
          <br />
          Click "Add MCP Server" to get started.
        </div>
      ) : (
        <>
          {filteredServers.length === 0 && servers.length > 0 && (
            <div className={styles.noServersFiltered}>No {serverFilter} servers found</div>
          )}

          {filteredServers.map(server => {
            const isExpanded = expandedServers.has(server.config.id)
            const hasTools = server.status === 'connected' && Boolean(server.tools?.length)

            return (
              <div key={server.config.id} className={styles.serverCard}>
                <div
                  className={styles.serverHeader}
                  onClick={() => hasTools && onToggleServerExpand(server.config.id)}
                  style={{ cursor: hasTools ? 'pointer' : 'default' }}
                >
                  <div className={styles.serverInfo}>
                    {hasTools && (
                      <span className={styles.expandIcon}>{isExpanded ? '▼' : '▶'}</span>
                    )}
                    {renderStatusIcon(server.status)}
                    <span className={styles.serverName}>{server.config.name}</span>
                    <span className={styles.transportBadge}>
                      {getTransportLabel(server.config.transport)}
                    </span>
                    <span className={`${styles.statusBadge} ${styles[server.status]}`}>
                      {server.status}
                    </span>
                    {hasTools && (
                      <span className={styles.toolCount}>{server.tools?.length || 0} tools</span>
                    )}
                  </div>

                  <div className={styles.serverActions} onClick={event => event.stopPropagation()}>
                    <button
                      className={styles.actionBtn}
                      onClick={() => onToggleConnection(server)}
                      disabled={isReadonly || actionLoading === server.config.id}
                      title={server.status === 'connected' ? 'Disconnect' : 'Connect'}
                    >
                      {actionLoading === server.config.id
                        ? '...'
                        : server.status === 'connected'
                          ? '⏹'
                          : '▶'}
                    </button>
                    <button
                      className={styles.actionBtn}
                      onClick={() => {
                        if (isReadonly) {
                          return
                        }
                        onEditServer(server.config)
                      }}
                      disabled={isReadonly}
                      title="Edit"
                    >
                      ⚙
                    </button>
                    <button
                      className={styles.actionBtn}
                      onClick={() => onDeleteServer(server.config.id)}
                      disabled={isReadonly || actionLoading === server.config.id}
                      title="Delete"
                    >
                      🗑
                    </button>
                  </div>
                </div>

                {server.config.description && (
                  <div className={styles.serverDescription}>{server.config.description}</div>
                )}

                {server.error && <div className={styles.serverError}>{server.error}</div>}

                {server.status !== 'connected' && !server.error && (
                  <div className={styles.connectionHint}>Click ▶ to connect and load tools</div>
                )}

                {hasTools && isExpanded && (
                  <div className={styles.toolsContainer}>
                    {server.tools?.map(tool => (
                      <div key={tool.name} className={styles.toolCard}>
                        <div className={styles.toolHeader}>
                          <span className={styles.toolIcon}>🔧</span>
                          <span className={styles.toolName}>{tool.name}</span>
                        </div>
                        {tool.description && (
                          <div className={styles.toolDescription}>{tool.description}</div>
                        )}
                        <div className={styles.toolParams}>
                          <span className={styles.paramsLabel}>Parameters:</span>
                          <MCPToolParameterList parameters={extractMCPToolParameters(tool)} />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}

          {registryTools.length > 0 && (
            <div className={`${styles.serverCard} ${styles.builtInSection}`}>
              <div
                className={styles.serverHeader}
                onClick={onToggleBuiltInExpanded}
                style={{ cursor: 'pointer' }}
              >
                <div className={styles.serverInfo}>
                  <span className={styles.expandIcon}>{builtInExpanded ? '▼' : '▶'}</span>
                  <span className={styles.statusDot} data-status="connected">●</span>
                  <span className={styles.serverName}>Built-in Tools</span>
                  <span className={styles.transportBadge}>Frontend</span>
                  <span className={`${styles.statusBadge} ${styles.connected}`}>active</span>
                  <span className={styles.toolCount}>{registryTools.length} tools</span>
                </div>
              </div>
              <div className={styles.serverDescription}>
                Executable tools registered in the frontend (web search, open web, etc.)
              </div>

              {builtInExpanded && (
                <div className={styles.toolsContainer}>
                  {registryTools.map(tool => (
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
                        <MCPToolParameterList parameters={extractRegisteredToolParameters(tool)} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {toolsDbTools.length > 0 && (
            <div className={`${styles.serverCard} ${styles.toolsDbSection}`}>
              <div
                className={styles.serverHeader}
                onClick={onToggleBuiltInExpanded}
                style={{ cursor: 'pointer' }}
              >
                <div className={styles.serverInfo}>
                  <span className={styles.expandIcon}>{builtInExpanded ? '▼' : '▶'}</span>
                  <span className={styles.statusDot} data-status="connected">●</span>
                  <span className={styles.serverName}>Semantic Router Tools</span>
                  <span className={styles.transportBadge}>Backend</span>
                  <span className={`${styles.statusBadge} ${styles.connected}`}>active</span>
                  <span className={styles.toolCount}>{toolsDbTools.length} tools</span>
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
                    toolsDbTools.map(tool => (
                      <div key={tool.tool.function.name} className={styles.toolCard}>
                        <div className={styles.toolHeader}>
                          <span className={styles.toolIcon}>🔧</span>
                          <span className={styles.toolName}>{tool.tool.function.name}</span>
                          <span className={styles.categoryBadge}>{tool.category}</span>
                        </div>
                        <div className={styles.toolDescription}>
                          {tool.tool.function.description}
                        </div>
                        {tool.tags.length > 0 && (
                          <div className={styles.toolTags}>
                            {tool.tags.map(tag => (
                              <span key={tag} className={styles.tagBadge}>{tag}</span>
                            ))}
                          </div>
                        )}
                        <div className={styles.toolParams}>
                          <span className={styles.paramsLabel}>Parameters:</span>
                          <MCPToolParameterList parameters={extractBuiltInToolParameters(tool)} />
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
  )
}

function renderStatusIcon(status: MCPServerState['status']) {
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
