import React, { useEffect, useId, useState } from 'react'

import type { RegisteredTool } from '../tools'
import type { MCPServerConfig, MCPServerState } from '../tools/mcp'
import styles from './MCPConfigPanel.module.css'
import { MCPListPagination } from './MCPListPagination'
import type { BuiltInTool, ServerFilter, ServerSort, UnifiedTool } from './mcpConfigPanelTypes'
import { MCP_NESTED_TOOLS_PAGE_SIZE, MCP_SERVERS_PAGE_SIZE } from './mcpConfigPanelTypes'
import {
  getMCPPageCount,
  getTransportLabel,
  paginateMCPItems,
  toUnifiedBuiltInTool,
  toUnifiedMCPTool,
  toUnifiedRegisteredTool,
} from './mcpConfigPanelUtils'

interface MCPServersSectionProps {
  actionLoading: string | null
  builtInExpanded: boolean
  expandedServers: Set<string>
  filteredServers: MCPServerState[]
  isReadonly: boolean
  registryTools: RegisteredTool[]
  serverFilter: ServerFilter
  serverSearch: string
  serverSort: ServerSort
  servers: MCPServerState[]
  serversSectionExpanded: boolean
  toolsDbExpanded: boolean
  toolsDbLoading: boolean
  toolsDbTools: BuiltInTool[]
  onDeleteServer: (id: string) => void
  onEditServer: (server: MCPServerConfig) => void
  onSelectTool: (tool: UnifiedTool) => void
  onServerFilterChange: (filter: ServerFilter) => void
  onServerSearchChange: (value: string) => void
  onServerSortChange: (sort: ServerSort) => void
  onToggleBuiltInExpanded: () => void
  onToggleConnection: (server: MCPServerState) => void
  onToggleServerExpand: (id: string) => void
  onToggleServersSection: () => void
  onToggleToolsDbExpanded: () => void
}

export const MCPServersSection: React.FC<MCPServersSectionProps> = ({
  actionLoading,
  builtInExpanded,
  expandedServers,
  filteredServers,
  isReadonly,
  registryTools,
  serverFilter,
  serverSearch,
  serverSort,
  servers,
  serversSectionExpanded,
  toolsDbExpanded,
  toolsDbLoading,
  toolsDbTools,
  onDeleteServer,
  onEditServer,
  onSelectTool,
  onServerFilterChange,
  onServerSearchChange,
  onServerSortChange,
  onToggleBuiltInExpanded,
  onToggleConnection,
  onToggleServerExpand,
  onToggleServersSection,
  onToggleToolsDbExpanded,
}) => {
  const contentId = useId()
  const searchInputId = useId()
  const [page, setPage] = useState(1)
  const connectedCount = servers.filter((server) => server.status === 'connected').length
  const pageCount = getMCPPageCount(filteredServers.length, MCP_SERVERS_PAGE_SIZE)
  const visibleServers = paginateMCPItems(filteredServers, page, MCP_SERVERS_PAGE_SIZE)

  useEffect(() => setPage(1), [serverFilter, serverSearch, serverSort])
  useEffect(() => {
    if (page > pageCount) setPage(pageCount)
  }, [page, pageCount])

  return (
    <section className={styles.mcpServersSection}>
      <div className={styles.sectionHeader}>
        <button
          type="button"
          className={styles.sectionToggle}
          onClick={onToggleServersSection}
          aria-expanded={serversSectionExpanded}
          aria-controls={contentId}
        >
          <span className={styles.expandIcon} aria-hidden="true">
            {serversSectionExpanded ? '▼' : '▶'}
          </span>
          <span>MCP Servers</span>
          <span className={styles.serverCountBadge}>
            {connectedCount} / {servers.length} connected
          </span>
        </button>
      </div>

      {serversSectionExpanded ? (
        <div id={contentId} className={styles.serverCatalog}>
          {servers.length > 0 ? (
            <>
              <div className={styles.catalogControls}>
                <div className={styles.searchField}>
                  <label htmlFor={searchInputId}>Search servers</label>
                  <div className={styles.toolSearchWrapper}>
                    <input
                      id={searchInputId}
                      type="search"
                      className={styles.toolSearchInput}
                      placeholder="Name, endpoint, or tool"
                      value={serverSearch}
                      onChange={(event) => onServerSearchChange(event.target.value)}
                    />
                    {serverSearch ? (
                      <button
                        type="button"
                        className={styles.clearSearchBtn}
                        onClick={() => onServerSearchChange('')}
                        aria-label="Clear server search"
                      >
                        ×
                      </button>
                    ) : null}
                  </div>
                </div>
                <label className={styles.catalogSelect}>
                  <span>Status</span>
                  <select
                    value={serverFilter}
                    onChange={(event) => onServerFilterChange(event.target.value as ServerFilter)}
                  >
                    <option value="all">All statuses</option>
                    <option value="connected">Connected</option>
                    <option value="disconnected">Not connected</option>
                  </select>
                </label>
                <label className={styles.catalogSelect}>
                  <span>Sort</span>
                  <select
                    value={serverSort}
                    onChange={(event) => onServerSortChange(event.target.value as ServerSort)}
                  >
                    <option value="name-asc">Name A–Z</option>
                    <option value="status">Connection status</option>
                    <option value="tools-desc">Most tools</option>
                  </select>
                </label>
              </div>
              <div className={styles.catalogMeta} aria-live="polite">
                <strong>{filteredServers.length}</strong> of {servers.length} servers
                <span>Client view · {MCP_SERVERS_PAGE_SIZE} per page</span>
              </div>
            </>
          ) : null}

          {servers.length === 0 &&
          registryTools.length === 0 &&
          toolsDbTools.length === 0 &&
          !toolsDbLoading ? (
            <div className={styles.empty}>
              {isReadonly
                ? 'No MCP servers or built-in tools are configured.'
                : 'No MCP servers are configured. Add a server to get started.'}
            </div>
          ) : null}

          {filteredServers.length === 0 && servers.length > 0 ? (
            <div className={styles.noServersFiltered}>
              No servers match the current search and status filter.
            </div>
          ) : null}

          {visibleServers.map((server) => {
            const isExpanded = expandedServers.has(server.config.id)
            const hasTools = server.status === 'connected' && Boolean(server.tools?.length)
            const serverToolsId = `mcp-server-tools-${server.config.id.replace(/[^a-zA-Z0-9_-]/g, '-')}`
            const unifiedTools = (server.tools || []).map((tool) => toUnifiedMCPTool(server, tool))

            return (
              <article key={server.config.id} className={styles.serverCard}>
                <div className={styles.serverHeader}>
                  <button
                    type="button"
                    className={styles.serverHeaderButton}
                    onClick={() => hasTools && onToggleServerExpand(server.config.id)}
                    disabled={!hasTools}
                    aria-expanded={hasTools ? isExpanded : undefined}
                    aria-controls={hasTools ? serverToolsId : undefined}
                  >
                    <span className={styles.serverInfo}>
                      <span className={styles.expandIcon} aria-hidden="true">
                        {hasTools ? (isExpanded ? '▼' : '▶') : '•'}
                      </span>
                      {renderStatusIcon(server.status)}
                      <span className={styles.serverName}>{server.config.name}</span>
                      <span className={styles.transportBadge}>
                        {getTransportLabel(server.config.transport)}
                      </span>
                      <span className={`${styles.statusBadge} ${styles[server.status]}`}>
                        {server.status}
                      </span>
                      {hasTools ? (
                        <span className={styles.toolCount}>{server.tools?.length || 0} tools</span>
                      ) : null}
                    </span>
                  </button>

                  <div className={styles.serverActions}>
                    <button
                      type="button"
                      className={styles.actionBtn}
                      onClick={() => onToggleConnection(server)}
                      disabled={isReadonly || Boolean(actionLoading)}
                      title={server.status === 'connected' ? 'Disconnect' : 'Connect'}
                      aria-label={`${server.status === 'connected' ? 'Disconnect' : 'Connect'} ${server.config.name}`}
                    >
                      {actionLoading === server.config.id
                        ? '…'
                        : server.status === 'connected'
                          ? '⏹'
                          : '▶'}
                    </button>
                    <button
                      type="button"
                      className={styles.actionBtn}
                      onClick={() => onEditServer(server.config)}
                      disabled={isReadonly || Boolean(actionLoading)}
                      title="Edit"
                      aria-label={`Edit ${server.config.name}`}
                    >
                      ⚙
                    </button>
                    <button
                      type="button"
                      className={styles.actionBtn}
                      onClick={() => onDeleteServer(server.config.id)}
                      disabled={isReadonly || Boolean(actionLoading)}
                      title="Delete"
                      aria-label={`Delete ${server.config.name}`}
                    >
                      🗑
                    </button>
                  </div>
                </div>

                {server.config.description ? (
                  <div className={styles.serverDescription}>{server.config.description}</div>
                ) : null}
                {server.error ? <div className={styles.serverError}>{server.error}</div> : null}
                {server.status !== 'connected' && !server.error ? (
                  <div className={styles.connectionHint}>
                    Connect this server to discover its tools.
                  </div>
                ) : null}
                {hasTools && isExpanded ? (
                  <ToolCollection
                    id={serverToolsId}
                    label={`${server.config.name} tools`}
                    tools={unifiedTools}
                    onSelectTool={onSelectTool}
                  />
                ) : null}
              </article>
            )
          })}

          {filteredServers.length > 0 ? (
            <MCPListPagination
              itemCount={filteredServers.length}
              itemLabel="servers"
              page={page}
              pageSize={MCP_SERVERS_PAGE_SIZE}
              onPageChange={setPage}
            />
          ) : null}

          {registryTools.length > 0 ? (
            <ToolSourceCard
              title="Built-in Tools"
              description="Executable tools registered in the dashboard frontend."
              badge="Frontend"
              expanded={builtInExpanded}
              tools={registryTools.map(toUnifiedRegisteredTool)}
              onToggle={onToggleBuiltInExpanded}
              onSelectTool={onSelectTool}
            />
          ) : null}

          {toolsDbLoading || toolsDbTools.length > 0 ? (
            <ToolSourceCard
              title="Semantic Router Tools"
              description="Tool definitions exposed by the Semantic Router tools database."
              badge="Backend"
              expanded={toolsDbExpanded}
              loading={toolsDbLoading}
              tools={toolsDbTools.map(toUnifiedBuiltInTool)}
              onToggle={onToggleToolsDbExpanded}
              onSelectTool={onSelectTool}
            />
          ) : null}
        </div>
      ) : null}
    </section>
  )
}

interface ToolSourceCardProps {
  badge: string
  description: string
  expanded: boolean
  loading?: boolean
  title: string
  tools: UnifiedTool[]
  onSelectTool: (tool: UnifiedTool) => void
  onToggle: () => void
}

function ToolSourceCard({
  badge,
  description,
  expanded,
  loading = false,
  title,
  tools,
  onSelectTool,
  onToggle,
}: ToolSourceCardProps) {
  const contentId = useId()
  return (
    <article className={`${styles.serverCard} ${styles.builtInSection}`}>
      <div className={styles.serverHeader}>
        <button
          type="button"
          className={styles.serverHeaderButton}
          onClick={onToggle}
          aria-expanded={expanded}
          aria-controls={contentId}
        >
          <span className={styles.serverInfo}>
            <span className={styles.expandIcon} aria-hidden="true">
              {expanded ? '▼' : '▶'}
            </span>
            <span className={styles.statusDot} data-status="connected" aria-hidden="true">
              ●
            </span>
            <span className={styles.serverName}>{title}</span>
            <span className={styles.transportBadge}>{badge}</span>
            <span className={`${styles.statusBadge} ${styles.connected}`}>
              {loading ? 'loading' : 'active'}
            </span>
            <span className={styles.toolCount}>{tools.length} tools</span>
          </span>
        </button>
      </div>
      <div className={styles.serverDescription}>{description}</div>
      {expanded ? (
        loading && tools.length === 0 ? (
          <div id={contentId} className={styles.toolsLoading} role="status">
            Loading tools…
          </div>
        ) : (
          <ToolCollection
            id={contentId}
            label={`${title} catalog`}
            tools={tools}
            onSelectTool={onSelectTool}
          />
        )
      ) : null}
    </article>
  )
}

interface ToolCollectionProps {
  id: string
  label: string
  tools: UnifiedTool[]
  onSelectTool: (tool: UnifiedTool) => void
}

function ToolCollection({ id, label, tools, onSelectTool }: ToolCollectionProps) {
  const [page, setPage] = useState(1)
  const pageCount = getMCPPageCount(tools.length, MCP_NESTED_TOOLS_PAGE_SIZE)
  const visibleTools = paginateMCPItems(tools, page, MCP_NESTED_TOOLS_PAGE_SIZE)

  useEffect(() => {
    if (page > pageCount) setPage(pageCount)
  }, [page, pageCount])

  return (
    <div id={id} className={styles.toolsContainer} aria-label={label}>
      {visibleTools.map((tool) => (
        <button
          type="button"
          key={tool.id}
          className={styles.toolCard}
          onClick={() => onSelectTool(tool)}
          aria-label={`View details for ${tool.name}`}
        >
          <span className={styles.toolHeader}>
            <span className={styles.toolIcon} aria-hidden="true">
              🔧
            </span>
            <span className={styles.toolName}>{tool.name}</span>
          </span>
          <span className={styles.toolDescription}>{tool.description || 'No description'}</span>
          <span className={styles.toolParams}>
            {tool.parameters.length} {tool.parameters.length === 1 ? 'parameter' : 'parameters'}
          </span>
        </button>
      ))}
      <MCPListPagination
        itemCount={tools.length}
        itemLabel="tools"
        page={page}
        pageSize={MCP_NESTED_TOOLS_PAGE_SIZE}
        onPageChange={setPage}
      />
    </div>
  )
}

function renderStatusIcon(status: MCPServerState['status']) {
  switch (status) {
    case 'connected':
      return (
        <span className={styles.statusDot} data-status="connected" aria-hidden="true">
          ●
        </span>
      )
    case 'connecting':
      return (
        <span className={styles.statusDot} data-status="connecting" aria-hidden="true">
          ◐
        </span>
      )
    case 'error':
      return (
        <span className={styles.statusDot} data-status="error" aria-hidden="true">
          ●
        </span>
      )
    default:
      return (
        <span className={styles.statusDot} data-status="disconnected" aria-hidden="true">
          ○
        </span>
      )
  }
}
