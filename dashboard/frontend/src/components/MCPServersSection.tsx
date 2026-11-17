import React, { useEffect, useState } from 'react'
import { Blocks, Pencil, Plug, Plus, Power, RefreshCw, Trash2, Unplug, Wrench } from 'lucide-react'

import type { MCPServerConfig, MCPServerState } from '../tools/mcp'
import {
  Badge,
  Button,
  DataTable,
  DataToolbar,
  DropdownMenuItem,
  DropdownMenuSeparator,
  EmptyState,
  IconTile,
  Segmented,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  StatusLabel,
  cn,
  type Column,
} from '../ui'
import { MCPListPagination } from './MCPListPagination'
import type { ServerFilter, ServerSort } from './mcpConfigPanelTypes'
import { MCP_SERVERS_PAGE_SIZE } from './mcpConfigPanelTypes'
import {
  SERVER_STATUS_META,
  TRANSPORT_META,
  getMCPPageCount,
  paginateMCPItems,
  serverTarget,
} from './mcpConfigPanelUtils'

interface MCPServersSectionProps {
  servers: MCPServerState[]
  filteredServers: MCPServerState[]
  serverFilter: ServerFilter
  serverSearch: string
  serverSort: ServerSort
  isReadonly: boolean
  actionLoading: string | null
  activeServerId?: string
  onServerFilterChange: (filter: ServerFilter) => void
  onServerSearchChange: (value: string) => void
  onServerSortChange: (sort: ServerSort) => void
  onOpenServer: (id: string) => void
  onToggleConnection: (server: MCPServerState) => void
  onToggleEnabled: (server: MCPServerState) => void
  onEditServer: (server: MCPServerConfig) => void
  onDeleteServer: (id: string) => void
  onAddServer: () => void
  onRefresh: () => void
}

export const MCPServersSection: React.FC<MCPServersSectionProps> = ({
  servers,
  filteredServers,
  serverFilter,
  serverSearch,
  serverSort,
  isReadonly,
  actionLoading,
  activeServerId,
  onServerFilterChange,
  onServerSearchChange,
  onServerSortChange,
  onOpenServer,
  onToggleConnection,
  onToggleEnabled,
  onEditServer,
  onDeleteServer,
  onAddServer,
  onRefresh,
}) => {
  const [page, setPage] = useState(1)
  const connectedCount = servers.filter((server) => server.status === 'connected').length
  const pageCount = getMCPPageCount(filteredServers.length, MCP_SERVERS_PAGE_SIZE)
  const visibleServers = paginateMCPItems(filteredServers, page, MCP_SERVERS_PAGE_SIZE)

  useEffect(() => setPage(1), [serverFilter, serverSearch, serverSort])
  useEffect(() => {
    if (page > pageCount) setPage(pageCount)
  }, [page, pageCount])

  const columns: Column<MCPServerState>[] = [
    {
      id: 'name',
      header: 'Server',
      cell: (row) => {
        const transport = TRANSPORT_META[row.config.transport] ?? TRANSPORT_META.stdio
        return (
          <div className="flex min-w-0 items-center gap-3">
            <IconTile
              icon={transport.icon}
              tone="slate"
              size="sm"
              className={cn(!row.config.enabled && 'opacity-60')}
            />
            <div className="flex min-w-0 flex-col">
              <span className="flex items-center gap-1.5">
                <span className="truncate font-mono text-[12.5px] font-medium">
                  {row.config.name}
                </span>
                {!row.config.enabled ? (
                  <Badge size="sm" tone="outline">
                    off
                  </Badge>
                ) : null}
              </span>
              <span className="max-w-[320px] truncate font-mono text-[11.5px] text-subtle-foreground">
                {serverTarget(row.config) || row.config.description || '—'}
              </span>
            </div>
          </div>
        )
      },
    },
    {
      id: 'transport',
      header: 'Transport',
      hideBelow: 'xl',
      cell: (row) => (
        <Badge tone="outline">
          {TRANSPORT_META[row.config.transport]?.label ?? row.config.transport}
        </Badge>
      ),
    },
    {
      id: 'status',
      header: 'Status',
      cell: (row) => {
        const meta = SERVER_STATUS_META[row.status] ?? SERVER_STATUS_META.disconnected
        return (
          <StatusLabel tone={meta.tone} pulse={meta.pulse}>
            {actionLoading === row.config.id ? 'Working…' : meta.label}
          </StatusLabel>
        )
      },
    },
    {
      id: 'tools',
      header: 'Tools',
      align: 'right',
      cell: (row) =>
        row.status === 'connected' ? (
          <span className="text-[13px] tabular">{row.tools?.length ?? 0}</span>
        ) : (
          <span className="text-xs text-subtle-foreground">—</span>
        ),
    },
  ]

  return (
    <section className="flex flex-col gap-3" aria-labelledby="mcp-servers-title">
      <div className="flex flex-col gap-0.5">
        <h3 id="mcp-servers-title" className="text-[15px] font-semibold tracking-[-0.012em]">
          MCP servers
        </h3>
        <p className="text-[13px] text-muted-foreground">
          Servers the dashboard connects to. Their tools become available to decisions.
        </p>
      </div>
      <div>
        <DataToolbar
          search={serverSearch}
          onSearchChange={onServerSearchChange}
          placeholder="Search servers or tools…"
          filters={
            servers.length ? (
              <>
                <Segmented
                  size="sm"
                  aria-label="Status"
                  value={serverFilter}
                  onValueChange={onServerFilterChange}
                  options={[
                    { value: 'all', label: 'All' },
                    { value: 'connected', label: 'Connected' },
                    { value: 'disconnected', label: 'Not connected' },
                  ]}
                />
                <Select
                  value={serverSort}
                  onValueChange={(value) => onServerSortChange(value as ServerSort)}
                >
                  <SelectTrigger size="sm" aria-label="Sort servers" className="min-w-36">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="name-asc">Name A–Z</SelectItem>
                    <SelectItem value="status">Connection status</SelectItem>
                    <SelectItem value="tools-desc">Most tools</SelectItem>
                  </SelectContent>
                </Select>
              </>
            ) : null
          }
          actions={
            <>
              <span className="text-xs text-muted-foreground">
                <span className="tabular">{connectedCount}</span> of{' '}
                <span className="tabular">{servers.length}</span> connected
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={onRefresh}
                aria-label="Refresh MCP data"
              >
                <RefreshCw />
              </Button>
            </>
          }
        />
        <DataTable
          data={visibleServers}
          columns={columns}
          getRowId={(row) => row.config.id}
          activeRowId={activeServerId}
          onRowClick={(row) => onOpenServer(row.config.id)}
          rowActions={(row) => (
            <>
              <DropdownMenuItem onSelect={() => onOpenServer(row.config.id)}>
                <Wrench />
                View tools
              </DropdownMenuItem>
              <DropdownMenuItem
                disabled={isReadonly || Boolean(actionLoading)}
                onSelect={() => onToggleConnection(row)}
                aria-label={`${row.status === 'connected' ? 'Disconnect' : 'Connect'} ${row.config.name}`}
              >
                {row.status === 'connected' ? <Unplug /> : <Plug />}
                {row.status === 'connected' ? 'Disconnect' : 'Connect'}
              </DropdownMenuItem>
              <DropdownMenuItem
                disabled={isReadonly || Boolean(actionLoading)}
                onSelect={() => onEditServer(row.config)}
                aria-label={`Edit ${row.config.name}`}
              >
                <Pencil />
                Edit
              </DropdownMenuItem>
              <DropdownMenuItem
                disabled={isReadonly || Boolean(actionLoading)}
                onSelect={() => onToggleEnabled(row)}
              >
                <Power />
                {row.config.enabled ? 'Turn off' : 'Turn on'}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                variant="destructive"
                disabled={isReadonly || Boolean(actionLoading)}
                onSelect={() => onDeleteServer(row.config.id)}
                aria-label={`Delete ${row.config.name}`}
              >
                <Trash2 />
                Delete
              </DropdownMenuItem>
            </>
          )}
          empty={
            <EmptyState
              compact
              icon={Blocks}
              tone="slate"
              title={servers.length ? 'No servers match' : 'No MCP servers yet'}
              description={
                servers.length
                  ? 'No servers match the current search and status filter.'
                  : isReadonly
                    ? 'No MCP servers are configured.'
                    : 'Add a server to give decisions tools such as search, files or calendars.'
              }
              actions={
                servers.length ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      onServerSearchChange('')
                      onServerFilterChange('all')
                    }}
                  >
                    Clear filters
                  </Button>
                ) : !isReadonly ? (
                  <Button size="sm" onClick={onAddServer}>
                    <Plus />
                    Add server
                  </Button>
                ) : null
              }
            />
          }
        />
      </div>
      {filteredServers.length > MCP_SERVERS_PAGE_SIZE ? (
        <MCPListPagination
          itemCount={filteredServers.length}
          itemLabel="servers"
          page={page}
          pageSize={MCP_SERVERS_PAGE_SIZE}
          onPageChange={setPage}
        />
      ) : null}
    </section>
  )
}
