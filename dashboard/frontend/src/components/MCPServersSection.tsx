import React, { useEffect, useMemo, useState } from 'react'
import { Blocks, Pencil, Plug, Plus, Power, Trash2, Unplug, Wrench } from 'lucide-react'

import type { MCPServerConfig, MCPServerState } from '../tools/mcp'
import {
  Badge,
  Button,
  DataTable,
  DataToolbar,
  DropdownMenuItem,
  DropdownMenuSeparator,
  EmptyState,
  FilterPill,
  IconTile,
  StatusLabel,
  cn,
  type Column,
  type DataTableSort,
} from '../ui'
import { MCPListPagination } from './MCPListPagination'
import type { ServerFilters } from './mcpConfigPanelTypes'
import { MCP_SERVERS_PAGE_SIZE } from './mcpConfigPanelTypes'
import {
  SERVER_STATUS_META,
  SERVER_STATUS_RANK,
  TRANSPORT_META,
  getMCPPageCount,
  paginateMCPItems,
  serverTarget,
} from './mcpConfigPanelUtils'

const SORT_VALUE: Record<string, (server: MCPServerState) => string | number> = {
  name: (server) => server.config.name.toLowerCase(),
  status: (server) => SERVER_STATUS_RANK[server.status] ?? 9,
  tools: (server) => server.tools?.length ?? 0,
}

function sortServers(servers: MCPServerState[], sort: DataTableSort | null): MCPServerState[] {
  const value = sort ? SORT_VALUE[sort.id] : undefined
  if (!sort || !value) return servers
  return [...servers].sort((left, right) => {
    const a = value(left)
    const b = value(right)
    const order =
      typeof a === 'number' && typeof b === 'number' ? a - b : String(a).localeCompare(String(b))
    return sort.dir === 'asc' ? order : -order
  })
}

interface MCPServersSectionProps {
  servers: MCPServerState[]
  filteredServers: MCPServerState[]
  serverFilters: ServerFilters
  serverSearch: string
  isReadonly: boolean
  actionLoading: string | null
  activeServerId?: string
  onServerFiltersChange: (filters: ServerFilters) => void
  onServerSearchChange: (value: string) => void
  onOpenServer: (id: string) => void
  onToggleConnection: (server: MCPServerState) => void
  onToggleEnabled: (server: MCPServerState) => void
  onEditServer: (server: MCPServerConfig) => void
  onDeleteServer: (id: string) => void
  onAddServer: () => void
}

export const MCPServersSection: React.FC<MCPServersSectionProps> = ({
  servers,
  filteredServers,
  serverFilters,
  serverSearch,
  isReadonly,
  actionLoading,
  activeServerId,
  onServerFiltersChange,
  onServerSearchChange,
  onOpenServer,
  onToggleConnection,
  onToggleEnabled,
  onEditServer,
  onDeleteServer,
  onAddServer,
}) => {
  const [page, setPage] = useState(1)
  const [sort, setSort] = useState<DataTableSort | null>(null)
  const connectedCount = servers.filter((server) => server.status === 'connected').length
  const sortedServers = useMemo(() => sortServers(filteredServers, sort), [filteredServers, sort])
  const pageCount = getMCPPageCount(sortedServers.length, MCP_SERVERS_PAGE_SIZE)
  const visibleServers = paginateMCPItems(sortedServers, page, MCP_SERVERS_PAGE_SIZE)
  const filtered =
    serverSearch.trim() !== '' ||
    serverFilters.transports.length > 0 ||
    serverFilters.statuses.length > 0

  useEffect(() => setPage(1), [serverFilters, serverSearch, sort])
  useEffect(() => {
    if (page > pageCount) setPage(pageCount)
  }, [page, pageCount])

  const columns: Column<MCPServerState>[] = [
    {
      id: 'name',
      header: 'Server',
      sortable: true,
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
      sortable: true,
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
      sortable: true,
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
                <FilterPill
                  label="Transport"
                  selected={serverFilters.transports}
                  onChange={(transports) => onServerFiltersChange({ ...serverFilters, transports })}
                  options={Object.entries(TRANSPORT_META).map(([value, meta]) => ({
                    value,
                    label: meta.label,
                    count: servers.filter((server) => server.config.transport === value).length,
                  }))}
                />
                <FilterPill
                  label="Status"
                  selected={serverFilters.statuses}
                  onChange={(statuses) => onServerFiltersChange({ ...serverFilters, statuses })}
                  options={(['connected', 'connecting', 'error', 'disconnected'] as const).map(
                    (value) => ({
                      value,
                      label: SERVER_STATUS_META[value].label,
                      count: servers.filter((server) => server.status === value).length,
                    }),
                  )}
                />
              </>
            ) : null
          }
          actions={
            <span className="text-xs text-muted-foreground">
              <span className="tabular">{connectedCount}</span> of{' '}
              <span className="tabular">{servers.length}</span> connected
            </span>
          }
        />
        <DataTable
          data={visibleServers}
          columns={columns}
          getRowId={(row) => row.config.id}
          sort={sort}
          onSortChange={setSort}
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
                  ? 'Clear the search or filters to see every server.'
                  : isReadonly
                    ? 'No MCP servers are configured.'
                    : 'Add a server to give decisions tools such as search, files or calendars.'
              }
              actions={
                servers.length && filtered ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      onServerSearchChange('')
                      onServerFiltersChange({ transports: [], statuses: [] })
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
      {sortedServers.length > MCP_SERVERS_PAGE_SIZE ? (
        <MCPListPagination
          itemCount={sortedServers.length}
          itemLabel="servers"
          page={page}
          pageSize={MCP_SERVERS_PAGE_SIZE}
          onPageChange={setPage}
        />
      ) : null}
    </section>
  )
}
