import React, { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Box, Blocks, Server, Wrench } from 'lucide-react'

import { Badge, Button, Card, DataToolbar, EmptyState, IconTile, Segmented } from '../ui'
import { MCPListPagination } from './MCPListPagination'
import type { ToolSourceFilter, UnifiedTool } from './mcpConfigPanelTypes'
import { MCP_TOOLS_PAGE_SIZE } from './mcpConfigPanelTypes'
import {
  TOOL_SOURCE_META,
  getMCPPageCount,
  paginateMCPItems,
  toolIcon,
} from './mcpConfigPanelUtils'

interface MCPAvailableToolsSectionProps {
  allAvailableTools: UnifiedTool[]
  filteredTools: UnifiedTool[]
  toolSearch: string
  toolSourceFilter: ToolSourceFilter
  onSearchChange: (value: string) => void
  onSelectTool: (tool: UnifiedTool) => void
  onSourceFilterChange: (filter: ToolSourceFilter) => void
}

const SOURCE_ICON = { mcp: Blocks, frontend: Box, backend: Server } as const

function ToolCard({ tool, onSelect }: { tool: UnifiedTool; onSelect: () => void }) {
  const Icon = toolIcon(tool.name)
  const source = TOOL_SOURCE_META[tool.sourceType]
  const SourceIcon = SOURCE_ICON[tool.sourceType]
  return (
    <Card
      interactive
      role="button"
      tabIndex={0}
      aria-label={`View details for ${tool.name}`}
      onClick={onSelect}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onSelect()
        }
      }}
      className="flex h-full w-full min-w-0 flex-col gap-3 p-4 outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
    >
      <IconTile icon={Icon} tone="slate" />
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="truncate font-mono text-[12.5px] font-semibold">{tool.name}</span>
        <p className="line-clamp-2 min-h-8 text-xs leading-relaxed text-muted-foreground">
          {tool.description || 'No description'}
        </p>
      </div>
      <div className="mt-auto flex items-center gap-1.5 border-t border-border pt-3">
        <Badge tone={source.tone} className="min-w-0">
          <SourceIcon />
          <span className="truncate">
            {tool.sourceType === 'mcp' ? `${source.label} · ${tool.source}` : source.label}
          </span>
        </Badge>
        <span className="ml-auto shrink-0 text-[11px] tabular text-subtle-foreground">
          {tool.parameters.length} {tool.parameters.length === 1 ? 'param' : 'params'}
        </span>
      </div>
    </Card>
  )
}

export const MCPAvailableToolsSection: React.FC<MCPAvailableToolsSectionProps> = ({
  allAvailableTools,
  filteredTools,
  toolSearch,
  toolSourceFilter,
  onSearchChange,
  onSelectTool,
  onSourceFilterChange,
}) => {
  const [page, setPage] = useState(1)
  const pageCount = getMCPPageCount(filteredTools.length, MCP_TOOLS_PAGE_SIZE)
  const visibleTools = paginateMCPItems(filteredTools, page, MCP_TOOLS_PAGE_SIZE)
  const count = (source: ToolSourceFilter) =>
    source === 'all'
      ? allAvailableTools.length
      : allAvailableTools.filter((tool) => tool.sourceType === source).length

  useEffect(() => setPage(1), [toolSearch, toolSourceFilter])
  useEffect(() => {
    if (page > pageCount) setPage(pageCount)
  }, [page, pageCount])

  return (
    <section className="@container flex flex-col gap-3" aria-labelledby="mcp-tools-title">
      <div className="flex flex-col gap-0.5">
        <h3 id="mcp-tools-title" className="text-[15px] font-semibold tracking-[-0.012em]">
          Available tools
        </h3>
        <p className="text-[13px] text-muted-foreground">
          Tools decisions can call. Built-in tools run in the dashboard; router tools come from the
          tools database.
        </p>
      </div>
      <DataToolbar
        search={toolSearch}
        onSearchChange={onSearchChange}
        placeholder="Search tools…"
        className="mb-0"
        filters={
          <>
            <Segmented
              size="sm"
              aria-label="Source"
              value={toolSourceFilter}
              onValueChange={onSourceFilterChange}
              options={[
                { value: 'all', label: `All · ${count('all')}` },
                { value: 'mcp', label: `MCP · ${count('mcp')}` },
                { value: 'frontend', label: `Built-in · ${count('frontend')}` },
                { value: 'backend', label: `Router · ${count('backend')}` },
              ]}
            />
          </>
        }
      />
      {filteredTools.length ? (
        <div className="grid grid-cols-1 gap-3 @[30rem]:grid-cols-2 @[40rem]:grid-cols-3 @[50rem]:grid-cols-4">
          <AnimatePresence initial={false} mode="popLayout">
            {visibleTools.map((tool) => (
              <motion.div
                key={tool.id}
                layout
                initial={{ opacity: 0, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.97 }}
                transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                className="flex min-w-0"
              >
                <ToolCard tool={tool} onSelect={() => onSelectTool(tool)} />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      ) : (
        <Card>
          <EmptyState
            compact
            icon={Wrench}
            tone="slate"
            title={allAvailableTools.length ? 'No tools match' : 'No tools available'}
            description={
              allAvailableTools.length
                ? 'No tools match the current search and source filter.'
                : 'Connect an MCP server or add built-in tools to see them here.'
            }
            actions={
              allAvailableTools.length ? (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    onSearchChange('')
                    onSourceFilterChange('all')
                  }}
                >
                  Clear filters
                </Button>
              ) : null
            }
          />
        </Card>
      )}
      {filteredTools.length > 0 ? (
        <MCPListPagination
          itemCount={filteredTools.length}
          itemLabel="tools"
          page={page}
          pageSize={MCP_TOOLS_PAGE_SIZE}
          onPageChange={setPage}
        />
      ) : null}
    </section>
  )
}
