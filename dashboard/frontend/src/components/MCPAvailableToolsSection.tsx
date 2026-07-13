import React, { useEffect, useId, useState } from 'react'

import styles from './MCPConfigPanel.module.css'
import { MCPListPagination } from './MCPListPagination'
import type { ToolSort, ToolSourceFilter, UnifiedTool } from './mcpConfigPanelTypes'
import { MCP_TOOLS_PAGE_SIZE } from './mcpConfigPanelTypes'
import { getMCPPageCount, paginateMCPItems } from './mcpConfigPanelUtils'

interface MCPAvailableToolsSectionProps {
  allAvailableTools: UnifiedTool[]
  filteredTools: UnifiedTool[]
  toolSearch: string
  toolSort: ToolSort
  toolSourceFilter: ToolSourceFilter
  toolsSectionExpanded: boolean
  onSearchChange: (value: string) => void
  onSelectTool: (tool: UnifiedTool) => void
  onSortChange: (sort: ToolSort) => void
  onSourceFilterChange: (filter: ToolSourceFilter) => void
  onToggleExpanded: () => void
}

export const MCPAvailableToolsSection: React.FC<MCPAvailableToolsSectionProps> = ({
  allAvailableTools,
  filteredTools,
  toolSearch,
  toolSort,
  toolSourceFilter,
  toolsSectionExpanded,
  onSearchChange,
  onSelectTool,
  onSortChange,
  onSourceFilterChange,
  onToggleExpanded,
}) => {
  const contentId = useId()
  const searchInputId = useId()
  const [page, setPage] = useState(1)
  const pageCount = getMCPPageCount(filteredTools.length, MCP_TOOLS_PAGE_SIZE)
  const visibleTools = paginateMCPItems(filteredTools, page, MCP_TOOLS_PAGE_SIZE)

  useEffect(() => setPage(1), [toolSearch, toolSort, toolSourceFilter])
  useEffect(() => {
    if (page > pageCount) setPage(pageCount)
  }, [page, pageCount])

  return (
    <section className={styles.availableToolsSection}>
      <div className={styles.sectionHeader}>
        <button
          type="button"
          className={styles.sectionToggle}
          onClick={onToggleExpanded}
          aria-expanded={toolsSectionExpanded}
          aria-controls={contentId}
        >
          <span className={styles.expandIcon} aria-hidden="true">
            {toolsSectionExpanded ? '▼' : '▶'}
          </span>
          <span>Available Tools</span>
          <span className={styles.toolCountBadge}>{allAvailableTools.length} tools</span>
        </button>
      </div>

      {toolsSectionExpanded ? (
        <div id={contentId}>
          {allAvailableTools.length === 0 ? (
            <div className={styles.noToolsAvailable}>
              <span className={styles.emptyIcon} aria-hidden="true">
                🔧
              </span>
              <p>No tools available</p>
              <span className={styles.emptyHint}>
                Connect an MCP server or add built-in tools to see them here.
              </span>
            </div>
          ) : (
            <>
              <div className={styles.catalogControls}>
                <div className={styles.searchField}>
                  <label htmlFor={searchInputId}>Search tools</label>
                  <div className={styles.toolSearchWrapper}>
                    <input
                      id={searchInputId}
                      type="search"
                      className={styles.toolSearchInput}
                      placeholder="Name, source, or parameter"
                      value={toolSearch}
                      onChange={(event) => onSearchChange(event.target.value)}
                    />
                    {toolSearch ? (
                      <button
                        type="button"
                        className={styles.clearSearchBtn}
                        onClick={() => onSearchChange('')}
                        aria-label="Clear tool search"
                      >
                        ×
                      </button>
                    ) : null}
                  </div>
                </div>
                <label className={styles.catalogSelect}>
                  <span>Source</span>
                  <select
                    value={toolSourceFilter}
                    onChange={(event) =>
                      onSourceFilterChange(event.target.value as ToolSourceFilter)
                    }
                  >
                    <option value="all">All sources</option>
                    <option value="mcp">MCP servers</option>
                    <option value="frontend">Frontend</option>
                    <option value="backend">Semantic Router</option>
                  </select>
                </label>
                <label className={styles.catalogSelect}>
                  <span>Sort</span>
                  <select
                    value={toolSort}
                    onChange={(event) => onSortChange(event.target.value as ToolSort)}
                  >
                    <option value="name-asc">Name A–Z</option>
                    <option value="source-asc">Source A–Z</option>
                    <option value="parameters-desc">Most parameters</option>
                  </select>
                </label>
              </div>

              <div className={styles.catalogMeta} aria-live="polite">
                <strong>{filteredTools.length}</strong> of {allAvailableTools.length} tools
                <span>Client view · {MCP_TOOLS_PAGE_SIZE} per page</span>
              </div>

              {filteredTools.length === 0 ? (
                <div className={styles.noToolsFound}>
                  No tools match the current search and source filter.
                </div>
              ) : (
                <>
                  <div className={styles.toolsGridWrapper}>
                    <div className={styles.toolsGrid}>
                      {visibleTools.map((tool) => (
                        <button
                          type="button"
                          key={tool.id}
                          className={`${styles.toolGridCard} ${styles[`source_${tool.sourceType}`]}`}
                          onClick={() => onSelectTool(tool)}
                          aria-label={`View details for ${tool.name}`}
                        >
                          <span className={styles.toolGridHeader}>
                            <span className={styles.toolGridIcon} aria-hidden="true">
                              🔧
                            </span>
                            <span className={styles.toolGridName}>{tool.name}</span>
                          </span>
                          <span className={styles.toolGridDesc}>
                            {tool.description || 'No description'}
                          </span>
                          <span className={styles.toolCardMeta}>
                            <span
                              className={`${styles.sourceTypeBadge} ${styles[tool.sourceType]}`}
                            >
                              {tool.sourceType === 'mcp'
                                ? 'MCP'
                                : tool.sourceType === 'frontend'
                                  ? 'Frontend'
                                  : 'Backend'}
                            </span>
                            <span className={styles.sourceNameBadge}>{tool.source}</span>
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                  <MCPListPagination
                    itemCount={filteredTools.length}
                    itemLabel="tools"
                    page={page}
                    pageSize={MCP_TOOLS_PAGE_SIZE}
                    onPageChange={setPage}
                  />
                </>
              )}
            </>
          )}
        </div>
      ) : null}
    </section>
  )
}
