import React from 'react'
import styles from './MCPConfigPanel.module.css'
import type { UnifiedTool } from './mcpConfigPanelTypes'
import { TOOLS_GRID_MAX_HEIGHT } from './mcpConfigPanelTypes'

interface MCPAvailableToolsSectionProps {
  allAvailableTools: UnifiedTool[]
  filteredTools: UnifiedTool[]
  toolSearch: string
  toolsSectionExpanded: boolean
  onSearchChange: (value: string) => void
  onSelectTool: (tool: UnifiedTool) => void
  onToggleExpanded: () => void
}

export const MCPAvailableToolsSection: React.FC<MCPAvailableToolsSectionProps> = ({
  allAvailableTools,
  filteredTools,
  toolSearch,
  toolsSectionExpanded,
  onSearchChange,
  onSelectTool,
  onToggleExpanded,
}) => {
  return (
    <div className={styles.availableToolsSection}>
      <div className={styles.sectionHeader} onClick={onToggleExpanded}>
        <div className={styles.sectionTitle}>
          <span className={styles.expandIcon}>{toolsSectionExpanded ? '▼' : '▶'}</span>
          <span>🧰 Available Tools</span>
          <span className={styles.toolCountBadge}>{allAvailableTools.length} tools</span>
        </div>
        {toolsSectionExpanded && allAvailableTools.length > 0 && (
          <div className={styles.toolSearchWrapper} onClick={event => event.stopPropagation()}>
            <input
              type="text"
              className={styles.toolSearchInput}
              placeholder="🔍 Search tools..."
              value={toolSearch}
              onChange={event => onSearchChange(event.target.value)}
            />
            {toolSearch && (
              <button className={styles.clearSearchBtn} onClick={() => onSearchChange('')}>
                ×
              </button>
            )}
          </div>
        )}
      </div>

      {toolsSectionExpanded && (
        <>
          {allAvailableTools.length === 0 && (
            <div className={styles.noToolsAvailable}>
              <span className={styles.emptyIcon}>🔧</span>
              <p>No tools available</p>
              <span className={styles.emptyHint}>
                Connect an MCP server or add built-in tools to see them here
              </span>
            </div>
          )}

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
                      onClick={() => onSelectTool(tool)}
                    >
                      <div className={styles.toolGridHeader}>
                        <span className={styles.toolGridIcon}>🔧</span>
                        <span className={styles.toolGridName}>{tool.name}</span>
                      </div>
                      <div className={styles.toolGridDesc}>
                        {tool.description.length > 60
                          ? `${tool.description.slice(0, 60)}...`
                          : tool.description || 'No description'}
                      </div>
                      <div className={styles.toolGridFooter}>
                        <span className={`${styles.sourceTypeBadge} ${styles[tool.sourceType]}`}>
                          {tool.sourceType === 'mcp'
                            ? '🔌 MCP'
                            : tool.sourceType === 'frontend'
                              ? '⚡ Frontend'
                              : '🌐 Backend'}
                        </span>
                        <span className={styles.sourceNameBadge}>{tool.source}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {filteredTools.length > 0 && (
                <div className={styles.toolsGridFooter}>
                  <span className={styles.toolCountInfo}>
                    {toolSearch
                      ? `${filteredTools.length} of ${allAvailableTools.length} tools`
                      : `${allAvailableTools.length} tools available`}
                  </span>
                  {filteredTools.length > 6 && (
                    <span className={styles.scrollHint}>↕ Scroll to see more</span>
                  )}
                </div>
              )}

              {filteredTools.length === 0 && toolSearch && (
                <div className={styles.noToolsFound}>No matching tools found</div>
              )}
            </>
          )}
        </>
      )}
    </div>
  )
}
