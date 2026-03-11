import React from 'react'
import styles from './MCPConfigPanel.module.css'
import type { UnifiedTool } from './mcpConfigPanelTypes'

interface MCPToolDetailModalProps {
  tool: UnifiedTool
  onClose: () => void
}

export const MCPToolDetailModal: React.FC<MCPToolDetailModalProps> = ({
  tool,
  onClose,
}) => {
  return (
    <div className={styles.dialogOverlay} onClick={onClose}>
      <div className={styles.toolDetailDialog} onClick={event => event.stopPropagation()}>
        <div className={styles.dialogHeader}>
          <div className={styles.toolDetailTitle}>
            <span className={styles.toolDetailIcon}>🔧</span>
            <h3>{tool.name}</h3>
            <span className={`${styles.sourceTypeBadge} ${styles[tool.sourceType]}`}>
              {tool.sourceType === 'mcp'
                ? '🔌 MCP'
                : tool.sourceType === 'frontend'
                  ? '⚡ Frontend'
                  : '🌐 Backend'}
            </span>
          </div>
          <button className={styles.closeBtn} onClick={onClose}>×</button>
        </div>

        <div className={styles.toolDetailContent}>
          <div className={styles.toolDetailSource}>
            <span className={styles.detailLabel}>Source:</span>
            <span className={styles.detailValue}>{tool.source}</span>
          </div>

          <div className={styles.toolDetailDescription}>
            <span className={styles.detailLabel}>Description:</span>
            <p>{tool.description || 'No description'}</p>
          </div>

          <div className={styles.toolDetailParams}>
            <span className={styles.detailLabel}>Parameters ({tool.parameters.length}):</span>
            {tool.parameters.length === 0 ? (
              <p className={styles.noParamsHint}>This tool requires no parameters</p>
            ) : (
              <div className={styles.paramDetailList}>
                {tool.parameters.map(parameter => (
                  <div key={parameter.name} className={styles.paramDetailItem}>
                    <div className={styles.paramDetailHeader}>
                      <span className={styles.paramDetailName}>{parameter.name}</span>
                      <span className={styles.paramDetailType}>({parameter.type})</span>
                      {parameter.required && (
                        <span className={styles.paramDetailRequired}>Required</span>
                      )}
                    </div>
                    {parameter.description && (
                      <div className={styles.paramDetailDesc}>{parameter.description}</div>
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
