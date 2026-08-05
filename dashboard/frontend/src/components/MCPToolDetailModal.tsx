import React, { useId } from 'react'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './MCPConfigPanel.module.css'
import type { UnifiedTool } from './mcpConfigPanelTypes'

interface MCPToolDetailModalProps {
  tool: UnifiedTool
  onClose: () => void
}

export const MCPToolDetailModal: React.FC<MCPToolDetailModalProps> = ({ tool, onClose }) => {
  const dialogId = useId()
  const titleId = `${dialogId}-title`
  const descriptionId = `${dialogId}-description`
  const dialogRef = useAccessibleDialog<HTMLDivElement>({ isOpen: true, onClose })

  return (
    <div className={styles.dialogOverlay} role="presentation" onMouseDown={onClose}>
      <div
        ref={dialogRef}
        className={styles.toolDetailDialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className={styles.dialogHeader}>
          <div className={styles.toolDetailTitle}>
            <span className={styles.toolDetailIcon}>🔧</span>
            <h3 id={titleId}>{tool.name}</h3>
            <span className={`${styles.sourceTypeBadge} ${styles[tool.sourceType]}`}>
              {tool.sourceType === 'mcp'
                ? '🔌 MCP'
                : tool.sourceType === 'frontend'
                  ? '⚡ Frontend'
                  : '🌐 Backend'}
            </span>
          </div>
          <button
            type="button"
            className={styles.closeBtn}
            onClick={onClose}
            aria-label="Close tool details"
            data-dialog-initial-focus
          >
            ×
          </button>
        </div>

        <div className={styles.toolDetailContent}>
          <div className={styles.toolDetailSource}>
            <span className={styles.detailLabel}>Source:</span>
            <span className={styles.detailValue}>{tool.source}</span>
          </div>

          <div className={styles.toolDetailDescription}>
            <span className={styles.detailLabel}>Description:</span>
            <p id={descriptionId}>{tool.description || 'No description'}</p>
          </div>

          <div className={styles.toolDetailParams}>
            <span className={styles.detailLabel}>Parameters ({tool.parameters.length}):</span>
            {tool.parameters.length === 0 ? (
              <p className={styles.noParamsHint}>This tool requires no parameters</p>
            ) : (
              <div className={styles.paramDetailList}>
                {tool.parameters.map((parameter) => (
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
          <button type="button" className={styles.cancelBtn} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
