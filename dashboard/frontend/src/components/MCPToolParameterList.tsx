import React from 'react'
import styles from './MCPConfigPanel.module.css'
import type { UnifiedToolParameter } from './mcpConfigPanelTypes'

interface MCPToolParameterListProps {
  parameters: UnifiedToolParameter[]
}

export const MCPToolParameterList: React.FC<MCPToolParameterListProps> = ({ parameters }) => {
  if (parameters.length === 0) {
    return <span className={styles.noParams}>No parameters</span>
  }

  return (
    <div className={styles.paramsList}>
      {parameters.map(parameter => (
        <div key={parameter.name} className={styles.paramItem}>
          <span className={styles.paramName}>{parameter.name}</span>
          <span className={styles.paramType}>({parameter.type})</span>
          {parameter.required && <span className={styles.paramRequired}>*</span>}
          {parameter.description && (
            <span className={styles.paramDesc}>{parameter.description}</span>
          )}
        </div>
      ))}
    </div>
  )
}
