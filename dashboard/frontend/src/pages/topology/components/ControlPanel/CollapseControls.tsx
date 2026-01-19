// ControlPanel/CollapseControls.tsx - Expand/Collapse all controls

import React from 'react'
import styles from './ControlPanel.module.css'

interface CollapseControlsProps {
  onExpandAll: () => void
  onCollapseAll: () => void
}

export const CollapseControls: React.FC<CollapseControlsProps> = ({
  onExpandAll,
  onCollapseAll,
}) => {
  return (
    <div className={styles.section}>
      <span className={styles.sectionTitle}>View Controls</span>
      <div className={styles.collapseControls}>
        <button
          className={styles.collapseBtn}
          onClick={onExpandAll}
          title="Expand All"
        >
          ➕ Expand All
        </button>
        <button
          className={styles.collapseBtn}
          onClick={onCollapseAll}
          title="Collapse All"
        >
          ➖ Collapse All
        </button>
      </div>
    </div>
  )
}
