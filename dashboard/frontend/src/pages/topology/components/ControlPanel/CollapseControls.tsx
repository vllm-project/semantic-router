// ControlPanel/CollapseControls.tsx - Expand/Collapse all controls

import React from 'react'
import ProductIcon from '../../../../components/ProductIcon'
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
    <div className={styles.collapseControls}>
      <button className={styles.collapseBtn} onClick={onExpandAll} title="Expand all">
        <ProductIcon name="expand" aria-hidden="true" /> Expand all
      </button>
      <button className={styles.collapseBtn} onClick={onCollapseAll} title="Collapse all">
        <ProductIcon name="minus" aria-hidden="true" /> Collapse all
      </button>
    </div>
  )
}
