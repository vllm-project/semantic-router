// ControlPanel/TopologyLegend.tsx - Legend component

import React from 'react'
import { SIGNAL_COLORS, NODE_COLORS, EDGE_COLORS } from '../../constants'
import styles from './ControlPanel.module.css'

export const TopologyLegend: React.FC = () => {
  return (
    <div className={styles.section}>
      <span className={styles.sectionTitle}>Legend</span>
      <div className={styles.legend}>
        {/* Node Types */}
        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: NODE_COLORS.client.background }}></span>
          <span>User Input</span>
        </div>

        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: SIGNAL_COLORS.keyword.background }}></span>
          <span>Signal Group</span>
        </div>

        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: NODE_COLORS.decision.normal.background }}></span>
          <span>Decision</span>
        </div>

        <div className={styles.legendItem}>
          <span 
            className={styles.legendColor} 
            style={{ 
              background: NODE_COLORS.decision.unreachable.background,
              border: '1px dashed #5D4037',
            }}
          ></span>
          <span>Unreachable</span>
        </div>

        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: NODE_COLORS.model.standard.background }}></span>
          <span>Model</span>
        </div>

        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: NODE_COLORS.model.reasoning.background }}></span>
          <span>Reasoning Model</span>
        </div>

        {/* Edge Types */}
        <div className={styles.legendItem}>
          <div
            className={styles.legendLine}
            style={{ background: EDGE_COLORS.reasoning }}
          ></div>
          <span>Reasoning Path</span>
        </div>

        <div className={styles.legendItem}>
          <div className={styles.legendLineDashed}></div>
          <span>Standard Path</span>
        </div>

        <div className={styles.legendItem}>
          <div
            className={styles.legendLine}
            style={{ background: EDGE_COLORS.highlighted }}
          ></div>
          <span>Highlighted</span>
        </div>
      </div>
    </div>
  )
}
