import { memo } from 'react'
import { NodeProps } from 'reactflow'
import styles from './CustomNodes.module.css'

interface MoreDecisionsNodeData {
  hiddenCount: number
  onExpand?: () => void
}

export const MoreDecisionsNode = memo<NodeProps<MoreDecisionsNodeData>>(({ data }) => {
  const { hiddenCount, onExpand } = data

  return (
    <div className={styles.moreDecisionsNode}>
      <div className={styles.moreDecisionsTitle}>+{hiddenCount} more decisions</div>
      <button
        type="button"
        className={styles.moreDecisionsButton}
        onClick={(event) => {
          event.stopPropagation()
          onExpand?.()
        }}
      >
        Expand
      </button>
    </div>
  )
})

MoreDecisionsNode.displayName = 'MoreDecisionsNode'
