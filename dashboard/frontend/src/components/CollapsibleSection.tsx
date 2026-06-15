import React, { useState } from 'react'
import styles from './CollapsibleSection.module.css'

interface CollapsibleSectionProps {
  id: string
  title: string
  content: React.ReactNode
  isTruncated?: boolean
  defaultExpanded?: boolean
  onToggle?: (isExpanded: boolean) => void
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  id,
  title,
  content,
  isTruncated = false,
  defaultExpanded = false,
  onToggle
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)

  const handleToggle = () => {
    const newExpanded = !isExpanded
    setIsExpanded(newExpanded)
    onToggle?.(newExpanded)
  }

  return (
    <div className={styles.container}>
      <button
        className={styles.header}
        onClick={handleToggle}
        aria-expanded={isExpanded}
        aria-controls={`collapsible-content-${id}`}
      >
        <span className={`${styles.icon} ${isExpanded ? styles.iconExpanded : ''}`}>
          ▶
        </span>
        <span className={styles.title}>
          {isExpanded ? 'Hide' : 'Show'} {title}
        </span>
        {isTruncated && (
          <span className={styles.truncatedBadge}>Truncated</span>
        )}
      </button>
      {isExpanded && (
        <div
          id={`collapsible-content-${id}`}
          className={styles.content}
          role="region"
          aria-labelledby={`collapsible-header-${id}`}
        >
          {content}
        </div>
      )}
    </div>
  )
}

export default CollapsibleSection
