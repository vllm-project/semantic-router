import React, { useState } from 'react'
import { ChevronRight } from 'lucide-react'

import { Badge, cn } from '../ui'

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
  onToggle,
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)

  const handleToggle = () => {
    const newExpanded = !isExpanded
    setIsExpanded(newExpanded)
    onToggle?.(newExpanded)
  }

  return (
    <div className="w-full">
      <button
        type="button"
        id={`collapsible-header-${id}`}
        className="flex w-full items-center gap-2 rounded-md border border-border bg-surface-subtle px-3 py-2 text-left text-[13px] font-medium text-foreground outline-none transition-colors hover:border-border-strong hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/40"
        onClick={handleToggle}
        aria-expanded={isExpanded}
        aria-controls={`collapsible-content-${id}`}
      >
        <ChevronRight
          aria-hidden="true"
          className={cn(
            'size-3.5 shrink-0 text-subtle-foreground transition-transform duration-200',
            isExpanded && 'rotate-90',
          )}
        />
        <span className="flex-1">
          {isExpanded ? 'Hide' : 'Show'} {title}
        </span>
        {isTruncated && (
          <Badge tone="warning" size="sm">
            Truncated
          </Badge>
        )}
      </button>
      {isExpanded && (
        <div
          id={`collapsible-content-${id}`}
          className="mt-2 animate-in fade-in-0 slide-in-from-top-1 duration-200"
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
