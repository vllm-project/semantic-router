import React from 'react'
import { Plus, Search } from 'lucide-react'

import { Badge, Button, InputGroup, cn } from '../ui'

interface TableHeaderProps {
  title: string
  icon?: string
  count?: number
  searchPlaceholder?: string
  searchValue?: string
  onSearchChange?: (value: string) => void
  onSecondaryAction?: () => void
  secondaryActionText?: string
  onAdd?: () => void
  addButtonText?: string
  disabled?: boolean
  variant?: 'default' | 'embedded'
}

const TableHeader: React.FC<TableHeaderProps> = ({
  title,
  icon,
  count,
  searchPlaceholder = 'Search...',
  searchValue = '',
  onSearchChange,
  onSecondaryAction,
  secondaryActionText = 'More',
  onAdd,
  addButtonText = 'Add New',
  disabled = false,
  variant = 'default',
}) => {
  const showSearch = Boolean(onSearchChange && searchPlaceholder.trim().length > 0)

  return (
    <div
      className={cn(
        'flex flex-wrap items-center justify-between gap-x-4 gap-y-2.5',
        variant === 'embedded' ? 'mb-3' : 'mb-4',
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        {icon ? (
          <span className="shrink-0 text-base leading-none" aria-hidden="true">
            {icon}
          </span>
        ) : null}
        <h3 className="truncate text-[15px] font-semibold tracking-[-0.012em] text-foreground">
          {title}
        </h3>
        {count !== undefined ? (
          <Badge tone="neutral" size="sm" className="tabular">
            {count} {count === 1 ? 'item' : 'items'}
          </Badge>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {showSearch ? (
          <InputGroup
            type="search"
            aria-label={searchPlaceholder}
            placeholder={searchPlaceholder}
            value={searchValue}
            onChange={(e) => onSearchChange?.(e.target.value)}
            leading={<Search />}
            containerClassName="w-full sm:w-64"
            className="h-8 [&::-webkit-search-cancel-button]:cursor-pointer"
          />
        ) : null}
        {onSecondaryAction && !disabled ? (
          <Button type="button" variant="outline" size="sm" onClick={onSecondaryAction}>
            {secondaryActionText}
          </Button>
        ) : null}
        {onAdd && !disabled ? (
          <Button type="button" size="sm" onClick={onAdd}>
            <Plus />
            {addButtonText}
          </Button>
        ) : null}
      </div>
    </div>
  )
}

export default TableHeader
