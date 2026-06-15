import React from 'react'
import styles from './TableHeader.module.css'

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
    <div className={`${styles.header} ${variant === 'embedded' ? styles.embedded : ''}`}>
      <div className={styles.titleSection}>
        {icon && <span className={styles.icon}>{icon}</span>}
        <h3 className={styles.title}>{title}</h3>
        {count !== undefined && (
          <span className={styles.badge}>{count} {count === 1 ? 'item' : 'items'}</span>
        )}
      </div>
      <div className={styles.actions}>
        {showSearch && (
          <input
            type="search"
            className={styles.searchInput}
            placeholder={searchPlaceholder}
            value={searchValue}
            onChange={(e) => onSearchChange?.(e.target.value)}
          />
        )}
        {onSecondaryAction && !disabled && (
          <button className={styles.secondaryButton} onClick={onSecondaryAction}>
            {secondaryActionText}
          </button>
        )}
        {onAdd && !disabled && (
          <button className={styles.addButton} onClick={onAdd}>
            {addButtonText}
          </button>
        )}
      </div>
    </div>
  )
}

export default TableHeader
