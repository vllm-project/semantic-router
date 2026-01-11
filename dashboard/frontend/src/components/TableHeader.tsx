import React from 'react'
import styles from './TableHeader.module.css'

interface TableHeaderProps {
  title: string
  icon?: string
  count?: number
  searchPlaceholder?: string
  searchValue?: string
  onSearchChange?: (value: string) => void
  onAdd?: () => void
  addButtonText?: string
}

const TableHeader: React.FC<TableHeaderProps> = ({
  title,
  icon,
  count,
  searchPlaceholder = 'Search...',
  searchValue = '',
  onSearchChange,
  onAdd,
  addButtonText = 'Add New'
}) => {
  return (
    <div className={styles.header}>
      <div className={styles.titleSection}>
        {icon && <span className={styles.icon}>{icon}</span>}
        <h3 className={styles.title}>{title}</h3>
        {count !== undefined && (
          <span className={styles.badge}>{count} {count === 1 ? 'item' : 'items'}</span>
        )}
      </div>
      <div className={styles.actions}>
        {onSearchChange && (
          <input
            type="search"
            className={styles.searchInput}
            placeholder={searchPlaceholder}
            value={searchValue}
            onChange={(e) => onSearchChange(e.target.value)}
          />
        )}
        {onAdd && (
          <button className={styles.addButton} onClick={onAdd}>
            {addButtonText}
          </button>
        )}
      </div>
    </div>
  )
}

export default TableHeader

