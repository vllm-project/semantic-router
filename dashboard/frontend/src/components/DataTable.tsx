import React, { useState } from 'react'
import styles from './DataTable.module.css'

export interface Column<T> {
  key: string
  header: string
  width?: string
  align?: 'left' | 'center' | 'right'
  render?: (row: T) => React.ReactNode
  sortable?: boolean
}

export interface DataTableProps<T> {
  columns: Column<T>[]
  data: T[]
  keyExtractor: (row: T) => string
  onView?: (row: T) => void
  onEdit?: (row: T) => void
  onDelete?: (row: T) => void
  expandable?: boolean
  renderExpandedRow?: (row: T) => React.ReactNode
  isRowExpanded?: (row: T) => boolean
  onToggleExpand?: (row: T) => void
  emptyMessage?: string
  className?: string
  readonly?: boolean
}

export function DataTable<T>({
  columns,
  data,
  keyExtractor,
  onView,
  onEdit,
  onDelete,
  expandable = false,
  renderExpandedRow,
  isRowExpanded,
  onToggleExpand,
  emptyMessage = 'No data available',
  className = '',
  readonly = false
}: DataTableProps<T>) {
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')

  // In readonly mode, disable edit and delete actions
  const effectiveOnEdit = readonly ? undefined : onEdit
  const effectiveOnDelete = readonly ? undefined : onDelete

  const handleSort = (columnKey: string) => {
    if (sortColumn === columnKey) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortColumn(columnKey)
      setSortDirection('asc')
    }
  }

  const sortedData = React.useMemo(() => {
    if (!sortColumn) return data

    return [...data].sort((a, b) => {
      const aValue = (a as any)[sortColumn]
      const bValue = (b as any)[sortColumn]

      if (aValue === bValue) return 0
      
      const comparison = aValue > bValue ? 1 : -1
      return sortDirection === 'asc' ? comparison : -comparison
    })
  }, [data, sortColumn, sortDirection])

  return (
    <div className={`${styles.tableContainer} ${className}`}>
      <table className={styles.table}>
        <thead className={styles.thead}>
          <tr>
            {expandable && <th className={styles.expandColumn}></th>}
            {columns.map((column) => (
              <th
                key={column.key}
                className={`${styles.th} ${column.sortable ? styles.sortable : ''}`}
                style={{ 
                  width: column.width,
                  textAlign: column.align || 'left'
                }}
                onClick={() => column.sortable && handleSort(column.key)}
              >
                {column.header}
                {column.sortable && sortColumn === column.key && (
                  <span className={styles.sortIcon}>
                    {sortDirection === 'asc' ? ' ↑' : ' ↓'}
                  </span>
                )}
              </th>
            ))}
            {(onView || onEdit || onDelete) && (
              <th className={`${styles.th} ${styles.actionsColumn}`}>Actions</th>
            )}
          </tr>
        </thead>
        <tbody className={styles.tbody}>
          {sortedData.length === 0 ? (
            <tr>
              <td 
                colSpan={columns.length + (expandable ? 1 : 0) + (onView || onEdit || onDelete ? 1 : 0)}
                className={styles.emptyState}
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            sortedData.map((row) => {
              const key = keyExtractor(row)
              const isExpanded = isRowExpanded?.(row) || false

              return (
                <React.Fragment key={key}>
                  <tr className={styles.tr}>
                    {expandable && (
                      <td className={styles.expandCell}>
                        <button
                          className={styles.expandButton}
                          onClick={() => onToggleExpand?.(row)}
                        >
                          <span className={`${styles.expandIcon} ${isExpanded ? styles.expanded : ''}`}>
                            ▶
                          </span>
                        </button>
                      </td>
                    )}
                    {columns.map((column) => (
                      <td
                        key={column.key}
                        className={styles.td}
                        style={{ textAlign: column.align || 'left' }}
                      >
                        {column.render ? column.render(row) : (row as any)[column.key]}
                      </td>
                    ))}
                    {(onView || onEdit || onDelete) && (
                      <td className={`${styles.td} ${styles.actionsCell}`}>
                        <div className={styles.actionButtons}>
                          {onView && (
                            <button
                              className={`${styles.actionButton} ${styles.viewButton}`}
                              onClick={() => onView(row)}
                            >
                              View
                            </button>
                          )}
                          {effectiveOnEdit && (
                            <button
                              className={`${styles.actionButton} ${styles.editButton}`}
                              onClick={() => effectiveOnEdit(row)}
                            >
                              Edit
                            </button>
                          )}
                          {effectiveOnDelete && (
                            <button
                              className={`${styles.actionButton} ${styles.deleteButton}`}
                              onClick={() => effectiveOnDelete(row)}
                            >
                              Delete
                            </button>
                          )}
                        </div>
                      </td>
                    )}
                  </tr>
                  {expandable && isExpanded && renderExpandedRow && (
                    <tr className={styles.expandedRow}>
                      <td colSpan={columns.length + 2}>
                        {renderExpandedRow(row)}
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              )
            })
          )}
        </tbody>
      </table>
    </div>
  )
}

